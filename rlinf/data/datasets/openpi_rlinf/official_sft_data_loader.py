# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapters for the official OpenPI PyTorch SFT data loader."""

from __future__ import annotations

import dataclasses
from typing import Any

from omegaconf import OmegaConf

from rlinf.config import SupportedModel
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id


def build_official_openpi_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the SFT loader provided by OpenPI for a LeRobot dataset."""
    del rank
    # Prefer torchcodec when FFmpeg libs are present; else PyAV fallback.
    # Cap BLAS/OpenMP threads before workers are spawned.
    _set_single_thread_blas_env()
    from rlinf.data.datasets.openpi_rlinf.pyav_video_patch import (
        apply_pyav_video_decode_patch,
    )
    from rlinf.data.datasets.openpi_rlinf.lerobot_hf_query_patch import (
        apply_lerobot_hf_query_patch,
    )

    apply_pyav_video_decode_patch()
    apply_lerobot_hf_query_patch()

    repo_id = resolve_lerobot_repo_id(data_paths)
    if repo_id is None:
        raise ValueError(
            "OpenPI SFT requires data.train_data_paths to be set to a local "
            "dataset path or LeRobot repo id."
        )

    import openpi.training.data_loader as openpi_data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    model_cfg = cfg.actor.model
    model_type = SupportedModel(model_cfg.model_type)
    batch_size = cfg.actor.micro_batch_size
    if eval_dataset:
        batch_size = cfg.actor.get("eval_batch_size", batch_size)

    config_name = str(model_cfg.openpi.config_name)
    config = get_openpi_config(
        config_name,
        model_path=model_cfg.model_path,
        batch_size=batch_size * world_size,
        repo_id=repo_id,
        data_kwargs=getattr(model_cfg, "openpi_data", None),
    )
    if model_type == SupportedModel.OPENPI_RLINF:
        config = dataclasses.replace(
            config,
            num_workers=int(
                OmegaConf.select(cfg, "data.num_workers", default=config.num_workers)
            ),
            seed=int(OmegaConf.select(cfg, "actor.seed", default=config.seed)),
        )
        _validate_openpi_rlinf_model_shape(model_cfg, config)

    # Spawned DataLoader workers: re-apply decode patch + force thread=1.
    _patch_openpi_dataloader_worker_init(openpi_data_loader)

    # Cobot Magic demos include episode_success=failure; drop those episodes.
    if "cobot" in config_name.lower():
        _patch_create_torch_dataset_drop_failures(openpi_data_loader)

    data_loader = openpi_data_loader.create_data_loader(
        config, framework="pytorch", shuffle=not eval_dataset
    )
    _boost_openpi_torch_dataloader_prefetch(data_loader, prefetch_factor=4)
    return data_loader, data_loader.data_config()


def _set_single_thread_blas_env() -> None:
    """Avoid OpenMP/BLAS oversubscription inside each DataLoader worker."""
    import os

    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"


def _openpi_worker_init_fn_with_pyav(worker_id: int) -> None:
    """Picklable DataLoader worker init: threads=1, decode patch, JAX env."""
    _set_single_thread_blas_env()
    from rlinf.data.datasets.openpi_rlinf.pyav_video_patch import (
        apply_pyav_video_decode_patch,
    )

    apply_pyav_video_decode_patch()
    from rlinf.data.datasets.openpi_rlinf.lerobot_hf_query_patch import (
        apply_lerobot_hf_query_patch,
    )

    apply_lerobot_hf_query_patch()
    # Match openpi.training.data_loader._worker_init_fn (JAX in workers).
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    del worker_id


def _patch_openpi_dataloader_worker_init(openpi_data_loader: Any) -> None:
    """Ensure each DataLoader worker re-applies decode patch and thread caps."""
    if getattr(openpi_data_loader._worker_init_fn, "_rlinf_pyav_patch", False):
        return

    _openpi_worker_init_fn_with_pyav._rlinf_pyav_patch = True  # type: ignore[attr-defined]
    openpi_data_loader._worker_init_fn = _openpi_worker_init_fn_with_pyav


def _boost_openpi_torch_dataloader_prefetch(
    data_loader: Any, *, prefetch_factor: int = 4
) -> None:
    """Rebuild OpenPI's inner torch DataLoader with a deeper prefetch queue."""
    import torch

    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None)
    if torch_loader is None or not isinstance(torch_loader, torch.utils.data.DataLoader):
        return
    if torch_loader.num_workers <= 0:
        return
    if getattr(torch_loader, "_rlinf_prefetch_factor", None) == prefetch_factor:
        return

    common = {
        "dataset": torch_loader.dataset,
        "num_workers": torch_loader.num_workers,
        "collate_fn": torch_loader.collate_fn,
        "pin_memory": True,
        "timeout": torch_loader.timeout,
        "worker_init_fn": torch_loader.worker_init_fn,
        "multiprocessing_context": torch_loader.multiprocessing_context,
        "generator": torch_loader.generator,
        "persistent_workers": True,
        "prefetch_factor": prefetch_factor,
    }
    # Mutually exclusive: batch_sampler vs batch_size/sampler/drop_last.
    if torch_loader.batch_sampler is not None and torch_loader.sampler is None:
        boosted = torch.utils.data.DataLoader(
            batch_sampler=torch_loader.batch_sampler, **common
        )
    else:
        boosted = torch.utils.data.DataLoader(
            batch_size=torch_loader.batch_size,
            shuffle=False,
            sampler=torch_loader.sampler,
            drop_last=torch_loader.drop_last,
            **common,
        )
    boosted._rlinf_prefetch_factor = prefetch_factor  # type: ignore[attr-defined]
    openpi_loader._data_loader = boosted


def _patch_create_torch_dataset_drop_failures(openpi_data_loader: Any) -> None:
    """Wrap OpenPI ``create_torch_dataset`` to keep only success Cobot episodes."""
    if getattr(openpi_data_loader.create_torch_dataset, "_rlinf_cobot_success_only", False):
        return

    import openpi.models.model as openpi_model
    import openpi.training.config as openpi_config
    import openpi.transforms as openpi_transforms

    from rlinf.data.datasets.openpi_rlinf.cobot_episode_filter import (
        list_success_episode_indices_from_meta,
    )

    original = openpi_data_loader.create_torch_dataset

    def create_torch_dataset_success_only(
        data_config: openpi_config.DataConfig,
        action_horizon: int,
        model_config: openpi_model.BaseModelConfig,
    ):
        repo_id = data_config.repo_id
        if repo_id is None:
            raise ValueError("Repo ID is not set. Cannot create dataset.")
        if repo_id == "fake":
            return original(data_config, action_horizon, model_config)

        lerobot_dataset = openpi_data_loader._import_lerobot_dataset()
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        episodes = list_success_episode_indices_from_meta(repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            episodes=episodes,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        if data_config.prompt_from_task:
            dataset = openpi_data_loader.TransformedDataset(
                dataset,
                [openpi_transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
            )
        return dataset

    create_torch_dataset_success_only._rlinf_cobot_success_only = True  # type: ignore[attr-defined]
    openpi_data_loader.create_torch_dataset = create_torch_dataset_success_only


def get_official_openpi_sft_num_batches(data_loader: Any) -> int:
    """Return the inner PyTorch ``DataLoader`` length used by OpenPI."""
    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None) or getattr(
        openpi_loader, "torch_loader", None
    )
    if torch_loader is None:
        raise TypeError(
            "OpenPI dataloader does not expose an inner torch DataLoader; "
            "cannot infer steps per epoch from len()."
        )
    return len(torch_loader)


def is_official_openpi_sft_dataloader(data_loader: Any) -> bool:
    """Return whether ``data_loader`` has OpenPI's loader wrapper layout."""
    return getattr(data_loader, "_data_loader", None) is not None


def _validate_openpi_rlinf_model_shape(model_cfg: Any, openpi_config: Any) -> None:
    """Keep the local Pi0 architecture consistent with the OpenPI config."""
    local_horizon = int(model_cfg.num_action_chunks)
    official_horizon = int(openpi_config.model.action_horizon)
    if local_horizon != official_horizon:
        raise ValueError(
            "openpi_rlinf SFT action horizon must match the official OpenPI "
            f"config: actor.model.num_action_chunks={local_horizon}, "
            f"{model_cfg.openpi.config_name}.model.action_horizon="
            f"{official_horizon}."
        )

    local_action_dim = int(model_cfg.openpi.model_action_dim)
    official_action_dim = int(openpi_config.model.action_dim)
    if local_action_dim != official_action_dim:
        raise ValueError(
            "openpi_rlinf SFT model action dim must match the official OpenPI "
            f"config: actor.model.openpi.model_action_dim={local_action_dim}, "
            f"{model_cfg.openpi.config_name}.model.action_dim="
            f"{official_action_dim}."
        )
