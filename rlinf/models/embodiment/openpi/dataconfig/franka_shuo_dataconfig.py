# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data config for Shuo's Franka EE LeRobot datasets (3 RGB + 7D EE).

Mirrors RL-Token ``LeRobotFrankaEEDataConfig`` used by
``pi05_stack_bowls_rc_franka_ee_local``.
"""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import franka_ee_shuo_policy


def _data_config_field_names() -> set[str]:
    return {field.name for field in dataclasses.fields(DataConfig)}


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaShuoDataConfig(DataConfigFactory):
    """LeRobot v2.1 Franka EE: 7D state/actions and three RGB videos.

    Expected raw LeRobot keys: ``global_image``, ``right_image``,
    ``wrist_image``, ``state``, ``actions``, ``frame_index``.
    """

    repo_ids: tuple[str, ...] = ()
    fps: int = 30
    default_prompt: str | None = None
    # Original stage-1 config does not convert abs EE pose to delta.
    extra_delta_transform: bool = False
    # Equalize sampling across concatenated datasets (OpenPI ConcatDataset).
    balance_datasets: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/global_image": "global_image",
                        "observation/right_image": "right_image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "frame_index": "frame_index",
                        "prompt": "prompt",
                        "dataset_root": "dataset_root",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                franka_ee_shuo_policy.FrankaEEShuoInputs(
                    model_type=model_config.model_type,
                    dataset_root=self.repo_id if isinstance(self.repo_id, str) else None,
                    fps=self.fps,
                )
            ],
            outputs=[franka_ee_shuo_policy.FrankaEEShuoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        base = self.create_base_config(assets_dirs, model_config)
        replace_kwargs: dict = {
            "repack_transforms": repack_transform,
            "data_transforms": data_transforms,
            "model_transforms": model_transforms,
            "action_sequence_keys": ("actions",),
        }
        if self.repo_ids:
            replace_kwargs["repo_id"] = self.repo_ids

        field_names = _data_config_field_names()
        if "skip_lerobot_video_decode" in field_names:
            replace_kwargs["skip_lerobot_video_decode"] = True
        if "balance_datasets" in field_names:
            replace_kwargs["balance_datasets"] = self.balance_datasets

        return dataclasses.replace(base, **replace_kwargs)
