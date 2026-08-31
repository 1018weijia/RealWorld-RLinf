#!/usr/bin/env python3
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
"""Metadata-driven open-loop evaluation for LeRobot policy servers.

The evaluator uses ``LeRobotDataset`` for both v2.1 and v3 datasets, so parquet
rows, independently packed camera videos, and per-episode timestamp offsets are
resolved by the same metadata-driven code used during SFT.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import requests
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "datasets"
INFER_CONFIG_DIR = REPO_ROOT / "toolkits/inference/config"
TRAIN_CONFIG_DIR = REPO_ROOT / "examples/sft/config"
DEFAULT_ACTION_DIMENSIONS = (
    [f"left_joint_{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(6)]
    + ["right_gripper"]
)
XROBOT_ACTION_DIMENSIONS = (
    [f"left_{name}" for name in ("x", "y", "z", "roll", "pitch", "yaw")]
    + ["left_gripper"]
    + [f"right_{name}" for name in ("x", "y", "z", "roll", "pitch", "yaw")]
    + ["right_gripper"]
)


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """A dataset/server mapping for one open-loop task."""

    name: str
    robot: str
    prompt: str
    port: int
    dataset: str
    cameras: dict[str, str]
    infer_config: str | None = None
    train_config: str | None = None
    state_indices: tuple[int, ...] | None = None
    action_indices: tuple[int, ...] | None = None
    action_dimensions: tuple[str, ...] | None = None

    @classmethod
    def from_mapping(cls, name: str, raw: dict[str, Any]) -> TaskSpec:
        required = ("robot", "prompt", "port", "dataset", "cameras")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"task {name!r} is missing required keys: {missing}")
        cameras = dict(raw["cameras"])
        if set(cameras) != {"high", "left_wrist", "right_wrist"}:
            raise ValueError(
                f"task {name!r} cameras must define high, left_wrist, right_wrist"
            )
        state_indices = raw.get("state_indices")
        action_indices = raw.get("action_indices")
        action_dimensions = raw.get("action_dimensions")
        return cls(
            name=name,
            robot=str(raw["robot"]),
            prompt=str(raw["prompt"]),
            port=int(raw["port"]),
            dataset=str(raw["dataset"]),
            cameras={str(key): str(value) for key, value in cameras.items()},
            infer_config=raw.get("infer_config"),
            train_config=raw.get("train_config"),
            state_indices=tuple(int(i) for i in state_indices)
            if state_indices is not None
            else None,
            action_indices=tuple(int(i) for i in action_indices)
            if action_indices is not None
            else None,
            action_dimensions=tuple(str(x) for x in action_dimensions)
            if action_dimensions is not None
            else None,
        )


_DOBOT_CAMERAS = {
    "high": "observation.images.top",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}
_XROBOT_CAMERAS = {
    "high": "observation.images.head",
    "left_wrist": "observation.images.left_arm",
    "right_wrist": "observation.images.right_arm",
}
_COBOT_CAMERAS = {
    "high": "observation.images.cam_high",
    "left_wrist": "observation.images.cam_left_wrist",
    "right_wrist": "observation.images.cam_right_wrist",
}


def _preset(
    name: str,
    robot: str,
    prompt: str,
    port: int,
    dataset: str,
    cameras: dict[str, str],
    infer_config: str | None = None,
    train_config: str | None = None,
    **kwargs: Any,
) -> TaskSpec:
    return TaskSpec(
        name=name,
        robot=robot,
        prompt=prompt,
        port=port,
        dataset=dataset,
        cameras=cameras,
        infer_config=infer_config,
        train_config=train_config,
        **kwargs,
    )


TASKS = {
    "cook": _preset(
        "cook",
        "dobot",
        "cook vegetable",
        8010,
        "Dobot/NormalData/dobot_cook_vegetable_fullV30",
        _DOBOT_CAMERAS,
        "dobot_cook_vegetable.yaml",
        "dobot_sft_openpi_rlinf_pi05_cook_vegetable.yaml",
        action_dimensions=tuple(DEFAULT_ACTION_DIMENSIONS),
    ),
    "pour": _preset(
        "pour",
        "dobot",
        "pour water",
        8011,
        "Dobot/NormalData/dobot_pour_water_fullV30",
        _DOBOT_CAMERAS,
        "dobot_pour_water.yaml",
        "dobot_sft_openpi_rlinf_pi05_pour_water_hf_v21.yaml",
        action_dimensions=tuple(DEFAULT_ACTION_DIMENSIONS),
    ),
    "tidy": _preset(
        "tidy",
        "dobot",
        "tidy up the desk",
        8012,
        "Dobot/NormalData/dobot_tidy_up_the_desk_fullV30",
        _DOBOT_CAMERAS,
        "dobot_tidy_up_the_desk.yaml",
        "dobot_sft_openpi_rlinf_pi05_tidy_up_the_desk.yaml",
        action_dimensions=tuple(DEFAULT_ACTION_DIMENSIONS),
    ),
    "towel": _preset(
        "towel",
        "dobot",
        "Fold towel",
        8013,
        "Dobot/NormalData/dobot_towel_fullV30",
        _DOBOT_CAMERAS,
        "dobot_towel.yaml",
        "dobot_sft_openpi_rlinf_pi05_towel.yaml",
        action_dimensions=tuple(DEFAULT_ACTION_DIMENSIONS),
    ),
    "ring": _preset(
        "ring",
        "xrobot",
        "put ring on the rod",
        8020,
        "XRobot",
        _XROBOT_CAMERAS,
        "xrobot_put_ring_on_the_rod.yaml",
        "xrobot_sft_openpi_rlinf_pi05_put_ring_on_the_rod.yaml",
        action_dimensions=tuple(XROBOT_ACTION_DIMENSIONS),
    ),
    "cobot_cube": _preset(
        "cobot_cube",
        "cobot",
        "put cube in drawer",
        8000,
        "Cobot_Magic_Arx-5/cobot_magic_cube_into_drawer_v1/lerobot",
        _COBOT_CAMERAS,
        "cobot_cube_into_drawer.yaml",
        "cobot_sft_openpi_rlinf_pi05.yaml",
        action_indices=tuple(range(14)),
        action_dimensions=tuple(DEFAULT_ACTION_DIMENSIONS),
    ),
}


def load_task_specs(path: Path | None) -> dict[str, TaskSpec]:
    """Return built-in presets merged with an optional YAML task mapping."""
    specs = dict(TASKS)
    if path is None:
        return specs
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"task config must be a mapping: {path}")
    task_mappings = raw.get("tasks", raw)
    if not isinstance(task_mappings, dict):
        raise ValueError(f"task config 'tasks' must be a mapping: {path}")
    for name, mapping in task_mappings.items():
        if not isinstance(mapping, dict):
            raise ValueError(f"task {name!r} must be a mapping")
        specs[str(name)] = TaskSpec.from_mapping(str(name), mapping)
    return specs


def apply_lerobot_compatibility_patches(robot: str) -> None:
    """Apply the same LeRobot compatibility patches used by OpenPI SFT."""
    from rlinf.data.datasets.openpi_rlinf.lerobot_hf_query_patch import (
        apply_lerobot_hf_query_patch,
    )
    from rlinf.data.datasets.openpi_rlinf.lerobot_list_feature_patch import (
        apply_lerobot_list_feature_patch,
    )
    from rlinf.data.datasets.openpi_rlinf.pyav_video_patch import (
        apply_pyav_video_decode_patch,
    )

    apply_pyav_video_decode_patch()
    apply_lerobot_hf_query_patch()
    apply_lerobot_list_feature_patch()
    if robot == "dobot":
        from rlinf.data.datasets.openpi_rlinf.dobot_lerobot_dataset_patch import (
            apply_dobot_lerobot_hf_dataset_patch,
        )

        apply_dobot_lerobot_hf_dataset_patch()


class V21EpisodeDataset:
    """Minimal metadata-driven reader for per-episode LeRobot v2.1 datasets."""

    def __init__(
        self,
        root: Path,
        info: dict[str, Any],
        episode: int,
        camera_keys: tuple[str, ...],
        action_horizon: int,
    ) -> None:
        self.root = root
        self.info = info
        self.episode = episode
        self.camera_keys = camera_keys
        self.action_horizon = action_horizon
        parquet = root / info["data_path"].format(
            episode_chunk=episode // int(info["chunks_size"]),
            episode_index=episode,
        )
        if not parquet.is_file():
            raise FileNotFoundError(parquet)
        self.columns = pq.read_table(parquet).to_pydict()
        self._captures: dict[str, cv2.VideoCapture] = {}
        for key in camera_keys:
            video = root / info["video_path"].format(
                episode_chunk=episode // int(info["chunks_size"]),
                episode_index=episode,
                video_key=key,
            )
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                self.close()
                raise RuntimeError(f"could not open v2.1 video: {video}")
            self._captures[key] = capture
        self.meta = type(
            "V21Metadata",
            (),
            {
                "video_keys": list(camera_keys),
                "episodes": {
                    episode: {
                        "dataset_from_index": 0,
                        "dataset_to_index": len(self),
                    }
                },
            },
        )()

    def __len__(self) -> int:
        return len(self.columns["action"])

    def _read_image(self, key: str, frame: int) -> torch.Tensor:
        capture = self._captures[key]
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"could not decode {key} frame {frame}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        end = len(self) - 1
        action_indices = np.clip(np.arange(index, index + self.action_horizon), 0, end)
        sample = {
            "observation.state": torch.as_tensor(
                self.columns["observation.state"][index]
            ),
            "action": torch.as_tensor(
                [self.columns["action"][int(i)] for i in action_indices]
            ),
            "timestamp": torch.as_tensor(self.columns.get("timestamp", [index])[index]),
        }
        sample.update({key: self._read_image(key, index) for key in self.camera_keys})
        return sample

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def __del__(self) -> None:
        self.close()


def create_episode_dataset(
    dataset_root: Path, spec: TaskSpec, episode: int, action_horizon: int
) -> tuple[Any, dict[str, Any]]:
    """Create a metadata-driven LeRobot view for one episode."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    info_path = dataset_root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text())
    total_episodes = int(info["total_episodes"])
    if not 0 <= episode < total_episodes:
        raise ValueError(
            f"episode {episode} is outside [0, {total_episodes}) for {dataset_root}"
        )
    fps = float(info["fps"])
    version = str(info.get("codebase_version", ""))
    if version.startswith("v2"):
        dataset = V21EpisodeDataset(
            dataset_root,
            info,
            episode,
            tuple(spec.cameras.values()),
            action_horizon,
        )
    else:
        apply_lerobot_compatibility_patches(spec.robot)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            str(dataset_root),
            episodes=[episode],
            delta_timestamps={"action": [step / fps for step in range(action_horizon)]},
        )
    if not len(dataset):
        raise ValueError(f"episode {episode} has no frames in {dataset_root}")
    missing = [
        key for key in spec.cameras.values() if key not in dataset.meta.video_keys
    ]
    if missing:
        raise ValueError(
            f"task {spec.name!r} camera features are absent from dataset: {missing}; "
            f"available={dataset.meta.video_keys}"
        )
    return dataset, info


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def select_dimensions(value: Any, indices: tuple[int, ...] | None) -> np.ndarray:
    """Convert a state/action array to numpy and optionally select its last axis."""
    array = _to_numpy(value)
    return array[..., list(indices)] if indices is not None else array


def image_b64(image: Any) -> str:
    """Encode a LeRobot image tensor/array as RGB JPEG base64."""
    array = _to_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"camera image must be rank 3, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"camera image must have 3 RGB channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255)
    array = np.asarray(array, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_payload(sample: dict[str, Any], spec: TaskSpec) -> dict[str, Any]:
    """Build the inference HTTP payload from one metadata-aligned sample."""
    state = select_dimensions(sample["observation.state"], spec.state_indices)
    if state.ndim != 1:
        raise ValueError(f"state must be 1-D after selection, got {state.shape}")
    return {
        "prompt": spec.prompt,
        "state": state.astype(np.float32).tolist(),
        "cam_high_b64": image_b64(sample[spec.cameras["high"]]),
        "cam_left_wrist_b64": image_b64(sample[spec.cameras["left_wrist"]]),
        "cam_right_wrist_b64": image_b64(sample[spec.cameras["right_wrist"]]),
    }


def wait_health(url: str, timeout: float, retries: int) -> dict[str, Any]:
    last: Exception | None = None
    for _ in range(retries):
        try:
            response = requests.get(f"{url}/health", timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # server startup can take several minutes
            last = exc
            time.sleep(2.0)
    raise RuntimeError(f"server is not healthy: {url}: {last}")


def audit_config(spec: TaskSpec) -> dict[str, Any]:
    """Ensure inference YAML uses the paired SFT preprocessing settings."""
    if not spec.infer_config or not spec.train_config:
        return {"status": "not_configured"}
    infer_path = INFER_CONFIG_DIR / spec.infer_config
    train_path = TRAIN_CONFIG_DIR / spec.train_config
    if not infer_path.is_file() or not train_path.is_file():
        return {
            "status": "config_missing",
            "inference_config": str(infer_path),
            "training_config": str(train_path),
        }
    infer, train = OmegaConf.load(infer_path), OmegaConf.load(train_path)
    pairs = {
        "action_dim": ("model.action_dim", "actor.model.action_dim"),
        "num_action_chunks": (
            "model.num_action_chunks",
            "actor.model.num_action_chunks",
        ),
        "config_name": ("model.openpi.config_name", "actor.model.openpi.config_name"),
        "num_images_in_input": (
            "model.openpi.num_images_in_input",
            "actor.model.openpi.num_images_in_input",
        ),
        "discrete_state_input": (
            "model.openpi.discrete_state_input",
            "actor.model.openpi.discrete_state_input",
        ),
        "max_token_len": (
            "model.openpi.max_token_len",
            "actor.model.openpi.max_token_len",
        ),
        "precision": ("precision", "actor.model.precision"),
        "prompt": ("default_prompt", "actor.model.openpi_data.default_prompt"),
        "norm_stats": ("norm_stats", "actor.model.openpi_data.norm_stats_path"),
    }
    settings = {
        key: tuple(
            str(OmegaConf.select(config, config_path))
            for config, config_path in ((infer, paths[0]), (train, paths[1]))
        )
        for key, paths in pairs.items()
    }
    mismatches = {
        key: {"inference": values[0], "training": values[1]}
        for key, values in settings.items()
        if values[0] != values[1]
    }
    norm_path = Path(str(infer.norm_stats))
    if not norm_path.is_file():
        raise FileNotFoundError(norm_path)
    if mismatches:
        raise ValueError(f"training/inference configuration mismatch: {mismatches}")
    return {
        "status": "matched",
        "inference_config": str(infer_path),
        "training_config": str(train_path),
        "checkpoint": str(infer.ckpt),
        "norm_stats": str(norm_path),
        "norm_stats_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "settings": {key: values[0] for key, values in settings.items()},
    }


def request_windows(
    num_frames: int, mode: str, chunk_size: int
) -> list[tuple[int, int]]:
    """Return ``(episode offset, actions consumed)`` inference windows."""
    if num_frames < 0 or chunk_size < 1:
        raise ValueError(
            "num_frames must be non-negative and chunk_size must be positive"
        )
    if mode == "single_step":
        return [(offset, 1) for offset in range(num_frames)]
    if mode != "chunk":
        raise ValueError(f"unsupported execution mode: {mode}")
    return [
        (offset, min(chunk_size, num_frames - offset))
        for offset in range(0, num_frames, chunk_size)
    ]


def action_labels(spec: TaskSpec, action_dim: int) -> list[str]:
    labels = list(spec.action_dimensions or ())
    if labels and len(labels) != action_dim:
        raise ValueError(
            f"task {spec.name!r} has {len(labels)} action labels for dim {action_dim}"
        )
    return labels or [f"action_{index}" for index in range(action_dim)]


def run_task(
    spec: TaskSpec,
    out_dir: Path,
    data_root: Path,
    episode: int,
    max_frames: int | None,
    timeout: float,
    mode: str,
    chunk_size: int | None,
    server_host: str,
) -> dict[str, Any]:
    """Replay one recorded episode against a running inference server."""
    dataset_root = Path(spec.dataset).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = data_root / dataset_root
    audit = audit_config(spec)
    effective_chunk_size = chunk_size or int(
        audit.get("settings", {}).get("num_action_chunks", 50)
    )
    dataset, info = create_episode_dataset(
        dataset_root, spec, episode, effective_chunk_size
    )
    num_frames = (
        min(len(dataset), max_frames) if max_frames is not None else len(dataset)
    )
    windows = request_windows(num_frames, mode, effective_chunk_size)
    url = f"http://{server_host}:{spec.port}"
    health = wait_health(url, timeout=10.0, retries=90)
    if health.get("robot") not in (None, spec.robot):
        raise ValueError(
            f"server robot={health.get('robot')!r} does not match task robot={spec.robot!r}"
        )

    pred, gt, timestamps, latencies, request_frames, offsets, chunks = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    action_dim: int | None = None
    for request_index, (offset, consume) in enumerate(windows):
        sample = dataset[offset]
        gt_chunk = select_dimensions(sample["action"], spec.action_indices)
        if gt_chunk.ndim == 1:
            gt_chunk = gt_chunk[None, :]
        if gt_chunk.ndim != 2 or gt_chunk.shape[0] < consume:
            raise ValueError(
                f"GT action chunk {gt_chunk.shape} cannot provide {consume} actions"
            )
        payload = build_payload(sample, spec)
        start = time.perf_counter()
        response = requests.post(f"{url}/infer", json=payload, timeout=timeout)
        response.raise_for_status()
        latency = time.perf_counter() - start
        body = response.json()
        returned = np.asarray(body["actions"], dtype=np.float32)
        if returned.ndim == 1:
            returned = returned[None, :]
        action_dim = int(gt_chunk.shape[1])
        if (
            returned.ndim != 2
            or returned.shape[1] != action_dim
            or returned.shape[0] < consume
        ):
            raise ValueError(
                f"server action chunk {returned.shape}; required [{consume}, {action_dim}]"
            )
        pred.extend(returned[:consume])
        gt.extend(gt_chunk[:consume].astype(np.float32))
        timestamp = float(
            np.asarray(_to_numpy(sample.get("timestamp", offset))).reshape(-1)[0]
        )
        timestamps.extend(
            timestamp + step / float(info["fps"]) for step in range(consume)
        )
        latencies.append(latency)
        request_frames.extend([offset] * consume)
        offsets.extend(range(consume))
        chunks.append(returned)
        print(
            f"{spec.name}: request {request_index + 1}, frames "
            f"{offset + 1}-{offset + consume}/{num_frames}, latency={latency:.2f}s",
            flush=True,
        )

    if action_dim is None:
        raise ValueError(f"task {spec.name!r} produced no inference windows")
    pred_arr = np.asarray(pred, dtype=np.float32)
    gt_arr = np.asarray(gt, dtype=np.float32)
    error = pred_arr - gt_arr
    episode_meta = dataset.meta.episodes[episode]
    stem = f"{spec.name}_episode_{episode:03d}_{mode}"
    metrics = {
        "task": spec.name,
        "robot": spec.robot,
        "episode": episode,
        "execution_mode": mode,
        "chunk_size": effective_chunk_size,
        "num_frames": num_frames,
        "num_inference_requests": len(latencies),
        "action_dim": action_dim,
        "action_dimensions": action_labels(spec, action_dim),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "per_dim_mae": np.abs(error).mean(axis=0).tolist(),
        "mean_latency_sec": float(np.mean(latencies)),
        "p95_latency_sec": float(np.percentile(latencies, 95)),
        "server_health": health,
        "server_action_spec": {
            "action_dim": action_dim,
            "returned_chunk_size": int(chunks[0].shape[0]),
        },
        "config_audit": audit,
        "dataset": {
            "root": str(dataset_root),
            "codebase_version": str(info.get("codebase_version")),
            "fps": float(info["fps"]),
            "camera_keys": spec.cameras,
            "dataset_from_index": int(episode_meta["dataset_from_index"]),
            "dataset_to_index": int(episode_meta["dataset_to_index"]),
        },
        "server": url,
    }
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        pred=pred_arr,
        gt=gt_arr,
        error=error,
        timestamp=np.asarray(timestamps),
        request_frame=np.asarray(request_frames),
        chunk_offset=np.asarray(offsets),
        inference_latency=np.asarray(latencies),
        action_chunks=np.asarray(chunks, dtype=object),
    )
    (out_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2))
    close = getattr(dataset, "close", None)
    if close is not None:
        close()
    return metrics


def plot_task(result: dict[str, Any], out_dir: Path) -> None:
    stem = (
        f"{result['task']}_episode_{result['episode']:03d}_{result['execution_mode']}"
    )
    data = np.load(out_dir / f"{stem}.npz", allow_pickle=True)
    pred, gt = data["pred"], data["gt"]
    figure, axes = plt.subplots(
        pred.shape[1], 1, figsize=(18, 2.1 * pred.shape[1]), sharex=True
    )
    boundaries = np.flatnonzero(data["chunk_offset"] == 0)
    for dimension, axis in enumerate(np.atleast_1d(axes)):
        axis.plot(
            gt[:, dimension],
            color="C0",
            linewidth=1.2,
            label="GT" if dimension == 0 else None,
        )
        axis.plot(
            pred[:, dimension],
            color="C3",
            linestyle="--",
            linewidth=1.1,
            label="prediction" if dimension == 0 else None,
        )
        for boundary in boundaries[1:]:
            axis.axvline(boundary, color="0.5", alpha=0.25, linewidth=0.8)
        axis.set_ylabel(result["action_dimensions"][dimension])
        axis.grid(alpha=0.22)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("recorded trajectory frame")
    figure.suptitle(
        f"{result['task']} | {result['execution_mode']} | chunk={result['chunk_size']} "
        f"| MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}"
    )
    figure.tight_layout()
    figure.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_results(results: list[dict[str, Any]], out_dir: Path) -> None:
    figure, axes = plt.subplots(
        len(results), 1, figsize=(16, 4.2 * len(results)), squeeze=False
    )
    for row, result in enumerate(results):
        stem = f"{result['task']}_episode_{result['episode']:03d}_{result['execution_mode']}"
        data = np.load(out_dir / f"{stem}.npz", allow_pickle=True)
        axis = axes[row, 0]
        for dimension in range(data["pred"].shape[1]):
            axis.plot(
                data["gt"][:, dimension],
                color=f"C{dimension % 10}",
                alpha=0.32,
                linewidth=0.8,
            )
            axis.plot(
                data["pred"][:, dimension],
                color=f"C{dimension % 10}",
                linestyle="--",
                linewidth=1.0,
            )
        for boundary in np.flatnonzero(data["chunk_offset"] == 0)[1:]:
            axis.axvline(boundary, color="0.5", alpha=0.2, linewidth=0.7)
        axis.set_title(
            f"{result['task']} | {result['execution_mode']} | "
            f"MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}"
        )
        axis.set_xlabel("recorded trajectory frame")
        axis.set_ylabel("absolute action")
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Recorded trajectory GT vs deployed inference (solid=GT, dashed=prediction)",
        y=1.0,
    )
    figure.tight_layout()
    figure.savefig(
        out_dir / f"open_loop_all_tasks_{results[0]['execution_mode']}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def parse_args(task_specs: dict[str, TaskSpec]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", type=Path, help="optional custom task YAML")
    parser.add_argument(
        "--tasks", nargs="+", default=list(task_specs), choices=list(task_specs)
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    parser.add_argument("--episode", type=int, default=0, help="recorded episode index")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--execution-mode", choices=("single_step", "chunk"), default="chunk"
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    # Parse task config separately so custom task names become valid argparse choices.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--task-config", type=Path)
    bootstrap_args, _ = bootstrap.parse_known_args()
    task_specs = load_task_specs(bootstrap_args.task_config)
    args = parse_args(task_specs)
    if args.episode < 0:
        raise SystemExit("--episode must be non-negative")
    if args.chunk_size is not None and args.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.tasks:
        try:
            result = run_task(
                task_specs[name],
                args.output_dir,
                args.data_root,
                args.episode,
                args.max_frames,
                args.timeout,
                args.execution_mode,
                args.chunk_size,
                args.server_host,
            )
            plot_task(result, args.output_dir)
            results.append(result)
        except Exception as exc:
            print(f"ERROR {name}: {exc}", flush=True)
            (args.output_dir / f"{name}_{args.execution_mode}_error.txt").write_text(
                str(exc)
            )
    if results:
        plot_results(results, args.output_dir)
        (args.output_dir / f"summary_{args.execution_mode}.json").write_text(
            json.dumps(results, indent=2)
        )
        print(json.dumps(results, indent=2))
    if len(results) != len(args.tasks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
