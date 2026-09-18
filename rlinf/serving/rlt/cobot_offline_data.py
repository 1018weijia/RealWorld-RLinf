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

"""LeRobot v3 Cobot episodes and portable, native RLT feature buffers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

JOINT_NAMES = [
    name
    for side in ("left", "right")
    for name in [*[f"{side}_joint_{i}.pos" for i in range(1, 7)], f"{side}_gripper.pos"]
]
CAMERAS = {
    "observation.images.cam_high": "image",
    "observation.images.cam_left_wrist": "wrist_image",
    "observation.images.cam_right_wrist": "side_image",
}
FORMAT = "cobot_rlinf_offline_v1"


def atomic_save(value: Any, path: Path) -> None:
    """Publish a complete shard/checkpoint, preserving any previous file on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    """Hash source data without retaining it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract(cfg) -> dict:
    """Identify the frozen feature space and Stage2 architecture across handoff."""
    from omegaconf import OmegaConf

    weights = Path(cfg.rlt_feature_model.model_path)
    if weights.is_dir():
        weights /= "actor/model_state_dict/full_weights.pt"
    return {
        "task": str(cfg.server.task_prompt),
        "feature_model": OmegaConf.to_container(cfg.rlt_feature_model, resolve=True),
        "actor_model": OmegaConf.to_container(cfg.actor.model, resolve=True),
        "weights_size": weights.stat().st_size,
        "weights_mtime_ns": weights.stat().st_mtime_ns,
        "norm_sha256": sha256(Path(cfg.rlt_feature_model.openpi_data.norm_stats_path)),
        "gamma": float(cfg.algorithm.gamma),
    }


class CobotLeRobotV3:
    """Read named dual-arm joints and per-episode video offsets from v3 shards."""

    def __init__(self, root: str | Path, prompt: str):
        import pyarrow.parquet as pq

        self.root = Path(root).resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["codebase_version"] != "v3.0":
            raise ValueError("This converter requires LeRobot v3.0")
        self.fps = float(self.info["fps"])
        if self.fps != 30:
            raise ValueError(
                "Cobot online execution and offline data must both use 30 Hz"
            )
        self.prompt = prompt
        features = self.info["features"]
        self.state_indices = [
            features["observation.state"]["names"].index(n) for n in JOINT_NAMES
        ]
        self.action_indices = [
            features["action"]["names"].index(n) for n in JOINT_NAMES
        ]
        for name in CAMERAS:
            if features[name]["dtype"] != "video":
                raise ValueError(f"Missing video feature {name}")
        self.episodes = []
        for path in sorted((self.root / "meta/episodes").rglob("*.parquet")):
            columns = [
                n for n in pq.read_schema(path).names if not n.startswith("stats/")
            ]
            self.episodes.extend(pq.read_table(path, columns=columns).to_pylist())
        self.episodes.sort(key=lambda row: row["episode_index"])
        if [r["episode_index"] for r in self.episodes] != list(
            range(self.info["total_episodes"])
        ):
            raise ValueError("Episode metadata is incomplete or duplicated")
        for row in self.episodes:
            if row["tasks"] != [prompt]:
                raise ValueError(
                    f"Task mismatch in episode {row['episode_index']}: {row['tasks']}"
                )
            if row.get("episode_success") not in ("success", "failure"):
                raise ValueError(
                    "Every episode needs an explicit success/failure label"
                )

    def table(self, row: dict) -> tuple[np.ndarray, np.ndarray]:
        """Read only this episode, including when a parquet file contains several."""
        import pyarrow.parquet as pq

        path = self.root / self.info["data_path"].format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
        )
        table = pq.read_table(
            path,
            columns=[
                "observation.state",
                "action",
                "frame_index",
                "index",
                "timestamp",
            ],
            filters=[("episode_index", "=", row["episode_index"])],
        ).to_pydict()
        length = int(row["length"])
        if table["frame_index"] != list(range(length)):
            raise ValueError(
                f"Missing or unordered frames in episode {row['episode_index']}"
            )
        if table["index"] != list(
            range(row["dataset_from_index"], row["dataset_to_index"])
        ):
            raise ValueError("Parquet global indices disagree with episode metadata")
        np.testing.assert_allclose(
            table["timestamp"], np.arange(length) / self.fps, atol=1e-4
        )
        state = np.asarray(table["observation.state"], np.float32)[
            :, self.state_indices
        ]
        actions = np.asarray(table["action"], np.float32)[:, self.action_indices]
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite robot state/action")
        return state, actions

    def frames(
        self, row: dict, camera: str, indices: list[int]
    ) -> dict[int, np.ndarray]:
        """Decode selected frames using v3's video timestamps, never file-local row guesses."""
        import av

        prefix = f"videos/{camera}"
        path = self.root / self.info["video_path"].format(
            video_key=camera,
            chunk_index=row[f"{prefix}/chunk_index"],
            file_index=row[f"{prefix}/file_index"],
        )
        offset = float(row[f"{prefix}/from_timestamp"])
        result = {}
        cursor = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(int(max(0, offset - 1) / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                if cursor == len(indices):
                    break
                target = offset + indices[cursor] / self.fps
                timestamp = float(frame.pts * frame.time_base)
                if timestamp < target - 0.51 / self.fps:
                    continue
                if abs(timestamp - target) > 0.51 / self.fps:
                    raise ValueError(
                        f"Video frame missing: {path}, wanted timestamp {target}, got {timestamp}"
                    )
                result[indices[cursor]] = frame.to_ndarray(format="rgb24")
                cursor += 1
        if len(result) != len(indices):
            raise ValueError(f"Truncated video: {path}")
        return result

    def fingerprint(self) -> str:
        """Content hash enables safe reuse of expensive per-episode feature shards."""
        digest = hashlib.sha256()
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".parquet", ".mp4"):
                digest.update(str(path.relative_to(self.root)).encode())
                digest.update(sha256(path).encode())
        return digest.hexdigest()


def chunk_starts(length: int, chunk: int = 30) -> list[int]:
    """Terminal-aligned full chunks; reserve the last recorded observation as endpoint."""
    return (
        list(range((length - 1) % chunk, length - chunk, chunk))
        if length > chunk
        else []
    )


def convert_episode(data: CobotLeRobotV3, row: dict, inference, gamma: float) -> dict:
    """Encode each boundary once, sharing the exact feature between adjacent chunks."""
    state, actions = data.table(row)
    chunk = inference.chunk_len
    starts = chunk_starts(len(state), chunk)
    if not starts:
        raise ValueError(f"Episode {row['episode_index']} has no complete transition")
    endpoints = [*starts, starts[-1] + chunk]
    videos = {name: data.frames(row, name, endpoints) for name in CAMERAS}
    features = []
    normalized = []
    for position, index in enumerate(endpoints):
        observation = {"state": state[index], "prompt": data.prompt}
        # Recorded previous targets are the causal command boundary. Do not
        # replace recorded actions or observations with projected trajectories.
        observation["previous_command"] = actions[index - 1] if index else state[index]
        observation.update(
            {target: videos[name].pop(index) for name, target in CAMERAS.items()}
        )
        encoded = inference.encode(observation)
        features.append(
            {
                k: v.detach().float().cpu()
                for k, v in inference.strip_private(encoded).items()
            }
        )
        if position < len(starts):
            action = inference.to_normalized_space(
                actions[index : index + chunk], encoded
            )
            restored = inference.to_robot_space(
                torch.as_tensor(action, device=inference.device), encoded
            )
            np.testing.assert_allclose(
                restored, actions[index : index + chunk], atol=5e-3, rtol=0
            )
            normalized.append(torch.from_numpy(action.reshape(-1)))
    count = len(starts)
    rewards = torch.zeros(count, chunk)
    success = row["episode_success"] == "success"
    rewards[-1, -1] = float(success)
    dones = torch.zeros(count, 1, dtype=torch.bool)
    dones[-1] = True
    returns = torch.zeros(count, 1)
    running = 0.0
    for i in reversed(range(count)):
        for reward in reversed(rewards[i].tolist()):
            running = reward + gamma * running
        returns[i] = running
    return {
        "curr_obs": {k: torch.cat([f[k] for f in features[:-1]]) for k in features[0]},
        "next_obs": {k: torch.cat([f[k] for f in features[1:]]) for k in features[0]},
        "actions": torch.stack(normalized),
        "rewards": rewards,
        "dones": dones,
        "terminations": dones.clone(),
        "bootstrap_mask": (~dones).float(),
        "mc_returns": returns,
        "success": torch.full((count, 1), success),
        "episode_index": torch.full((count,), row["episode_index"], dtype=torch.long),
    }


def concatenate(rows: list[dict]) -> dict:
    """Concatenate a tensor tree along its leading sample axis."""
    return {
        k: concatenate([r[k] for r in rows])
        if isinstance(v, dict)
        else torch.cat([r[k] for r in rows])
        for k, v in rows[0].items()
    }


class OfflineBuffer:
    """Immutable features with episode-separated training/validation indices."""

    def __init__(self, payload: dict):
        if payload.get("format") != FORMAT:
            raise ValueError("Not a native Cobot RLinf offline buffer")
        self.payload = payload
        self.rows = payload["rows"]
        self.size = len(self.rows["actions"])
        if not self.size:
            raise ValueError("Offline buffer is empty")
        model = payload["contract"]["actor_model"]
        chunk, dim = int(model["num_action_chunks"]), int(model["action_dim"])
        expected = {
            "actions": (self.size, chunk * dim),
            "rewards": (self.size, chunk),
            "dones": (self.size, 1),
            "terminations": (self.size, 1),
            "bootstrap_mask": (self.size, 1),
            "mc_returns": (self.size, 1),
            "success": (self.size, 1),
            "episode_index": (self.size,),
        }
        for key, shape in expected.items():
            value = self.rows[key]
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(
                    f"Invalid offline field {key}: expected finite {shape}"
                )
        for group in ("curr_obs", "next_obs"):
            shapes = {
                "z_rl": (self.size, int(model["z_dim"])),
                "proprio": (self.size, int(model["proprio_dim"])),
                "ref_chunk": (self.size, int(model["ref_num_action_chunks"]), dim),
                "ref_candidates": (
                    self.size,
                    int(model["expo_num_base_samples"]),
                    int(model["ref_num_action_chunks"]),
                    dim,
                ),
            }
            if model.get("joint_motion"):
                shapes["motion_context"] = (self.size, 42)
            for key, shape in shapes.items():
                value = self.rows[group][key]
                if tuple(value.shape) != shape or not torch.isfinite(value).all():
                    raise ValueError(
                        f"Invalid offline field {group}.{key}: expected finite {shape}"
                    )
            if (
                model.get("joint_motion")
                and (self.rows[group]["motion_context"][:, :14].abs() < 1e-8).any()
            ):
                raise ValueError(
                    "Offline motion_context contains a zero physical action scale"
                )
        if (
            self.rows["dones"].dtype != torch.bool
            or self.rows["success"].dtype != torch.bool
        ):
            raise ValueError("Offline terminal/outcome masks must be boolean")
        validation = torch.isin(
            self.rows["episode_index"], torch.tensor(payload["validation_episodes"])
        )
        self.train_indices = (~validation).nonzero().flatten()
        self.validation_indices = validation.nonzero().flatten()
        if not len(self.train_indices):
            raise ValueError("Offline training partition is empty")

    def sample(
        self, count: int, device: torch.device, *, validation: bool = False
    ) -> dict:
        """Sample uniformly; offline rows never participate in mutable online PER."""
        pool = self.validation_indices if validation else self.train_indices
        if not len(pool):
            raise ValueError("Requested an empty offline partition")
        indices = pool[torch.randint(len(pool), (count,))]
        return self.select(indices, device)

    @torch.no_grad()
    def cache_joint_references(
        self, model, device: torch.device, batch_size: int = 256
    ) -> None:
        """Cache deterministic motion projections in memory, never in source replay.

        They depend only on immutable observations and the checked motion contract,
        not actor/critic weights. This avoids repeated 30-step GPU scans per update.
        """
        rows = dict(self.rows)
        for group in ("curr_obs", "next_obs"):
            original = self.payload["rows"][group]
            refs, candidates = [], []
            for start in range(0, self.size, batch_size):
                obs = {
                    k: v[start : start + batch_size].to(device)
                    for k, v in original.items()
                }
                refs.append(model._get_ref_chunk(obs).cpu())
                candidates.append(model._get_ref_candidates(obs).cpu())
            rows[group] = {
                **original,
                "joint_projected_ref": torch.cat(refs),
                "joint_projected_candidates": torch.cat(candidates),
            }
        self.rows = rows

    def fixed_validation(self, count: int, device: torch.device, *, seed: int) -> dict:
        """Choose held-out rows without replacement or changes to training RNG."""
        pool = self.validation_indices
        generator = torch.Generator().manual_seed(seed)
        indices = pool[torch.randperm(len(pool), generator=generator)[:count]]
        return self.select(indices, device)

    def select(self, indices: torch.Tensor, device: torch.device) -> dict:
        """Materialize immutable rows at explicit indices on the requested device."""

        def select(tree):
            return {
                k: select(v) if isinstance(v, dict) else v[indices].to(device)
                for k, v in tree.items()
            }

        return select(self.rows)
