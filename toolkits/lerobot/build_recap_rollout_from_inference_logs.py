#!/usr/bin/env python3
"""Build a RECAP-ready LeRobot v2.1 rollout dataset from Franka inference logs.

The input is the `recap_episode_labels.csv` produced by `label_pnp_session.py`.
Each row points at one `session_async_*` directory containing `actions.json` and
`videos/cam_{high,side,wrist}.mp4`.

By default this creates a compact dataset whose rows are aligned to video
frames, with video files symlinked into the LeRobot directory. Use
`--alignment action_frames --video-mode reencode` to mirror the older
`convert_inference_log_to_lerobot.py` behavior of expanding every low-level
action step and resampling videos to match.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CAM_TO_FEATURE = {
    "cam_high": "global_image",
    "cam_side": "right_image",
    "cam_wrist": "wrist_image",
}
FEATURE_ALIASES = {"image": "global_image"}
RPY_WRAP_CENTERS = (math.pi, 0.0, 0.0)


@dataclass(frozen=True)
class EpisodeSpec:
    source_index: int
    episode_index: int
    session_dir: Path
    task: str
    prompt: str
    is_success: bool


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "success", "s"}


def normalize_rpy_to_centers(rpy_values, centers=RPY_WRAP_CENTERS) -> np.ndarray:
    import numpy as np

    rpy_arr = np.asarray(rpy_values, dtype=np.float64)
    centers_arr = np.asarray(centers, dtype=np.float64)
    return ((rpy_arr - centers_arr + np.pi) % (2 * np.pi)) - np.pi + centers_arr


def stats_block(values: np.ndarray) -> dict:
    import numpy as np

    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "count": [int(len(values))],
    }


def compute_returns_for_episode(
    episode_length: int,
    is_success: bool,
    gamma: float,
    failure_reward: float,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    rewards = np.full(episode_length, -1.0, dtype=np.float32)
    rewards[-1] = 0.0 if is_success else failure_reward
    returns = np.zeros(episode_length, dtype=np.float32)
    returns[-1] = rewards[-1]
    for t in range(episode_length - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns, rewards


def load_episode_specs(labels_csv: Path, task_id: str | None) -> list[EpisodeSpec]:
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    specs: list[EpisodeSpec] = []
    for row in rows:
        if task_id and row.get("task_id") != task_id:
            continue
        specs.append(
            EpisodeSpec(
                source_index=len(specs),
                episode_index=len(specs),
                session_dir=Path(row["session_dir"]).expanduser(),
                task=row.get("task") or row.get("prompt") or "perform the task",
                prompt=row.get("prompt") or row.get("task") or "perform the task",
                is_success=parse_bool(row.get("is_success", "0")),
            )
        )
    return specs


def load_log_arrays(log_path: Path, unwrap_rpy: bool) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    log = json.loads(log_path.read_text(encoding="utf-8"))
    chunks = sorted(log["chunks"], key=lambda c: int(c["chunk_id"]))
    actions = np.asarray(
        [action for chunk in chunks for action in chunk["actions"]], dtype=np.float32
    )
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"{log_path}: expected actions with shape (N, 7), got {actions.shape}")

    states = np.empty_like(actions)
    states[0] = np.asarray(chunks[0]["state_sent"], dtype=np.float32)
    states[1:] = actions[:-1]

    if unwrap_rpy:
        actions[:, 3:6] = normalize_rpy_to_centers(actions[:, 3:6]).astype(np.float32)
        states[:, 3:6] = normalize_rpy_to_centers(states[:, 3:6]).astype(np.float32)
    return states, actions


def video_frame_count(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise RuntimeError(f"Invalid frame count for video: {video_path}")
    return n


def sample_to_length(values: np.ndarray, n_out: int) -> np.ndarray:
    import numpy as np

    if len(values) == n_out:
        return values
    if n_out == 1:
        return values[[0]]
    idx = np.rint(np.linspace(0, len(values) - 1, n_out)).astype(np.int64)
    return values[idx]


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link/copy mode: {mode}")


def read_video_resampled(video_path: Path, n_out: int) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_src <= 0:
        raise RuntimeError(f"Invalid frame count for video: {video_path}")
    wanted = [0] if n_out == 1 else np.rint(np.linspace(0, n_src - 1, n_out)).astype(int)
    wanted = list(map(int, wanted))

    frames: list[np.ndarray] = []
    cur_idx = -1
    cur_frame = None
    for target in wanted:
        while cur_idx < target:
            ok, frame = cap.read()
            if not ok:
                break
            cur_idx += 1
            cur_frame = frame
        if cur_frame is None:
            raise RuntimeError(f"Could not read frame {target} from {video_path}")
        frames.append(cur_frame.copy())
    cap.release()
    return frames


def write_video(frames: Iterable[np.ndarray], dst: Path, fps: int) -> None:
    import cv2

    frames = list(frames)
    dst.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def materialize_video(
    session_dir: Path,
    videos_dir: Path,
    episode_index: int,
    feature: str,
    source_cam: str,
    n_frames: int,
    alignment: str,
    video_mode: str,
    fps: int,
) -> str:
    src = session_dir / "videos" / f"{source_cam}.mp4"
    dst = videos_dir / feature / f"episode_{episode_index:06d}.mp4"
    if alignment == "action_frames":
        if video_mode != "reencode":
            raise ValueError("action_frames alignment requires --video-mode reencode")
        write_video(read_video_resampled(src, n_frames), dst, fps)
    elif video_mode in {"symlink", "copy"}:
        link_or_copy(src, dst, video_mode)
    elif video_mode == "reencode":
        write_video(read_video_resampled(src, n_frames), dst, fps)
    else:
        raise ValueError(f"Unsupported video mode: {video_mode}")
    return f"videos/chunk-000/{feature}/episode_{episode_index:06d}.mp4"


def build_dataset(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    labels_csv = Path(args.labels_csv).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    specs = load_episode_specs(labels_csv, args.task_id)
    if not specs:
        raise ValueError(f"No episodes selected from {labels_csv}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    if output.exists():
        shutil.rmtree(output)

    data_dir = output / "data" / "chunk-000"
    videos_dir = output / "videos" / "chunk-000"
    meta_dir = output / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    episodes = []
    episodes_stats = []
    global_states = []
    global_actions = []
    returns_sidecar = []
    all_returns = []
    all_rewards = []
    total_frames = 0
    feature_keys = ["image", "global_image", "right_image", "wrist_image"]

    for spec in specs:
        log_path = spec.session_dir / "actions.json"
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        states, actions = load_log_arrays(log_path, unwrap_rpy=args.unwrap_rpy)

        if args.alignment == "video_frames":
            counts = [
                video_frame_count(spec.session_dir / "videos" / f"{cam}.mp4")
                for cam in CAM_TO_FEATURE
            ]
            if len(set(counts)) != 1:
                raise ValueError(f"{spec.session_dir}: camera frame counts differ: {counts}")
            n_frames = counts[0]
            states = sample_to_length(states, n_frames)
            actions = sample_to_length(actions, n_frames)
        else:
            n_frames = len(actions)

        video_paths = {}
        for cam, feature in CAM_TO_FEATURE.items():
            video_paths[feature] = materialize_video(
                spec.session_dir,
                videos_dir,
                spec.episode_index,
                feature,
                cam,
                n_frames,
                args.alignment,
                args.video_mode,
                args.fps,
            )
        for alias, source_feature in FEATURE_ALIASES.items():
            src_path = videos_dir / source_feature / f"episode_{spec.episode_index:06d}.mp4"
            dst_path = videos_dir / alias / f"episode_{spec.episode_index:06d}.mp4"
            if args.video_mode == "copy":
                link_or_copy(src_path, dst_path, "copy")
            else:
                link_or_copy(src_path, dst_path, "symlink")
            video_paths[alias] = f"videos/chunk-000/{alias}/episode_{spec.episode_index:06d}.mp4"

        rows = []
        for frame_index in range(n_frames):
            rows.append(
                {
                    **video_paths,
                    "state": states[frame_index].astype(np.float32).tolist(),
                    "actions": actions[frame_index].astype(np.float32).tolist(),
                    "task": spec.prompt,
                    "task_index": 0,
                    "episode_index": spec.episode_index,
                    "frame_index": frame_index,
                    "timestamp": frame_index / args.fps,
                    "index": total_frames + frame_index,
                    "is_success": spec.is_success,
                    "source_session": spec.session_dir.name,
                }
            )
        pd.DataFrame(rows).to_parquet(
            data_dir / f"episode_{spec.episode_index:06d}.parquet", index=False
        )

        episodes.append(
            {
                "episode_index": spec.episode_index,
                "tasks": [spec.prompt],
                "length": int(n_frames),
            }
        )
        episodes_stats.append(
            {
                "episode_index": spec.episode_index,
                "stats": {"state": stats_block(states), "action": stats_block(actions)},
            }
        )
        returns_arr, rewards_arr = compute_returns_for_episode(
            episode_length=n_frames,
            is_success=spec.is_success,
            gamma=args.gamma,
            failure_reward=args.failure_reward,
        )
        returns_sidecar.append(
            pd.DataFrame(
                {
                    "episode_index": np.full(n_frames, spec.episode_index, dtype=np.int64),
                    "frame_index": np.arange(n_frames, dtype=np.int64),
                    "return": returns_arr,
                    "reward": rewards_arr,
                    "prompt": np.full(n_frames, spec.prompt, dtype=object),
                }
            )
        )
        all_returns.append(returns_arr)
        all_rewards.append(rewards_arr)
        global_states.append(states)
        global_actions.append(actions)
        total_frames += n_frames
        print(
            f"episode {spec.episode_index:06d}: {spec.session_dir.name} "
            f"frames={n_frames} success={int(spec.is_success)}"
        )

    state_values = np.concatenate(global_states, axis=0)
    action_values = np.concatenate(global_actions, axis=0)
    video_feature = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": args.fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": args.robot_type,
        "total_episodes": len(specs),
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "total_videos": len(specs) * len(feature_keys),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {"train": f"0:{len(specs)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            **{feature: video_feature for feature in feature_keys},
            "state": {"dtype": "float32", "shape": [7], "names": ["state"]},
            "actions": {"dtype": "float32", "shape": [7], "names": ["actions"]},
            "task": {"dtype": "string", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "is_success": {"dtype": "bool", "shape": [1], "names": None},
            "source_session": {"dtype": "string", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for item in episodes:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (meta_dir / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for item in episodes_stats:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    task = specs[0].prompt
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (meta_dir / "tasks.json").write_text(
        json.dumps({task: [0]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    stats = {
        "state": {k: v for k, v in stats_block(state_values).items() if k != "count"},
        "action": {k: v for k, v in stats_block(action_values).items() if k != "count"},
    }
    if args.returns_tag:
        returns_values = np.concatenate(all_returns, axis=0)
        rewards_values = np.concatenate(all_rewards, axis=0)
        stats["return"] = {
            k: v for k, v in stats_block(returns_values[:, None]).items() if k != "count"
        }
        stats["reward"] = {
            k: v for k, v in stats_block(rewards_values[:, None]).items() if k != "count"
        }
    (meta_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.returns_tag:
        returns_path = meta_dir / f"returns_{args.returns_tag}.parquet"
        pd.concat(returns_sidecar, ignore_index=True).to_parquet(returns_path, index=False)
        print(f"wrote returns sidecar: {returns_path}")
    shutil.copy2(labels_csv, meta_dir / "recap_episode_labels.csv")
    print(f"wrote {output} episodes={len(specs)} frames={total_frames}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="Franka")
    parser.add_argument("--returns-tag", default="fail300")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--failure-reward", type=float, default=-300.0)
    parser.add_argument(
        "--alignment",
        choices=["video_frames", "action_frames"],
        default="video_frames",
        help="video_frames keeps original videos; action_frames expands every low-level action.",
    )
    parser.add_argument(
        "--video-mode",
        choices=["symlink", "copy", "reencode"],
        default="symlink",
        help="Use reencode with --alignment action_frames.",
    )
    parser.add_argument("--unwrap-rpy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    build_dataset(parser.parse_args())


if __name__ == "__main__":
    main()
