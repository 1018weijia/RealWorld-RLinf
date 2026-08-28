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
"""Open-loop trajectory comparison for Dobot/XRobot inference servers.

For one episode per task, replay recorded observations to the HTTP inference
server and compare the returned absolute action chunk with the recorded GT
action. The first action in each returned chunk is used at the corresponding
observation time; this avoids counting overlapping chunk predictions multiple
times while preserving the deployed server path.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import requests
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "datasets"
INFER_CONFIG_DIR = REPO_ROOT / "toolkits/inference/config"
TRAIN_CONFIG_DIR = REPO_ROOT / "examples/sft/config"
ACTION_DIMENSIONS = (
    [f"left_joint_{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(6)]
    + ["right_gripper"]
)
TASKS: dict[str, dict[str, Any]] = {
    "cook": {
        "robot": "dobot",
        "prompt": "cook vegetable",
        "port": 8010,
        "dataset": "Dobot/NormalData/dobot_cook_vegetable_fullV30",
        "episode": 0,
        "infer_config": "dobot_cook_vegetable.yaml",
        "train_config": "dobot_sft_openpi_rlinf_pi05_cook_vegetable.yaml",
    },
    "pour": {
        "robot": "dobot",
        "prompt": "pour water",
        "port": 8011,
        "dataset": "Dobot/NormalData/dobot_pour_water_fullV30",
        "episode": 0,
        "infer_config": "dobot_pour_water.yaml",
        "train_config": "dobot_sft_openpi_rlinf_pi05_pour_water.yaml",
    },
    "tidy": {
        "robot": "dobot",
        "prompt": "tidy up the desk",
        "port": 8012,
        "dataset": "Dobot/NormalData/dobot_tidy_up_the_desk_fullV30",
        "episode": 0,
        "infer_config": "dobot_tidy_up_the_desk.yaml",
        "train_config": "dobot_sft_openpi_rlinf_pi05_tidy_up_the_desk.yaml",
    },
    "towel": {
        "robot": "dobot",
        "prompt": "Fold towel",
        "port": 8013,
        "dataset": "Dobot/NormalData/dobot_towel_fullV30",
        "episode": 0,
        "infer_config": "dobot_towel.yaml",
        "train_config": "dobot_sft_openpi_rlinf_pi05_towel.yaml",
    },
    "ring": {
        "robot": "xrobot",
        "prompt": "put ring on the rod",
        "port": 8020,
        "dataset": "XRobot",
        "episode": 0,
    },
}


def episode_parquet(dataset: Path, robot: str, episode: int) -> Path:
    if robot == "xrobot":
        return dataset / "data/chunk-000" / f"episode_{episode:06d}.parquet"
    return dataset / "data/chunk-000" / f"file-{episode:03d}.parquet"


def episode_video(dataset: Path, robot: str, episode: int, view: str) -> Path:
    if robot == "xrobot":
        return (
            dataset
            / "videos/chunk-000"
            / f"observation.images.{view}"
            / f"episode_{episode:06d}.mp4"
        )
    return (
        dataset
        / "videos"
        / f"observation.images.{view}"
        / "chunk-000"
        / f"file-{episode:03d}.mp4"
    )


def image_b64(cap: cv2.VideoCapture, frame_idx: int) -> str:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not decode frame {frame_idx}")
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not ok:
        raise RuntimeError(f"could not encode frame {frame_idx}")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


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


def audit_config(spec: dict[str, Any]) -> dict[str, Any]:
    """Ensure inference YAML uses the SFT model preprocessing settings."""
    if "infer_config" not in spec:
        return {"status": "not_configured"}
    infer_path, train_path = (
        INFER_CONFIG_DIR / spec["infer_config"],
        TRAIN_CONFIG_DIR / spec["train_config"],
    )
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
            str(OmegaConf.select(config, path))
            for config, path in ((infer, paths[0]), (train, paths[1]))
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
    return (
        [(i, 1) for i in range(num_frames)]
        if mode == "single_step"
        else [
            (i, min(chunk_size, num_frames - i))
            for i in range(0, num_frames, chunk_size)
        ]
    )


def run_task(
    name: str,
    spec: dict[str, Any],
    out_dir: Path,
    data_root: Path,
    episode: int,
    max_frames: int | None,
    timeout: float,
    mode: str,
    chunk_size: int | None,
    server_host: str,
) -> dict[str, Any]:
    dataset = data_root / str(spec["dataset"])
    parquet_path = episode_parquet(dataset, spec["robot"], episode)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    columns = pq.read_table(parquet_path).to_pydict()
    episode_indices = np.asarray(columns.get("episode_index", []))
    selected = (
        np.flatnonzero(episode_indices == episode)
        if episode_indices.size
        else np.arange(len(columns["action"]))
    )
    if not len(selected):
        raise ValueError(f"episode {episode} not found in {parquet_path}")
    if max_frames is not None:
        selected = selected[:max_frames]
    action_dim = len(columns["action"][int(selected[0])])
    audit = audit_config(spec)
    effective_chunk_size = chunk_size or int(
        audit.get("settings", {}).get("num_action_chunks", 50)
    )
    views = {
        "xrobot": {"top": "head", "left": "left_arm", "right": "right_arm"},
        "dobot": {"top": "top", "left": "left_wrist", "right": "right_wrist"},
    }[spec["robot"]]
    caps = {
        key: cv2.VideoCapture(str(episode_video(dataset, spec["robot"], episode, view)))
        for key, view in views.items()
    }
    try:
        if not all(cap.isOpened() for cap in caps.values()):
            raise RuntimeError(f"could not open all videos for {name}: {caps}")
        url, health = f"http://{server_host}:{spec['port']}", None
        health = wait_health(url, timeout=10.0, retries=90)
        pred, gt, timestamps, latencies, request_frames, offsets, chunks = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for request_index, (offset, consume) in enumerate(
            request_windows(len(selected), mode, effective_chunk_size)
        ):
            frame = int(selected[offset])
            payload = {
                "prompt": spec["prompt"],
                "state": [float(x) for x in columns["observation.state"][frame]],
                "cam_high_b64": image_b64(caps["top"], frame),
                "cam_left_wrist_b64": image_b64(caps["left"], frame),
                "cam_right_wrist_b64": image_b64(caps["right"], frame),
            }
            start = time.perf_counter()
            response = requests.post(f"{url}/infer", json=payload, timeout=timeout)
            response.raise_for_status()
            latency = time.perf_counter() - start
            returned = np.asarray(response.json()["actions"], dtype=np.float32)
            if returned.ndim == 1:
                returned = returned.reshape(1, -1)
            if (
                returned.ndim != 2
                or returned.shape[1] != action_dim
                or returned.shape[0] < consume
            ):
                raise ValueError(
                    f"invalid returned action chunk {returned.shape}; required [{consume}, {action_dim}]"
                )
            frames, used = selected[offset : offset + consume], returned[:consume]
            pred.extend(used)
            gt.extend(
                np.asarray(
                    [columns["action"][int(index)] for index in frames],
                    dtype=np.float32,
                )
            )
            timestamps.extend(
                float(columns.get("timestamp", [int(index)])[int(index)])
                for index in frames
            )
            latencies.append(latency)
            request_frames.extend([frame] * consume)
            offsets.extend(range(consume))
            chunks.append(returned)
            print(
                f"{name}: request {request_index + 1}, frames {offset + 1}-{offset + consume}/{len(selected)}, latency={latency:.2f}s",
                flush=True,
            )
        pred_arr, gt_arr = (
            np.asarray(pred, dtype=np.float32),
            np.asarray(gt, dtype=np.float32),
        )
        error = pred_arr - gt_arr
        stem = f"{name}_episode_{episode:03d}_{mode}"
        metrics = {
            "task": name,
            "robot": spec["robot"],
            "episode": episode,
            "execution_mode": mode,
            "chunk_size": effective_chunk_size,
            "num_frames": len(selected),
            "num_inference_requests": len(latencies),
            "action_dim": action_dim,
            "action_dimensions": ACTION_DIMENSIONS
            if action_dim == 14
            else [f"action_{i}" for i in range(action_dim)],
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
            "parquet": str(parquet_path),
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
        return metrics
    finally:
        for cap in caps.values():
            cap.release()


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
        f"{result['task']} | {result['execution_mode']} | chunk={result['chunk_size']} | MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}"
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
            f"{result['task']} | {result['execution_mode']} | MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    parser.add_argument("--episode", type=int, default=0, help="recorded episode index")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="optional cap for debugging"
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--execution-mode", choices=("single_step", "chunk"), default="chunk"
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    args = parser.parse_args()
    if args.episode < 0:
        parser.error("--episode must be non-negative")
    if args.chunk_size is not None and args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if not args.data_root.is_dir():
        parser.error(f"--data-root does not exist: {args.data_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.tasks:
        try:
            result = run_task(
                name,
                TASKS[name],
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
