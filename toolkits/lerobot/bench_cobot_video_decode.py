#!/usr/bin/env python3
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

"""Benchmark Cobot video decode: official LeRobot backends vs RLinf PyAV patch.

Usage (from repo root, with HF_LEROBOT_HOME set)::

    python toolkits/lerobot/bench_cobot_video_decode.py
"""

from __future__ import annotations

import importlib
import os
import statistics
import time
from pathlib import Path

import av
import numpy as np
import torch
import tyro


def _time_calls(fn, repeats: int) -> list[float]:
    fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _summarize(name: str, times: list[float], n_frames: int) -> None:
    mean = statistics.mean(times)
    p50 = statistics.median(times)
    p90 = sorted(times)[max(0, int(0.9 * len(times)) - 1)]
    fps = n_frames / mean if mean > 0 else float("inf")
    print(
        f"{name:32s}  mean={mean * 1000:7.1f} ms  "
        f"p50={p50 * 1000:7.1f} ms  p90={p90 * 1000:7.1f} ms  "
        f"~{fps:6.1f} frames/s"
    )


def _reload_video_utils():
    import lerobot.datasets.video_utils as vu

    return importlib.reload(vu)


def _decode_pyav_range_style(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
) -> torch.Tensor:
    """Mimic LeRobot torchvision/pyav: one seek to min(ts), then read forward.

    Official ``decode_video_frames_torchvision(..., backend='pyav')`` does this
    via ``torchvision.io.VideoReader``. That API is missing in some torch/
    torchvision wheels, so this reimplements the same access pattern in raw PyAV
    for a fair algorithmic comparison against RLinf's per-timestamp seek.
    """
    ts_sorted = sorted(timestamps)
    first_ts, last_ts = ts_sorted[0], ts_sorted[-1]
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    time_base = float(stream.time_base) if stream.time_base else 1.0 / 30.0
    container.seek(
        int(first_ts / time_base), stream=stream, any_frame=False, backward=True
    )

    loaded_frames: list[np.ndarray] = []
    loaded_ts: list[float] = []
    for frame in container.decode(video=0):
        t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
        if t > last_ts + tolerance_s:
            break
        loaded_frames.append(frame.to_ndarray(format="rgb24"))
        loaded_ts.append(t)
    container.close()

    if not loaded_ts:
        raise RuntimeError(f"No frames loaded from {video_path}")

    loaded_ts_arr = np.asarray(loaded_ts)
    frames_out = []
    for query_t in timestamps:
        idx = int(np.argmin(np.abs(loaded_ts_arr - query_t)))
        if abs(loaded_ts_arr[idx] - query_t) > tolerance_s + 1e-3:
            raise AssertionError(
                f"ts {query_t} not within tol (nearest={loaded_ts_arr[idx]})"
            )
        frames_out.append(loaded_frames[idx])
    arr = np.stack(frames_out, axis=0)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)


def main(
    repo_id: str = "cobot_magic/cube_into_drawer",
    n_queries: int = 16,
    repeats: int = 3,
    seed: int = 0,
    batch_items: int = 64,
) -> None:
    """Compare decode backends on Cobot Magic LeRobot v3 videos."""
    home = Path(
        os.environ.get(
            "HF_LEROBOT_HOME",
            Path.home() / ".cache" / "huggingface" / "lerobot",
        )
    )
    root = home / repo_id
    videos = sorted((root / "videos").rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"No mp4 under {root / 'videos'}")

    rng = np.random.default_rng(seed)
    video_path = videos[0]
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30.0)
        n_frames_est = int(stream.frames or (fps * 60))
        duration_s = n_frames_est / fps

    nearby = [
        float(x) for x in np.linspace(1.0, 1.0 + (n_queries - 1) / fps, n_queries)
    ]
    # Random order inside a short window (realistic for one getitem action
    # chunk). Full-video scatter would force range-decode to scan hours of MP4.
    window_lo, window_hi = 10.0, 40.0
    scattered = [
        float(x) for x in rng.uniform(window_lo, window_hi, size=n_queries)
    ]
    patterns = {
        "nearby_window": nearby,
        "scattered_in_30s_window": scattered,
    }

    print(f"video={video_path}")
    print(
        f"size={video_path.stat().st_size / 1e6:.1f} MB  "
        f"fps≈{fps:.1f}  duration≈{duration_s:.1f}s"
    )
    print(f"n_queries={n_queries}  repeats={repeats}")
    print()

    vu = _reload_video_utils()
    tol = 1.0 / fps + 1e-3

    from rlinf.data.datasets.openpi_rlinf.pyav_video_patch import (
        _decode_video_frames_pyav,
    )

    for pattern_name, timestamps in patterns.items():
        print(f"=== pattern: {pattern_name} ({len(timestamps)} timestamps) ===")

        try:

            def run_official_pyav(ts=timestamps):
                return vu.decode_video_frames(
                    video_path, ts, tolerance_s=tol, backend="pyav"
                )

            times = _time_calls(run_official_pyav, repeats)
            _summarize("official torchvision+pyav", times, len(timestamps))
            out = run_official_pyav()
            print(f"  shape={tuple(out.shape)} dtype={out.dtype}")
        except Exception as e:
            print(
                "official torchvision+pyav      FAIL: "
                f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"
            )

        try:

            def run_torchcodec(ts=timestamps):
                return vu.decode_video_frames(
                    video_path, ts, tolerance_s=tol, backend="torchcodec"
                )

            times = _time_calls(run_torchcodec, repeats)
            _summarize("official torchcodec", times, len(timestamps))
        except Exception as e:
            print(
                "official torchcodec            FAIL: "
                f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"
            )

        try:

            def run_range(ts=timestamps):
                return _decode_pyav_range_style(video_path, ts, tolerance_s=tol)

            times = _time_calls(run_range, repeats)
            _summarize("pyav range (official-style)", times, len(timestamps))
        except Exception as e:
            print(f"pyav range (official-style)    FAIL: {type(e).__name__}: {e}")

        try:

            def run_rlinf(ts=timestamps):
                return _decode_video_frames_pyav(video_path, ts, tolerance_s=tol)

            times = _time_calls(run_rlinf, repeats)
            _summarize("rlinf seek-per-timestamp", times, len(timestamps))
        except Exception as e:
            print(f"rlinf seek-per-timestamp       FAIL: {type(e).__name__}: {e}")

        print()

    cams = []
    for cam in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        cam_vids = sorted((root / "videos" / cam).rglob("*.mp4"))
        if cam_vids:
            cams.append(cam_vids[0])
    if len(cams) == 3:
        ts1 = [5.0]
        print("=== SFT-like: 3 cameras x 1 frame @ t=5.0s ===")

        def run_rlinf_3cam():
            for p in cams:
                _decode_video_frames_pyav(p, ts1, tolerance_s=tol)

        def run_range_3cam():
            for p in cams:
                _decode_pyav_range_style(p, ts1, tolerance_s=tol)

        times = _time_calls(run_range_3cam, repeats)
        _summarize("pyav range 3-cam", times, 3)
        times = _time_calls(run_rlinf_3cam, repeats)
        _summarize("rlinf 3-cam", times, 3)
        print()

    print(
        f"=== SFT batch-like: {batch_items} items x 1 cam x 1 frame (scattered) ==="
    )
    query_ts = [
        float(x)
        for x in rng.uniform(1.0, min(120.0, max(2.0, duration_s - 0.5)), size=batch_items)
    ]

    def run_batch_rlinf():
        for t in query_ts:
            _decode_video_frames_pyav(video_path, [t], tolerance_s=tol)

    def run_batch_range():
        for t in query_ts:
            _decode_pyav_range_style(video_path, [t], tolerance_s=tol)

    n_rep = max(1, repeats // 2)
    times = _time_calls(run_batch_range, n_rep)
    _summarize("pyav range batch-like", times, batch_items)
    times = _time_calls(run_batch_rlinf, n_rep)
    _summarize("rlinf batch-like", times, batch_items)
    print()

    print("Takeaways:")
    print(
        "- LeRobot v3.0 format is fine; decoding goes through "
        "lerobot.datasets.video_utils.decode_video_frames."
    )
    print(
        "- Official default backend is torchcodec (needs system FFmpeg shared "
        "libs). On this host it fails, so training uses the RLinf PyAV patch."
    )
    print(
        "- Official fallback backend='pyav' uses torchvision.io.VideoReader; "
        "this torchvision wheel has no VideoReader, so that path also fails."
    )
    print(
        "- Slow SFT steps come mainly from many random single-frame seeks into "
        "long MP4s (3 cams/sample), not from inventing a non-LeRobot format."
    )


if __name__ == "__main__":
    tyro.cli(main)
