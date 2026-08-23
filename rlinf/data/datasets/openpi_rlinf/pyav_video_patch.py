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

"""Patch LeRobot video decoding to use PyAV when torchcodec/FFmpeg libs are missing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_PATCHED = False


def _decode_video_frames_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
    **kwargs,
) -> torch.Tensor:
    """Decode frames with PyAV; returns float tensor ``[T, C, H, W]`` in ``[0, 1]``."""
    del backend, kwargs
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    time_base = float(stream.time_base) if stream.time_base else 1.0 / 30.0

    frames_out: list[np.ndarray] = []
    for query_t in timestamps:
        target_pts = int(query_t / time_base) if time_base > 0 else 0
        container.seek(target_pts, stream=stream, any_frame=False, backward=True)

        best = None
        best_dt = None
        for frame in container.decode(video=0):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            dt = abs(t - query_t)
            if best is None or dt < best_dt:
                best = frame
                best_dt = dt
            if t >= query_t:
                break

        if best is None or best_dt is None or best_dt > tolerance_s + 1e-3:
            container.close()
            raise AssertionError(
                f"Timestamp {query_t} not found near any frame "
                f"(nearest_dt={best_dt}, tol={tolerance_s}) in {video_path}"
            )
        frames_out.append(best.to_ndarray(format="rgb24"))

    container.close()
    arr = np.stack(frames_out, axis=0)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)


def _torchcodec_available() -> bool:
    """Return True if torchcodec can load with the current FFmpeg shared libs."""
    try:
        import torchcodec  # noqa: F401
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        return True
    except Exception:
        return False


def apply_pyav_video_decode_patch() -> None:
    """Prefer official torchcodec; fall back to PyAV when FFmpeg libs are missing.

    When ``LD_LIBRARY_PATH`` points at a conda-forge FFmpeg prefix (see
    ``toolkits/lerobot/env_ffmpeg_torchcodec.sh``), torchcodec loads and this
    patch is a no-op so LeRobot keeps its default decoder cache path.
    """
    global _PATCHED
    if _PATCHED:
        return

    if _torchcodec_available():
        _PATCHED = True
        return

    import lerobot.datasets.video_utils as video_utils

    video_utils.decode_video_frames = _decode_video_frames_pyav
    if hasattr(video_utils, "decode_video_frames_torchcodec"):
        video_utils.decode_video_frames_torchcodec = _decode_video_frames_pyav
    if hasattr(video_utils, "decode_video_frames_torchvision"):
        video_utils.decode_video_frames_torchvision = _decode_video_frames_pyav
    if hasattr(video_utils, "get_safe_default_codec"):
        video_utils.get_safe_default_codec = lambda: "pyav"
    _PATCHED = True
