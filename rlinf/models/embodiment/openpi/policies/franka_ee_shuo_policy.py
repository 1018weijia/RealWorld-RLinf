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
"""Franka EE 3-camera transforms matching RL-Token ``franka_ee_policy.py``."""

import dataclasses
from pathlib import Path

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_franka_ee_shuo_example() -> dict:
    """Creates a random input example for the local Franka EE policy."""
    return {
        "observation/state": np.random.rand(7).astype(np.float32),
        "observation/global_image": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation/right_image": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation/wrist_image": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "prompt": "Stack three bowls together.",
    }


def _as_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _resolve_video_path(video_path: str, dataset_root: str | None) -> Path:
    path = Path(video_path)
    if path.is_file():
        return path
    if dataset_root is not None:
        candidate = Path(dataset_root).expanduser() / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve video path {video_path!r} with dataset_root={dataset_root!r}"
    )


def _decode_video_frame(
    video_path: str, frame_index: int, dataset_root: str | None, fps: int
) -> np.ndarray:
    import av

    resolved = _resolve_video_path(video_path, dataset_root)
    container = av.open(str(resolved))
    stream = container.streams.video[0]
    try:
        if frame_index > 0 and stream.time_base is not None:
            target_pts = int((frame_index / fps) / float(stream.time_base))
            container.seek(target_pts, any_frame=False, backward=True, stream=stream)

        for decoded_offset, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and stream.time_base is not None:
                decoded_index = int(round(float(frame.pts * stream.time_base) * fps))
                if decoded_index >= frame_index:
                    return frame.to_ndarray(format="rgb24")
            elif decoded_offset == 0:
                return frame.to_ndarray(format="rgb24")
    finally:
        container.close()

    raise ValueError(f"Could not decode frame {frame_index} from {resolved}")


def _parse_image(
    image, *, frame_index: int | None, dataset_root: str | None, fps: int
) -> np.ndarray:
    if isinstance(image, bytes):
        image = image.decode("utf-8")
    if isinstance(image, str):
        return _decode_video_frame(image, frame_index or 0, dataset_root, fps)

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaEEShuoInputs(transforms.DataTransformFn):
    """Maps local 7D Franka EE LeRobot samples into OpenPI model inputs.

    Cameras: ``global_image`` → ``base_0_rgb``, ``wrist_image`` →
    ``left_wrist_0_rgb``, ``right_image`` → ``right_wrist_0_rgb``. All three
    views are unmasked, matching ``pi05_stack_bowls_rc_franka_ee_local``.
    """

    model_type: _model.ModelType
    dataset_root: str | None = None
    fps: int = 30

    def __call__(self, data: dict) -> dict:
        frame_index = _as_int(data.get("frame_index", 0))
        dataset_root = data.get("dataset_root", self.dataset_root)
        base_image = _parse_image(
            data["observation/global_image"],
            frame_index=frame_index,
            dataset_root=dataset_root,
            fps=self.fps,
        )
        right_image = _parse_image(
            data["observation/right_image"],
            frame_index=frame_index,
            dataset_root=dataset_root,
            fps=self.fps,
        )
        wrist_image = _parse_image(
            data["observation/wrist_image"],
            frame_index=frame_index,
            dataset_root=dataset_root,
            fps=self.fps,
        )

        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaEEShuoOutputs(transforms.DataTransformFn):
    """Maps model action chunks back to the 7D Franka EE action space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
