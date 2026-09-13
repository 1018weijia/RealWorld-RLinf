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

"""Mapping between the installed DesktopClient protocol and X2Robot EE RLT."""

from __future__ import annotations

import base64
from typing import Any, Mapping

import cv2
import numpy as np

ACTION_DIM = 14
ACTION_SCHEMA = "x2robot-ee14-v1"
CAMERA_KEYS = ("image", "wrist_image", "side_image")
EXECUTION_RECEIPT_KEY = "rlt_execution"


class MappingError(RuntimeError):
    """An observation/action cannot be interpreted without ambiguity."""


def _get(mapping: Mapping[Any, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    encoded = key.encode()
    if encoded in mapping:
        return mapping[encoded]
    raise MappingError(f"missing field {key!r}")


def _vector(value: Any, name: str, length: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise MappingError(f"{name} is not a float vector") from exc
    if result.shape != (length,):
        raise MappingError(f"{name} has shape {result.shape}, expected ({length},)")
    if not np.isfinite(result).all():
        raise MappingError(f"{name} contains NaN or Inf")
    return result


def _decode_rgb(value: Any, name: str, resize_to: tuple[int, int] | None) -> np.ndarray:
    if not isinstance(value, str) or not value:
        raise MappingError(f"{name} is missing or not base64 JPEG")
    try:
        raw = base64.b64decode(value, validate=True)
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        raise MappingError(f"{name} is not a valid image") from exc
    if bgr is None or bgr.ndim != 3 or bgr.shape[-1] != 3:
        raise MappingError(f"{name} is not a three-channel image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if resize_to is not None:
        width, height = resize_to
        if width <= 0 or height <= 0:
            raise MappingError(f"invalid resize target {resize_to}")
        source_height, source_width = rgb.shape[:2]
        scale = min(width / source_width, height / source_height)
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        resized = cv2.resize(rgb, (resized_width, resized_height))
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        top = (height - resized_height) // 2
        left = (width - resized_width) // 2
        canvas[top : top + resized_height, left : left + resized_width] = resized
        rgb = canvas
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def observation_to_rlt(
    observation: Mapping[Any, Any],
    *,
    task: str,
    resize_to: tuple[int, int] | None = (224, 224),
) -> dict[str, Any]:
    """Convert one DesktopClient observation to the raw RLT observation map."""
    state = _get(observation, "state")
    views = _get(observation, "views")
    if not isinstance(state, Mapping) or not isinstance(views, Mapping):
        raise MappingError("state and views must be mappings")
    state_14 = np.concatenate(
        [
            _vector(_get(state, "follow_left_position"), "left position", 3),
            _vector(_get(state, "follow_left_rotation"), "left rotation", 3),
            _vector(_get(state, "follow_left_gripper"), "left gripper", 1),
            _vector(_get(state, "follow_right_position"), "right position", 3),
            _vector(_get(state, "follow_right_rotation"), "right rotation", 3),
            _vector(_get(state, "follow_right_gripper"), "right gripper", 1),
        ]
    ).astype(np.float32, copy=False)
    return {
        "image": _decode_rgb(_get(views, "camera_front"), "camera_front", resize_to),
        "wrist_image": _decode_rgb(
            _get(views, "camera_left"), "camera_left", resize_to
        ),
        "side_image": _decode_rgb(
            _get(views, "camera_right"), "camera_right", resize_to
        ),
        "state": state_14,
        "prompt": str(task),
    }


def split_ee_actions(
    actions: Any,
    *,
    chunk_length: int,
    gripper_limits: tuple[float, float] = (-0.1, 4.6),
) -> tuple[dict[str, Any], np.ndarray]:
    """Validate robot-space EE14 actions and build DesktopClient output fields."""
    array = np.array(actions, dtype=np.float32, copy=True)
    if array.shape != (int(chunk_length), ACTION_DIM):
        raise MappingError(
            f"actions have shape {array.shape}, expected ({chunk_length}, {ACTION_DIM})"
        )
    if not np.isfinite(array).all():
        raise MappingError("actions contain NaN or Inf")
    low, high = map(float, gripper_limits)
    array[:, (6, 13)] = np.clip(array[:, (6, 13)], low, high)
    output = {
        "follow_left_position": array[:, 0:3],
        "follow_left_rotation": array[:, 3:6],
        "follow_left_gripper": array[:, 6],
        "follow_right_position": array[:, 7:10],
        "follow_right_rotation": array[:, 10:13],
        "follow_right_gripper": array[:, 13],
        "model_output_text": "rlt_stage2_x2robot_ee",
    }
    return output, array


def execution_receipt(
    observation: Mapping[Any, Any], *, chunk_length: int
) -> tuple[np.ndarray | None, bool]:
    """Read an optional receipt produced by a takeover-aware robot client."""
    try:
        receipt = _get(observation, EXECUTION_RECEIPT_KEY)
    except MappingError:
        return None, False
    if not isinstance(receipt, Mapping):
        raise MappingError(f"{EXECUTION_RECEIPT_KEY} must be a mapping")
    try:
        actions = _get(receipt, "action_chunk")
    except MappingError:
        actions = None
    intervention = bool(
        receipt.get("intervention", receipt.get(b"intervention", False))
    )
    if actions is None:
        if intervention:
            raise MappingError("intervention receipt is missing action_chunk")
        return None, False
    _, actual = split_ee_actions(actions, chunk_length=chunk_length)
    return actual, intervention
