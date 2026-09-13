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

"""Convert V2 measured end-pose history into an X2Robot EE14 receipt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .mapping import ACTION_DIM, MappingError


@dataclass(frozen=True)
class ExecutionReceipt:
    action_chunk: np.ndarray
    source_samples: int
    source_start_wall_ns: int
    source_end_wall_ns: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "action_chunk": self.action_chunk,
            "intervention": True,
            "source": "v2-measured-end-pose",
            "source_samples": self.source_samples,
            "source_start_wall_ns": self.source_start_wall_ns,
            "source_end_wall_ns": self.source_end_wall_ns,
        }


def _scalar_gripper(sample: Mapping[str, Any], key: str) -> float:
    value = sample.get(key)
    if value is None:
        raise MappingError(f"V2 sample is missing {key}")
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.isfinite(array[0]):
        raise MappingError(f"V2 sample {key} is empty or non-finite")
    return float(array[0])


def _pose(sample: Mapping[str, Any], side: str) -> tuple[np.ndarray, np.ndarray]:
    value = sample.get(side)
    if not isinstance(value, Mapping):
        raise MappingError(f"V2 sample is missing {side} pose")
    xyz = np.asarray(value.get("xyz"), dtype=np.float64).reshape(-1)
    quat = np.asarray(value.get("quat_xyzw"), dtype=np.float64).reshape(-1)
    if xyz.shape != (3,) or quat.shape != (4,):
        raise MappingError(f"V2 {side} pose must be xyz(3)+quat_xyzw(4)")
    if not np.isfinite(xyz).all() or not np.isfinite(quat).all():
        raise MappingError(f"V2 {side} pose contains NaN or Inf")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8 or abs(norm - 1.0) > 0.10:
        raise MappingError(f"V2 {side} quaternion norm is invalid: {norm}")
    return xyz, quat / norm


def _validated_rows(
    history: Sequence[Mapping[str, Any]], *, start_wall_ns: int | None
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    previous_ns = -1
    for sample in history:
        if not isinstance(sample, Mapping):
            raise MappingError("V2 history contains a non-mapping sample")
        if sample.get("source_kind") != "end_pose":
            raise MappingError("RLT receipt requires V2 source_kind=end_pose")
        stamp = int(sample.get("t_wall_ns", 0))
        if stamp <= previous_ns:
            raise MappingError("V2 history timestamps are not strictly increasing")
        previous_ns = stamp
        if start_wall_ns is None or stamp >= int(start_wall_ns):
            rows.append(sample)
    if not rows:
        raise MappingError("V2 history has no samples in the pending RLT window")
    return rows


def receipt_from_v2_history(
    history: Sequence[Mapping[str, Any]],
    *,
    chunk_length: int,
    start_wall_ns: int | None = None,
) -> ExecutionReceipt:
    """Resample measured dual-arm poses to one fixed-length EE14 chunk.

    The V2 ring is sampled at 50 Hz while the current policy executes at 30 Hz.
    Timestamp interpolation preserves the physical path and converts quaternion
    feedback to the same xyz+rpy+gripper representation used by the policy.
    """
    if chunk_length <= 0:
        raise MappingError("chunk_length must be positive")
    rows = _validated_rows(history, start_wall_ns=start_wall_ns)
    stamps = np.asarray([int(row["t_wall_ns"]) for row in rows], dtype=np.float64)
    values = np.empty((len(rows), ACTION_DIM), dtype=np.float64)
    left_quats = []
    right_quats = []
    for index, row in enumerate(rows):
        left_xyz, left_quat = _pose(row, "left")
        right_xyz, right_quat = _pose(row, "right")
        values[index, 0:3] = left_xyz
        values[index, 6] = _scalar_gripper(row, "left_gripper")
        values[index, 7:10] = right_xyz
        values[index, 13] = _scalar_gripper(row, "right_gripper")
        left_quats.append(left_quat)
        right_quats.append(right_quat)
    values[:, 3:6] = np.unwrap(
        Rotation.from_quat(np.asarray(left_quats)).as_euler("xyz"), axis=0
    )
    values[:, 10:13] = np.unwrap(
        Rotation.from_quat(np.asarray(right_quats)).as_euler("xyz"), axis=0
    )
    if len(rows) == 1:
        output = np.repeat(values, chunk_length, axis=0)
    else:
        target = np.linspace(stamps[0], stamps[-1], chunk_length)
        output = np.column_stack(
            [
                np.interp(target, stamps, values[:, column])
                for column in range(ACTION_DIM)
            ]
        )
    if not np.isfinite(output).all():
        raise MappingError("resampled V2 execution receipt contains NaN or Inf")
    return ExecutionReceipt(
        action_chunk=output.astype(np.float32),
        source_samples=len(rows),
        source_start_wall_ns=int(stamps[0]),
        source_end_wall_ns=int(stamps[-1]),
    )


def state_from_v2_sample(sample: Mapping[str, Any]) -> np.ndarray:
    """Convert one measured V2 sample to X2Robot xyz+rpy+gripper EE14."""
    if sample.get("source_kind") != "end_pose":
        raise MappingError("RLT terminal state requires V2 source_kind=end_pose")
    left_xyz, left_quat = _pose(sample, "left")
    right_xyz, right_quat = _pose(sample, "right")
    state = np.concatenate(
        [
            left_xyz,
            Rotation.from_quat(left_quat).as_euler("xyz"),
            [_scalar_gripper(sample, "left_gripper")],
            right_xyz,
            Rotation.from_quat(right_quat).as_euler("xyz"),
            [_scalar_gripper(sample, "right_gripper")],
        ]
    )
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise MappingError("V2 terminal state is not finite EE14")
    return state.astype(np.float32)
