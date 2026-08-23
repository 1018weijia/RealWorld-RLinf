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

"""Speed up LeRobot v3 delta queries for vector columns (e.g. action chunks)."""

from __future__ import annotations

from typing import Any

import torch

_PATCHED = False


def _is_contiguous(indices: list[int]) -> bool:
    if len(indices) <= 1:
        return True
    return indices[-1] - indices[0] + 1 == len(indices)


def _fast_query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
    """Row-first / slice batch fetch; avoids slow ``hf_dataset[key][indices]``."""
    result: dict[str, Any] = {}
    for key, q_idx in query_indices.items():
        if key in self.meta.video_keys:
            continue
        relative_indices = (
            q_idx
            if self._absolute_to_relative_idx is None
            else [self._absolute_to_relative_idx[idx] for idx in q_idx]
        )
        if not relative_indices:
            continue

        if _is_contiguous(relative_indices):
            lo = relative_indices[0]
            hi = relative_indices[-1] + 1
            rows = self.hf_dataset[lo:hi]
            result[key] = torch.stack(rows[key])
            continue

        try:
            rows = self.hf_dataset[relative_indices]
            result[key] = torch.stack(rows[key])
        except (KeyError, TypeError, IndexError):
            result[key] = torch.stack(self.hf_dataset[relative_indices][key])
    return result


def apply_lerobot_hf_query_patch() -> None:
    """Replace LeRobotDataset._query_hf_dataset with the fast implementation."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if getattr(LeRobotDataset._query_hf_dataset, "_rlinf_fast_hf_query", False):
        _PATCHED = True
        return

    _fast_query_hf_dataset._rlinf_fast_hf_query = True  # type: ignore[attr-defined]
    LeRobotDataset._query_hf_dataset = _fast_query_hf_dataset
    _PATCHED = True
