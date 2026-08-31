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
"""Unit tests for action-chunk replay scheduling."""

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[2] / "toolkits/inference/open_loop/run_open_loop.py"
)
SPEC = importlib.util.spec_from_file_location("open_loop", MODULE_PATH)
assert SPEC and SPEC.loader
open_loop = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = open_loop
SPEC.loader.exec_module(open_loop)


def test_chunk_request_windows_handle_final_partial_chunk():
    assert open_loop.request_windows(123, "chunk", 50) == [(0, 50), (50, 50), (100, 23)]


def test_single_step_request_windows():
    assert open_loop.request_windows(3, "single_step", 50) == [(0, 1), (1, 1), (2, 1)]


def test_request_windows_reject_invalid_mode():
    import pytest

    with pytest.raises(ValueError, match="unsupported execution mode"):
        open_loop.request_windows(3, "invalid", 50)


def test_task_spec_requires_all_logical_cameras():
    import pytest

    with pytest.raises(ValueError, match="cameras must define"):
        open_loop.TaskSpec.from_mapping(
            "custom",
            {
                "robot": "custom",
                "prompt": "do task",
                "port": 9000,
                "dataset": "/tmp/data",
                "cameras": {"high": "observation.images.head"},
            },
        )


def test_task_spec_supports_custom_camera_mapping_and_indices():
    spec = open_loop.TaskSpec.from_mapping(
        "custom",
        {
            "robot": "custom",
            "prompt": "do task",
            "port": 9000,
            "dataset": "/tmp/data",
            "cameras": {
                "high": "observation.images.head",
                "left_wrist": "observation.images.left",
                "right_wrist": "observation.images.right",
            },
            "state_indices": [0, 2],
            "action_indices": [1, 3],
            "action_dimensions": ["a", "b"],
        },
    )
    assert spec.state_indices == (0, 2)
    assert spec.action_indices == (1, 3)
    assert spec.cameras["right_wrist"] == "observation.images.right"


def test_build_payload_uses_metadata_aligned_camera_features():
    import torch

    spec = open_loop.TaskSpec.from_mapping(
        "custom",
        {
            "robot": "custom",
            "prompt": "do task",
            "port": 9000,
            "dataset": "/tmp/data",
            "cameras": {
                "high": "head",
                "left_wrist": "left",
                "right_wrist": "right",
            },
        },
    )
    sample = {
        "observation.state": torch.arange(14, dtype=torch.float32),
        "head": torch.zeros(3, 4, 4),
        "left": torch.ones(3, 4, 4),
        "right": torch.full((3, 4, 4), 0.5),
    }
    payload = open_loop.build_payload(sample, spec)
    assert payload["state"] == list(range(14))
    assert payload["cam_high_b64"] != payload["cam_left_wrist_b64"]
    assert payload["cam_left_wrist_b64"] != payload["cam_right_wrist_b64"]


def test_evaluator_has_no_manual_video_shard_formula():
    source = MODULE_PATH.read_text()
    assert "episode // chunks_size" not in source
    assert "episode_video(" not in source
