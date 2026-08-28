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
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[2] / "toolkits/inference/open_loop/run_open_loop.py"
)
SPEC = importlib.util.spec_from_file_location("open_loop", MODULE_PATH)
assert SPEC and SPEC.loader
open_loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(open_loop)


def test_chunk_request_windows_handle_final_partial_chunk():
    assert open_loop.request_windows(123, "chunk", 50) == [(0, 50), (50, 50), (100, 23)]


def test_single_step_request_windows():
    assert open_loop.request_windows(3, "single_step", 50) == [(0, 1), (1, 1), (2, 1)]
