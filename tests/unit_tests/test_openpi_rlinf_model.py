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

"""Unit tests for OpenPI RLinf model construction helpers."""

from types import SimpleNamespace

import pytest
import torch

from rlinf.models.embodiment.openpi_rlinf import _resolve_pi0_dtype


@pytest.mark.parametrize(
    ("precision", "torch_dtype", "expected_target", "expected_pi0"),
    [
        ("fp32", None, torch.float32, "float32"),
        ("bf16", None, torch.bfloat16, "bfloat16"),
        ("bf16", torch.float32, torch.float32, "float32"),
        (None, None, None, "bfloat16"),
    ],
)
def test_resolve_pi0_dtype(precision, torch_dtype, expected_target, expected_pi0):
    cfg = SimpleNamespace(precision=precision)

    target_dtype, pi0_dtype = _resolve_pi0_dtype(cfg, torch_dtype)

    assert target_dtype == expected_target
    assert pi0_dtype == expected_pi0
