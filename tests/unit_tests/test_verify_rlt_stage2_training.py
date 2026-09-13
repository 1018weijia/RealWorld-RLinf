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
"""Verdict logic for the RLT Stage 2 training check."""

import importlib.util
from pathlib import Path

import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "toolkits"
    / "realworld_check"
    / "verify_rlt_stage2_training.py"
)
_SPEC = importlib.util.spec_from_file_location("verify_rlt_stage2", _MODULE_PATH)
verify = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify)


def _weights(value: float) -> dict[str, torch.Tensor]:
    return {"head.weight": torch.full((2, 2), value), "head.bias": torch.zeros(2)}


def test_moved_weights_and_finite_losses_pass():
    result = verify.summarize(_weights(0.0), _weights(0.25), {"sac/critic_loss": 0.04})
    assert result["ok"]
    assert result["tensors_changed"] == 1
    assert result["max_delta"] == 0.25


def test_unchanged_weights_fail():
    """A burst that touches nothing must not be reported as training."""
    result = verify.summarize(_weights(0.5), _weights(0.5), {"sac/critic_loss": 0.04})
    assert not result["ok"]
    assert result["tensors_changed"] == 0
    assert result["max_delta"] == 0.0


def test_nan_loss_fails_even_when_weights_move():
    result = verify.summarize(
        _weights(0.0), _weights(0.25), {"sac/critic_loss": float("nan")}
    )
    assert not result["ok"]
    assert not result["losses_finite"]


def test_infinite_loss_fails():
    result = verify.summarize(
        _weights(0.0), _weights(0.25), {"sac/actor_loss": float("inf")}
    )
    assert not result["ok"]
    assert not result["losses_finite"]


def test_metrics_without_losses_are_treated_as_finite():
    """Schedule-only bursts report no loss; that alone must not fail."""
    result = verify.summarize(
        _weights(0.0), _weights(0.1), {"rlt/critic_updates_run": 55.0}
    )
    assert result["losses_finite"]
    assert result["ok"]
