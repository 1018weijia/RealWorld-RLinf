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

import math

import torch
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager


def _make_manager() -> FSDPModelManager:
    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager._cfg = OmegaConf.create(
        {
            "optim": {"clip_grad": 1.0},
            "fsdp_config": {"sharding_strategy": "no_shard"},
        }
    )
    manager._dp_group = None
    manager.last_optimizer_group_grad_norms = {}
    return manager


def test_clip_grad_norm_by_optimizer_group():
    manager = _make_manager()
    rlt_param = torch.nn.Parameter(torch.zeros(2))
    vla_param = torch.nn.Parameter(torch.zeros(2))
    rlt_param.grad = torch.tensor([3.0, 4.0])
    vla_param.grad = torch.tensor([0.3, 0.4])
    manager.optimizer = torch.optim.SGD(
        [
            {"params": [rlt_param], "group_name": "rlt"},
            {"params": [vla_param], "group_name": "vla"},
        ],
        lr=0.1,
    )

    total_norm = manager._clip_grad_norm_by_optimizer_group()

    assert math.isclose(total_norm, math.sqrt(25.25), rel_tol=1e-6)
    assert math.isclose(rlt_param.grad.norm().item(), 1.0, rel_tol=2e-6)
    assert math.isclose(vla_param.grad.norm().item(), 0.5, rel_tol=1e-6)
    assert manager.last_optimizer_group_grad_norms == {"rlt": 5.0, "vla": 0.5}


def test_optimizer_step_keeps_legacy_global_clip_by_default():
    class _Scaler:
        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            pass

        def update(self):
            pass

    class _Strategy:
        def clip_grad_norm_(self, model):
            return 7.0

    manager = _make_manager()
    manager.optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1)
    manager.optimizer_steps = 0
    manager.critic_warmup_steps = 0
    manager.grad_scaler = _Scaler()
    manager._strategy = _Strategy()
    manager.model = torch.nn.Linear(1, 1)
    manager.last_optimizer_group_grad_norms = {"stale": 1.0}

    grad_norm, learning_rates = manager.optimizer_step()

    assert grad_norm == 7.0
    assert learning_rates == [0.1]
    assert manager.last_optimizer_group_grad_norms == {}
