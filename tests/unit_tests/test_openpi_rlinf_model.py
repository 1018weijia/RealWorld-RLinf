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
from torch import nn

from rlinf.models.embodiment.openpi_rlinf import _resolve_pi0_dtype
from rlinf.models.embodiment.openpi_rlinf.sft_action_model import (
    OpenPiPytorchSFTActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import OpenPiPytorchRLTConfig


@pytest.mark.parametrize(
    (
        "precision",
        "torch_dtype",
        "compute_precision",
        "expected_target",
        "expected_pi0",
    ),
    [
        ("fp32", None, None, torch.float32, "float32"),
        ("bf16", None, None, torch.bfloat16, "bfloat16"),
        ("bf16", torch.float32, None, torch.float32, "float32"),
        (None, None, None, None, "bfloat16"),
        ("fp32", None, "bf16", torch.float32, "bfloat16"),
    ],
)
def test_resolve_pi0_dtype(
    precision, torch_dtype, compute_precision, expected_target, expected_pi0
):
    cfg = SimpleNamespace(
        precision=precision,
        openpi=SimpleNamespace(compute_precision=compute_precision),
    )

    target_dtype, pi0_dtype = _resolve_pi0_dtype(cfg, torch_dtype)

    assert target_dtype == expected_target
    assert pi0_dtype == expected_pi0


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.k_proj = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.v_proj = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.o_proj = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


class _TinyExpertBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _TinyAttention()
        self.pre_attention_norms = nn.ModuleList([nn.LayerNorm(2), nn.LayerNorm(2)])
        self.pre_ffw_norms = nn.ModuleList([nn.LayerNorm(2), nn.LayerNorm(2)])
        self.mlps = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


class _TinyPi0(nn.Module):
    """Small parameter tree with the same action-expert names as Pi0."""

    def __init__(self):
        super().__init__()
        self.img = nn.Linear(2, 2)
        self.llm = nn.Module()
        self.llm.embedder = nn.Embedding(4, 2)
        self.llm.layers = nn.ModuleList([_TinyExpertBlock(), _TinyExpertBlock()])
        self.llm.final_norms = nn.ModuleList([nn.LayerNorm(2), nn.LayerNorm(2)])
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.time_mlp_in = nn.Linear(2, 2)
        self.time_mlp_out = nn.Linear(2, 2)


def test_legacy_rlt_honors_action_expert_scope():
    model = OpenPiPytorchSFTActionModel(
        _TinyPi0(),
        num_steps=5,
        action_env_dim=2,
        rlt_cfg=OpenPiPytorchRLTConfig(
            use_rlt=True,
            rlt_architecture="legacy",
            rlt_input_dim=2,
            rlt_embed_dim=2,
            rlt_prefix_seq_len=2,
            rlt_num_layers=1,
            rlt_num_heads=1,
            vla_finetune_scope="action_expert",
        ),
    )

    trainable_vla = {
        name
        for name, parameter in model.model.named_parameters()
        if parameter.requires_grad
    }
    expert_markers = (
        ".attn.q_proj.1.",
        ".attn.k_proj.1.",
        ".attn.v_proj.1.",
        ".attn.o_proj.1.",
        ".pre_attention_norms.1.",
        ".pre_ffw_norms.1.",
        ".mlps.1.",
    )
    expected_trainable = {
        name
        for name, _ in model.model.named_parameters()
        if any(marker in name for marker in expert_markers)
        or name.startswith("llm.final_norms.1.")
        or name.startswith(
            ("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.")
        )
    }
    assert trainable_vla == expected_trainable
    assert "llm.layers.1.attn.q_proj.0.weight" not in trainable_vla
    assert "llm.layers.1.attn.q_proj.1.weight" in trainable_vla
    assert model._vla_trainable_param_count == sum(
        parameter.numel()
        for parameter in model.model.parameters()
        if parameter.requires_grad
    )
    assert all(parameter.requires_grad for parameter in model.rlt_module.parameters())
