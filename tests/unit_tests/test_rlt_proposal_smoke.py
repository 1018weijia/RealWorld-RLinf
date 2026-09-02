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

"""Minimal proposal smokes: Stage-2 losses, progress head, branch/transition."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from rlinf.algorithms.rlt.losses import (
    compute_q_node1_gap,
    compute_rlt_actor_loss,
    compute_rlt_critic_loss,
)
from rlinf.algorithms.rlt.offline_demo_transitions import make_synthetic_offline_batch
from rlinf.algorithms.rlt.progress_head import (
    ProgressHeadEnsemble,
    progress_d2_ranking_loss,
    progress_head_loss,
    voc_progress_labels,
)
from rlinf.algorithms.rlt.transition import (
    RLT_BRANCH_FIELDS,
    RLT_OBS_KEYS,
    annotate_rlt_branch_fields,
    extract_rlt_obs_from_forward_inputs,
    update_rlt_transitions,
)
from rlinf.data.schema.embodied_trajectory_builder import EmbodiedTrajectoryBuilder
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy

Z_DIM, PROPRIO_DIM, ACTION_DIM, CHUNK_LEN, BATCH = 16, 4, 4, 3, 4


def _policy() -> RLTMLPPolicy:
    torch.manual_seed(0)
    return RLTMLPPolicy(
        z_dim=Z_DIM,
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        num_action_chunks=CHUNK_LEN,
    )


def _batch(**extra) -> dict:
    batch = {
        "curr_obs": {
            "z_rl": torch.randn(BATCH, Z_DIM),
            "proprio": torch.randn(BATCH, PROPRIO_DIM),
            "ref_chunk": torch.randn(BATCH, CHUNK_LEN, ACTION_DIM),
        },
        "next_obs": {
            "z_rl": torch.randn(BATCH, Z_DIM),
            "proprio": torch.randn(BATCH, PROPRIO_DIM),
            "ref_chunk": torch.randn(BATCH, CHUNK_LEN, ACTION_DIM),
        },
        "actions": torch.randn(BATCH, CHUNK_LEN, ACTION_DIM),
        "rewards": torch.randn(BATCH, CHUNK_LEN),
        "terminations": torch.zeros(BATCH, 1, dtype=torch.bool),
        "intervene_flags": torch.zeros(BATCH, CHUNK_LEN, dtype=torch.bool),
    }
    batch.update(extra)
    return batch


def test_stage2_losses_bootstrap_and_offline():
    model = _policy()
    target = copy.deepcopy(model)
    batch = _batch()
    critic_loss, critic_m = compute_rlt_critic_loss(
        model=model, target_model=target, batch=batch, gamma=0.99
    )
    actor_loss, _, actor_m = compute_rlt_actor_loss(
        model=model,
        batch=batch,
        chunk_len=CHUNK_LEN,
        action_dim=ACTION_DIM,
        q_weight=0.1,
        bc_weight=1.0,
    )
    assert torch.isfinite(critic_loss) and torch.isfinite(actor_loss)
    assert "q_data" in critic_m and "bc_loss" in actor_m

    loss_on, _ = compute_rlt_critic_loss(
        model=model,
        target_model=target,
        batch={**batch, "bootstrap_mask": torch.ones(BATCH, 1)},
        gamma=0.99,
    )
    loss_off, _ = compute_rlt_critic_loss(
        model=model,
        target_model=target,
        batch={**batch, "bootstrap_mask": torch.zeros(BATCH, 1)},
        gamma=0.99,
    )
    assert not torch.allclose(loss_on, loss_off)

    gap = compute_q_node1_gap(
        model=model,
        curr_obs={k: v[:2] for k, v in batch["curr_obs"].items()},
        human_actions=torch.randn(2, CHUNK_LEN, ACTION_DIM),
        bad_actions=torch.randn(2, CHUNK_LEN, ACTION_DIM),
    )
    assert abs(gap["q_node1_gap"] - (gap["q_node1_human"] - gap["q_node1_bad"])) < 1e-5

    offline = make_synthetic_offline_batch(
        num_chunks=4,
        z_dim=Z_DIM,
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        chunk_length=CHUNK_LEN,
        success=True,
        seed=1,
    )
    assert offline["bootstrap_mask"][-1].item() == 0.0
    assert offline["rewards"][-1, -1].item() == 1.0


def test_progress_head_voc_and_d2():
    torch.manual_seed(0)
    head = ProgressHeadEnsemble(
        embedding_dim=Z_DIM, num_heads=3, num_bins=8, hidden_dim=32, mlp_layers=1, dropout=0.0
    )
    labels = voc_progress_labels(BATCH)
    out = head(torch.randn(BATCH, Z_DIM))
    loss, metrics = progress_head_loss(out, labels, torch.ones(BATCH, 3))
    assert torch.isfinite(loss) and "progress_ce_loss" in metrics
    d2_loss, d2_m = progress_d2_ranking_loss(
        torch.tensor([0.7, 0.4]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.6, 0.4]),
        torch.tensor([1.0, 0.0]),
        margin=0.05,
    )
    assert torch.isfinite(d2_loss) and d2_m["progress_d2_rank_loss"] > 0.0


def test_branch_fields_and_transition_update():
    assert "bootstrap_mask" in RLT_BRANCH_FIELDS
    assert Trajectory().bootstrap_mask is None

    builder = EmbodiedTrajectoryBuilder()
    fields = annotate_rlt_branch_fields(
        batch_size=2, bootstrap_mask=0.0, progress_label=0.5, progress_mask=True
    )
    builder.append_transitions(
        {
            "z_rl": torch.randn(2, 8),
            "proprio": torch.randn(2, 4),
            "ref_chunk": torch.randn(2, 3, 4),
        },
        {
            "z_rl": torch.randn(2, 8),
            "proprio": torch.randn(2, 4),
            "ref_chunk": torch.randn(2, 3, 4),
        },
        branch_fields=fields,
    )
    builder.actions.append(torch.zeros(2, 12))
    builder.rewards.append(torch.zeros(2, 3))
    traj = builder.to_trajectory()
    assert float(traj.bootstrap_mask.mean()) == 0.0

    curr = {
        "z_rl": torch.randn(2, 8),
        "proprio": torch.randn(2, 4),
        "ref_chunk": torch.randn(2, 3, 4),
    }
    nxt = {k: torch.randn_like(v) for k, v in curr.items()}
    fi = {**curr, **{f"rlt_transition_{k}": v for k, v in nxt.items()}}
    pairs = []

    class _B:
        def append_transitions(self, a, b, branch_fields=None):
            pairs.append((a, b, branch_fields))

    pending = [curr]
    update_rlt_transitions(
        0, pending, [_B()], SimpleNamespace(forward_inputs=fi), cache_current=True
    )
    assert len(pairs) == 1
    assert set(pending[0].keys()) == set(RLT_OBS_KEYS)

    try:
        extract_rlt_obs_from_forward_inputs({"z_rl": torch.zeros(1, 2)})
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "Missing RLT forward_inputs keys" in str(exc)
