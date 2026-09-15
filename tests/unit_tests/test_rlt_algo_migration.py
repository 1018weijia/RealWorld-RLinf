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

"""Algorithm migration A–E: delayed Polyak, ensemble backup, gripper, rank, clip."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt import losses as rlt_losses
from rlinf.algorithms.rlt.losses import (
    compute_rlt_critic_loss,
    critic_intervention_rank_loss,
    critic_pairwise_rank_loss,
    sample_disjoint_q_indices,
)
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import (
    DirectGaussianActor,
    EnsembleQCritic,
    RLTTD3MLPPolicy,
    clip_action,
    flat_gripper_indices,
)
from rlinf.serving.rlt.trainer import RLTStage2Trainer


class _DummyOpt:
    def __init__(self) -> None:
        self.param_groups = [
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3}
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        del set_to_none

    def step(self) -> None:
        return None


def test_polyak_only_runs_on_actor_step() -> None:
    trainer = object.__new__(RLTStage2Trainer)
    trainer.target_update_on_actor_step = True
    trainer.update_step = 1
    trainer.cfg = OmegaConf.create(
        {
            "algorithm": {"target_update_freq": 1},
            "actor": {
                "optim": {"clip_grad": 10.0},
                "critic_optim": {"clip_grad": 10.0},
            },
        }
    )
    trainer.qf_optimizer = _DummyOpt()
    trainer.optimizer = _DummyOpt()
    trainer._sample_batch = lambda: {}
    loss = torch.tensor(0.0, requires_grad=True)
    trainer.forward_critic = lambda _batch: (loss, {})
    trainer.forward_actor = lambda _batch: (loss, torch.tensor(0.0), {})
    updates: list[bool] = []
    trainer.soft_update_target_model = lambda: updates.append(True)

    critic_metrics = trainer.update_once(train_actor=False)
    assert updates == []
    assert critic_metrics["target_updated"] == 0.0

    actor_metrics = trainer.update_once(train_actor=True)
    assert updates == [True]
    assert actor_metrics["target_updated"] == 1.0

    trainer.target_update_on_actor_step = False
    updates.clear()
    trainer.update_once(train_actor=False)
    assert updates == [True]


def test_disjoint_q_indices_refuse_undersized_ensemble() -> None:
    torch.manual_seed(0)
    groups = sample_disjoint_q_indices(10, 2, 2)
    assert len(groups) == 2
    assert len(groups[0]) == len(groups[1]) == 2
    assert set(groups[0]).isdisjoint(groups[1])
    with pytest.raises(ValueError, match="need num_qs"):
        sample_disjoint_q_indices(2, 2, 2)


def test_expo_decoupled_uses_disjoint_heads_and_reports_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TargetModel:
        def __call__(self, *, forward_type, obs, actions, **_kwargs):
            del forward_type, obs
            # Four heads: two optimistic, two pessimistic on the first dim.
            first = actions[:, :1]
            return torch.cat(
                [first + 1.0, first + 1.0, first - 1.0, first - 1.0], dim=-1
            )

    class CurrentModel:
        def __call__(self, *, forward_type, obs, actions=None, **_kwargs):
            del forward_type
            if actions is None:
                return torch.zeros(obs["z_rl"].shape[0], 2), None, None
            q = obs["z_rl"][:, :1]
            return q.repeat(1, 4)

    curr = {
        "z_rl": torch.zeros(2, 1),
        "proprio": torch.zeros(2, 1),
        "ref_chunk": torch.zeros(2, 2, 1),
    }
    nxt = {key: torch.zeros_like(value) for key, value in curr.items()}
    candidates = torch.zeros(2, 3, 2)
    candidates[0, :, 0] = torch.tensor([0.1, 0.9, 0.4])
    candidates[1, :, 0] = torch.tensor([0.2, 0.3, -0.1])
    batch = {
        "curr_obs": curr,
        "next_obs": nxt,
        "actions": torch.zeros(2, 2, 1),
        "rewards": torch.zeros(2, 2),
        "dones": torch.zeros(2, 1, dtype=torch.bool),
        "terminations": torch.zeros(2, 1, dtype=torch.bool),
        "bootstrap_mask": torch.ones(2, 1),
    }
    kwargs = {
        "model": CurrentModel(),
        "target_model": TargetModel(),
        "batch": batch,
        "gamma": 1.0,
        "use_done_key": True,
        "next_actions_fn": lambda _obs: candidates,
        "critic_num_min_qs": 2,
    }
    monkeypatch.setattr(
        rlt_losses, "sample_q_indices", lambda *_args, **_kwargs: [0, 1]
    )
    monkeypatch.setattr(
        rlt_losses,
        "sample_disjoint_q_indices",
        lambda *_args, **_kwargs: ([0, 1], [2, 3]),
    )
    _, expo = compute_rlt_critic_loss(td_backup="expo", **kwargs)
    _, decoupled = compute_rlt_critic_loss(td_backup="expo_decoupled", **kwargs)
    assert expo["td_backup"] == 1.0
    assert decoupled["td_backup"] == 2.0
    assert "expo_backup_optimism_gap_mean" not in expo
    assert decoupled["expo_backup_optimism_gap_mean"] == pytest.approx(2.0)
    assert decoupled["target_q"] < expo["target_q"]


def test_inward_clip_keeps_optimizer_inward_gradient() -> None:
    high = torch.tensor([1.5], requires_grad=True)
    clip_action(high, -1.0, 1.0, gradient_mode="inward").backward(torch.tensor([1.0]))
    assert high.grad.item() == pytest.approx(1.0)

    outward = torch.tensor([1.5], requires_grad=True)
    clip_action(outward, -1.0, 1.0, gradient_mode="inward").backward(
        torch.tensor([-1.0])
    )
    assert outward.grad.item() == pytest.approx(0.0)

    hard = torch.tensor([1.5], requires_grad=True)
    clip_action(hard, -1.0, 1.0, gradient_mode="hard").backward(torch.tensor([1.0]))
    assert hard.grad.item() == pytest.approx(0.0)


def test_gripper_absolute_output_ignores_reference() -> None:
    assert flat_gripper_indices(14, 14) == (6, 13)
    actor = DirectGaussianActor(
        state_dim=2,
        action_chunk_dim=14,
        hidden_dim=4,
        num_hidden_layers=1,
        sigma=0.0,
        edit_scale=0.4,
        action_dim=14,
        gripper_absolute_output=True,
        gripper_output_scale=1.0,
    )
    state = torch.zeros(1, 2)
    reference = torch.full((1, 14), 0.5)
    action = actor.mean(state, reference)
    assert torch.allclose(action[0, :6], reference[0, :6])
    assert torch.allclose(action[0, 7:13], reference[0, 7:13])
    assert action[0, 6].item() == pytest.approx(0.0)
    assert action[0, 13].item() == pytest.approx(0.0)


def test_constructor_defaults_stay_twin_q_and_scale_0_2() -> None:
    model = RLTTD3MLPPolicy(
        z_dim=4,
        proprio_dim=2,
        action_dim=3,
        num_action_chunks=2,
    )
    assert model.q_head.num_qs == 2
    assert isinstance(model.q_head, EnsembleQCritic)
    assert model.actor.edit_scale == pytest.approx(0.2)
    assert model.actor.action_clip_gradient_mode == "hard"
    assert model.ACTOR_SEMANTIC_VERSION == 2


def test_rank_slope_scales_margin_by_action_distance() -> None:
    class QModel:
        def __call__(self, *, forward_type, obs, actions, **_kwargs):
            del forward_type, obs
            q = actions[:, :1]
            return torch.cat([q, q], dim=-1)

    obs = {"z_rl": torch.zeros(2, 1)}
    ref = torch.zeros(2, 2)
    far = torch.tensor([[1.0, 0.0], [0.5, 0.0]])
    near = torch.zeros(2, 2)
    _, constant = critic_pairwise_rank_loss(
        model=QModel(),
        curr_obs=obs,
        ref_chunk=ref,
        positive_action=far,
        negative_action=near,
        action_mask=torch.ones(2, 2),
        confidence=torch.ones(2),
        margin=0.1,
        rank_slope=0.0,
    )
    _, sloped = critic_pairwise_rank_loss(
        model=QModel(),
        curr_obs=obs,
        ref_chunk=ref,
        positive_action=far,
        negative_action=near,
        action_mask=torch.ones(2, 2),
        confidence=torch.ones(2),
        margin=0.1,
        rank_slope=0.2,
    )
    assert constant["preference_margin_mean"] == pytest.approx(0.1)
    assert sloped["preference_margin_mean"] > constant["preference_margin_mean"]


def test_intervention_rank_local_probes_and_empty_rows() -> None:
    class QModel:
        def __call__(self, *, forward_type, obs, actions, **_kwargs):
            del forward_type, obs
            q = actions[:, :1]
            return torch.cat([q, q], dim=-1)

    obs = {
        "z_rl": torch.zeros(2, 1),
        "proprio": torch.zeros(2, 1),
        "ref_chunk": torch.zeros(2, 2, 1),
    }
    human = torch.tensor([[1.0], [0.0]])
    negative = torch.zeros(2, 1)
    flags = torch.tensor([1.0, 0.0])
    loss, metrics = critic_intervention_rank_loss(
        model=QModel(),
        curr_obs=obs,
        human_action=human,
        negative_action=negative,
        intervene_flags=flags,
        slope=0.3,
        local_steps=(0.5,),
    )
    assert metrics["intervention_rank_active"] == 1.0
    assert metrics["intervention_rank_pairs"] == 1.0
    assert metrics["intervention_rank_local_probes"] == 1.0
    assert torch.isfinite(loss)

    empty, empty_metrics = critic_intervention_rank_loss(
        model=QModel(),
        curr_obs=obs,
        human_action=human,
        negative_action=negative,
        intervene_flags=torch.zeros(2),
        slope=0.3,
    )
    assert empty_metrics["intervention_rank_active"] == 0.0
    assert empty.item() == 0.0


def test_td_target_clip_min_records_fraction() -> None:
    class Twin:
        def __call__(self, *, forward_type, obs, actions=None, **_kwargs):
            del forward_type
            batch = obs["z_rl"].shape[0]
            if actions is None:
                return torch.zeros(batch, 2), None, None
            return torch.full((batch, 2), -8.0)

    curr = {
        "z_rl": torch.zeros(2, 1),
        "proprio": torch.zeros(2, 1),
        "ref_chunk": torch.zeros(2, 2, 1),
    }
    batch = {
        "curr_obs": curr,
        "next_obs": curr,
        "actions": torch.zeros(2, 2, 1),
        "rewards": torch.zeros(2, 2),
        "dones": torch.zeros(2, 1, dtype=torch.bool),
        "terminations": torch.zeros(2, 1, dtype=torch.bool),
        "bootstrap_mask": torch.ones(2, 1),
    }
    _, metrics = compute_rlt_critic_loss(
        model=Twin(),
        target_model=Twin(),
        batch=batch,
        gamma=0.99,
        use_done_key=True,
        td_target_clip_min=-1.0,
    )
    assert metrics["td_target_clipped_frac"] == pytest.approx(1.0)
    assert metrics["target_q"] == pytest.approx(-1.0)
    assert metrics["td_target_unclipped"] < -1.0
