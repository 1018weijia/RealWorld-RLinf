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
    actor_pairwise_preference_loss,
    compute_q_node1_gap,
    compute_rlt_actor_loss,
    compute_rlt_critic_loss,
    critic_pairwise_rank_loss,
)
from rlinf.algorithms.rlt.offline_demo_transitions import make_synthetic_offline_batch
from rlinf.algorithms.rlt.preference import RewindPreferenceBuffer
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
    branch_fields_from_env_info,
    extract_rlt_obs_from_forward_inputs,
    update_rlt_transitions,
)
from rlinf.data.schema.embodied_trajectory_builder import EmbodiedTrajectoryBuilder
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer
from rlinf.envs.realworld.cobot.control import MockRewindAdapter
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
        embedding_dim=Z_DIM,
        num_heads=3,
        num_bins=8,
        hidden_dim=32,
        mlp_layers=1,
        dropout=0.0,
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


def test_realworld_intervention_metadata_contract():
    fields = branch_fields_from_env_info(
        {"rlt_bootstrap_mask": torch.tensor([0.0, 1.0])},
        batch_size=2,
        device=torch.device("cpu"),
        intervene_flags=torch.tensor([[True, False], [False, False]]),
    )
    assert fields["bootstrap_mask"].reshape(-1).tolist() == [0.0, 1.0]
    assert fields["branch_id"].reshape(-1).tolist() == [6, 0]
    assert fields["terminal_type"].reshape(-1).tolist() == [4, 0]
    assert fields["action_source"].reshape(-1).tolist() == [2, 1]


def test_intervention_branch_metadata_distinguishes_executed_action():
    fields = branch_fields_from_env_info(
        {"rlt_bootstrap_mask": torch.ones(1)},
        batch_size=1,
        device=torch.device("cpu"),
        intervene_flags=torch.zeros(1, 2, dtype=torch.bool),
    )
    assert fields["action_source"].item() == 1
    assert fields["bootstrap_mask"].item() == 1.0


def test_rewind_preference_losses_and_buffer():
    model = _policy()
    batch = _batch()
    preference = RewindPreferenceBuffer(capacity=2)
    preference.add(
        curr_obs={key: value[0] for key, value in batch["curr_obs"].items()},
        ref_chunk=batch["curr_obs"]["ref_chunk"][0],
        positive_action=batch["actions"][0],
        negative_action=batch["actions"][1],
        action_mask=torch.ones(CHUNK_LEN, ACTION_DIM, dtype=torch.bool),
        confidence=0.8,
    )
    sampled = preference.sample(4, torch.device("cpu"))
    assert sampled is not None and sampled["positive_action"].shape[0] == 1
    critic_loss, critic_metrics = critic_pairwise_rank_loss(
        model=model,
        curr_obs=sampled["curr_obs"],
        ref_chunk=sampled["ref_chunk"],
        positive_action=sampled["positive_action"],
        negative_action=sampled["negative_action"],
        action_mask=sampled["action_mask"],
        confidence=sampled["confidence"],
        margin=0.1,
    )
    action_mean, _, _ = model.sac_forward(sampled["curr_obs"], deterministic=True)
    actor_loss, actor_metrics = actor_pairwise_preference_loss(
        action_mean=action_mean,
        ref_chunk=sampled["ref_chunk"],
        positive_action=sampled["positive_action"],
        negative_action=sampled["negative_action"],
        action_mask=sampled["action_mask"],
        confidence=sampled["confidence"],
        fixed_std=model.fixed_std,
        beta=1.0,
    )
    assert torch.isfinite(critic_loss) and torch.isfinite(actor_loss)
    assert critic_metrics["preference_pair_count"] == 1.0
    assert "preference_logprob_gap" in actor_metrics


def test_mock_cobot_rewind_state_machine():
    adapter = MockRewindAdapter(action_dim=2, task="test", history_size=2)
    adapter.reset()
    adapter.execute([0.2, 0.3])
    adapter.execute([0.4, 0.5])
    adapter.request_rewind_exit(chunks_rewound=1, terminal_reward=-2.0)
    event = adapter.poll_rewind_event()
    assert event is not None and event.mode == "exit"
    assert torch.allclose(
        torch.as_tensor(adapter.rewind_chunks(event.chunks_rewound).state),
        torch.tensor([0.2, 0.3]),
    )
    adapter.request_rewind_credit(
        bad_chunks=1, terminal_reward=-1.0, prefix_reward=-0.1
    )
    credit = adapter.poll_rewind_event()
    assert (
        credit is not None and credit.mode == "credit" and credit.prefix_reward == -0.1
    )
    adapter.stop("test")
    assert adapter.execute([0.0, 0.0]).terminated


def test_rewind_event_branch_fields():
    event = SimpleNamespace(
        mode="exit",
        chunks_rewound=2,
        terminal_reward=-2.0,
        prefix_reward=-0.1,
        confidence=0.7,
    )
    fields = branch_fields_from_env_info(
        {"rlt_rewind_event": event, "rlt_recovery_root": True},
        batch_size=1,
        device=torch.device("cpu"),
    )
    assert fields["rewind_mode"].item() == 1
    assert fields["rewind_chunks"].item() == 2
    assert fields["recovery_root"].item() is True


def _rewind_trajectory() -> Trajectory:
    """One-session d,e,f,g trace with executable next-action fields."""
    steps, envs = 4, 1
    actions = torch.arange(steps * CHUNK_LEN * ACTION_DIM, dtype=torch.float32).reshape(
        steps, envs, -1
    )
    obs = {
        "z_rl": torch.randn(steps, envs, Z_DIM),
        "proprio": torch.randn(steps, envs, PROPRIO_DIM),
        "ref_chunk": torch.randn(steps, envs, CHUNK_LEN, ACTION_DIM),
    }
    return Trajectory(
        max_episode_length=steps,
        model_weights_id="rewind",
        actions=actions,
        rewards=torch.zeros(steps, envs, CHUNK_LEN),
        terminations=torch.zeros(steps, envs, 1, dtype=torch.bool),
        curr_obs=obs,
        next_obs={key: value.clone() for key, value in obs.items()},
        bootstrap_mask=torch.ones(steps, envs, 1),
        branch_id=torch.zeros(steps, envs, 1, dtype=torch.long),
        terminal_type=torch.zeros(steps, envs, 1, dtype=torch.long),
        next_action_override=torch.zeros_like(actions),
        next_action_override_mask=torch.zeros(steps, envs, 1, dtype=torch.bool),
        record_transition=torch.ones(steps, envs, 1, dtype=torch.bool),
        rewind_episode_id=torch.ones(steps, envs, 1, dtype=torch.long),
        rewind_session_id=torch.zeros(steps, envs, 1, dtype=torch.long),
        rewind_env_id=torch.zeros(steps, envs, 1, dtype=torch.long),
        rewind_chunk_id=torch.arange(steps).reshape(steps, envs, 1),
    )


def test_replay_buffer_filters_event_only_rows():
    trajectory = _rewind_trajectory()
    trajectory.record_transition[2] = False
    buffer = TrajectoryReplayBuffer(auto_save=False, sample_window_size=10)
    buffer.add_trajectories([trajectory])
    batch = buffer.sample(20)
    assert batch["record_transition"].bool().all()
    assert not (batch["rewind_chunk_id"].reshape(-1) == 2).any()


def test_critic_target_prefers_executed_next_action_override():
    class _Target:
        def __call__(self, *, forward_type, obs, actions):
            del forward_type, obs
            values = actions.reshape(actions.shape[0], -1).sum(dim=-1)
            return torch.stack([values, values], dim=-1)

    class _Model(_Target):
        def __call__(self, *, forward_type, obs, actions=None):
            if actions is None:
                return (
                    torch.zeros(obs["z_rl"].shape[0], CHUNK_LEN * ACTION_DIM),
                    None,
                    None,
                )
            return super().__call__(forward_type=forward_type, obs=obs, actions=actions)

    batch = _batch(rewards=torch.zeros(BATCH, CHUNK_LEN))
    override = torch.full_like(batch["actions"], 2.0)
    batch["next_action_override"] = override
    batch["next_action_override_mask"] = torch.ones(BATCH, 1, dtype=torch.bool)
    loss_override, metrics = compute_rlt_critic_loss(
        model=_Model(), target_model=_Target(), batch=batch, gamma=0.9
    )
    batch.pop("next_action_override")
    batch.pop("next_action_override_mask")
    loss_policy, _ = compute_rlt_critic_loss(
        model=_Model(), target_model=_Target(), batch=batch, gamma=0.9
    )
    assert (
        torch.isfinite(loss_override) and metrics["next_action_override_ratio"] == 1.0
    )
    assert not torch.allclose(loss_override, loss_policy)
