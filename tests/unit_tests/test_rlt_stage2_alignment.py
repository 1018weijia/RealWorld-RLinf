# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Golden Stage 2 alignment tests against rlt-openpi remote-franka semantics."""

from __future__ import annotations

import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.losses import compute_rlt_critic_loss
from rlinf.algorithms.rlt.preference import RewindPreferenceBuffer
from rlinf.algorithms.rlt.transition import (
    ACTION_SOURCE_HUMAN,
    ACTION_SOURCE_POLICY,
    branch_fields_from_env_info,
)
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer
from rlinf.data.storage.replay.dataset import ReplayBufferDataset
from rlinf.envs.realworld.cobot.cobot_env import CobotEnv
from rlinf.envs.realworld.cobot.control import (
    CobotRewindEvent,
    CobotStepResult,
    MockRewindAdapter,
)
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import (
    DirectGaussianActor,
    RLTTD3MLPPolicy,
)
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

Z_DIM = 3
PROPRIO_DIM = 2
ACTION_DIM = 2
CHUNK_LEN = 3


def _cobot_override(**updates):
    config = {
        "is_dummy": True,
        "action_dim": ACTION_DIM,
        "state_dim": ACTION_DIM,
        "image_keys": ["image", "wrist_image", "side_image"],
        "session_id": 101,
        "rewind_history_chunks": 8,
    }
    config.update(updates)
    return config


def _realworld_cfg():
    return OmegaConf.create(
        {
            "override_cfg": _cobot_override(),
            "video_cfg": {"save_video": False},
            "seed": 0,
            "use_fixed_reset_state_ids": False,
            "auto_reset": True,
            "ignore_terminations": False,
            "group_size": 1,
            "main_image_key": "image",
            "max_episode_steps": 96,
            "manual_episode_control_only": False,
            "init_params": {"id": "CobotEnv-v1"},
            "rlt_intervention_metadata": True,
        }
    )


def _obs(steps: int = 1):
    return {
        "z_rl": torch.arange(steps * Z_DIM, dtype=torch.float32).reshape(
            steps, 1, Z_DIM
        ),
        "proprio": torch.zeros(steps, 1, PROPRIO_DIM),
        "ref_chunk": torch.zeros(steps, 1, CHUNK_LEN, ACTION_DIM),
    }


def _trajectory(
    chunk_ids,
    *,
    sources=None,
    records=None,
    dones=None,
    episode_id: int = 1,
    session_id: int = 101,
):
    chunk_ids = list(chunk_ids)
    steps = len(chunk_ids)
    actions = torch.arange(steps * CHUNK_LEN * ACTION_DIM, dtype=torch.float32).reshape(
        steps, 1, CHUNK_LEN, ACTION_DIM
    )
    actions = actions / max(float(actions.max()), 1.0)
    if sources is None:
        sources = [ACTION_SOURCE_POLICY] * steps
    if records is None:
        records = [True] * steps
    if dones is None:
        dones = [False] * steps
    obs = _obs(steps)
    return Trajectory(
        max_episode_length=steps,
        model_weights_id="alignment",
        actions=actions,
        rewards=torch.zeros(steps, 1, CHUNK_LEN),
        dones=torch.tensor(dones, dtype=torch.bool).reshape(steps, 1, 1),
        terminations=torch.tensor(dones, dtype=torch.bool).reshape(steps, 1, 1),
        truncations=torch.zeros(steps, 1, 1, dtype=torch.bool),
        curr_obs=obs,
        next_obs={key: value.clone() for key, value in obs.items()},
        intervene_flags=torch.zeros(steps, 1, CHUNK_LEN, dtype=torch.bool),
        bootstrap_mask=torch.ones(steps, 1, 1),
        branch_id=torch.zeros(steps, 1, 1, dtype=torch.long),
        terminal_type=torch.zeros(steps, 1, 1, dtype=torch.long),
        action_source=torch.tensor(sources).reshape(steps, 1, 1),
        recovery_root=torch.zeros(steps, 1, 1, dtype=torch.bool),
        next_action_override=torch.zeros_like(actions),
        next_action_override_mask=torch.zeros(steps, 1, 1, dtype=torch.bool),
        record_transition=torch.tensor(records, dtype=torch.bool).reshape(steps, 1, 1),
        rewind_episode_id=torch.full((steps, 1, 1), episode_id, dtype=torch.long),
        rewind_session_id=torch.full((steps, 1, 1), session_id, dtype=torch.long),
        rewind_env_id=torch.zeros(steps, 1, 1, dtype=torch.long),
        rewind_chunk_id=torch.tensor(chunk_ids).reshape(steps, 1, 1),
    )


def _policy():
    return RLTTD3MLPPolicy(
        z_dim=Z_DIM,
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        num_action_chunks=CHUNK_LEN,
        ref_num_action_chunks=CHUNK_LEN,
        mlp_hidden_dim=8,
        mlp_num_hidden_layers=1,
        actor_noise_sigma=0.2,
        residual_scale=0.2,
        expo_num_base_samples=4,
        expo_num_edit_samples=4,
    )


def _rewind_worker():
    worker = object.__new__(RLTACFSDPPolicy)
    worker.replay_buffer = TrajectoryReplayBuffer(
        auto_save=False, sample_window_size=20
    )
    worker.rewind_preference_buffer = RewindPreferenceBuffer(16)
    worker._pending_rewind_forks = {}
    worker._rewind_rows = {}
    worker._recovery_root_pending = {}
    worker._processed_rewind_events = set()
    worker._committed_transition_ids = set()
    worker._processed_episode_ends = set()
    worker.log_warning = lambda *_args, **_kwargs: None
    return worker


def test_cobot_schema_and_sync_vector_reset():
    env = CobotEnv(_cobot_override())
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    vector = gym.vector.SyncVectorEnv([lambda: CobotEnv(_cobot_override())])
    vector_obs, _ = vector.reset()
    assert vector.observation_space.contains(vector_obs)
    vector.close()
    env.close()


def test_event_only_chunk_executes_no_policy_action_and_recovers_at_boundary():
    env = RealWorldEnv(_realworld_cfg(), 1, 0, 1, None)
    env.reset()
    first = torch.tensor([[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]])
    env.chunk_step(first)
    adapter = env.env.envs[0].unwrapped._adapter
    calls_before = adapter.execute_calls
    adapter.request_rewind_exit(chunks_rewound=1, terminal_reward=-1.0)

    obs_list, rewards, terms, truncs, infos_list = env.chunk_step(
        torch.full_like(first, 0.9)
    )
    info = infos_list[-1]
    assert adapter.execute_calls == calls_before
    assert rewards.shape == terms.shape == truncs.shape == (1, CHUNK_LEN)
    assert not info["record_transition"].any()
    assert len(info["rewind_events"]) == 1
    assert info["rewind_events"][0].chunk_id == 0
    assert torch.allclose(obs_list[-1]["states"], torch.zeros(1, ACTION_DIM))

    recovery = torch.tensor([[[0.4, 0.5], [0.5, 0.6], [0.6, 0.7]]])
    _, _, _, _, recovery_infos = env.chunk_step(recovery)
    assert adapter.execute_calls == calls_before + CHUNK_LEN
    assert recovery_infos[-1]["rlt_recovery_root"].all()
    env.close()


def test_credit_event_does_not_move_robot_and_adapter_identity_restores():
    adapter = MockRewindAdapter(ACTION_DIM, "test", session_id=77)
    adapter.reset()
    adapter.on_action_chunk_begin()
    adapter.execute([0.2, 0.3])
    adapter.on_action_chunk_end(True)
    state = adapter.state_dict()
    adapter.request_rewind_credit(bad_chunks=1, terminal_reward=-1.0, prefix_reward=0.1)
    before = adapter.observe().state.copy()
    event = adapter.poll_rewind_event()
    assert event.mode == "credit" and event.chunk_id == 0
    assert (adapter.observe().state == before).all()

    restored = MockRewindAdapter(ACTION_DIM, "test", session_id=1)
    restored.load_state_dict(state)
    restored.on_action_chunk_begin()
    result = restored.execute([0.4, 0.5])
    assert result.info["rlt_session_id"] == 77
    assert result.info["rlt_chunk_id"] == 1


def test_transition_metadata_rejects_partial_and_preserves_override():
    override = torch.full((1, CHUNK_LEN, ACTION_DIM), 0.25)
    fields = branch_fields_from_env_info(
        {
            "record_transition": torch.tensor([[True, False, True]]),
            "next_action_override": override,
            "next_action_override_mask": torch.tensor([True]),
        },
        batch_size=1,
        device=torch.device("cpu"),
    )
    assert not fields["record_transition"].item()
    assert fields["next_action_override_mask"].dtype == torch.bool
    assert torch.equal(fields["next_action_override"], override)
    with pytest.raises(ValueError, match="batch mismatch"):
        branch_fields_from_env_info(
            {"rlt_bootstrap_mask": torch.ones(2)},
            batch_size=3,
            device=torch.device("cpu"),
        )


def test_mixed_takeover_chunk_records_only_normalized_executed_actions():
    class TakeoverAdapter(MockRewindAdapter):
        def execute(self, action):
            result = super().execute(action)
            info = dict(result.info)
            if self.execute_calls == 2:
                executed = -np.asarray(action, dtype=np.float32)
                self._state = executed.copy()
                info.update(
                    executed_action=executed,
                    intervene_action=executed,
                    intervene_flag=True,
                )
                result = CobotStepResult(self.observe(), result.reward, info=info)
            return result

    env = RealWorldEnv(_realworld_cfg(), 1, 0, 1, None)
    adapter = TakeoverAdapter(ACTION_DIM, "test", session_id=101)
    env.env.envs[0].unwrapped._adapter = adapter
    env.reset()
    proposal = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])
    _, _, _, _, infos = env.chunk_step(proposal)
    executed = infos[-1]["intervene_action"].reshape_as(proposal)
    assert torch.equal(executed[:, 0], proposal[:, 0])
    assert torch.equal(executed[:, 1], -proposal[:, 1])
    assert torch.equal(executed[:, 2], proposal[:, 2])
    assert infos[-1]["intervene_flag"].reshape(-1).tolist() == [False, True, False]
    assert infos[-1]["record_transition"].all()
    env.close()


@pytest.mark.parametrize(
    ("safety_fault", "terminated", "truncated", "expected_bootstrap"),
    [(True, True, False, 0.0), (False, False, True, 1.0)],
)
def test_fault_or_timeout_stops_and_discards_partial_chunk(
    safety_fault, terminated, truncated, expected_bootstrap
):
    class InterruptingAdapter(MockRewindAdapter):
        def execute(self, action):
            result = super().execute(action)
            return CobotStepResult(
                result.observation,
                result.reward,
                terminated=terminated,
                truncated=truncated,
                info={**result.info, "rlt_safety_fault": safety_fault},
            )

    cfg = _realworld_cfg()
    cfg.auto_reset = False
    env = RealWorldEnv(cfg, 1, 0, 1, None)
    adapter = InterruptingAdapter(ACTION_DIM, "test", session_id=101)
    env.env.envs[0].unwrapped._adapter = adapter
    env.reset()
    proposal = torch.full((1, CHUNK_LEN, ACTION_DIM), 0.2)
    _, _, terms, truncs, infos = env.chunk_step(proposal)
    assert adapter.execute_calls == 1
    assert not infos[-1]["record_transition"].all()
    assert float(infos[-1]["rlt_bootstrap_mask"][0]) == expected_bootstrap
    assert bool(terms.any()) is terminated
    assert bool(truncs.any()) is truncated
    env.close()


@pytest.mark.parametrize("source", [ACTION_SOURCE_HUMAN, ACTION_SOURCE_POLICY])
def test_first_valid_post_rewind_action_becomes_positive(source):
    worker = _rewind_worker()
    history = _trajectory([0, 1])
    worker.replay_buffer.add_trajectories([history])
    worker._append_replay_rows(history, 0)
    worker._patch_rewind_event(
        CobotRewindEvent(
            mode="exit",
            chunks_rewound=1,
            terminal_reward=-1.0,
            episode_id=1,
            session_id=101,
            chunk_id=1,
        )
    )
    replacement = _trajectory(
        [1, 2, 3],
        sources=[ACTION_SOURCE_HUMAN, source, source],
        records=[True, False, True],
    )
    worker._consume_rewind_preferences(replacement)
    sampled = worker.rewind_preference_buffer.sample(1, torch.device("cpu"))
    assert len(worker.rewind_preference_buffer) == 1
    assert torch.equal(sampled["positive_action"][0], replacement.actions[2, 0])
    anchor = worker.replay_buffer._load_trajectory(0, "alignment")
    assert anchor.next_action_override_mask[0, 0].item()
    assert torch.equal(anchor.next_action_override[0, 0], replacement.actions[2, 0])
    worker._consume_rewind_preferences(replacement)
    assert len(worker.rewind_preference_buffer) == 1


def test_residual_actor_and_checkpoint_semantic_golden():
    actor = DirectGaussianActor(
        state_dim=2,
        action_chunk_dim=3,
        hidden_dim=4,
        num_hidden_layers=1,
        sigma=0.2,
        edit_scale=0.2,
    )
    state = torch.zeros(2, 2)
    reference = torch.tensor([[0.1, -0.2, 0.3], [0.9, -0.9, 0.0]])
    assert torch.equal(actor.mean(state, reference), reference)
    last = [
        module for module in actor.mlp.modules() if isinstance(module, torch.nn.Linear)
    ][-1]
    with torch.no_grad():
        last.bias.fill_(2.0)
    # OpenPI quantile normalization lets a valid reference exceed [-1, 1], so
    # the residual is bounded by the configured clip, not by unit range.
    assert (actor.action_clip_min, actor.action_clip_max) == (-1.4, 1.4)
    expected = (reference + 0.2 * torch.tanh(torch.tensor(2.0))).clamp(-1.4, 1.4)
    assert torch.allclose(actor.mean(state, reference), expected)

    tight = DirectGaussianActor(
        state_dim=2,
        action_chunk_dim=3,
        hidden_dim=4,
        num_hidden_layers=1,
        sigma=0.0,
        edit_scale=0.2,
        action_clip_min=-0.5,
        action_clip_max=0.5,
    )
    assert torch.equal(tight.mean(state, reference), reference.clamp(-0.5, 0.5))

    model = _policy()
    assert model.actor_semantic_version.item() == 2
    legacy = model.state_dict()
    legacy.pop("actor_semantic_version")
    with pytest.raises(RuntimeError, match="direct-action checkpoints"):
        model.load_state_dict(legacy, strict=False)


def test_expo_selects_best_of_four_base_and_four_edits():
    class FirstCoordinateCritic(torch.nn.Module):
        def forward(self, _state, action):
            value = action[:, :1]
            return torch.cat([value, value], dim=-1)

    model = _policy()
    model.selection_critic = FirstCoordinateCritic()
    candidates = torch.zeros(1, 4, CHUNK_LEN, ACTION_DIM)
    candidates[0, :, 0, 0] = torch.tensor([0.1, 0.9, 0.4, 0.2])
    obs = {
        "z_rl": torch.zeros(1, Z_DIM),
        "proprio": torch.zeros(1, PROPRIO_DIM),
        "ref_chunk": candidates[:, 0],
        "ref_candidates": candidates,
    }
    action, selected_base = model.select_expo_action(obs, exploration=False)
    assert model.build_expo_candidates(obs, exploration=False)[0].shape[1] == 8
    assert action[0, 0].item() == pytest.approx(0.9)
    assert selected_base[0, 0].item() == pytest.approx(0.9)


def test_huber_twin_loss_and_expo_override_match_franka_golden():
    class CurrentModel:
        def __call__(self, *, forward_type, obs, actions=None, **_kwargs):
            del forward_type
            if actions is None:
                return (
                    torch.zeros(obs["z_rl"].shape[0], CHUNK_LEN * ACTION_DIM),
                    None,
                    None,
                )
            q = obs["z_rl"][:, :1]
            return torch.cat([q, q], dim=-1)

    class TargetModel:
        def __call__(self, *, forward_type, obs, actions, **_kwargs):
            del forward_type, obs
            q = actions[:, :1]
            return torch.cat([q, q], dim=-1)

    curr = {
        "z_rl": torch.tensor([[0.0], [2.0]]),
        "proprio": torch.zeros(2, 1),
        "ref_chunk": torch.zeros(2, CHUNK_LEN, ACTION_DIM),
    }
    nxt = {key: torch.zeros_like(value) for key, value in curr.items()}
    candidates = torch.zeros(2, 3, CHUNK_LEN * ACTION_DIM)
    candidates[0, :, 0] = torch.tensor([0.1, 0.9, 0.4])
    candidates[1, :, 0] = torch.tensor([0.2, 0.3, -0.1])
    override = torch.zeros(2, CHUNK_LEN * ACTION_DIM)
    override[0, 0] = 0.75
    batch = {
        "curr_obs": curr,
        "next_obs": nxt,
        "actions": torch.zeros(2, CHUNK_LEN, ACTION_DIM),
        "rewards": torch.zeros(2, CHUNK_LEN),
        "dones": torch.zeros(2, 1, dtype=torch.bool),
        "terminations": torch.zeros(2, 1, dtype=torch.bool),
        "bootstrap_mask": torch.ones(2, 1),
        "next_action_override": override,
        "next_action_override_mask": torch.tensor([[True], [False]]),
    }
    loss, metrics = compute_rlt_critic_loss(
        model=CurrentModel(),
        target_model=TargetModel(),
        batch=batch,
        gamma=1.0,
        use_done_key=True,
        next_actions_fn=lambda _obs: candidates,
        critic_loss_type="huber",
        critic_huber_delta=0.5,
    )
    assert metrics["target_q"] == pytest.approx((0.75 + 0.3) / 2)
    expected = (
        torch.nn.functional.huber_loss(
            torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
            torch.tensor([[0.75, 0.75], [0.3, 0.3]]),
            delta=0.5,
            reduction="none",
        )
        .sum(dim=-1)
        .mean()
    )
    assert torch.allclose(loss, expected)


def test_per_sampling_priority_checkpoint_and_intervention_split(tmp_path):
    replay = TrajectoryReplayBuffer(
        seed=0,
        auto_save=False,
        sample_window_size=20,
        use_per=True,
        per_alpha=1.0,
        per_beta=0.4,
    )
    replay.add_trajectories([_trajectory(range(10))])
    replay.update_priorities(
        torch.arange(10) * 0,
        torch.arange(10),
        torch.tensor([1e-6, 1e-6, 1e-6, 100.0] + [1e-6] * 6),
    )
    batch = replay.sample(200)
    assert (batch["_replay_row_index"] == 3).sum() > 150
    assert torch.all(batch["weights"] <= 1.0)
    replay.save_checkpoint(str(tmp_path / "replay"))
    restored = TrajectoryReplayBuffer(
        auto_save=False, sample_window_size=20, use_per=True
    )
    restored.load_checkpoint(str(tmp_path / "replay"))
    assert restored.total_samples == 10
    assert restored._priorities[(0, 3)] > restored._priorities[(0, 2)]

    demo = TrajectoryReplayBuffer(
        seed=1, auto_save=False, sample_window_size=20, use_per=True
    )
    demo.add_trajectories([_trajectory([20, 21])])
    dataset = ReplayBufferDataset(replay, demo, 8, 1, 1, allow_empty_demo=True)
    mixed = next(iter(dataset))
    assert mixed["_per_online_count"].item() == 4
    assert mixed["weights"].shape[0] == 8


def test_transition_and_episode_dedup_controls_utd_budget():
    worker = _rewind_worker()
    trajectory = _trajectory([0, 1], dones=[False, True])
    first = worker._deduplicate_trajectory(trajectory)
    second = worker._deduplicate_trajectory(trajectory)
    assert worker._trajectory_transition_count(first) == 2
    assert worker._trajectory_transition_count(second) == 0
    assert worker._trajectory_completed_episodes(first) == 1
    assert worker._trajectory_completed_episodes(second) == 0

    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "update_epoch": 1,
                "replay_buffer": {"min_buffer_size": 2},
            }
        }
    )
    worker.rlt_schedule_cfg = OmegaConf.create(
        {
            "warmup_min_size": 250,
            "warmup_post_collect_updates": 0,
            "utd_ratio": 5,
            "episode_boundary_only": True,
            "max_updates_per_train_step": 0,
        }
    )
    worker.update_step = 0
    worker._warmup_ready_total_transitions = None
    worker._warmup_ready_total_episodes = None
    counters = {
        "min_replay_size": 250.0,
        "min_demo_size": 0.0,
        "transitions_since_train": 250.0,
        "episodes_since_train": 0.0,
        "total_transitions_added": 250.0,
        "total_episodes_added": 0.0,
    }
    worker._global_rlt_counters = lambda: counters
    updates, metrics = worker._rlt_updates_to_run()
    assert updates == 0 and metrics["rlt/skip_reason"] == 4.0
    counters = {
        **counters,
        "episodes_since_train": 1.0,
        "total_episodes_added": 1.0,
    }
    worker._global_rlt_counters = lambda: counters
    updates, _ = worker._rlt_updates_to_run()
    assert updates == 250 * 5


def test_worker_rewind_and_schedule_checkpoint_roundtrip(tmp_path, monkeypatch):
    def fake_parent_save(_self, save_base_path, _step):
        (Path(save_base_path) / "sac_components").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(EmbodiedSACFSDPPolicy, "save_checkpoint", fake_parent_save)
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy,
        "load_checkpoint",
        lambda _self, _load_base_path: None,
    )

    worker = _rewind_worker()
    worker._rank = 0
    worker.update_step = 17
    worker.transitions_since_train = 3
    worker.episodes_since_train = 1
    worker.total_transitions_added = 303
    worker.total_episodes_added = 9
    worker._warmup_ready_total_transitions = 250
    worker._warmup_ready_total_episodes = 7
    worker.pending_update_budget = 1498
    key = (1, 101, 0)
    worker._pending_rewind_forks[key] = {"fork_chunk_id": 4, "confidence": 0.8}
    worker._rewind_rows[key] = [(0, 0, 4)]
    worker._recovery_root_pending[key] = 4
    worker._processed_rewind_events.add((1, 101, 0, 4, "exit"))
    worker._committed_transition_ids.add((1, 101, 0, 4))
    worker._processed_episode_ends.add(key)
    worker.save_checkpoint(str(tmp_path), worker.update_step)

    restored = _rewind_worker()
    restored._rank = 0
    restored.update_step = 0
    restored.transitions_since_train = 0
    restored.episodes_since_train = 0
    restored.total_transitions_added = 0
    restored.total_episodes_added = 0
    restored._warmup_ready_total_transitions = None
    restored._warmup_ready_total_episodes = None
    restored.pending_update_budget = 0
    restored.load_checkpoint(str(tmp_path))

    assert restored._pending_rewind_forks == worker._pending_rewind_forks
    assert restored._rewind_rows == worker._rewind_rows
    assert restored._recovery_root_pending == worker._recovery_root_pending
    assert restored._processed_rewind_events == worker._processed_rewind_events
    assert restored._committed_transition_ids == worker._committed_transition_ids
    assert restored._processed_episode_ends == worker._processed_episode_ends
    assert restored.update_step == 17
    assert restored.transitions_since_train == 3
    assert restored.episodes_since_train == 1
    assert restored.total_transitions_added == 303
    assert restored.total_episodes_added == 9
    assert restored._warmup_ready_total_transitions == 250
    assert restored._warmup_ready_total_episodes == 7
    assert restored.pending_update_budget == 1498


def test_dummy_env_replay_loss_and_weight_sync_path(monkeypatch):
    env = RealWorldEnv(_realworld_cfg(), 1, 0, 1, None)
    boundary_obs, _ = env.reset()
    actions = torch.tensor([[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]])
    next_obs_list, rewards, terms, truncs, infos = env.chunk_step(actions)
    assert infos[-1]["record_transition"].all()

    policy = _policy()
    rollout_policy = copy.deepcopy(policy)
    with torch.no_grad():
        for parameter in policy.actor.parameters():
            parameter.add_(0.01)
    rollout_policy.load_state_dict(policy.state_dict())
    rlt_obs = {
        "z_rl": torch.zeros(1, Z_DIM),
        "proprio": boundary_obs["states"].float(),
        "ref_chunk": torch.zeros(1, CHUNK_LEN, ACTION_DIM),
    }
    next_rlt_obs = {
        **rlt_obs,
        "proprio": next_obs_list[-1]["states"].float(),
    }
    trajectory = _trajectory([0])
    trajectory.curr_obs = {key: value.unsqueeze(0) for key, value in rlt_obs.items()}
    trajectory.next_obs = {
        key: value.unsqueeze(0) for key, value in next_rlt_obs.items()
    }
    trajectory.actions = actions.unsqueeze(0)
    trajectory.rewards = rewards.unsqueeze(0)
    trajectory.terminations = terms[:, -1:].unsqueeze(0)
    trajectory.truncations = truncs[:, -1:].unsqueeze(0)
    trajectory.dones = (terms | truncs)[:, -1:].unsqueeze(0)
    worker = _rewind_worker()
    worker.cfg = OmegaConf.create({"env": {"train": {"env_type": "realworld"}}})
    worker.demo_buffer = None
    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_rlt_ac_policy_worker.all_reduce_dict",
        lambda values, **_kwargs: values,
    )
    added, completed = worker._ingest_rollout_trajectories([trajectory])
    assert added == 1 and completed == 0
    assert worker.replay_buffer.total_samples == 1
    batch = worker.replay_buffer.sample(1)
    loss, _ = compute_rlt_critic_loss(
        model=policy,
        target_model=copy.deepcopy(policy),
        batch=batch,
        gamma=0.99,
        use_done_key=True,
        critic_loss_type="huber",
        critic_huber_delta=0.5,
    )
    assert torch.isfinite(loss)
    assert torch.equal(
        rollout_policy.actor_semantic_version, policy.actor_semantic_version
    )
    env.close()
