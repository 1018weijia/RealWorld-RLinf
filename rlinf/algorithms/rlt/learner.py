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

"""Worker-free RLT Stage 2 learner cores.

This module holds the RLT actor-critic losses, rollout-to-replay ingestion,
rewind credit patching, recovery-preference construction and the warmup/UTD
schedule.  Nothing here imports :mod:`rlinf.scheduler`, so the same semantics
back both the Ray ``Worker`` path (ManiSkill simulator) and the single-process
WebSocket server used for real-robot Stage 2.

Hosts must provide the following attribute surface:

``cfg``
    Full RLinf ``DictConfig``.
``device`` / ``torch_dtype``
    Training device and dtype.
``model`` / ``target_model``
    Online and target policies exposing the embodied ``ForwardType`` API.
``replay_buffer`` / ``demo_buffer``
    :class:`~rlinf.data.storage.replay.TrajectoryReplayBuffer` instances
    (``demo_buffer`` may be ``None``).
``rewind_preference_buffer``
    :class:`~rlinf.algorithms.rlt.preference.RewindPreferenceBuffer`.
``update_step``
    Number of critic updates already applied.

Distributed reductions and logging are routed through overridable hooks so a
single-process host needs no ``torch.distributed`` process group.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import torch

from rlinf.algorithms.rlt import losses as rlt_losses
from rlinf.algorithms.rlt.transition import (
    ACTION_SOURCE_HUMAN,
    ACTION_SOURCE_POLICY,
    use_simulator_transition_replay,
)
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import (
    DEFAULT_ACTION_CLIP_MAX,
    DEFAULT_ACTION_CLIP_MIN,
)
from rlinf.utils.metric_utils import (
    collect_trajectory_replay_metrics,
    trajectory_has_bool_tensor,
)

logger = logging.getLogger(__name__)


def _optional_float(value: Any) -> float | None:
    """Parse a config number, treating empty / null as unset."""
    if value is None or value == "":
        return None
    return float(value)


@dataclass(frozen=True)
class RLTRewindEvent:
    """One operator rewind decision, as consumed by :class:`RLTRewindCore`.

    Attributes:
        mode: ``"exit"`` for a physical rewind that also seeds a preference
            pair, ``"credit"`` for a replay-only correction.
        chunks_rewound: Number of trailing chunks on the discarded branch.
        terminal_reward: Reward written to the last bad chunk.
        prefix_reward: Reward written to the chunk before the bad branch,
            ``"credit"`` mode only.
        confidence: Weight of the resulting preference pair.
        episode_id: Episode the rewind belongs to.
        session_id: Client session the rewind belongs to.
        env_id: Robot index within the session.
        chunk_id: Chunk at the fork point.
    """

    mode: str
    chunks_rewound: int
    terminal_reward: float
    prefix_reward: float = 0.0
    confidence: float = 1.0
    episode_id: int = 0
    session_id: int = 0
    env_id: int = 0
    chunk_id: int = 0


class RLTHostHooks:
    """Injection points that differ between the Ray worker and the server."""

    def _rlt_warn(self, message: str) -> None:
        """Emit a learner warning. Workers route this to ``log_warning``."""
        logger.warning(message)

    def _rlt_reduce_sum(self, values: dict[str, float]) -> dict[str, float]:
        """Sum-reduce schedule counters across learner ranks."""
        return dict(values)

    def _rlt_reduce_min(self, values: dict[str, float]) -> dict[str, float]:
        """Min-reduce buffer watermarks across learner ranks."""
        return dict(values)

    @property
    def _rlt_metric_reducer(self):
        """Reducer handed to :func:`collect_trajectory_replay_metrics`."""
        return None


class RLTLossCore(RLTHostHooks):
    """RLT actor-critic losses over replay batches.

    Forward types follow the existing off-policy actor-critic API, while the
    RLT objective disables entropy/alpha and uses a fixed-std actor, min-Q
    critic target, Q1 actor objective, and BC regularization.
    """

    @staticmethod
    def _flatten_chunk(tensor: torch.Tensor) -> torch.Tensor:
        return rlt_losses.flatten_chunk(tensor)

    def _chunk_shape(self) -> tuple[int, int]:
        chunk_len = int(self.cfg.actor.model.num_action_chunks)
        action_dim = int(self.cfg.actor.model.action_dim)
        return chunk_len, action_dim

    def _action_clip_bounds(self) -> dict[str, float]:
        """Resolve the residual-action bounds shared by the actor and critic."""
        model_cfg = self.cfg.actor.model
        return {
            "action_clip_min": float(
                model_cfg.get("action_clip_min", DEFAULT_ACTION_CLIP_MIN)
            ),
            "action_clip_max": float(
                model_cfg.get("action_clip_max", DEFAULT_ACTION_CLIP_MAX)
            ),
        }

    def _ref_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        chunk_len, action_dim = self._chunk_shape()
        ref_chunk = self._flatten_chunk(obs["ref_chunk"]).reshape(
            obs["ref_chunk"].shape[0], -1, action_dim
        )
        return ref_chunk[:, :chunk_len].reshape(ref_chunk.shape[0], -1)

    @staticmethod
    def _require_twin_q(all_q_values: torch.Tensor) -> None:
        rlt_losses.require_twin_q(all_q_values)

    def _min_twin_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        return rlt_losses.min_twin_q(all_q_values)

    def _q1(self, all_q_values: torch.Tensor) -> torch.Tensor:
        return rlt_losses.q1(all_q_values)

    def _discounted_chunk_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.to(self.torch_dtype)
        return rlt_losses.discounted_chunk_rewards(
            rewards, float(self.cfg.algorithm.gamma)
        )

    def _bc_metrics(
        self,
        pi: torch.Tensor,
        actions: torch.Tensor,
        ref_chunk: torch.Tensor,
        intervene_flags: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        chunk_len, action_dim = self._chunk_shape()
        return rlt_losses.compute_rlt_bc_loss(
            pi=pi,
            actions=actions,
            ref_chunk=ref_chunk,
            intervene_flags=intervene_flags,
            chunk_len=chunk_len,
            action_dim=action_dim,
        )

    def _actor_objective_weights(self) -> tuple[float, float, dict[str, float]]:
        """Resolve RLT actor-objective BC/Q weights."""
        schedule_cfg = self.cfg.algorithm.get("actor_weight_schedule", {})
        schedule_enabled = bool(schedule_cfg.get("enable", False))
        if not schedule_enabled:
            bc_weight = float(self.cfg.algorithm.get("bc_weight", 1.0))
            q_weight = float(self.cfg.algorithm.get("q_weight", 1.0))
            return (
                bc_weight,
                q_weight,
                {
                    "bc_weight": bc_weight,
                    "q_weight": q_weight,
                    "actor_weight_schedule_enabled": 0.0,
                    "actor_weight_in_warmup": 0.0,
                    "actor_weight_ramp_progress": 1.0,
                },
            )

        weight_warmup_updates = int(schedule_cfg.get("warmup_updates", 0))
        ramp_updates = int(schedule_cfg.get("ramp_updates", 0))
        in_warmup = int(self.update_step) < weight_warmup_updates
        warmup_bc_weight = float(
            schedule_cfg.get(
                "warmup_bc_weight",
                self.cfg.algorithm.get("bc_weight", 1.0),
            )
        )
        warmup_q_weight = float(
            schedule_cfg.get(
                "warmup_q_weight",
                self.cfg.algorithm.get("q_weight", 1.0),
            )
        )
        online_bc_weight = float(
            schedule_cfg.get(
                "online_bc_weight",
                self.cfg.algorithm.get("bc_weight", 1.0),
            )
        )
        online_q_weight = float(
            schedule_cfg.get(
                "online_q_weight",
                self.cfg.algorithm.get("q_weight", 1.0),
            )
        )
        if in_warmup:
            bc_weight = warmup_bc_weight
            q_weight = warmup_q_weight
            ramp_progress = 0.0
        elif ramp_updates > 0:
            ramp_progress = min(
                1.0,
                max(
                    0.0,
                    float(int(self.update_step) - weight_warmup_updates + 1)
                    / float(ramp_updates),
                ),
            )
            bc_weight = warmup_bc_weight + ramp_progress * (
                online_bc_weight - warmup_bc_weight
            )
            q_weight = warmup_q_weight + ramp_progress * (
                online_q_weight - warmup_q_weight
            )
        else:
            bc_weight = online_bc_weight
            q_weight = online_q_weight
            ramp_progress = 1.0

        metrics = {
            "bc_weight": bc_weight,
            "q_weight": q_weight,
            "actor_weight_schedule_enabled": 1.0,
            "actor_weight_in_warmup": float(in_warmup),
            "actor_weight_ramp_progress": ramp_progress,
        }
        return bc_weight, q_weight, metrics

    def _rl_algo_td_backup(self) -> str:
        raw = str(self.cfg.algorithm.get("rl_algo_td_backup") or "td3").lower()
        if raw in ("expo_double", "decoupled"):
            return "expo_decoupled"
        if raw not in ("td3", "expo", "expo_decoupled"):
            raise ValueError(
                "algorithm.rl_algo_td_backup must be td3, expo or expo_decoupled, "
                f"got {raw!r}"
            )
        return raw

    def _validate_td_backup_ensemble(self) -> None:
        """Refuse ``expo_decoupled`` when the critic cannot split two subsets."""
        backup = self._rl_algo_td_backup()
        num_qs = int(self.cfg.actor.model.get("critic_num_qs", 2))
        num_min = int(self.cfg.actor.model.get("critic_num_min_qs", 2))
        if backup == "expo_decoupled" and num_qs < 2 * num_min:
            raise ValueError(
                "expo_decoupled needs critic_num_qs >= 2 * critic_num_min_qs "
                f"(got {num_qs} < 2*{num_min})"
            )

    def _next_actions_for_critic_target(self, next_obs):
        backup = self._rl_algo_td_backup()
        if backup in ("expo", "expo_decoupled"):
            target = self.target_model
            edits = self.cfg.algorithm.get("expo_backup_num_edit_samples")
            original = target.expo_num_edit_samples
            if edits is not None:
                target.expo_num_edit_samples = int(edits)
            try:
                candidates, _ = target.build_expo_candidates(
                    next_obs, exploration=False
                )
            finally:
                target.expo_num_edit_samples = original
            return candidates
        return self.model(
            forward_type=ForwardType.SAC,
            obs=next_obs,
        )

    def _preference_batch(self):
        preference_cfg = self.cfg.algorithm.get("rewind_preference", {}) or {}
        if not bool(preference_cfg.get("enable", False)):
            return None
        return self.rewind_preference_buffer.sample(
            int(preference_cfg.get("batch_size", 32)), self.device
        )

    def _add_critic_preference_loss(
        self, loss: torch.Tensor, metrics: dict[str, float]
    ):
        preference_cfg = self.cfg.algorithm.get("rewind_preference", {}) or {}
        pair_batch = self._preference_batch()
        min_pairs = int(preference_cfg.get("min_pairs", 1))
        if pair_batch is None or len(self.rewind_preference_buffer) < min_pairs:
            metrics["preference_critic_active"] = 0.0
            return loss
        preference_loss, preference_metrics = rlt_losses.critic_pairwise_rank_loss(
            model=self.model,
            curr_obs=pair_batch["curr_obs"],
            ref_chunk=pair_batch["ref_chunk"],
            positive_action=pair_batch["positive_action"],
            negative_action=pair_batch["negative_action"],
            action_mask=pair_batch["action_mask"],
            confidence=pair_batch["confidence"],
            margin=float(preference_cfg.get("rank_margin", 0.1)),
            rank_slope=float(preference_cfg.get("rank_slope", 0.0)),
        )
        weight = float(preference_cfg.get("critic_weight", 0.0))
        metrics.update(preference_metrics)
        metrics["preference_critic_weight"] = weight
        metrics["preference_critic_active"] = 1.0
        return loss + weight * preference_loss

    def _add_intervention_rank_loss(
        self,
        loss: torch.Tensor,
        metrics: dict[str, float],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        rank_cfg = self.cfg.algorithm.get("intervention_rank", {}) or {}
        if not bool(rank_cfg.get("enable", False)):
            metrics["intervention_rank_active"] = 0.0
            return loss
        flags = batch.get("intervene_flags")
        if flags is None:
            metrics["intervention_rank_active"] = 0.0
            return loss
        vs_actor = bool(rank_cfg.get("vs_actor", True))
        vs_reference = bool(rank_cfg.get("vs_reference", False))
        if not (vs_actor or vs_reference):
            metrics["intervention_rank_active"] = 0.0
            return loss
        steps = rank_cfg.get("local_steps", ()) or ()
        if isinstance(steps, str):
            steps = tuple(float(part) for part in steps.split(",") if part.strip())
        else:
            steps = tuple(float(part) for part in steps)
        negatives: list[tuple[str, torch.Tensor]] = []
        if vs_reference:
            negatives.append(("reference", self._ref_chunk(batch["curr_obs"])))
        if vs_actor:
            with torch.no_grad():
                actor_neg, _, _ = self.model(
                    forward_type=ForwardType.SAC,
                    obs=batch["curr_obs"],
                    deterministic=True,
                )
            negatives.append(("actor", actor_neg))
        weight = float(rank_cfg.get("weight", 0.0))
        total = loss.new_zeros(())
        active = 0.0
        for name, negative in negatives:
            rank_loss, rank_metrics = rlt_losses.critic_intervention_rank_loss(
                model=self.model,
                curr_obs=batch["curr_obs"],
                human_action=batch["actions"],
                negative_action=negative,
                intervene_flags=flags,
                slope=float(rank_cfg.get("slope", 0.3)),
                min_distance=float(rank_cfg.get("min_distance", 0.0)),
                local_steps=steps,
                local_weight=float(rank_cfg.get("local_weight", 1.0)),
            )
            total = total + rank_loss
            active = max(
                active, float(rank_metrics.get("intervention_rank_active", 0.0))
            )
            for key, value in rank_metrics.items():
                suffix = key.removeprefix("intervention_rank_")
                metrics[f"intervention_rank_{name}_{suffix}"] = value
        metrics["intervention_rank_active"] = active
        metrics["intervention_rank_weight"] = weight
        return loss + weight * total

    def _add_actor_preference_loss(
        self, loss: torch.Tensor, metrics: dict[str, float]
    ) -> torch.Tensor:
        """Add the rewind Bradley-Terry term to the actor objective.

        Mirrors :meth:`_add_critic_preference_loss`. The pair buffer is filled
        by rewind recovery; without this term those pairs only move Q and
        never push the actor toward the recovered chunk.
        """
        preference_cfg = self.cfg.algorithm.get("rewind_preference", {}) or {}
        pair_batch = self._preference_batch()
        min_pairs = int(preference_cfg.get("min_pairs", 1))
        if pair_batch is None or len(self.rewind_preference_buffer) < min_pairs:
            metrics["preference_actor_active"] = 0.0
            return loss
        mean_action, _, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=pair_batch["curr_obs"],
            deterministic=True,
        )
        preference_loss, preference_metrics = rlt_losses.actor_pairwise_preference_loss(
            action_mean=mean_action,
            ref_chunk=pair_batch["ref_chunk"],
            positive_action=pair_batch["positive_action"],
            negative_action=pair_batch["negative_action"],
            action_mask=pair_batch["action_mask"],
            confidence=pair_batch["confidence"],
            fixed_std=float(self.cfg.actor.model.actor_noise_sigma),
            beta=float(preference_cfg.get("actor_beta", 1.0)),
        )
        weight = float(preference_cfg.get("actor_weight", 0.0))
        metrics.update(preference_metrics)
        metrics["preference_actor_weight"] = weight
        metrics["preference_actor_active"] = 1.0
        return loss + weight * preference_loss

    def _per_beta(self) -> float:
        replay_cfg = self.cfg.algorithm.replay_buffer
        start = float(replay_cfg.get("per_beta_start", 0.4))
        end = float(replay_cfg.get("per_beta_end", 1.0))
        steps = max(1, int(replay_cfg.get("per_beta_anneal_steps", 50000)))
        progress = min(1.0, float(getattr(self, "update_step", 0)) / steps)
        return start + progress * (end - start)

    def _update_replay_priorities(
        self, batch: dict[str, torch.Tensor], td_errors: torch.Tensor
    ) -> dict[str, float]:
        replay_cfg = self.cfg.algorithm.replay_buffer
        if not bool(replay_cfg.get("prioritized", False)):
            return {}
        ids = batch.get("_replay_trajectory_id")
        rows = batch.get("_replay_row_index")
        if ids is None or rows is None:
            raise ValueError("PER batch is missing replay row handles")
        online_count = int(
            torch.as_tensor(batch.get("_per_online_count", len(td_errors))).item()
        )
        online_count = max(0, min(online_count, len(td_errors)))
        if online_count:
            self.replay_buffer.update_priorities(
                ids[:online_count], rows[:online_count], td_errors[:online_count]
            )
        if online_count < len(td_errors):
            if self.demo_buffer is None:
                raise ValueError("PER batch contains demo rows without a demo buffer")
            self.demo_buffer.update_priorities(
                ids[online_count:], rows[online_count:], td_errors[online_count:]
            )
        beta = self._per_beta()
        self.replay_buffer.set_per_beta(beta)
        if self.demo_buffer is not None:
            self.demo_buffer.set_per_beta(beta)
        return {
            "per_beta": beta,
            "per_td_error_mean": float(td_errors.detach().float().mean().item()),
            "per_weight_mean": float(batch["weights"].float().mean().item()),
        }

    def forward_critic(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        # Ensure reward dtype matches worker training dtype before loss helpers.
        batch = {
            **batch,
            "rewards": batch["rewards"].to(self.torch_dtype),
        }
        critic_loss, metrics = rlt_losses.compute_rlt_critic_loss(
            model=self.model,
            target_model=self.target_model,
            batch=batch,
            gamma=float(self.cfg.algorithm.gamma),
            bootstrap_type=bootstrap_type,
            use_crossq=use_crossq,
            use_done_key=bool(
                self.cfg.algorithm.get(
                    "use_done_key", use_simulator_transition_replay(self.cfg)
                )
            ),
            next_actions_fn=self._next_actions_for_critic_target,
            critic_loss_type=self.cfg.algorithm.get("critic_loss", "mse"),
            critic_huber_delta=float(self.cfg.algorithm.get("critic_huber_delta", 0.5)),
            intervention_noise_sigma=float(
                self.cfg.algorithm.get("intervention_critic_action_noise_sigma", 0.0)
            ),
            intervention_noise_clip=float(
                self.cfg.algorithm.get("intervention_critic_action_noise_clip", 0.0)
            ),
            rewind_noise_sigma=float(
                self.cfg.algorithm.get("rewind_critic_action_noise_sigma", 0.0)
            ),
            rewind_noise_clip=float(
                self.cfg.algorithm.get("rewind_critic_action_noise_clip", 0.0)
            ),
            td_backup=self._rl_algo_td_backup(),
            critic_num_min_qs=int(self.cfg.actor.model.get("critic_num_min_qs", 2)),
            expo_backup_num_edit_samples=self.cfg.algorithm.get(
                "expo_backup_num_edit_samples"
            ),
            td_target_clip_min=_optional_float(
                self.cfg.algorithm.get("td_target_clip_min")
            ),
            td_target_clip_max=_optional_float(
                self.cfg.algorithm.get("td_target_clip_max")
            ),
            **self._action_clip_bounds(),
        )
        td_errors = metrics.pop("_td_errors", None)
        if td_errors is not None:
            metrics.update(self._update_replay_priorities(batch, td_errors))
        critic_loss = self._add_critic_preference_loss(critic_loss, metrics)
        critic_loss = self._add_intervention_rank_loss(critic_loss, metrics, batch)
        return critic_loss, metrics

    def forward_actor(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        chunk_len, action_dim = self._chunk_shape()
        bc_weight, q_weight, weight_metrics = self._actor_objective_weights()
        actor_loss, entropy, metrics = rlt_losses.compute_rlt_actor_loss(
            model=self.model,
            batch=batch,
            chunk_len=chunk_len,
            action_dim=action_dim,
            q_weight=q_weight,
            bc_weight=bc_weight,
            reference_dropout_prob=reference_dropout_prob,
            use_crossq=use_crossq,
            actor_q_aggregation=str(
                self.cfg.algorithm.get("actor_q_aggregation", "min")
            ),
        )
        metrics.update(weight_metrics)
        actor_loss = self._add_actor_preference_loss(actor_loss, metrics)
        return actor_loss, entropy, metrics

    def forward_alpha(self, batch):
        del batch
        raise NotImplementedError(
            "RLT AC disables entropy/alpha training. Use "
            "algorithm.entropy_tuning.alpha_type=fixed_alpha."
        )


class RLTReplayCore(RLTHostHooks):
    """Shared rollout-to-replay ingestion for every RLT host."""

    @staticmethod
    def _trajectory_transition_count(traj: Trajectory) -> int:
        if traj.actions is None:
            return 0
        if isinstance(traj.record_transition, torch.Tensor):
            steps, envs = traj.actions.shape[:2]
            return int(
                traj.record_transition.reshape(steps, envs, -1)
                .bool()
                .all(dim=-1)
                .sum()
                .item()
            )
        return int(traj.actions.shape[0] * traj.actions.shape[1])

    def _trajectory_completed_episodes(self, traj: Trajectory) -> int:
        dones = traj.dones
        if dones is None or traj.actions is None:
            return 0
        steps, envs = traj.actions.shape[:2]
        if dones.shape[0] == steps + 1:
            dones = dones[1:]
        else:
            dones = dones[:steps]
        done_rows = dones.reshape(steps, envs, -1).bool().any(dim=-1)
        if isinstance(traj.record_transition, torch.Tensor):
            done_rows &= (
                traj.record_transition.reshape(steps, envs, -1).bool().all(dim=-1)
            )

        identity_fields = (
            traj.rewind_episode_id,
            traj.rewind_session_id,
            traj.rewind_env_id,
        )
        if not all(isinstance(value, torch.Tensor) for value in identity_fields):
            return int(done_rows.sum().item())
        identities = [
            value.reshape(steps, envs, -1)[..., 0] for value in identity_fields
        ]
        if not any(bool(value.ne(0).any()) for value in identities):
            return int(done_rows.sum().item())

        processed = getattr(self, "_processed_episode_ends", set())
        completed = 0
        for step, env in done_rows.nonzero(as_tuple=False).tolist():
            key = tuple(int(value[step, env]) for value in identities)
            if key in processed:
                continue
            processed.add(key)
            completed += 1
        self._processed_episode_ends = processed
        return completed

    def _deduplicate_trajectory(self, traj: Trajectory) -> Trajectory:
        """Drop retry duplicates using committed robot transition identity."""

        if traj.actions is None or not isinstance(traj.record_transition, torch.Tensor):
            return traj
        steps, envs = traj.actions.shape[:2]
        identity_fields = (
            traj.rewind_episode_id,
            traj.rewind_session_id,
            traj.rewind_env_id,
            traj.rewind_chunk_id,
        )
        if not all(isinstance(value, torch.Tensor) for value in identity_fields):
            return traj
        identities = [
            value.reshape(steps, envs, -1)[..., 0] for value in identity_fields
        ]
        if not any(bool(value.ne(0).any()) for value in identities):
            return traj

        record = traj.record_transition.reshape(steps, envs, -1).bool().all(dim=-1)
        committed = getattr(self, "_committed_transition_ids", set())
        duplicate_rows: list[tuple[int, int]] = []
        for step, env in record.nonzero(as_tuple=False).tolist():
            key = tuple(int(value[step, env]) for value in identities)
            if key in committed:
                duplicate_rows.append((step, env))
            else:
                committed.add(key)
        self._committed_transition_ids = committed
        if not duplicate_rows:
            return traj

        filtered = copy.copy(traj)
        filtered.record_transition = traj.record_transition.clone()
        for step, env in duplicate_rows:
            filtered.record_transition[step, env] = False
        return filtered

    @staticmethod
    def _transition_reward_value(traj: Trajectory) -> float | None:
        rewards = traj.rewards
        if not isinstance(rewards, torch.Tensor) or rewards.numel() == 0:
            return None
        return float(rewards.detach().float().reshape(-1).sum().item())

    @staticmethod
    def _transition_done_value(traj: Trajectory) -> bool | None:
        dones = traj.dones
        if not isinstance(dones, torch.Tensor) or dones.numel() == 0:
            return None
        return bool(dones.detach().to(torch.bool).reshape(-1).any().item())

    @staticmethod
    def _row_tensor(tensor: torch.Tensor, idx: int) -> torch.Tensor:
        return tensor[idx].detach().clone().unsqueeze(0).unsqueeze(0).cpu().contiguous()

    @staticmethod
    def _step_env_tensor(
        tensor: torch.Tensor, step_idx: int, env_idx: int
    ) -> torch.Tensor:
        return (
            tensor[step_idx, env_idx]
            .detach()
            .clone()
            .unsqueeze(0)
            .unsqueeze(0)
            .cpu()
            .contiguous()
        )

    def _row_tensor_dict(
        self,
        tensor_dict: dict[str, object],
        idx: int,
    ) -> dict[str, torch.Tensor]:
        row_dict = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor) and idx < value.shape[0]:
                row_dict[key] = self._row_tensor(value, idx)
        return row_dict

    def _rlt_obs_from_flat_dict(
        self,
        flat: dict,
        dict_key: str,
        idx: int,
    ) -> dict[str, torch.Tensor] | None:
        value = flat.get(dict_key)
        if not isinstance(value, dict):
            return None
        obs = self._row_tensor_dict(value, idx)
        return obs if obs else None

    @staticmethod
    def _flat_record_transition(flat: dict, idx: int) -> bool:
        forward_inputs = flat.get("forward_inputs")
        if not isinstance(forward_inputs, dict):
            return False
        record_transition = forward_inputs.get("record_transition")
        if not isinstance(record_transition, torch.Tensor):
            return False
        if idx >= record_transition.shape[0]:
            return False
        return bool(record_transition[idx].detach().to(torch.bool).reshape(-1).all())

    def _transition_replay_trajectories(
        self,
        trajectory: Trajectory,
    ) -> tuple[list[Trajectory], int]:
        if (
            trajectory.actions is None
            or trajectory.rewards is None
            or self.replay_buffer is None
        ):
            return [], 0

        flat = self.replay_buffer._flatten_trajectory(trajectory)
        actions = flat.get("actions")
        rewards = flat.get("rewards")
        if not isinstance(actions, torch.Tensor) or not isinstance(
            rewards, torch.Tensor
        ):
            return [], 0

        tensor_fields = (
            "actions",
            "intervene_flags",
            "rewards",
            "terminations",
            "truncations",
            "dones",
            "prev_logprobs",
            "prev_values",
            "versions",
        )
        dict_fields = ("forward_inputs",)
        replay_trajectories = []
        completed_episodes = 0
        traj_len = int(trajectory.actions.shape[0])
        bsz = int(trajectory.actions.shape[1])
        num_rows = int(actions.shape[0])
        auto_reset = bool(self.cfg.env.train.get("auto_reset", False))

        for env_idx in range(bsz):
            for t in range(traj_len):
                idx = t * bsz + env_idx
                if idx >= num_rows:
                    break
                if not self._flat_record_transition(flat, idx):
                    continue

                transition = Trajectory(
                    max_episode_length=1,
                    model_weights_id=trajectory.model_weights_id,
                )
                for field_name in tensor_fields:
                    value = flat.get(field_name)
                    if isinstance(value, torch.Tensor) and idx < value.shape[0]:
                        setattr(transition, field_name, self._row_tensor(value, idx))
                for field_name in dict_fields:
                    value = flat.get(field_name)
                    if isinstance(value, dict):
                        setattr(
                            transition, field_name, self._row_tensor_dict(value, idx)
                        )

                curr_obs = self._rlt_obs_from_flat_dict(flat, "curr_obs", idx)
                if curr_obs is None:
                    raise ValueError(
                        "RLT transition replay requires curr_obs. Ensure "
                        "update_rlt_transitions() populated transition obs "
                        f"before replay ingestion, got row index {idx}."
                    )
                transition.curr_obs = curr_obs

                # Dones have one extra initial slot, so transition t reads
                # terminal flags from t+1. Rewards are already action-aligned
                # by EmbodiedTrajectoryBuilder because the initial empty reward is
                # skipped and the final reward is appended after rollout.
                done_idx = min(
                    t + 1,
                    int(trajectory.dones.shape[0]) - 1
                    if isinstance(trajectory.dones, torch.Tensor)
                    else traj_len - 1,
                )
                for done_field in ("dones", "terminations", "truncations"):
                    done_value = getattr(trajectory, done_field, None)
                    if (
                        isinstance(done_value, torch.Tensor)
                        and done_idx < done_value.shape[0]
                        and env_idx < done_value.shape[1]
                    ):
                        setattr(
                            transition,
                            done_field,
                            self._step_env_tensor(done_value, done_idx, env_idx),
                        )

                is_done = (
                    isinstance(transition.dones, torch.Tensor)
                    and transition.dones.reshape(-1).to(torch.bool).any()
                )
                if is_done:
                    next_obs = curr_obs
                else:
                    next_obs = self._rlt_obs_from_flat_dict(flat, "next_obs", idx)
                if next_obs is not None:
                    transition.next_obs = next_obs
                else:
                    raise ValueError(
                        "RLT transition replay requires next_obs for non-terminal "
                        "transitions. Ensure update_rlt_transitions() populated "
                        f"transition obs before replay ingestion, got row index {idx}."
                    )

                replay_trajectories.append(transition)
                if is_done:
                    completed_episodes += 1
                    if not auto_reset:
                        break

        return replay_trajectories, completed_episodes

    def _transition_replay_metrics(
        self,
        replay_trajectories: list[Trajectory],
    ) -> dict[str, float]:
        metrics = {"replay/transition_count": float(len(replay_trajectories))}
        reward_values = [
            reward
            for traj in replay_trajectories
            if (reward := self._transition_reward_value(traj)) is not None
        ]
        if reward_values:
            metrics["replay/reward_mean"] = float(
                sum(reward_values) / len(reward_values)
            )
            metrics["replay/reward_positive_rate"] = float(
                sum(reward > 0.0 for reward in reward_values) / len(reward_values)
            )
        done_values = [
            done
            for traj in replay_trajectories
            if (done := self._transition_done_value(traj)) is not None
        ]
        if done_values:
            metrics["replay/done_rate"] = float(
                sum(bool(done) for done in done_values) / len(done_values)
            )
        return metrics

    def _ingest_rollout_trajectories(
        self,
        recv_list: list[Trajectory],
    ) -> tuple[int, int]:
        self._last_replay_metrics = {}
        recv_list = [self._deduplicate_trajectory(traj) for traj in recv_list]
        reducer = self._rlt_metric_reducer

        if use_simulator_transition_replay(self.cfg):
            replay_list = []
            completed = 0
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                transition_trajs, completed_count = (
                    self._transition_replay_trajectories(traj)
                )
                replay_list.extend(transition_trajs)
                completed += completed_count
            self._last_replay_metrics = {
                **self._transition_replay_metrics(replay_list),
                **collect_trajectory_replay_metrics(recv_list, reducer=reducer),
            }
            self.replay_buffer.add_trajectories(replay_list)
            for traj in recv_list:
                self._ingest_rewind_events(traj.rewind_events)
                self._consume_rewind_preferences(traj)

            if self.demo_buffer is not None:
                intervene_traj_list = [
                    traj
                    for traj in replay_list
                    if trajectory_has_bool_tensor(traj.intervene_flags)
                ]
                if len(intervene_traj_list) > 0:
                    self.demo_buffer.add_trajectories(intervene_traj_list)

            return len(replay_list), completed

        replay_start = self.replay_buffer._trajectory_counter
        self.replay_buffer.add_trajectories(recv_list)
        for offset, trajectory in enumerate(recv_list):
            if replay_start + offset in self.replay_buffer._trajectory_index:
                self._append_replay_rows(trajectory, replay_start + offset)
        for trajectory in recv_list:
            self._ingest_rewind_events(trajectory.rewind_events)
            self._consume_rewind_preferences(trajectory)

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)

            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

        added = sum(self._trajectory_transition_count(traj) for traj in recv_list)
        completed = sum(self._trajectory_completed_episodes(traj) for traj in recv_list)
        self._last_replay_metrics = collect_trajectory_replay_metrics(
            recv_list, reducer=reducer
        )
        return added, completed


class RLTRewindCore(RLTHostHooks):
    """Rewind credit patching and recovery-preference construction."""

    def init_rlt_rewind_state(self) -> None:
        """Initialize the per-session rewind bookkeeping maps."""
        self._pending_rewind_forks: dict[tuple[int, int, int], dict[str, object]] = {}
        self._rewind_rows: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
        self._recovery_root_pending: dict[tuple[int, int, int], int] = {}
        self._processed_rewind_events: set[tuple[int, int, int, int, str]] = set()
        self._committed_transition_ids: set[tuple[int, int, int, int]] = set()
        self._processed_episode_ends: set[tuple[int, int, int]] = set()

    @staticmethod
    def _row_key(episode_id: int, session_id: int, env_id: int) -> tuple[int, int, int]:
        # chunk_id is intentionally absent: all of an episode's rows share one
        # key and are ordered by chunk_id within it, which is what lets rewind
        # address a prefix of the episode.
        return (episode_id, session_id, env_id)

    def _append_replay_rows(self, trajectory: Trajectory, trajectory_id: int) -> None:
        """Index confirmed replay rows by robot-provided session identity."""
        if trajectory.actions is None:
            return
        flat = self.replay_buffer._flatten_trajectory(trajectory)
        episode_ids = flat.get("rewind_episode_id")
        session_ids = flat.get("rewind_session_id")
        env_ids = flat.get("rewind_env_id")
        chunk_ids = flat.get("rewind_chunk_id")
        records = flat.get("record_transition")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (episode_ids, session_ids, env_ids, chunk_ids)
        ):
            return
        for row in range(trajectory.actions.shape[0] * trajectory.actions.shape[1]):
            if isinstance(records, torch.Tensor) and not bool(
                records[row].reshape(-1)[0]
            ):
                continue
            key = self._row_key(
                int(episode_ids[row].reshape(-1)[0]),
                int(session_ids[row].reshape(-1)[0]),
                int(env_ids[row].reshape(-1)[0]),
            )
            self._rewind_rows.setdefault(key, []).append(
                (trajectory_id, row, int(chunk_ids[row].reshape(-1)[0]))
            )

    def _patch_rows(
        self, trajectory_id: int, updates: dict[int, dict[str, torch.Tensor]]
    ) -> None:
        self.replay_buffer.patch_trajectory_rows(trajectory_id, updates)

    def _patch_rewind_event(self, event: object) -> None:
        """Apply remote-franka credit only to real rows in one session."""
        key = self._row_key(
            int(event.episode_id), int(event.session_id), int(event.env_id)
        )
        rows = [
            row
            for row in self._rewind_rows.get(key, [])
            if row[2] <= int(event.chunk_id)
        ]
        count = min(int(event.chunks_rewound), len(rows))
        if count <= 0:
            self._rlt_warn(f"Ignoring rewind event without bad rows: {key}")
            return
        bad_rows = rows[-count:]
        grouped: dict[int, dict[int, dict[str, torch.Tensor]]] = {}

        def row_field(tid: int, row: int, name: str) -> torch.Tensor:
            trajectory = self.replay_buffer._load_trajectory(
                tid, self.replay_buffer._trajectory_index[tid]["model_weights_id"]
            )
            value = self.replay_buffer._flatten_trajectory(trajectory)[name][
                row
            ].clone()
            return value

        terminal_tid, terminal_row, _ = bad_rows[-1]
        terminal_reward = row_field(terminal_tid, terminal_row, "rewards")
        terminal_reward.reshape(-1)[-1] = float(event.terminal_reward)
        grouped.setdefault(terminal_tid, {})[terminal_row] = {
            "rewards": terminal_reward,
            "bootstrap_mask": torch.zeros_like(
                row_field(terminal_tid, terminal_row, "bootstrap_mask")
            ),
            "terminal_type": torch.full_like(
                row_field(terminal_tid, terminal_row, "terminal_type"), 3
            ),
            "branch_id": torch.full_like(
                row_field(terminal_tid, terminal_row, "branch_id"), 2
            ),
            "next_action_override_mask": torch.zeros_like(
                row_field(terminal_tid, terminal_row, "next_action_override_mask")
            ),
        }
        for current, successor in zip(bad_rows[:-1], bad_rows[1:]):
            tid, row, _ = current
            successor_tid, successor_row, _ = successor
            successor_action = row_field(successor_tid, successor_row, "actions")
            grouped.setdefault(tid, {})[row] = {
                "bootstrap_mask": torch.ones_like(
                    row_field(tid, row, "bootstrap_mask")
                ),
                "branch_id": torch.full_like(row_field(tid, row, "branch_id"), 2),
                "next_action_override": successor_action,
                "next_action_override_mask": torch.ones_like(
                    row_field(tid, row, "next_action_override_mask")
                ),
            }
        if event.mode == "credit" and len(rows) > count:
            predecessor_tid, predecessor_row, _ = rows[-count - 1]
            prefix_reward = row_field(predecessor_tid, predecessor_row, "rewards")
            prefix_reward.reshape(-1)[-1] = float(event.prefix_reward)
            grouped.setdefault(predecessor_tid, {})[predecessor_row] = {
                "rewards": prefix_reward,
                "bootstrap_mask": torch.zeros_like(
                    row_field(predecessor_tid, predecessor_row, "bootstrap_mask")
                ),
                "next_action_override_mask": torch.zeros_like(
                    row_field(
                        predecessor_tid, predecessor_row, "next_action_override_mask"
                    )
                ),
            }
        for tid, updates in grouped.items():
            self._patch_rows(tid, updates)
        if event.mode == "exit" and len(rows) > count:
            anchor_tid, anchor_row, _ = rows[-count - 1]
            bad_tid, bad_row, _ = bad_rows[0]
            anchor_trajectory = self.replay_buffer._load_trajectory(
                anchor_tid,
                self.replay_buffer._trajectory_index[anchor_tid]["model_weights_id"],
            )
            anchor_flat = self.replay_buffer._flatten_trajectory(anchor_trajectory)
            bad_trajectory = self.replay_buffer._load_trajectory(
                bad_tid,
                self.replay_buffer._trajectory_index[bad_tid]["model_weights_id"],
            )
            bad_flat = self.replay_buffer._flatten_trajectory(bad_trajectory)
            self._pending_rewind_forks[key] = {
                "anchor": (anchor_tid, anchor_row),
                "curr_obs": {
                    name: value[anchor_row]
                    for name, value in anchor_flat["next_obs"].items()
                },
                "ref_chunk": anchor_flat["next_obs"]["ref_chunk"][anchor_row],
                "negative_action": bad_flat["actions"][bad_row],
                "confidence": float(event.confidence),
                "fork_chunk_id": int(event.chunk_id),
            }
        elif event.mode == "credit":
            self._recovery_root_pending[key] = int(event.chunk_id)

    def _ingest_rewind_events(self, events: list[object] | None) -> None:
        for event in events or []:
            event_key = (
                int(event.episode_id),
                int(event.session_id),
                int(event.env_id),
                int(event.chunk_id),
                str(event.mode),
            )
            if event_key in self._processed_rewind_events:
                continue
            self._processed_rewind_events.add(event_key)
            self._patch_rewind_event(event)

    def _consume_rewind_preferences(self, trajectory: Trajectory) -> None:
        """Attach the first stored post-fork human or policy replacement."""
        if trajectory.actions is None:
            return
        flat = self.replay_buffer._flatten_trajectory(trajectory)
        required = (
            "actions",
            "curr_obs",
            "action_source",
            "record_transition",
            "rewind_episode_id",
            "rewind_session_id",
            "rewind_env_id",
            "rewind_chunk_id",
        )
        if any(key not in flat for key in required):
            return
        for row in range(flat["actions"].shape[0]):
            key = self._row_key(
                int(flat["rewind_episode_id"][row].reshape(-1)[0]),
                int(flat["rewind_session_id"][row].reshape(-1)[0]),
                int(flat["rewind_env_id"][row].reshape(-1)[0]),
            )
            fork = self._pending_rewind_forks.get(key)
            if fork is None:
                if key in self._recovery_root_pending:
                    fork_chunk_id = self._recovery_root_pending[key]
                    rows = [
                        item
                        for item in self._rewind_rows.get(key, [])
                        if item[2] > fork_chunk_id
                    ]
                    if rows:
                        recovery_tid, recovery_row, _ = rows[0]
                        self._patch_rows(
                            recovery_tid,
                            {
                                recovery_row: {
                                    "recovery_root": torch.ones((1,), dtype=torch.bool),
                                }
                            },
                        )
                        del self._recovery_root_pending[key]
                continue
            if not bool(flat["record_transition"][row].reshape(-1)[0]):
                continue
            chunk_id = int(flat["rewind_chunk_id"][row].reshape(-1)[0])
            if chunk_id <= int(fork["fork_chunk_id"]):
                continue
            action_source = int(flat["action_source"][row].reshape(-1)[0])
            if action_source not in (ACTION_SOURCE_HUMAN, ACTION_SOURCE_POLICY):
                continue
            self.rewind_preference_buffer.add(
                curr_obs=fork["curr_obs"],
                ref_chunk=fork["ref_chunk"],
                positive_action=flat["actions"][row],
                negative_action=fork["negative_action"],
                confidence=fork["confidence"],
                session_key=key,
            )
            anchor_tid, anchor_row = fork["anchor"]
            self._patch_rows(
                anchor_tid,
                {
                    anchor_row: {
                        "next_action_override": flat["actions"][row],
                        "next_action_override_mask": torch.ones((1,), dtype=torch.bool),
                    }
                },
            )
            del self._pending_rewind_forks[key]

    def rlt_rewind_state_dict(self) -> dict[str, object]:
        """Serialize rewind/dedup bookkeeping for checkpointing."""
        return {
            "preference_buffer": self.rewind_preference_buffer.state_dict(),
            "pending_forks": self._pending_rewind_forks,
            "rewind_rows": self._rewind_rows,
            "recovery_root_pending": self._recovery_root_pending,
            "processed_rewind_events": self._processed_rewind_events,
            "committed_transition_ids": self._committed_transition_ids,
            "processed_episode_ends": self._processed_episode_ends,
        }

    def load_rlt_rewind_state_dict(self, state: dict[str, object]) -> None:
        """Restore rewind/dedup bookkeeping saved by a previous run."""
        self.rewind_preference_buffer.load_state_dict(state["preference_buffer"])
        self._pending_rewind_forks = {
            key: value
            for key, value in state.get("pending_forks", {}).items()
            if "fork_chunk_id" in value
        }
        self._rewind_rows = state.get("rewind_rows", {})
        recovery_pending = state.get("recovery_root_pending", {})
        self._recovery_root_pending = (
            recovery_pending if isinstance(recovery_pending, dict) else {}
        )
        self._processed_rewind_events = state.get("processed_rewind_events", set())
        self._committed_transition_ids = state.get("committed_transition_ids", set())
        self._processed_episode_ends = state.get("processed_episode_ends", set())


class RLTScheduleCore(RLTHostHooks):
    """Warmup and update-to-data schedule shared by every RLT host."""

    def init_rlt_schedule_state(self) -> None:
        """Initialize warmup/UTD counters from ``algorithm.rlt_schedule``."""
        self.rlt_schedule_cfg = self.cfg.algorithm.get("rlt_schedule", {}) or {}
        self.use_rlt_schedule = bool(self.rlt_schedule_cfg.get("enable", False))
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        self.total_transitions_added = 0
        self.total_episodes_added = 0
        self._warmup_ready_total_transitions: int | None = None
        self._warmup_ready_total_episodes: int | None = None
        self.pending_update_budget = 0

    def _update_rollout_ingest_counters(self, added: int, completed: int) -> None:
        if not getattr(self, "use_rlt_schedule", False):
            return
        if not hasattr(self, "transitions_since_train"):
            return
        self.transitions_since_train += added
        self.episodes_since_train += completed
        self.total_transitions_added += added
        self.total_episodes_added += completed

    def _global_rlt_counters(self) -> dict[str, float]:
        summed = self._rlt_reduce_sum(
            {
                "transitions_since_train": float(self.transitions_since_train),
                "episodes_since_train": float(self.episodes_since_train),
                "total_transitions_added": float(self.total_transitions_added),
                "total_episodes_added": float(self.total_episodes_added),
            }
        )
        minimums = self._rlt_reduce_min(
            {
                "min_replay_size": float(self.replay_buffer.total_samples),
                "min_demo_size": float(
                    0 if self.demo_buffer is None else self.demo_buffer.total_samples
                ),
            }
        )
        summed.update(minimums)
        return summed

    def _rlt_updates_to_run(self) -> tuple[int, dict[str, float]]:
        replay_cfg = self.cfg.algorithm.replay_buffer
        schedule_cfg = self.rlt_schedule_cfg
        min_buffer_size = int(
            schedule_cfg.get("warmup_min_size", replay_cfg.get("min_buffer_size", 1))
        )
        counters = self._global_rlt_counters()
        buffer_ready = counters["min_replay_size"] >= min_buffer_size
        warmup_required_updates = int(
            schedule_cfg.get("warmup_post_collect_updates", 0)
        )
        if buffer_ready and self._warmup_ready_total_transitions is None:
            self._warmup_ready_total_transitions = int(
                counters["total_transitions_added"]
            )
            self._warmup_ready_total_episodes = int(counters["total_episodes_added"])

        train_every_transitions = int(schedule_cfg.get("train_every_transitions", 0))
        train_every_episodes = int(schedule_cfg.get("train_every_episodes", 0))
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        utd_ratio = int(schedule_cfg.get("utd_ratio", 0))
        episode_boundary_only = bool(schedule_cfg.get("episode_boundary_only", False))
        max_updates = int(schedule_cfg.get("max_updates_per_train_step", 0))

        updates_to_run = 0
        skip_reason = 0
        desired_total_updates = 0
        pending_updates = 0
        updates_scheduled = 0
        if update_epoch <= 0 and utd_ratio <= 0:
            skip_reason = 3
        elif not buffer_ready:
            skip_reason = 1
        elif episode_boundary_only and counters["episodes_since_train"] <= 0:
            skip_reason = 4
        else:
            online_transitions = max(
                int(counters["total_transitions_added"])
                - int(self._warmup_ready_total_transitions or 0),
                0,
            )
            online_episodes = max(
                int(counters["total_episodes_added"])
                - int(self._warmup_ready_total_episodes or 0),
                0,
            )
            if utd_ratio > 0:
                desired_total_updates = (
                    int(counters["total_transitions_added"]) * utd_ratio
                )
            elif train_every_transitions <= 0 and train_every_episodes <= 0:
                online_cycles = online_transitions
            else:
                transition_cycles = (
                    online_transitions // train_every_transitions
                    if train_every_transitions > 0
                    else 0
                )
                episode_cycles = (
                    online_episodes // train_every_episodes
                    if train_every_episodes > 0
                    else 0
                )
                online_cycles = max(transition_cycles, episode_cycles)
            if utd_ratio <= 0:
                desired_total_updates = (
                    warmup_required_updates + online_cycles * update_epoch
                )
            pending_updates = max(desired_total_updates - int(self.update_step), 0)
            updates_scheduled = pending_updates
            updates_to_run = pending_updates
            if max_updates > 0:
                updates_to_run = min(updates_to_run, max_updates)
            if updates_to_run <= 0:
                skip_reason = 2
        self.pending_update_budget = int(pending_updates)

        metrics = {
            "rlt/update_step": float(self.update_step),
            "rlt/ready_for_online": float(
                int(self.update_step) >= warmup_required_updates
            ),
            "rlt/warmup_required_updates": float(warmup_required_updates),
            "rlt/update_epoch": float(update_epoch),
            "rlt/utd_ratio": float(utd_ratio),
            "rlt/episode_boundary_only": float(episode_boundary_only),
            "rlt/max_updates_per_train_step": float(max_updates),
            "rlt/train_every_transitions": float(train_every_transitions),
            "rlt/train_every_episodes": float(train_every_episodes),
            "rlt/desired_total_updates": float(desired_total_updates),
            "rlt/pending_update_budget": float(self.pending_update_budget),
            "rlt/updates_scheduled": float(updates_scheduled),
            "rlt/updates_to_run": float(updates_to_run),
            "rlt/critic_updates_run": 0.0,
            "rlt/actor_updates_run": 0.0,
            "rlt/should_train": float(updates_to_run > 0),
            "rlt/skip_reason": float(skip_reason),
            "rlt/global_min_replay_size": float(counters["min_replay_size"]),
            "rlt/min_replay_buffer_size": float(min_buffer_size),
            "rlt/global_transitions_since_train": float(
                counters["transitions_since_train"]
            ),
            "rlt/global_total_transitions_added": float(
                counters["total_transitions_added"]
            ),
        }
        metrics.update(getattr(self, "_last_replay_metrics", {}))
        return updates_to_run, metrics

    def rlt_schedule_state_dict(self) -> dict[str, object]:
        """Serialize warmup/UTD counters for checkpointing."""
        return {
            "update_step": self.update_step,
            "transitions_since_train": self.transitions_since_train,
            "episodes_since_train": self.episodes_since_train,
            "total_transitions_added": self.total_transitions_added,
            "total_episodes_added": self.total_episodes_added,
            "warmup_ready_total_transitions": self._warmup_ready_total_transitions,
            "warmup_ready_total_episodes": self._warmup_ready_total_episodes,
            "pending_update_budget": self.pending_update_budget,
        }

    def load_rlt_schedule_state_dict(self, schedule: dict[str, object]) -> None:
        """Restore warmup/UTD counters saved by a previous run."""
        self.update_step = int(schedule.get("update_step", self.update_step))
        self.transitions_since_train = int(
            schedule.get("transitions_since_train", self.transitions_since_train)
        )
        self.episodes_since_train = int(
            schedule.get("episodes_since_train", self.episodes_since_train)
        )
        self.total_transitions_added = int(
            schedule.get("total_transitions_added", self.total_transitions_added)
        )
        self.total_episodes_added = int(
            schedule.get("total_episodes_added", self.total_episodes_added)
        )
        self._warmup_ready_total_transitions = schedule.get(
            "warmup_ready_total_transitions", self._warmup_ready_total_transitions
        )
        self._warmup_ready_total_episodes = schedule.get(
            "warmup_ready_total_episodes", self._warmup_ready_total_episodes
        )
        self.pending_update_budget = int(
            schedule.get("pending_update_budget", self.pending_update_budget)
        )


class RLTLearnerCore(RLTLossCore, RLTReplayCore, RLTRewindCore, RLTScheduleCore):
    """Aggregate of every Worker-free RLT learner concern."""
