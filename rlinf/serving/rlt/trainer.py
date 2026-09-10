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

"""Single-process RLT Stage 2 trainer.

This is the non-Ray counterpart of
:class:`~rlinf.workers.actor.fsdp_rlt_td3_policy_worker.RLTTD3FSDPPolicy`.  It
owns the Stage 2 model, its target, both optimizers and the replay buffers, and
it inherits every algorithm-level behaviour from
:mod:`rlinf.algorithms.rlt.learner` so the two hosts cannot drift apart.

There is no FSDP, no process group and no gradient accumulation across ranks:
one robot produces one transition at a time, so a single device holds the whole
Stage 2 head (a few MLPs on top of frozen Stage 1 features).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.algorithms.rlt.learner import RLTLearnerCore
from rlinf.algorithms.rlt.preference import RewindPreferenceBuffer
from rlinf.algorithms.rlt.transition import annotate_rlt_branch_fields
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.utils.metric_utils import append_to_dict

logger = logging.getLogger(__name__)


class RLTStage2Trainer(RLTLearnerCore):
    """Own the Stage 2 model, buffers and update loop in one process.

    Args:
        cfg: Full RLinf config. Only ``algorithm``, ``actor`` and
            ``runner.logger.log_path`` are read.
        model: Stage 2 policy (``RLTTD3MLPPolicy``).
        target_model: Frozen copy of ``model`` used for TD targets.
        device: Device both models live on.
        torch_dtype: Training dtype, ``float32`` for Q-learning stability.
    """

    def __init__(
        self,
        cfg: DictConfig,
        *,
        model: torch.nn.Module,
        target_model: torch.nn.Module,
        device: torch.device,
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.target_model = target_model
        self.device = device
        self.torch_dtype = torch_dtype
        self.update_step = 0
        self._last_replay_metrics: dict[str, float] = {}

        self.target_model.requires_grad_(False)
        self.target_update_type = cfg.algorithm.get("target_update_type", "all")
        if self.target_update_type not in ("all", "q_head_only"):
            raise ValueError(f"{self.target_update_type=} is not supported")
        self.critic_actor_ratio = int(cfg.algorithm.get("critic_actor_ratio", 1))
        self.tau = float(cfg.algorithm.tau)
        self.batch_size = int(cfg.actor.global_batch_size)

        self._build_buffers()
        self._build_optimizers()
        self.init_rlt_schedule_state()
        self.init_rlt_rewind_state()
        self.soft_update_target_model(tau=1.0)

    # ------------------------------------------------------------- setup

    def _build_buffers(self) -> None:
        replay_cfg = self.cfg.algorithm.replay_buffer
        seed = int(self.cfg.actor.get("seed", 1234))
        auto_save_path = replay_cfg.get("auto_save_path", None) or os.path.join(
            self.cfg.runner.logger.log_path, "replay_buffer/server"
        )
        shared = {
            "seed": seed,
            "use_per": replay_cfg.get("prioritized", False),
            "per_alpha": replay_cfg.get("per_alpha", 0.6),
            "per_beta": replay_cfg.get("per_beta_start", 0.4),
            "per_eps": replay_cfg.get("per_eps", 1e-6),
        }
        self.replay_buffer = TrajectoryReplayBuffer(
            enable_cache=replay_cfg.enable_cache,
            cache_size=replay_cfg.cache_size,
            sample_window_size=replay_cfg.sample_window_size,
            auto_save=replay_cfg.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=replay_cfg.get("trajectory_format", "pt"),
            **shared,
        )

        self.demo_buffer = None
        demo_cfg = self.cfg.algorithm.get("demo_buffer", None)
        if demo_cfg is not None:
            demo_save_path = demo_cfg.get("auto_save_path", None) or os.path.join(
                self.cfg.runner.logger.log_path, "demo_buffer/server"
            )
            self.demo_buffer = TrajectoryReplayBuffer(
                enable_cache=demo_cfg.enable_cache,
                cache_size=demo_cfg.cache_size,
                sample_window_size=demo_cfg.sample_window_size,
                auto_save=demo_cfg.get("auto_save", False),
                auto_save_path=demo_save_path,
                trajectory_format="pt",
                **shared,
            )

        preference_cfg = self.cfg.algorithm.get("rewind_preference", {}) or {}
        self.rewind_preference_buffer = RewindPreferenceBuffer(
            int(preference_cfg.get("capacity", 1024))
        )

    def _build_optimizers(self) -> None:
        """Split actor and critic parameters into their configured optimizers.

        Mirrors the worker's ``param_filters={"critic": [...]}`` split so the
        two hosts train the same parameter groups with the same learning rates.
        """
        critic_filters = ("q_head", "encoders", "encoder", "state_proj")
        actor_params, critic_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name for token in critic_filters):
                critic_params.append(param)
            else:
                actor_params.append(param)
        if not actor_params or not critic_params:
            raise ValueError(
                "Stage 2 model produced an empty actor or critic parameter group "
                f"(actor={len(actor_params)}, critic={len(critic_params)})"
            )

        self.optimizer = self._adamw(actor_params, self.cfg.actor.optim)
        self.qf_optimizer = self._adamw(critic_params, self.cfg.actor.critic_optim)

    @staticmethod
    def _adamw(params, optim_cfg) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params,
            lr=float(optim_cfg.lr),
            betas=(
                float(optim_cfg.get("adam_beta1", 0.9)),
                float(optim_cfg.get("adam_beta2", 0.999)),
            ),
            eps=float(optim_cfg.get("adam_eps", 1e-8)),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )

    # --------------------------------------------------------- ingestion

    def add_transition(
        self,
        *,
        curr_obs: dict[str, torch.Tensor],
        next_obs: dict[str, torch.Tensor],
        action_chunk: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        bootstrap_mask: float,
        intervention: bool,
        identity,
        action_source: int,
    ) -> tuple[int, int]:
        """Append one executed chunk as a single-row replay trajectory.

        Args:
            curr_obs: Stage 2 observation before the chunk.
            next_obs: Stage 2 observation after the chunk.
            action_chunk: Executed chunk in the replay action space,
                ``[chunk_len, action_dim]``.
            rewards: Per-step rewards, ``[chunk_len]``.
            done: Whether the episode terminated on this chunk.
            bootstrap_mask: ``0.0`` to cut bootstrapping (hard terminal).
            intervention: Whether a human drove any step of the chunk.
            identity: :class:`~rlinf.serving.rlt.protocol.ChunkIdentity`.
            action_source: ``ACTION_SOURCE_*`` code for replay diagnostics.

        Returns:
            ``(transitions_added, episodes_completed)``.
        """
        trajectory = self._build_transition_trajectory(
            curr_obs=curr_obs,
            next_obs=next_obs,
            action_chunk=action_chunk,
            rewards=rewards,
            done=done,
            bootstrap_mask=bootstrap_mask,
            intervention=intervention,
            identity=identity,
            action_source=action_source,
        )
        added, completed = self._ingest_rollout_trajectories([trajectory])
        self._update_rollout_ingest_counters(added, completed)
        return added, completed

    def _build_transition_trajectory(
        self,
        *,
        curr_obs: dict[str, torch.Tensor],
        next_obs: dict[str, torch.Tensor],
        action_chunk: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        bootstrap_mask: float,
        intervention: bool,
        identity,
        action_source: int,
    ) -> Trajectory:
        chunk_len, action_dim = self._chunk_shape()
        actions = torch.as_tensor(action_chunk, dtype=torch.float32).reshape(
            1, 1, chunk_len * action_dim
        )
        reward_row = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
        if reward_row.numel() != chunk_len:
            raise ValueError(
                f"rewards must have {chunk_len} entries, got {reward_row.numel()}"
            )
        rewards_t = reward_row.reshape(1, 1, chunk_len)
        done_t = torch.tensor([[[bool(done)]]], dtype=torch.bool)

        trajectory = Trajectory(max_episode_length=1)
        trajectory.actions = actions
        trajectory.intervene_flags = torch.full_like(
            actions, bool(intervention), dtype=torch.bool
        )
        trajectory.rewards = rewards_t
        trajectory.dones = done_t
        trajectory.terminations = done_t.clone()
        trajectory.truncations = torch.zeros_like(done_t)
        trajectory.versions = torch.full(
            (1, 1, 1), float(self.update_step), dtype=torch.float32
        )
        trajectory.curr_obs = {
            key: _to_replay_row(key, value) for key, value in curr_obs.items()
        }
        trajectory.next_obs = {
            key: _to_replay_row(key, value) for key, value in next_obs.items()
        }

        branch_fields = annotate_rlt_branch_fields(
            batch_size=1,
            bootstrap_mask=float(bootstrap_mask),
            action_source=int(action_source),
            rewind_episode_id=int(identity.episode_id),
            rewind_session_id=int(identity.session_id),
            rewind_env_id=int(identity.env_id),
            rewind_chunk_id=int(identity.chunk_id),
            next_action_override=torch.zeros_like(actions.reshape(1, -1)),
            record_transition=True,
        )
        for key, value in branch_fields.items():
            setattr(trajectory, key, value.reshape(1, 1, -1).contiguous())
        return trajectory

    # ---------------------------------------------------------- training

    def train(self, num_updates: int) -> dict[str, float]:
        """Run ``num_updates`` critic updates (actor every ``critic_actor_ratio``).

        Args:
            num_updates: Number of critic updates to apply.

        Returns:
            Mean metrics over the burst, including the RLT schedule diagnostics.
        """
        _, schedule_metrics = self._rlt_updates_to_run()
        if num_updates <= 0:
            return self._reduce_metrics(schedule_metrics)

        if not self.replay_buffer.is_ready(1):
            logger.warning("Skipping RLT update burst: replay buffer is empty")
            return self._reduce_metrics(schedule_metrics)

        self.model.train()
        metrics: dict[str, list[float]] = {}
        critic_updates = 0
        actor_updates = 0
        for _ in range(int(num_updates)):
            train_actor = int(self.update_step) % self.critic_actor_ratio == 0
            append_to_dict(metrics, self.update_once(train_actor=train_actor))
            self.update_step += 1
            critic_updates += 1
            actor_updates += int(train_actor)

        schedule_metrics["rlt/critic_updates_run"] = float(critic_updates)
        schedule_metrics["rlt/actor_updates_run"] = float(actor_updates)
        schedule_metrics["rlt/updates_to_run"] = float(num_updates)
        schedule_metrics["rlt/should_train"] = 1.0
        append_to_dict(metrics, schedule_metrics)
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        return self._reduce_metrics(metrics)

    def update_once(self, *, train_actor: bool) -> dict[str, float]:
        """Apply one critic update and, when scheduled, one actor update."""
        batch = self._sample_batch()

        self.qf_optimizer.zero_grad(set_to_none=True)
        critic_loss, critic_metrics = self.forward_critic(batch)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.qf_optimizer.param_groups for p in group["params"]],
            max_norm=float(self.cfg.actor.critic_optim.clip_grad),
        )
        self.qf_optimizer.step()

        metrics = {
            "sac/critic_loss": float(critic_loss.detach().item()),
            "critic/grad_norm": float(critic_grad_norm),
            "critic/lr": float(self.qf_optimizer.param_groups[0]["lr"]),
            **{f"critic/{key}": value for key, value in critic_metrics.items()},
        }

        if train_actor:
            self.optimizer.zero_grad(set_to_none=True)
            actor_loss, entropy, actor_metrics = self.forward_actor(batch)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                max_norm=float(self.cfg.actor.optim.clip_grad),
            )
            self.optimizer.step()
            metrics.update(
                {
                    "sac/actor_loss": float(actor_loss.detach().item()),
                    "actor/entropy": float(entropy.detach().item()),
                    "actor/grad_norm": float(actor_grad_norm),
                    "actor/lr": float(self.optimizer.param_groups[0]["lr"]),
                    **{f"actor/{key}": value for key, value in actor_metrics.items()},
                }
            )

        if self.update_step % int(self.cfg.algorithm.get("target_update_freq", 1)) == 0:
            self.soft_update_target_model()
        return metrics

    def _sample_batch(self) -> dict[str, Any]:
        """Draw one mixed replay/demo batch and move it to the device."""
        from rlinf.utils.nested_dict_process import put_tensor_device

        demo_ratio = float(self.cfg.algorithm.get("demo_batch_ratio", 0.0))
        demo_size = 0
        if self.demo_buffer is not None and demo_ratio > 0.0:
            if self.demo_buffer.is_ready(1):
                demo_size = min(int(self.batch_size * demo_ratio), self.batch_size - 1)
        online_size = self.batch_size - demo_size

        batch = self.replay_buffer.sample(online_size)
        if demo_size:
            demo_batch = self.demo_buffer.sample(demo_size)
            batch = self._concat_batches(batch, demo_batch, online_size)
        return put_tensor_device(batch, device=self.device)

    @staticmethod
    def _concat_batches(
        online: dict[str, Any], demo: dict[str, Any], online_size: int
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for key, value in online.items():
            other = demo.get(key)
            if torch.is_tensor(value) and torch.is_tensor(other):
                merged[key] = torch.cat([value, other], dim=0)
            elif isinstance(value, dict) and isinstance(other, dict):
                merged[key] = {
                    name: torch.cat([tensor, other[name]], dim=0)
                    for name, tensor in value.items()
                    if torch.is_tensor(tensor) and torch.is_tensor(other.get(name))
                }
            else:
                merged[key] = value
        merged["_per_online_count"] = torch.tensor(online_size)
        return merged

    @staticmethod
    def _reduce_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        reduced: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                if not value:
                    continue
                reduced[key] = float(np.mean([float(item) for item in value]))
            elif torch.is_tensor(value):
                reduced[key] = float(value.detach().float().mean().item())
            else:
                reduced[key] = float(value)
        return reduced

    @torch.no_grad()
    def soft_update_target_model(self, tau: float | None = None) -> None:
        """EMA the target network and refresh the frozen EXPO selection critic."""
        tau = self.tau if tau is None else float(tau)
        for (online_name, online), (target_name, target) in zip(
            self.model.named_parameters(), self.target_model.named_parameters()
        ):
            assert online_name == target_name
            if "q_head" not in online_name and self.target_update_type != "all":
                target.data.copy_(online.data)
            else:
                target.data.mul_(1.0 - tau).add_(online.data, alpha=tau)

        if hasattr(self.model, "sync_selection_critic_from") and hasattr(
            self.target_model, "q_head"
        ):
            self.model.sync_selection_critic_from(self.target_model.q_head)

    # ------------------------------------------------------ checkpointing

    def save(self, save_dir: str) -> None:
        """Write the model, target, optimizers, buffers and RLT bookkeeping."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "target_model": self.target_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "qf_optimizer": self.qf_optimizer.state_dict(),
                "rewind": self.rlt_rewind_state_dict(),
                "schedule": self.rlt_schedule_state_dict(),
            },
            os.path.join(save_dir, "stage2_state.pt"),
        )
        self.replay_buffer.save_checkpoint(os.path.join(save_dir, "replay_buffer"))
        if self.demo_buffer is not None:
            self.demo_buffer.save_checkpoint(os.path.join(save_dir, "demo_buffer"))
        logger.info("Saved RLT Stage 2 state to %s", save_dir)

    def load(self, load_dir: str) -> None:
        """Restore everything :meth:`save` wrote.

        Raises:
            FileNotFoundError: ``load_dir`` has no ``stage2_state.pt``.
        """
        state_path = os.path.join(load_dir, "stage2_state.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"No RLT Stage 2 checkpoint at {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.target_model.load_state_dict(state["target_model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.qf_optimizer.load_state_dict(state["qf_optimizer"])
        self.load_rlt_rewind_state_dict(state["rewind"])
        self.load_rlt_schedule_state_dict(state.get("schedule", {}))

        replay_dir = os.path.join(load_dir, "replay_buffer")
        if os.path.exists(replay_dir):
            self.replay_buffer.load_checkpoint(replay_dir)
        demo_dir = os.path.join(load_dir, "demo_buffer")
        if self.demo_buffer is not None and os.path.exists(demo_dir):
            self.demo_buffer.load_checkpoint(demo_dir)
        logger.info(
            "Restored RLT Stage 2 state from %s (update_step=%d replay=%d)",
            load_dir,
            self.update_step,
            self.replay_buffer.total_samples,
        )


def _to_replay_row(key: str, value) -> torch.Tensor:
    """Reshape one observation entry into a ``[T=1, B=1, ...]`` replay row.

    ``RLTStage2Inference.encode`` returns tensors that already carry a batch
    axis of 1, so that axis is replaced rather than added.

    Args:
        key: Observation key, used only in the error message.
        value: Tensor or array for a single sample, with a leading batch axis.

    Returns:
        A contiguous CPU tensor shaped ``[1, 1, ...]``.

    Raises:
        ValueError: The entry has no batch axis, or a batch larger than one.
            Guessing either way would reshape real data into the wrong slots.
    """
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == 0 or tensor.shape[0] != 1:
        raise ValueError(
            f"Observation {key!r} has shape {tuple(tensor.shape)}; the server "
            "ingests one transition at a time and expects a leading batch axis "
            "of 1, as produced by RLTStage2Inference.encode."
        )
    return tensor.reshape(1, 1, *tensor.shape[1:]).contiguous()
