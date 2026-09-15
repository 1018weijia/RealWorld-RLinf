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

"""Cal-QL pretraining and persistent offline replay for the native WS trainer."""

from __future__ import annotations

import copy
import math
import os
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from rlinf.algorithms.rlt.learner import _optional_float
from rlinf.algorithms.rlt.losses import compute_rlt_critic_loss
from rlinf.serving.rlt.cobot_offline_data import OfflineBuffer, atomic_save, contract
from rlinf.serving.rlt.trainer import RLTStage2Trainer


def conservative_gap(policy_q, other_q, data_q, returns, temperature: float = 1.0):
    """Calibrate only policy proposals with behavior returns, then apply CQL."""
    calibrated = torch.maximum(policy_q, returns[:, None, :])
    values = torch.cat([calibrated, other_q], dim=1)
    partition = temperature * (
        torch.logsumexp(values / temperature, dim=1) - math.log(values.shape[1])
    )
    return (partition - data_q).mean()


class RLTOfflineTrainer(RLTStage2Trainer):
    """Use the identical actor/critic and checkpoint for offline and online RL.

    Offline data stays separate from online/HIL replay, so PER and rewind can
    never mutate demonstrations. Online updates blend TD losses from both
    partitions; Cal-QL and successful-demo BC are active only during pretraining.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offline_buffer = None
        self.offline_total_updates = 0
        self.offline_mode = False
        self.offline_options = dict(self.cfg.get("offline", {}) or {})
        self.log_alpha = torch.nn.Parameter(torch.tensor(1.0, device=self.device))
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=float(self.cfg.actor.critic_optim.lr)
        )

    def attach_offline_buffer(
        self, payload: dict, *, allow_actor_reconfiguration: bool = False
    ) -> None:
        """Reject data encoded with another task, model, normalization or horizon."""
        expected = contract(self.cfg)
        actual = payload["contract"]
        if allow_actor_reconfiguration:
            if not self.offline_mode:
                raise ValueError(
                    "Actor reconfiguration requires fresh offline training"
                )
            # Only these two scalars are independent of the cached features,
            # reference candidates, measured actions, and MC returns.
            adjusted = copy.deepcopy(actual)
            for key in ("actor_noise_sigma", "residual_scale"):
                adjusted["actor_model"][key] = expected["actor_model"][key]
        else:
            adjusted = actual
        if adjusted != expected:
            raise ValueError(
                "Offline buffer differs from current task/Stage1/norm stats/Stage2 configuration"
            )
        if actual != expected:
            # Keep the source contract and arrays untouched. Checkpoint copies
            # carry the training contract so online resume stays strict.
            payload = dict(payload)
            payload["conversion_contract"] = payload.get("conversion_contract", actual)
            payload["contract"] = expected
        self.offline_buffer = OfflineBuffer(payload)

    def _sample_batch(self):
        if self.offline_buffer is None:
            return super()._sample_batch()
        if self.offline_mode:
            return {
                "offline": self.offline_buffer.sample(self.batch_size, self.device),
                "online": None,
                "ratio": 1.0,
            }
        ratio = float(self.cfg.algorithm.get("offline_sample_ratio", 0.1))
        if not 0 <= ratio < 1:
            raise ValueError("Online offline_sample_ratio must be in [0, 1)")
        count = int(self.batch_size * ratio)
        original_size = self.batch_size
        try:
            self.batch_size -= count
            online = super()._sample_batch()
        finally:
            self.batch_size = original_size
        return {
            "online": online,
            "offline": self.offline_buffer.sample(count, self.device)
            if count
            else None,
            "ratio": count / original_size,
        }

    def _offline_td(self, batch):
        return compute_rlt_critic_loss(
            model=self.model,
            target_model=self.target_model,
            batch=batch,
            gamma=float(self.cfg.algorithm.gamma),
            use_done_key=True,
            next_actions_fn=self._next_actions_for_critic_target,
            critic_loss_type=str(self.cfg.algorithm.critic_loss),
            critic_huber_delta=float(self.cfg.algorithm.critic_huber_delta),
            td_backup=self._rl_algo_td_backup(),
            critic_num_min_qs=int(self.cfg.actor.model.get("critic_num_min_qs", 2)),
            td_target_clip_min=_optional_float(
                self.cfg.algorithm.get("td_target_clip_min")
            ),
            td_target_clip_max=_optional_float(
                self.cfg.algorithm.get("td_target_clip_max")
            ),
            **self._action_clip_bounds(),
        )[0]

    def _calql(self, batch):
        model = self.model
        obs = batch["curr_obs"]
        state, ref = model._state(obs), model._get_ref_chunk(obs)
        count = int(self.offline_options.get("num_policy_actions", 4))
        random_count = int(self.offline_options.get("num_random_actions", 4))
        if count < 1 or random_count < 1:
            raise ValueError("Cal-QL requires policy and local-random proposals")
        with torch.no_grad():
            policy = torch.stack(
                [
                    model.actor(
                        state, ref, deterministic=False, apply_action_noise=True
                    )
                    for _ in range(count)
                ],
                dim=1,
            )
            noise = (
                torch.rand(len(ref), random_count, ref.shape[-1], device=self.device)
                * 2
                - 1
            )
            random = (
                ref[:, None]
                + noise * float(self.offline_options.get("random_scale", 0.2))
            ).clamp(model.action_clip_min, model.action_clip_max)
            other = torch.cat([model._get_ref_candidates(obs), random], dim=1)

        def q(actions):
            repeated = state[:, None].expand(-1, actions.shape[1], -1)
            return model.q_head(
                repeated.reshape(-1, state.shape[-1]),
                actions.reshape(-1, ref.shape[-1]),
            ).reshape(len(ref), actions.shape[1], -1)

        data_q = model.q_head(state, batch["actions"])
        gap = conservative_gap(q(policy), q(other), data_q, batch["mc_returns"])
        return gap, data_q.mean()

    def forward_critic(self, batch):
        if "online" not in batch:
            return super().forward_critic(batch)
        ratio = batch["ratio"]
        loss = torch.zeros((), device=self.device)
        metrics = {}
        if batch["online"] is not None:
            native, metrics = super().forward_critic(batch["online"])
            loss = (1 - ratio) * native
        if batch["offline"] is not None:
            td = self._offline_td(batch["offline"])
            loss = loss + ratio * td
            metrics["offline_td_loss"] = float(td.detach())
        if self.offline_mode:
            gap, data_q = self._calql(batch["offline"])
            weight = min(
                1.0,
                (self.offline_total_updates + 1)
                / max(1, int(self.offline_options.get("warmup_steps", 1000))),
            )
            excess = gap - float(self.offline_options.get("target_gap", 0.05))
            alpha = self.log_alpha.exp().clamp(max=1e6)
            loss = loss + weight * alpha.detach() * excess
            self.alpha_optimizer.zero_grad(set_to_none=True)
            (-weight * alpha * excess.detach()).backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha.clamp_(-20, math.log(1e6))
            metrics.update(
                calql_gap=float(gap.detach()),
                calql_alpha=float(alpha.detach()),
                q_data=float(data_q.detach()),
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite critic loss")
        return loss, metrics

    def forward_actor(self, batch):
        if "online" not in batch:
            return super().forward_actor(batch)
        if not self.offline_mode:
            online, entropy, metrics = super().forward_actor(batch["online"])
            if batch["offline"] is None:
                return online, entropy, metrics
            # Match the online actor objective; no demonstration BC during online RL.
            offline, _, _ = super().forward_actor(batch["offline"])
            return (
                (1 - batch["ratio"]) * online + batch["ratio"] * offline,
                entropy,
                metrics,
            )
        data = batch["offline"]
        model = self.model
        state, reference = (
            model._state(data["curr_obs"]),
            model._get_ref_chunk(data["curr_obs"]),
        )
        prediction = model.actor.mean(state, reference)
        # Project successful demonstrations into the bounded residual support.
        target = torch.maximum(
            torch.minimum(data["actions"], reference + model.actor.edit_scale),
            reference - model.actor.edit_scale,
        )
        target = target.clamp(model.action_clip_min, model.action_clip_max)
        success = data["success"].float().reshape(-1)
        bc = (
            F.mse_loss(prediction, target, reduction="none").mean(-1) * success
        ).sum() / success.sum().clamp_min(1)
        parameters = list(model.q_head.parameters())
        for p in parameters:
            p.requires_grad_(False)
        try:
            q_loss = -model.q_head(state, prediction).mean()
        finally:
            for p in parameters:
                p.requires_grad_(True)
        steps = self.offline_total_updates
        total = max(1, int(self.offline_options.get("steps", 40000)))
        bc_weight = 1.0 - 0.9 * min(1, steps / total)
        warmup = int(self.offline_options.get("warmup_steps", 1000))
        q_weight = 0.1 * min(1.0, max(0.0, (steps - warmup) / max(1, warmup)))
        loss = bc_weight * bc + q_weight * q_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite actor loss")
        return (
            loss,
            torch.zeros((), device=self.device),
            {
                "demo_bc": float(bc.detach()),
                "q_loss": float(q_loss.detach()),
                "bc_weight": bc_weight,
                "q_weight": q_weight,
                "success_fraction": float(success.mean()),
                "demo_reachable_fraction": float(
                    ((data["actions"] - reference).abs() <= model.actor.edit_scale)
                    .all(-1)
                    .float()
                    .mean()
                ),
            },
        )

    @torch.no_grad()
    def validate_offline(self) -> dict[str, float]:
        """Evaluate held-out episodes without advancing training or PER state."""
        if not len(self.offline_buffer.validation_indices):
            return {}
        batch = self.offline_buffer.sample(
            self.batch_size, self.device, validation=True
        )
        td = self._offline_td(batch)
        state, ref = (
            self.model._state(batch["curr_obs"]),
            self.model._get_ref_chunk(batch["curr_obs"]),
        )
        action = self.model.actor.mean(state, ref)
        return {
            "validation/td_loss": float(td),
            "validation/action_mse": float(F.mse_loss(action, batch["actions"])),
            "validation/q_data": float(
                self.model.q_head(state, batch["actions"]).mean()
            ),
            "validation/mc_return": float(batch["mc_returns"].mean()),
        }

    def save(self, save_dir: str) -> None:
        destination = Path(save_dir)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        super().save(str(staging))
        if self.offline_buffer is not None:
            atomic_save(self.offline_buffer.payload, staging / "offline_buffer.pt")
            atomic_save(
                {
                    "offline_total_updates": self.offline_total_updates,
                    "log_alpha": self.log_alpha.detach().cpu(),
                    "alpha_optimizer": self.alpha_optimizer.state_dict(),
                    "options": self.offline_options,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                },
                staging / "offline_state.pt",
            )
        os.rename(staging, destination)

    def load(self, load_dir: str) -> None:
        offline = Path(load_dir) / "offline_buffer.pt"
        if offline.exists():
            self.attach_offline_buffer(
                torch.load(offline, map_location="cpu", weights_only=False)
            )
            state = torch.load(
                Path(load_dir) / "offline_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.offline_total_updates = int(state["offline_total_updates"])
            self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
            if self.offline_mode:
                keys = (
                    "warmup_steps",
                    "target_gap",
                    "num_policy_actions",
                    "num_random_actions",
                    "random_scale",
                )
                if any(
                    self.offline_options.get(k) != state["options"].get(k) for k in keys
                ):
                    raise ValueError("Offline resume options differ from saved run")
                torch.set_rng_state(state["rng"])
                if state["cuda_rng"] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(state["cuda_rng"])
        super().load(load_dir)
