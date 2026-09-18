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

from rlinf.algorithms.rlt.losses import compute_rlt_critic_loss
from rlinf.serving.rlt.cobot_offline_data import OfflineBuffer, atomic_save, contract
from rlinf.serving.rlt.trainer import RLTStage2Trainer


def joint_loss_options(options: dict) -> dict:
    """Resolve an opt-in loss recipe without changing legacy checkpoints."""
    version = options.get("objective_version", "legacy")
    if version == "legacy":
        return {}
    if version != "joint-loss-v3":
        raise ValueError(f"Unknown offline objective {version}")
    result = {
        "objective_version": version,
        "calql_alpha": 0.1,
        "calql_calibration": "policy_only",
        "bc_weight": 1.0,
        "q_weight": 0.01,
        "q_warmup_steps": 2000,
        "q_ramp_steps": 2000,
        "mc_warmup_steps": 0,
        "q_aggregation": "min",
        "gradient_every": 500,
        "validation_seed": 918,
        "online_bc_weight": 1.0,
        "online_q_weight_max": 0.01,
    }
    result.update({key: options[key] for key in result if key in options})
    for key, value in result.items():
        if isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0):
            raise ValueError(f"offline.{key} must be finite and nonnegative")
    if result["q_aggregation"] not in ("min", "mean"):
        raise ValueError("offline.q_aggregation must be min or mean")
    if result["calql_calibration"] not in ("policy_only", "reachable_family"):
        raise ValueError(
            "offline.calql_calibration must be policy_only or reachable_family"
        )
    if result["calql_alpha"] > 0.2:
        raise ValueError("joint-loss-v3 requires fixed calql_alpha <= 0.2")
    return result


def joint_budget_bc(
    model,
    obs: dict,
    prediction: torch.Tensor,
    target: torch.Tensor,
    success: torch.Tensor,
) -> torch.Tensor:
    """Fit successful feasible arm targets in units of each physical edit budget."""
    from rlinf.models.embodiment.mlp_policy.cobot_joint_motion import ARM

    motion = model.joint_motion
    scale, _, _ = motion.context(model._state(obs))
    error = (prediction - target).reshape(-1, motion.chunk_len, 14) * scale[:, None]
    per_sample = (error[:, :, ARM] / motion.residual_rad).square().mean(dim=(1, 2))
    mask = success.float().reshape(-1)
    return (per_sample * mask).sum() / mask.sum().clamp_min(1)


def conservative_gap(
    policy_q,
    other_q,
    data_q,
    returns,
    temperature: float = 1.0,
    *,
    calibrate_other: bool = False,
):
    """Calibrate proposals inside the penalty, without constraining network Q.

    calibrate_other is only for the joint actor's projected references and local
    knot proposals: all belong to its executable policy family, not a separate
    uniform-action distribution. Legacy random-action candidates stay unchanged.
    """
    calibrated = torch.maximum(policy_q, returns[:, None, :])
    if calibrate_other:
        other_q = torch.maximum(other_q, returns[:, None, :])
    values = torch.cat([calibrated, other_q], dim=1)
    partition = temperature * (
        torch.logsumexp(values / temperature, dim=1) - math.log(values.shape[1])
    )
    return (partition - data_q).mean()


class CobotOfflineTrainer(RLTStage2Trainer):
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
        self.objective = joint_loss_options(self.offline_options)
        if self.objective and getattr(self.model, "joint_motion", None) is None:
            raise ValueError("joint-loss-v3 requires the Cobot joint motion actor")
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
        if self.objective and bool(
            self.offline_options.get("cache_joint_references", True)
        ):
            self.offline_buffer.cache_joint_references(self.model, self.device)

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
            if getattr(model, "joint_motion", None) is not None:
                policy = model.actor(
                    state.repeat_interleave(count, dim=0),
                    ref.repeat_interleave(count, dim=0),
                    deterministic=False,
                    apply_action_noise=True,
                ).reshape(len(ref), count, -1)
            else:
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
            if getattr(model, "joint_motion", None) is not None:
                random = model.local_random_candidates(obs, random_count, reference=ref)
            other = torch.cat([model._get_ref_candidates(obs), random], dim=1)

        def q(actions):
            repeated = state[:, None].expand(-1, actions.shape[1], -1)
            return model.q_head(
                repeated.reshape(-1, state.shape[-1]),
                actions.reshape(-1, ref.shape[-1]),
            ).reshape(len(ref), actions.shape[1], -1)

        data_q = model.q_head(state, batch["actions"])
        gap = conservative_gap(
            q(policy),
            q(other),
            data_q,
            batch["mc_returns"],
            calibrate_other=self.objective.get("calql_calibration")
            == "reachable_family",
        )
        return gap, data_q.mean()

    def forward_critic(self, batch):
        if "online" not in batch:
            return super().forward_critic(batch)
        if self.offline_mode and self.objective:
            return self._forward_joint_critic(batch["offline"])
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

    def _forward_joint_critic(self, data: dict):
        options = self.objective
        obs = data["curr_obs"]
        q_data = self.model.q_head(self.model._state(obs), data["actions"])
        mc = (
            F.huber_loss(
                q_data,
                data["mc_returns"].expand_as(q_data),
                delta=float(self.cfg.algorithm.critic_huber_delta),
                reduction="none",
            )
            .sum(-1)
            .mean()
        )
        mc_phase = self.offline_total_updates < int(options["mc_warmup_steps"])
        metrics = {
            "mc_loss": float(mc.detach()),
            "mc_warmup": float(mc_phase),
            "q_data": float(q_data.detach().mean()),
            "calql_alpha": float(options["calql_alpha"]),
        }
        if mc_phase:
            return mc, metrics
        td = self._offline_td(data)
        weight = min(
            1.0,
            (self.offline_total_updates - int(options["mc_warmup_steps"]) + 1)
            / max(1, int(self.offline_options.get("warmup_steps", 1000))),
        )
        gap, _ = self._calql(data)
        conservative = (
            weight
            * float(options["calql_alpha"])
            * (gap - float(self.offline_options.get("target_gap", 0.05)))
        )
        loss = td + conservative
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite joint critic loss")
        metrics.update(
            offline_td_loss=float(td.detach()),
            calql_gap=float(gap.detach()),
            calql_effective_alpha=weight * float(options["calql_alpha"]),
            calql_loss=float(conservative.detach()),
        )
        return loss, metrics

    def _joint_actor_terms(self, data: dict):
        model = self.model
        obs = data["curr_obs"]
        state, reference = model._state(obs), model._get_ref_chunk(obs)
        prediction = model.actor.mean(state, reference)
        target = model.demo_target(obs, data["actions"], reference=reference).detach()
        bc = joint_budget_bc(model, obs, prediction, target, data["success"])
        parameters = list(model.q_head.parameters())
        requires_grad = [p.requires_grad for p in parameters]
        for p in parameters:
            p.requires_grad_(False)
        try:
            values = model.q_head(state, prediction)
            q_loss = -(
                values.min(-1).values
                if self.objective["q_aggregation"] == "min"
                else values.mean(-1)
            ).mean()
        finally:
            for p, required in zip(parameters, requires_grad):
                p.requires_grad_(required)
        return bc, q_loss

    def _forward_joint_actor(self, data: dict):
        options = self.objective
        bc, q_loss = self._joint_actor_terms(data)
        ramp = min(
            1.0,
            max(
                0.0,
                (self.offline_total_updates - int(options["q_warmup_steps"]))
                / max(1, int(options["q_ramp_steps"])),
            ),
        )
        bc_weight, q_weight = (
            float(options["bc_weight"]),
            float(options["q_weight"]) * ramp,
        )
        weighted_bc, weighted_q = bc_weight * bc, q_weight * q_loss
        loss = weighted_bc + weighted_q
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite joint actor loss")
        metrics = {
            "demo_bc": float(bc.detach()),
            "q_loss": float(q_loss.detach()),
            "bc_weight": bc_weight,
            "q_weight": q_weight,
            "weighted_bc": float(weighted_bc.detach()),
            "weighted_q": float(weighted_q.detach()),
            "success_fraction": float(data["success"].float().mean()),
        }
        every = int(options["gradient_every"])
        if every and self.offline_total_updates % every == 0:
            parameters = list(self.model.actor.parameters())

            def norm(term):
                gradients = torch.autograd.grad(
                    term, parameters, retain_graph=True, allow_unused=True
                )
                return (
                    sum(
                        float(g.detach().square().sum())
                        for g in gradients
                        if g is not None
                    )
                    ** 0.5
                )

            bc_norm, q_norm = norm(weighted_bc), norm(weighted_q)
            metrics.update(
                bc_grad_norm=bc_norm,
                q_grad_norm=q_norm,
                q_to_bc_grad_ratio=q_norm / max(bc_norm, 1e-12),
            )
        return loss, torch.zeros((), device=self.device), metrics

    def forward_actor(self, batch):
        if "online" not in batch:
            return super().forward_actor(batch)
        if self.offline_mode and self.objective:
            return self._forward_joint_actor(batch["offline"])
        if not self.offline_mode:
            online, entropy, metrics = super().forward_actor(batch["online"])
            if batch["offline"] is None:
                return online, entropy, metrics
            # Legacy uses only the online objective; v3 also retains a demo anchor.
            offline, _, _ = super().forward_actor(batch["offline"])
            if self.objective:
                bc, _ = self._joint_actor_terms(batch["offline"])
                anchor = float(self.objective["online_bc_weight"]) * bc
                metrics["offline_demo_bc"] = float(bc.detach())
                metrics["offline_weighted_bc"] = float(anchor.detach())
                return (
                    (1 - batch["ratio"]) * online + batch["ratio"] * offline + anchor,
                    entropy,
                    metrics,
                )
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
        if getattr(model, "joint_motion", None) is not None:
            target = model.demo_target(
                data["curr_obs"], data["actions"], reference=reference
            ).detach()
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
        reachable = (
            ((data["actions"] - reference).abs() <= model.actor.edit_scale)
            .all(-1)
            .float()
            .mean()
        )
        if getattr(model, "joint_motion", None) is not None:
            from rlinf.models.embodiment.mlp_policy.cobot_joint_motion import ARM

            motion = model.joint_motion
            scale, _, _ = motion.context(state)
            delta = (
                (data["actions"] - reference).reshape(-1, motion.chunk_len, 14)
                * scale[:, None]
            )[:, :, ARM]
            reachable = (delta.abs() <= motion.residual_rad).float().mean()
        return (
            loss,
            torch.zeros((), device=self.device),
            {
                "demo_bc": float(bc.detach()),
                "q_loss": float(q_loss.detach()),
                "bc_weight": bc_weight,
                "q_weight": q_weight,
                "success_fraction": float(success.mean()),
                "demo_reachable_fraction": float(reachable),
            },
        )

    @torch.no_grad()
    def validate_offline(self) -> dict[str, float]:
        """Evaluate held-out episodes without advancing training or PER state."""
        if not len(self.offline_buffer.validation_indices):
            return {}
        if self.objective:
            devices = (
                [
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                ]
                if self.device.type == "cuda"
                else []
            )
            # Validation never advances training noise or sampling RNGs.
            with torch.random.fork_rng(devices=devices):
                seed = int(self.objective["validation_seed"])
                torch.random.default_generator.manual_seed(seed)
                for device in devices:
                    torch.cuda.default_generators[device].manual_seed(seed)
                batch = self.offline_buffer.fixed_validation(
                    self.batch_size,
                    self.device,
                    seed=int(self.objective["validation_seed"]),
                )
                return self._validate_batch(batch)
        batch = self.offline_buffer.sample(
            self.batch_size, self.device, validation=True
        )
        return self._validate_batch(batch)

    def _validate_batch(self, batch: dict) -> dict[str, float]:
        td = self._offline_td(batch)
        state, ref = (
            self.model._state(batch["curr_obs"]),
            self.model._get_ref_chunk(batch["curr_obs"]),
        )
        action = self.model.actor.mean(state, ref)
        metrics = {
            "validation/td_loss": float(td),
            "validation/action_mse": float(F.mse_loss(action, batch["actions"])),
            "validation/q_data": float(
                self.model.q_head(state, batch["actions"]).mean()
            ),
            "validation/mc_return": float(batch["mc_returns"].mean()),
        }
        motion = getattr(self.model, "joint_motion", None)
        if motion is not None:
            from rlinf.models.embodiment.mlp_policy.cobot_joint_motion import ARM

            scale, offset, anchor = motion.context(state)
            physical = (
                action.reshape(-1, motion.chunk_len, 14) * scale[:, None]
                + offset[:, None]
            )
            velocity, acceleration = motion.differences(
                physical[:, :, ARM], anchor[:, ARM]
            )
            baseline = motion.physical(ref, state)
            metrics.update(
                {
                    "validation/max_velocity_rad_s": float(
                        velocity.abs().max() * motion.hz
                    ),
                    "validation/max_acceleration_rad_s2": float(
                        acceleration.abs().max() * motion.hz**2
                    ),
                    "validation/max_residual_rad": float(
                        (physical[:, :, ARM] - baseline[:, :, ARM]).abs().max()
                    ),
                }
            )
            if self.objective:
                target = self.model.demo_target(
                    batch["curr_obs"], batch["actions"], reference=ref
                )
                q_data = self.model.q_head(state, batch["actions"])
                logits = self.model.actor.mlp(torch.cat((state, ref), dim=-1))
                metrics.update(
                    {
                        "validation/budget_bc": float(
                            joint_budget_bc(
                                self.model,
                                batch["curr_obs"],
                                action,
                                target,
                                batch["success"],
                            )
                        ),
                        "validation/q_actor": float(
                            self.model.q_head(state, action).mean()
                        ),
                        "validation/q_base": float(
                            self.model.q_head(state, ref).mean()
                        ),
                        "validation/q_mc_mae": float(
                            (q_data - batch["mc_returns"]).abs().mean()
                        ),
                        "validation/knot_saturation": float(
                            (logits.tanh().abs() > 0.99).float().mean()
                        ),
                        "validation/sample_count": float(len(action)),
                    }
                )
        return metrics

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
                    "objective": self.objective,
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
            saved_objective = joint_loss_options(state.get("objective", {}))
            if self.offline_mode and saved_objective != self.objective:
                raise ValueError(
                    "Offline resume loss recipe differs; start a fresh run"
                )
            if not self.offline_mode:
                self.objective = saved_objective
                if self.objective and not bool(self.cfg.server.get("eval_only", False)):
                    if float(self.cfg.algorithm.q_weight) > float(
                        self.objective["online_q_weight_max"]
                    ):
                        raise ValueError(
                            "joint-loss-v3 online handoff requires algorithm.q_weight<=0.01"
                        )
                    if float(self.cfg.algorithm.get("offline_sample_ratio", 0.0)) <= 0:
                        raise ValueError(
                            "joint-loss-v3 online handoff requires offline replay for BC"
                        )
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
