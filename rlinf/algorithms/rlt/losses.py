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

"""Pure RLT Stage-2 actor/critic loss helpers (no Worker dependency)."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def flatten_chunk(tensor: Tensor) -> Tensor:
    if tensor.dim() <= 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


def require_twin_q(all_q_values: Tensor) -> None:
    if all_q_values.shape[-1] < 2:
        raise ValueError(
            "RLT Stage 2 requires at least two Q heads for twin-Q training, "
            f"got Q shape {tuple(all_q_values.shape)}."
        )


def min_twin_q(all_q_values: Tensor) -> Tensor:
    require_twin_q(all_q_values)
    return torch.minimum(all_q_values[..., 0:1], all_q_values[..., 1:2])


def q1(all_q_values: Tensor) -> Tensor:
    require_twin_q(all_q_values)
    return all_q_values[..., 0:1]


def discounted_chunk_rewards(rewards: Tensor, gamma: float) -> Tensor:
    rewards = rewards.reshape(rewards.shape[0], -1)
    chunk_len = rewards.shape[-1]
    discounts = torch.pow(
        torch.as_tensor(gamma, device=rewards.device, dtype=rewards.dtype),
        torch.arange(chunk_len, device=rewards.device, dtype=rewards.dtype),
    )
    return torch.sum(rewards * discounts, dim=-1, keepdim=True)


def compute_rlt_bc_loss(
    pi: Tensor,
    actions: Tensor,
    ref_chunk: Tensor,
    intervene_flags: Tensor | None,
    *,
    chunk_len: int,
    action_dim: int,
) -> tuple[Tensor, dict[str, float]]:
    """BC toward human action when intervening, else toward VLA reference."""
    pi_chunk = flatten_chunk(pi).reshape(-1, chunk_len, action_dim)
    action_chunk = flatten_chunk(actions).reshape(-1, chunk_len, action_dim)
    bc_ref_chunk = flatten_chunk(ref_chunk).reshape(ref_chunk.shape[0], -1, action_dim)[
        :, :chunk_len
    ]
    batch_size = pi_chunk.shape[0]

    if intervene_flags is None:
        human_mask = torch.zeros(
            (batch_size, chunk_len), dtype=torch.bool, device=pi_chunk.device
        )
    else:
        flags = flatten_chunk(intervene_flags).to(device=pi_chunk.device).bool()
        if flags.shape[-1] == chunk_len:
            human_mask = flags.reshape(batch_size, chunk_len)
        else:
            human_mask = flags.reshape(batch_size, chunk_len, action_dim).any(dim=-1)

    bc_target = torch.where(human_mask[..., None], action_chunk, bc_ref_chunk)
    bc_error = torch.mean(torch.square(pi_chunk - bc_target), dim=-1)
    bc_loss = torch.mean(bc_error)

    policy_mask = ~human_mask
    ref_error = torch.mean(torch.square(pi_chunk - bc_ref_chunk), dim=-1)
    human_error = torch.mean(torch.square(pi_chunk - action_chunk), dim=-1)
    bc_ref = torch.sum(ref_error * policy_mask.to(ref_error.dtype)) / torch.clamp(
        torch.sum(policy_mask.to(ref_error.dtype)), min=1.0
    )
    bc_human = torch.sum(human_error * human_mask.to(human_error.dtype)) / torch.clamp(
        torch.sum(human_mask.to(human_error.dtype)), min=1.0
    )

    human_ratio = torch.mean(human_mask.to(torch.float32)).item()
    metrics = {
        "bc_loss": bc_loss.detach().item(),
        "bc_ref_loss": bc_ref.detach().item(),
        "bc_human_loss": bc_human.detach().item(),
        "human_mask_ratio": human_ratio,
        "policy_mask_ratio": 1.0 - human_ratio,
    }
    return bc_loss, metrics


def resolve_bootstrap_mask(
    batch: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return per-sample bootstrap mask ``[B, 1]`` (default all ones)."""
    mask = batch.get("bootstrap_mask", None)
    if mask is None:
        return torch.ones((batch_size, 1), device=device, dtype=dtype)
    mask = torch.as_tensor(mask, device=device, dtype=dtype)
    return mask.reshape(batch_size, -1)[:, :1]


def compute_rlt_critic_loss(
    *,
    model: Any,
    target_model: Any,
    batch: dict[str, Any],
    gamma: float,
    bootstrap_type: str = "standard",
    use_crossq: bool = False,
    use_done_key: bool = False,
    next_actions_fn: Callable[[dict[str, Tensor]], tuple[Tensor, ...]] | None = None,
    critic_loss_type: str = "mse",
    critic_huber_delta: float = 0.5,
    intervention_noise_sigma: float = 0.0,
    intervention_noise_clip: float = 0.0,
    rewind_noise_sigma: float = 0.0,
    rewind_noise_clip: float = 0.0,
) -> tuple[Tensor, dict[str, Any]]:
    """Twin-Q TD loss with EXPO expected-max and optional PER weights."""
    from rlinf.models.embodiment.base_policy import ForwardType

    curr_obs = batch["curr_obs"]
    next_obs = batch["next_obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    done_source = batch["dones"] if use_done_key else batch["terminations"]
    done_source = done_source.to(dtype=rewards.dtype)
    not_done = ~done_source.reshape(done_source.shape[0], -1).bool().any(
        dim=-1, keepdim=True
    )

    with torch.no_grad():
        if next_actions_fn is None:
            next_actions, _, _ = model(
                forward_type=ForwardType.SAC,
                obs=next_obs,
            )
        else:
            next_out = next_actions_fn(next_obs)
            next_actions = next_out[0] if isinstance(next_out, tuple) else next_out

        override = batch.get("next_action_override")
        override_mask = batch.get("next_action_override_mask")
        mask = None
        if override is not None and override_mask is not None:
            override = flatten_chunk(
                torch.as_tensor(
                    override, device=next_actions.device, dtype=next_actions.dtype
                )
            )
            mask = (
                torch.as_tensor(override_mask, device=next_actions.device)
                .reshape(-1, 1)
                .bool()
            )

        if next_actions.dim() == 3:
            if use_crossq:
                raise ValueError("EXPO candidate targets do not support CrossQ")
            batch_size, candidate_count, action_size = next_actions.shape
            candidate_obs = {
                key: (
                    value.repeat_interleave(candidate_count, dim=0)
                    if torch.is_tensor(value) and value.shape[0] == batch_size
                    else value
                )
                for key, value in next_obs.items()
            }
            candidate_q = target_model(
                forward_type=ForwardType.SAC_Q,
                obs=candidate_obs,
                actions=next_actions.reshape(-1, action_size),
            ).reshape(batch_size, candidate_count, -1)
            q_next = (
                torch.minimum(candidate_q[..., 0], candidate_q[..., 1])
                .max(dim=1, keepdim=True)
                .values
            )
            if override is not None and mask is not None:
                if override.shape != (batch_size, action_size):
                    raise ValueError(
                        "next_action_override must match EXPO action shape"
                    )
                override_q = target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=override,
                )
                q_next = torch.where(mask, min_twin_q(override_q), q_next)
        elif not use_crossq:
            if override is not None and mask is not None:
                if override.shape != next_actions.shape:
                    raise ValueError(
                        "next_action_override must match critic target action shape"
                    )
                next_actions = torch.where(
                    mask.expand_as(next_actions), override, next_actions
                )
            all_qf_next_target = target_model(
                forward_type=ForwardType.SAC_Q,
                obs=next_obs,
                actions=next_actions,
            )
            q_next = min_twin_q(all_qf_next_target)
        else:
            _, all_qf_next = model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_actions,
            )
            q_next = min_twin_q(all_qf_next.detach())

        reward_target = discounted_chunk_rewards(rewards, gamma)
        reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
        bootstrap_discount = gamma**reward_horizon
        bootstrap_mask = resolve_bootstrap_mask(
            batch,
            batch_size=reward_target.shape[0],
            device=reward_target.device,
            dtype=reward_target.dtype,
        )
        if bootstrap_type == "always":
            bootstrap_gate = bootstrap_mask
        elif bootstrap_type == "standard":
            bootstrap_gate = not_done.to(dtype=reward_target.dtype) * bootstrap_mask
        else:
            raise NotImplementedError(f"{bootstrap_type=} is not supported!")
        target_q_values = reward_target + bootstrap_gate * bootstrap_discount * q_next

    critic_actions = flatten_chunk(actions)

    def _augment_selected(
        values: Tensor,
        selected: Tensor | None,
        sigma: float,
        noise_clip: float,
    ) -> Tensor:
        if selected is None or sigma <= 0.0:
            return values
        selected = (
            torch.as_tensor(selected, device=values.device)
            .reshape(values.shape[0], -1)
            .bool()
            .any(dim=-1, keepdim=True)
        )
        noise = torch.randn_like(values) * float(sigma)
        if noise_clip > 0.0:
            noise = noise.clamp(-float(noise_clip), float(noise_clip))
        return torch.where(selected, (values + noise).clamp(-1.0, 1.0), values)

    critic_actions = _augment_selected(
        critic_actions,
        batch.get("intervene_flags"),
        intervention_noise_sigma,
        intervention_noise_clip,
    )
    bad_mask = batch.get("rewind_bad_action")
    if bad_mask is None and batch.get("branch_id") is not None:
        bad_mask = torch.as_tensor(batch["branch_id"]) == 2
    critic_actions = _augment_selected(
        critic_actions,
        bad_mask,
        rewind_noise_sigma,
        rewind_noise_clip,
    )

    if not use_crossq:
        all_data_q_values = model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=critic_actions,
        )
    else:
        all_data_q_values, _ = model(
            forward_type=ForwardType.CROSSQ_Q,
            obs=curr_obs,
            actions=actions,
            next_obs=next_obs,
            next_actions=next_actions,
        )

    target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
    expanded_target = target_q_values.expand_as(all_data_q_values)
    loss_kind = str(critic_loss_type).lower().replace("smooth_l1", "huber")
    if loss_kind == "huber":
        elementwise_loss = F.huber_loss(
            all_data_q_values,
            expanded_target,
            delta=float(critic_huber_delta),
            reduction="none",
        )
    elif loss_kind == "mse":
        elementwise_loss = F.mse_loss(
            all_data_q_values, expanded_target, reduction="none"
        )
    else:
        raise ValueError("critic_loss_type must be 'huber' or 'mse'")
    per_sample_loss = elementwise_loss.sum(dim=-1)
    weights = batch.get("weights")
    if weights is not None:
        weights = torch.as_tensor(
            weights, device=per_sample_loss.device, dtype=per_sample_loss.dtype
        ).reshape(-1)
        critic_loss = (per_sample_loss * weights).mean()
    else:
        critic_loss = per_sample_loss.mean()
    td_errors = (all_data_q_values.detach() - expanded_target).abs().mean(dim=-1)
    metrics = {
        "q_data": all_data_q_values.mean().item(),
        "bootstrap_mask_mean": float(bootstrap_mask.mean().item()),
        "next_action_override_ratio": float(
            0.0
            if batch.get("next_action_override_mask") is None
            else torch.as_tensor(batch["next_action_override_mask"])
            .float()
            .mean()
            .item()
        ),
        "target_q": target_q_values.mean().item(),
        "expo_candidate_count": float(
            next_actions.shape[1] if next_actions.dim() == 3 else 1
        ),
        "_td_errors": td_errors,
    }
    return critic_loss, metrics


def compute_rlt_actor_loss(
    *,
    model: Any,
    batch: dict[str, Any],
    chunk_len: int,
    action_dim: int,
    q_weight: float = 1.0,
    bc_weight: float = 1.0,
    reference_dropout_prob: float = 0.0,
    use_crossq: bool = False,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Actor objective: ``-q_weight * Q1(pi) + bc_weight * BC``."""
    from rlinf.models.embodiment.base_policy import ForwardType

    curr_obs = batch["curr_obs"]
    pi, log_pi, _ = model(
        forward_type=ForwardType.SAC,
        obs=curr_obs,
        apply_reference_dropout=True,
        reference_dropout_prob=reference_dropout_prob,
    )
    if log_pi.ndim == 1:
        log_pi = log_pi.unsqueeze(-1)
    log_pi = log_pi.sum(dim=-1, keepdim=True)

    if not use_crossq:
        all_qf_pi = model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=pi,
            detach_encoder=True,
        )
    else:
        all_qf_pi, _ = model(
            forward_type=ForwardType.CROSSQ_Q,
            obs=curr_obs,
            actions=pi,
            next_obs=None,
            next_actions=None,
            detach_encoder=True,
        )

    num_q_values = all_qf_pi.shape[-1]
    metrics = {
        f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
        for q_id in range(num_q_values)
    }
    qf_pi = q1(all_qf_pi)
    metrics["q_pi"] = qf_pi.mean().item()

    ref_chunk = (
        flatten_chunk(curr_obs["ref_chunk"])
        .reshape(curr_obs["ref_chunk"].shape[0], -1, action_dim)[:, :chunk_len]
        .reshape(curr_obs["ref_chunk"].shape[0], -1)
    )

    bc_loss, rlt_metrics = compute_rlt_bc_loss(
        pi=pi,
        actions=batch["actions"],
        ref_chunk=ref_chunk,
        intervene_flags=batch.get("intervene_flags", None),
        chunk_len=chunk_len,
        action_dim=action_dim,
    )
    metrics.update(rlt_metrics)

    entropy = -log_pi.mean()
    actor_loss = -q_weight * qf_pi.mean() + bc_weight * bc_loss
    metrics["bc_weight"] = float(bc_weight)
    metrics["q_weight"] = float(q_weight)
    metrics["action_ref_abs_mean"] = (
        (flatten_chunk(pi) - flatten_chunk(ref_chunk)).abs().mean().detach().item()
    )
    metrics["weighted_q"] = (q_weight * qf_pi.mean()).detach().item()
    metrics["weighted_bc"] = (bc_weight * bc_loss).detach().item()
    metrics["reference_dropout_prob"] = float(reference_dropout_prob)
    return actor_loss, entropy, metrics


@torch.no_grad()
def compute_q_node1_gap(
    *,
    model: Any,
    curr_obs: dict[str, Tensor],
    human_actions: Tensor,
    bad_actions: Tensor,
) -> dict[str, float]:
    """Diagnostic ``Q(s1, a_human) - Q(s1, a_bad)`` on a shared node-1 state."""
    from rlinf.models.embodiment.base_policy import ForwardType

    q_human = min_twin_q(
        model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=human_actions,
        )
    )
    q_bad = min_twin_q(
        model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=bad_actions,
        )
    )
    gap = q_human - q_bad
    return {
        "q_node1_human": float(q_human.mean().item()),
        "q_node1_bad": float(q_bad.mean().item()),
        "q_node1_gap": float(gap.mean().item()),
    }


def _preference_weights(mask: Tensor, confidence: Tensor) -> Tensor:
    """Return one confidence weight per preference pair."""
    active = mask.reshape(mask.shape[0], -1).any(dim=-1)
    if not bool(active.all()):
        raise ValueError(
            "each preference pair must select at least one action dimension"
        )
    return confidence.reshape(-1).to(dtype=torch.float32)


def _masked_preference_actions(
    ref_chunk: Tensor, action: Tensor, action_mask: Tensor
) -> Tensor:
    ref = flatten_chunk(ref_chunk)
    value = flatten_chunk(action)
    mask = flatten_chunk(action_mask).to(device=value.device, dtype=value.dtype)
    if ref.shape != value.shape or mask.shape != value.shape:
        raise ValueError("preference action, reference, and mask shapes must match")
    return mask * value + (1.0 - mask) * ref


def critic_pairwise_rank_loss(
    *,
    model: Any,
    curr_obs: dict[str, Tensor],
    ref_chunk: Tensor,
    positive_action: Tensor,
    negative_action: Tensor,
    action_mask: Tensor,
    confidence: Tensor,
    margin: float,
) -> tuple[Tensor, dict[str, float]]:
    """Per-head hinge ranking matching remote-franka rewind preference."""
    from rlinf.models.embodiment.base_policy import ForwardType

    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    positive = _masked_preference_actions(ref_chunk, positive_action, action_mask)
    negative = _masked_preference_actions(ref_chunk, negative_action, action_mask)
    q_positive = model(forward_type=ForwardType.SAC_Q, obs=curr_obs, actions=positive)
    q_negative = model(forward_type=ForwardType.SAC_Q, obs=curr_obs, actions=negative)
    if q_positive.shape != q_negative.shape or q_positive.shape[-1] < 2:
        raise ValueError("preference ranking requires matching twin-Q outputs")
    weights = _preference_weights(action_mask, confidence).to(q_positive.device)
    per_head = F.relu(float(margin) - q_positive + q_negative)
    per_pair = per_head.reshape(per_head.shape[0], -1).mean(dim=-1)
    loss = torch.sum(per_pair * weights) / torch.clamp(weights.sum(), min=1.0)
    return loss, {
        "preference_critic_loss": float(loss.detach().item()),
        "preference_q_gap": float((q_positive - q_negative).mean().detach().item()),
        "preference_pair_count": float(weights.numel()),
    }


def actor_pairwise_preference_loss(
    *,
    action_mean: Tensor,
    ref_chunk: Tensor,
    positive_action: Tensor,
    negative_action: Tensor,
    action_mask: Tensor,
    confidence: Tensor,
    fixed_std: float,
    beta: float,
) -> tuple[Tensor, dict[str, float]]:
    """Fixed-std Gaussian Bradley-Terry loss with masked reference suffix."""
    if fixed_std <= 0.0:
        raise ValueError("fixed_std must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    positive = _masked_preference_actions(ref_chunk, positive_action, action_mask)
    negative = _masked_preference_actions(ref_chunk, negative_action, action_mask)
    mean = flatten_chunk(action_mean)
    mask = flatten_chunk(action_mask).to(device=mean.device, dtype=mean.dtype)
    std = torch.as_tensor(fixed_std, device=mean.device, dtype=mean.dtype)
    log_positive = (-0.5 * ((positive - mean) / std).square() * mask).sum(dim=-1)
    log_negative = (-0.5 * ((negative - mean) / std).square() * mask).sum(dim=-1)
    weights = _preference_weights(action_mask, confidence).to(mean.device, mean.dtype)
    raw = -F.logsigmoid(float(beta) * (log_positive - log_negative))
    loss = torch.sum(raw * weights) / torch.clamp(weights.sum(), min=1.0)
    return loss, {
        "preference_actor_loss": float(loss.detach().item()),
        "preference_logprob_gap": float(
            (log_positive - log_negative).mean().detach().item()
        ),
    }
