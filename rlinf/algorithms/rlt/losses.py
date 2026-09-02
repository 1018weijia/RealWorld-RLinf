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
) -> tuple[Tensor, dict[str, float]]:
    """Twin-Q MSE with optional per-transition ``bootstrap_mask``."""
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

        if not use_crossq:
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

    if not use_crossq:
        all_data_q_values = model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=actions,
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
    critic_loss = F.mse_loss(
        all_data_q_values, target_q_values.expand_as(all_data_q_values)
    )
    metrics = {
        "q_data": all_data_q_values.mean().item(),
        "bootstrap_mask_mean": float(bootstrap_mask.mean().item()),
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

    ref_chunk = flatten_chunk(curr_obs["ref_chunk"]).reshape(
        curr_obs["ref_chunk"].shape[0], -1, action_dim
    )[:, :chunk_len].reshape(curr_obs["ref_chunk"].shape[0], -1)

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
