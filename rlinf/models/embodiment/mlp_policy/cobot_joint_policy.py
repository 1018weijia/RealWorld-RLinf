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


"""Cobot-specific residual actor; the original Franka TD3 path is unchanged."""

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rlinf.models.embodiment.mlp_policy.cobot_joint_motion import ARM, JointMotion
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import (
    RLTTD3MLPPolicy,
    _make_td3_mlp,
)


class JointResidualActor(nn.Module):
    """Predict six bounded knots per joint instead of thirty independent edits."""

    def __init__(
        self,
        state_dim: int,
        chunk_len: int,
        config: dict,
        hidden_dim: int,
        layers: int,
        sigma: float,
    ) -> None:
        super().__init__()
        self.motion = JointMotion(config, chunk_len)
        self.sigma = float(sigma)
        # Legacy metadata only; joint_motion contains the actual physical budgets.
        self.edit_scale = 0.0
        self.mlp = _make_td3_mlp(
            input_dim=state_dim + chunk_len * 14,
            output_dim=self.motion.knots * 12,
            hidden_dim=hidden_dim,
            num_hidden_layers=layers,
        )
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        x: torch.Tensor,
        a_tilde: torch.Tensor,
        *,
        deterministic: bool = False,
        apply_action_noise: bool | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if kwargs.get("apply_ref_dropout") and (kwargs.get("ref_dropout") or 0) > 0:
            raise ValueError(
                "Joint motion v2 requires the physical reference; dropout is unsupported"
            )
        # Every caller supplies _get_ref_chunk/_get_ref_candidates, which project
        # once. Repeating the causal scan per Cal-QL proposal dominates GPU time.
        base = a_tilde
        logits = self.mlp(torch.cat((x, base), dim=-1))
        if apply_action_noise is None:
            apply_action_noise = not deterministic
        if apply_action_noise and self.sigma > 0:
            logits = logits + torch.randn_like(logits) * self.sigma
        return self.motion.apply_to_base(base, logits, x)

    def mean(self, x: torch.Tensor, a_tilde: torch.Tensor) -> torch.Tensor:
        return self.forward(x, a_tilde, deterministic=True, apply_action_noise=False)


class CobotJointTD3Policy(RLTTD3MLPPolicy):
    """The transform context is visible to both actor and critic and saved in replay."""

    ACTOR_SEMANTIC_VERSION = 3

    def __init__(self, *, joint_motion: dict, **kwargs: Any) -> None:
        if kwargs["action_dim"] != 14 or kwargs["proprio_dim"] != 14:
            raise ValueError("Cobot motion v2 requires physical left/right joint14")
        if kwargs.get("ref_action_dropout", 0) != 0:
            raise ValueError("Cobot motion v2 requires ref_action_dropout=0")
        z_dim = int(kwargs["z_dim"])
        super().__init__(**{**kwargs, "z_dim": z_dim + 42})
        self.z_dim = z_dim
        self.actor = JointResidualActor(
            self.state_dim,
            self.chunk_len,
            joint_motion,
            kwargs.get("mlp_hidden_dim", 256),
            kwargs.get("mlp_num_hidden_layers", 2),
            kwargs.get("actor_noise_sigma", 0.025),
        )

    @property
    def joint_motion(self) -> JointMotion:
        return self.actor.motion

    def _state(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        context = obs["motion_context"].reshape(-1, 42)
        return torch.cat((super()._state(obs), context), dim=-1)

    def _get_ref_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.joint_motion.project_reference(
            super()._get_ref_chunk(obs), self._state(obs)
        )

    def _get_ref_candidates(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        raw = super()._get_ref_candidates(obs)
        state = self._state(obs)[:, None].expand(-1, raw.shape[1], -1)
        return self.joint_motion.project_reference(
            raw.flatten(0, 1), state.flatten(0, 1)
        ).reshape(raw.shape)

    def demo_target(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        *,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reachable BC target only; the critic keeps the unmodified recorded action.

        Use 80% of the physical edit budget to avoid training tanh to infinity.
        Down/up-sampling matches the actor knot parameterization. This operation
        never changes observations, rewards or stored demonstrations.
        """
        state = self._state(obs)
        motion = self.joint_motion
        base = motion.physical(
            self._get_ref_chunk(obs) if reference is None else reference, state
        )
        scale, offset, _ = motion.context(state)
        physical = (
            actions.reshape(-1, self.chunk_len, 14) * scale[:, None] + offset[:, None]
        )
        edit = (physical[:, :, ARM] - base[:, :, ARM]).clamp(
            -0.8 * motion.residual_rad, 0.8 * motion.residual_rad
        )
        knots = F.interpolate(
            edit.transpose(1, 2), size=motion.knots, mode="linear", align_corners=True
        )
        ratio = (knots.transpose(1, 2) / motion.residual_rad).clamp(-0.8, 0.8)
        return motion.normalized(
            motion.edit(base, motion.residual(torch.atanh(ratio)), state), state
        )

    def local_random_candidates(
        self,
        obs: dict[str, torch.Tensor],
        count: int,
        *,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cal-QL proposals obey the same joint contract as executable actions."""
        state = self._state(obs)
        reference = self._get_ref_chunk(obs) if reference is None else reference
        repeated_state = state.repeat_interleave(count, dim=0)
        repeated_ref = reference.repeat_interleave(count, dim=0)
        logits = torch.randn(
            len(reference) * count,
            self.joint_motion.knots * 12,
            device=reference.device,
        )
        return self.joint_motion.apply_to_base(
            repeated_ref, logits, repeated_state
        ).reshape(len(reference), count, -1)
