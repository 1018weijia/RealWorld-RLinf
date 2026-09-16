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


"""Differentiable joint-space trajectory contract for synchronous Cobot chunks.

Units are radians and seconds for the twelve arm joints. Grippers retain the
Stage1 command, bounded to the driver's [0, 1]; they are not residual outputs.
This limits commanded trajectories, not measured robot motion or collisions.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

VERSION = "cobot-joint-motion-v2"
ARM = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


class JointMotion(nn.Module):
    """Project a reference, then scale a smooth bounded edit into its slack.

    A single scale per joint and chunk preserves the edit's temporal shape.
    The base uses a configurable fraction of the velocity/acceleration limits
    to leave slack for learning. Both paths finish with zero command velocity.
    Context is [signed affine scale, affine offset, previous command], each 14D.
    """

    def __init__(self, config: dict, chunk_len: int = 30) -> None:
        super().__init__()
        self.config = dict(config)
        if self.config.get("version") != VERSION:
            raise ValueError("Unsupported Cobot motion contract")
        self.chunk_len = int(chunk_len)
        self.hz = float(config["control_hz"])
        self.reserve = float(config["base_limit_fraction"])
        self.knots = int(config["residual_knots"])
        if self.chunk_len < 4 or not 2 <= self.knots <= self.chunk_len - 1:
            raise ValueError("Cobot motion requires 2..chunk_length-1 residual knots")
        if not 0 < self.reserve < 1 or self.hz != 30:
            raise ValueError("Expected 30 Hz and 0 < base_limit_fraction < 1")
        for name in ("residual_rad", "velocity_rad_s", "acceleration_rad_s2"):
            value = torch.as_tensor(config[name], dtype=torch.float32)
            if (
                value.shape != (12,)
                or not torch.isfinite(value).all()
                or (value <= 0).any()
            ):
                raise ValueError(
                    f"{name} must contain twelve positive finite joint limits"
                )
            self.register_buffer(name, value)

    @staticmethod
    def context(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode the explicit physical transform; no assumptions about Aloha signs."""
        return x[..., -42:].split(14, dim=-1)

    def get_extra_state(self) -> dict:
        return self.config

    def set_extra_state(self, state: dict) -> None:
        if state != self.config:
            raise ValueError(
                "Checkpoint joint motion limits differ from configuration; retrain instead"
            )

    @staticmethod
    def differences(
        actions: torch.Tensor, anchor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Include the first command and acceleration from the preceding hold."""
        velocity = torch.diff(torch.cat((anchor[:, None], actions), dim=1), dim=1)
        acceleration = torch.diff(
            torch.cat((torch.zeros_like(velocity[:, :1]), velocity), dim=1), dim=1
        )
        return velocity, acceleration

    def base(self, reference: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return a feasible robot-space reference; zero residual reproduces this."""
        scale, offset, anchor = self.context(x)
        physical = (
            reference.reshape(-1, self.chunk_len, 14) * scale[:, None] + offset[:, None]
        )
        previous = anchor[:, ARM]
        velocity = torch.zeros_like(previous)
        vmax = self.velocity_rad_s / self.hz * self.reserve
        amax = self.acceleration_rad_s2 / self.hz**2 * self.reserve
        steps = []
        for t in range(self.chunk_len):
            # The remaining-step envelope guarantees braking to zero at the
            # last step while keeping the acceleration interval nonempty.
            cap = torch.minimum(vmax, (self.chunk_len - 1 - t) * amax)
            lower = torch.maximum(-cap, velocity - amax)
            upper = torch.minimum(cap, velocity + amax)
            velocity = torch.maximum(
                lower, torch.minimum(upper, physical[:, t, ARM] - previous)
            )
            previous = previous + velocity
            steps.append(previous)
        result = physical.clone()
        result[:, :, ARM] = torch.stack(steps, dim=1)
        result[:, :, [6, 13]] = physical[:, :, [6, 13]].clamp(0, 1)
        return result

    def edit(
        self, base: torch.Tensor, residual: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Scale a physical edit to satisfy all velocity and acceleration inequalities."""
        _, _, anchor = self.context(x)
        velocity, acceleration = self.differences(base[:, :, ARM], anchor[:, ARM])
        dv, da = self.differences(residual, torch.zeros_like(anchor[:, ARM]))
        factors = [torch.ones_like(residual[:, 0])]
        for current, delta, limit in (
            (velocity, dv, self.velocity_rad_s / self.hz),
            (acceleration, da, self.acceleration_rad_s2 / self.hz**2),
        ):
            # Solve -limit <= current + lambda * delta <= limit for lambda.
            slack = torch.where(delta >= 0, limit - current, limit + current)
            ratios = slack.clamp_min(0) / delta.abs().clamp_min(1e-12)
            factors.append(ratios.amin(dim=1))
        factor = torch.stack(factors).amin(dim=0).clamp(0, 1)
        result = base.clone()
        result[:, :, ARM] = base[:, :, ARM] + residual * factor[:, None]
        return result

    def residual(self, logits: torch.Tensor) -> torch.Tensor:
        """Interpolate bounded physical knots and duplicate the last edit for a hold."""
        knots = torch.tanh(logits.reshape(-1, self.knots, 12))
        smooth = F.interpolate(
            knots.transpose(1, 2),
            size=self.chunk_len - 1,
            mode="linear",
            align_corners=True,
        )
        smooth = torch.cat((smooth, smooth[:, :, -1:]), dim=-1).transpose(1, 2)
        # Ease into the edit from zero at the command boundary. Otherwise the
        # first-step acceleration bound would shrink the entire chunk's edit
        # almost to zero even for a constant, useful physical offset.
        phase = torch.linspace(
            0, 1, self.chunk_len - 1, device=logits.device, dtype=logits.dtype
        )
        envelope = phase.square() * (3 - 2 * phase)
        envelope = torch.cat((envelope, envelope[-1:]))
        return smooth * self.residual_rad * envelope[None, :, None]

    def normalized(self, physical: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Invert the calibrated, signed physical transform in float32."""
        scale, offset, _ = self.context(x)
        return ((physical - offset[:, None]) / scale[:, None]).flatten(1)

    def physical(self, normalized: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Decode an already projected reference without projecting it again."""
        scale, offset, _ = self.context(x)
        return (
            normalized.reshape(-1, self.chunk_len, 14) * scale[:, None]
            + offset[:, None]
        )

    def apply_to_base(
        self, base: torch.Tensor, logits: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Edit a feasible normalized base returned by project_reference."""
        return self.normalized(
            self.edit(self.physical(base, x), self.residual(logits), x), x
        )

    def project_reference(
        self, reference: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        return self.normalized(self.base(reference, x), x)

    def forward(
        self, reference: torch.Tensor, logits: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        return self.normalized(
            self.edit(self.base(reference, x), self.residual(logits), x), x
        )
