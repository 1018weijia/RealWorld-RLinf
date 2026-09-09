# Copyright 2026 The RLinf Authors.

"""Bounded, checkpointable fork-action preferences for RLT rewind."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _cpu_clone(value: Tensor) -> Tensor:
    return value.detach().cpu().clone().contiguous()


@dataclass(frozen=True)
class RewindPreference:
    """A human replacement action preferred to a bad fork action."""

    curr_obs: dict[str, Tensor]
    ref_chunk: Tensor
    positive_action: Tensor
    negative_action: Tensor
    action_mask: Tensor
    confidence: Tensor
    session_key: tuple[int, int, int] | None = None


class RewindPreferenceBuffer:
    """FIFO buffer that stores only fork-chunk rewind comparisons."""

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items: deque[RewindPreference] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        *,
        curr_obs: dict[str, Tensor],
        ref_chunk: Tensor,
        positive_action: Tensor,
        negative_action: Tensor,
        action_mask: Tensor | None = None,
        confidence: float | Tensor = 1.0,
        session_key: tuple[int, int, int] | None = None,
    ) -> None:
        if positive_action.shape != negative_action.shape:
            raise ValueError("positive_action and negative_action must share shape")
        mask = (
            torch.ones_like(positive_action, dtype=torch.bool)
            if action_mask is None
            else action_mask.bool()
        )
        if mask.shape != positive_action.shape:
            raise ValueError("action_mask must match action chunks")
        confidence_tensor = torch.as_tensor(confidence, dtype=torch.float32).reshape(1)
        if not 0.0 < float(confidence_tensor.item()) <= 1.0:
            raise ValueError("confidence must be in (0, 1]")
        self._items.append(
            RewindPreference(
                curr_obs={key: _cpu_clone(value) for key, value in curr_obs.items()},
                ref_chunk=_cpu_clone(ref_chunk),
                positive_action=_cpu_clone(positive_action),
                negative_action=_cpu_clone(negative_action),
                action_mask=_cpu_clone(mask),
                confidence=_cpu_clone(confidence_tensor),
                session_key=session_key,
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Any] | None:
        if not self._items:
            return None
        count = min(int(batch_size), len(self._items))
        indices = torch.randperm(len(self._items))[:count].tolist()
        items = [self._items[index] for index in indices]
        return {
            "curr_obs": {
                key: torch.stack([item.curr_obs[key] for item in items]).to(device)
                for key in items[0].curr_obs
            },
            "ref_chunk": torch.stack([item.ref_chunk for item in items]).to(device),
            "positive_action": torch.stack([item.positive_action for item in items]).to(
                device
            ),
            "negative_action": torch.stack([item.negative_action for item in items]).to(
                device
            ),
            "action_mask": torch.stack([item.action_mask for item in items]).to(device),
            "confidence": torch.cat([item.confidence for item in items]).to(device),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "items": list(self._items)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self._items = deque(state.get("items", []), maxlen=self.capacity)
