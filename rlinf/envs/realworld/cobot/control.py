# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Protocols for connecting a real Cobot controller to RLinf."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class CobotObservation:
    """Observation captured at a chunk boundary."""

    images: dict[str, np.ndarray]
    state: np.ndarray
    task: str
    timestamp: float | None = None


@dataclass(frozen=True)
class CobotStepResult:
    """Result of executing one normalized low-level action."""

    observation: CobotObservation
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class CobotControlAdapter(Protocol):
    """Local robot/control contract required by a real-world environment.

    Implementations must perform validation, clipping, watchdog handling, and
    emergency-stop locally. The learner may be disconnected without making
    those safety operations unavailable.
    """

    @property
    def action_dim(self) -> int:
        """Number of normalized action coordinates accepted by ``execute``."""

    def reset(self) -> CobotObservation:
        """Reset or safely prepare the robot and return the initial observation."""

    def observe(self) -> CobotObservation:
        """Capture an observation without executing an action."""

    def execute(self, action: Sequence[float]) -> CobotStepResult:
        """Validate and execute one normalized low-level action.

        The adapter owns normalized-to-robot conversion, clipping, timing, and
        local safety. ``executed_action`` in the result must be the normalized
        action actually sent to the robot.
        """

    def stop(self, reason: str) -> None:
        """Stop motion immediately and record the local stop reason."""

    def close(self) -> None:
        """Release hardware resources."""


@dataclass(frozen=True)
class CobotRewindEvent:
    """A local rewind decision emitted at an action-chunk boundary."""

    mode: Literal["exit", "credit"]
    chunks_rewound: int
    terminal_reward: float
    prefix_reward: float = 0.0
    confidence: float = 1.0
    episode_id: int = 0
    session_id: int = 0
    env_id: int = 0
    chunk_id: int = 0
    rewind_exit_mode: str = "physical"
    safety_state: str = "ok"

    def __post_init__(self) -> None:
        if self.chunks_rewound < 0:
            raise ValueError("chunks_rewound must be non-negative")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must be in (0, 1]")
        if self.mode not in ("exit", "credit"):
            raise ValueError("mode must be 'exit' or 'credit'")
        if self.rewind_exit_mode not in ("physical", "credit_only"):
            raise ValueError("unsupported rewind_exit_mode")


class RewindableCobotControlAdapter(CobotControlAdapter, Protocol):
    """Cobot adapter extension for local, chunk-aligned physical rewind."""

    def rewind_chunks(self, count: int) -> CobotObservation:
        """Reverse-play at most ``count`` completed chunks and return observation."""

    def poll_rewind_event(self) -> CobotRewindEvent | None:
        """Return a completed operator rewind decision once, if one exists."""


class MockRewindAdapter:
    """Deterministic local rewind adapter for tests and integration smoke tests."""

    def __init__(
        self,
        action_dim: int,
        task: str,
        history_size: int = 12,
        session_id: int | None = None,
    ):
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self.action_dim = int(action_dim)
        self._task = task
        self._history_size = int(history_size)
        self._state = np.zeros(self.action_dim, dtype=np.float32)
        self._history: list[np.ndarray] = []
        self._pending_event: CobotRewindEvent | None = None
        self._stopped = False
        self._episode_id = 0
        self._session_id = int(time.time_ns() if session_id is None else session_id)
        self._chunk_id = 0
        self._chunk_start_state: np.ndarray | None = None
        self.execute_calls = 0

    def _observation(self) -> CobotObservation:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        return CobotObservation(
            images={"image": image, "wrist_image": image, "side_image": image},
            state=self._state.copy(),
            task=self._task,
        )

    def reset(self) -> CobotObservation:
        self._state.fill(0.0)
        self._history.clear()
        self._pending_event = None
        self._stopped = False
        self._episode_id += 1
        self._chunk_id = 0
        self._chunk_start_state = None
        self.execute_calls = 0
        return self._observation()

    def observe(self) -> CobotObservation:
        return self._observation()

    def execute(self, action: Sequence[float]) -> CobotStepResult:
        if self._stopped:
            return CobotStepResult(
                self._observation(),
                0.0,
                terminated=True,
                info={
                    "executed_action": self._state.copy(),
                    "record_transition": False,
                    "rlt_safety_fault": True,
                    "rlt_episode_id": self._episode_id,
                    "rlt_session_id": self._session_id,
                    "rlt_env_id": 0,
                    "rlt_chunk_id": self._chunk_id,
                },
            )
        implicit_chunk = self._chunk_start_state is None
        if implicit_chunk:
            self.on_action_chunk_begin()
        self.execute_calls += 1
        self._state = validate_action(action, self.action_dim)
        result = CobotStepResult(
            self._observation(),
            0.0,
            info={
                "executed_action": self._state.copy(),
                "rlt_episode_id": self._episode_id,
                "rlt_session_id": self._session_id,
                "rlt_env_id": 0,
                "rlt_chunk_id": self._chunk_id,
            },
        )
        if implicit_chunk:
            self.on_action_chunk_end(True)
        return result

    def on_action_chunk_begin(self) -> None:
        self._chunk_start_state = self._state.copy()

    def on_action_chunk_end(self, committed: bool) -> None:
        if committed and self._chunk_start_state is not None:
            self._history.append(self._chunk_start_state)
            self._history = self._history[-self._history_size :]
            self._chunk_id += 1
        self._chunk_start_state = None

    def rewind_chunks(self, count: int) -> CobotObservation:
        if count < 0:
            raise ValueError("count must be non-negative")
        actual = min(int(count), len(self._history))
        if actual:
            self._state = self._history[-actual].copy()
            del self._history[-actual:]
        return self._observation()

    def request_rewind_exit(
        self, *, chunks_rewound: int, terminal_reward: float, confidence: float = 1.0
    ) -> None:
        self._pending_event = CobotRewindEvent(
            mode="exit",
            chunks_rewound=chunks_rewound,
            terminal_reward=terminal_reward,
            confidence=confidence,
            episode_id=self._episode_id,
            session_id=self._session_id,
            chunk_id=self._chunk_id - 1,
        )

    def request_rewind_credit(
        self,
        *,
        bad_chunks: int,
        terminal_reward: float,
        prefix_reward: float,
        confidence: float = 1.0,
    ) -> None:
        self._pending_event = CobotRewindEvent(
            mode="credit",
            chunks_rewound=bad_chunks,
            terminal_reward=terminal_reward,
            prefix_reward=prefix_reward,
            confidence=confidence,
            episode_id=self._episode_id,
            session_id=self._session_id,
            chunk_id=self._chunk_id - 1,
            rewind_exit_mode="credit_only",
        )

    def poll_rewind_event(self) -> CobotRewindEvent | None:
        event, self._pending_event = self._pending_event, None
        return event

    def stop(self, reason: str) -> None:
        del reason
        self._stopped = True

    def close(self) -> None:
        self._stopped = True

    def state_dict(self) -> dict[str, Any]:
        """Return identity/history state for controller-side persistence tests."""

        return {
            "action_dim": self.action_dim,
            "state": self._state.copy(),
            "history": [value.copy() for value in self._history],
            "pending_event": self._pending_event,
            "stopped": self._stopped,
            "episode_id": self._episode_id,
            "session_id": self._session_id,
            "chunk_id": self._chunk_id,
            "execute_calls": self.execute_calls,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore controller identity without reusing committed chunk IDs."""

        if int(state["action_dim"]) != self.action_dim:
            raise ValueError("Cannot restore Cobot state with a different action_dim")
        self._state = validate_action(state["state"], self.action_dim)
        self._history = [
            validate_action(value, self.action_dim)
            for value in state.get("history", [])
        ][-self._history_size :]
        self._pending_event = state.get("pending_event")
        self._stopped = bool(state.get("stopped", False))
        self._episode_id = int(state.get("episode_id", 0))
        self._session_id = int(state["session_id"])
        self._chunk_id = int(state.get("chunk_id", 0))
        self.execute_calls = int(state.get("execute_calls", 0))
        self._chunk_start_state = None


def validate_action(action: Sequence[float], action_dim: int) -> np.ndarray:
    """Validate a normalized action before passing it to a controller."""

    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.size != action_dim:
        raise ValueError(f"Expected action_dim={action_dim}, got {value.size}.")
    if not np.isfinite(value).all():
        raise ValueError("Cobot actions must contain only finite values.")
    if np.any(value < -1.0) or np.any(value > 1.0):
        raise ValueError("Cobot actions must be normalized to [-1, 1].")
    return value
