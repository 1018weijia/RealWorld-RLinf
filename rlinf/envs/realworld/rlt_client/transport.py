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

"""Robot-side data contract for the RLT Stage 2 client loop.

This module deliberately fixes only *what data crosses the boundary*, never
*how the robot is driven*. Cobot and Franka differ in nearly every control
detail — chunk playback rate, gripper encoding, teleoperation hardware, how a
physical rewind is performed — so a transport is free to implement
:meth:`RLTRobotTransport.execute_chunk` however its stack requires, as long as
it reports back the shapes described here.

The shapes the server depends on:

- images are ``uint8`` ``[H, W, 3]`` in RGB, one entry per camera key
- state is ``float32`` ``[proprio_dim]`` in robot units
- an executed chunk is ``float32`` ``[chunk_len, action_dim]`` in the same
  space the server handed out, or ``None`` when nothing was overridden
- rewards are ``float32`` ``[steps_executed]``, which may be shorter than the
  chunk when an operator cut it short
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ChunkIdentity:
    """Where a chunk sits in the run, used for replay indexing and rewind.

    Attributes:
        episode_id: Episode index within the session.
        session_id: Unique per server run, so a restart cannot collide with
            rows written before it.
        env_id: Robot index, ``0`` for a single-arm setup.
        chunk_id: Chunk index within the episode, starting at 1.
    """

    episode_id: int = 0
    session_id: int = 0
    env_id: int = 0
    chunk_id: int = 0


@dataclass(frozen=True)
class RobotObservation:
    """One observation captured at a chunk boundary, while the robot is still.

    Attributes:
        images: Camera key to ``uint8`` ``[H, W, 3]`` RGB frame. Keys must
            match the server's ``camera_keys`` handshake field.
        state: ``float32`` ``[proprio_dim]`` proprioception in robot units.
        task: Language instruction for the frozen Stage 1 VLA.
        timestamp: Capture time, for latency diagnostics only.
    """

    images: dict[str, np.ndarray]
    state: np.ndarray
    task: str
    timestamp: float | None = None

    def to_payload(self) -> dict[str, Any]:
        """Flatten into the ``observation`` mapping the server expects."""
        payload: dict[str, Any] = {
            key: np.ascontiguousarray(value) for key, value in self.images.items()
        }
        payload["state"] = np.asarray(self.state, dtype=np.float32)
        payload["prompt"] = self.task
        return payload


@dataclass
class ChunkExecutionResult:
    """Outcome of running one action chunk on the robot.

    Attributes:
        observation: Observation captured after the chunk finished.
        rewards: ``float32`` ``[steps_executed]``. Shorter than the chunk when
            an operator interrupted it; the server zero-pads the tail.
        steps_executed: Steps actually run, ``<= chunk_len``.
        terminated: Task reached a terminal state, so no bootstrap value.
        truncated: Stopped for a reason unrelated to the task (budget, abort),
            which still bootstraps.
        intervention: An operator overrode the commanded actions.
        executed_chunk: ``float32`` ``[chunk_len, action_dim]`` actually sent to
            the robot, or ``None`` when the server's chunk ran unmodified.
        info: Free-form diagnostics forwarded to the server unchanged.
    """

    observation: RobotObservation
    rewards: np.ndarray
    steps_executed: int
    terminated: bool = False
    truncated: bool = False
    intervention: bool = False
    executed_chunk: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Whether the episode ended, for either reason."""
        return bool(self.terminated or self.truncated)

    @property
    def bootstrap_mask(self) -> float:
        """``0`` on a true terminal, ``1`` otherwise.

        Truncation is not a terminal state: the value function should still
        bootstrap through it, or every timeout would be learned as failure.
        """
        return 0.0 if self.terminated else 1.0


@dataclass(frozen=True)
class OperatorEvent:
    """A decision the human operator made between chunks.

    Attributes:
        kind: ``"success"``, ``"failure"``, ``"rewind_exit"``,
            ``"rewind_credit"``, ``"discard"`` or ``"abort"``.
        chunks: Chunks the decision applies to, for the rewind kinds.
        terminal_reward: Reward written to the last bad chunk.
        prefix_reward: Reward written just before the bad branch, credit-only.
        confidence: Weight of the preference pair the rewind produces.
        note: Free-form operator annotation.
    """

    kind: str
    chunks: int = 0
    terminal_reward: float = 0.0
    prefix_reward: float = 0.0
    confidence: float = 1.0
    note: str = ""


@runtime_checkable
class RLTRobotTransport(Protocol):
    """Everything the RLT client loop needs from a robot stack.

    Implementations own all safety-critical behaviour locally: clipping,
    watchdogs, emergency stop and rewind playback must keep working when the
    learner is unreachable.
    """

    @property
    def action_dim(self) -> int:
        """Action coordinates per step, matching the server handshake."""

    @property
    def chunk_len(self) -> int:
        """Steps per action chunk, matching the server handshake."""

    @property
    def proprio_dim(self) -> int:
        """Proprioception width of :attr:`RobotObservation.state`."""

    @property
    def camera_keys(self) -> tuple[str, ...]:
        """Camera keys published in :attr:`RobotObservation.images`."""

    def reset(self) -> RobotObservation:
        """Bring the robot to a start pose and return the first observation."""

    def observe(self) -> RobotObservation:
        """Capture an observation without moving."""

    def execute_chunk(
        self, action_chunk: np.ndarray, identity: ChunkIdentity
    ) -> ChunkExecutionResult:
        """Run one chunk to completion or until an operator interrupts it.

        Args:
            action_chunk: ``float32`` ``[chunk_len, action_dim]`` in robot units.
            identity: Where this chunk sits in the run.

        Returns:
            What actually happened, including any operator override.
        """

    def poll_operator_event(self) -> OperatorEvent | None:
        """Return one pending operator decision, consuming it.

        Returns:
            The next event, or ``None`` when the operator has done nothing.
        """

    def rewind_chunks(self, count: int) -> RobotObservation:
        """Physically reverse at most ``count`` chunks and observe.

        Transports without physical rewind should raise
        :class:`NotImplementedError`; the loop then falls back to credit-only
        rewind, which needs no robot motion.
        """

    def stop(self, reason: str) -> None:
        """Halt motion immediately and record why."""

    def close(self) -> None:
        """Release hardware resources."""
