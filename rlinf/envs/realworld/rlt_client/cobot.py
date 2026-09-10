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

"""Cobot Magic transport for the RLT Stage 2 client loop.

Bridges the chunk-level contract the loop speaks to the step-level
:class:`~rlinf.envs.realworld.cobot.control.CobotControlAdapter` the Cobot
stack already provides. The adapter keeps ownership of everything
safety-critical — clipping, watchdogs, emergency stop, rewind playback — so
this class is only a shape adapter and an operator-event translator.

A Franka skeleton lives at the bottom of the file. Franka differs in its
control protocol, not in its data contract, so it implements the same
:class:`~rlinf.envs.realworld.rlt_client.transport.RLTRobotTransport`
interface and reuses the loop unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np

from rlinf.envs.realworld.cobot.control import (
    CobotControlAdapter,
    CobotObservation,
    CobotRewindEvent,
)
from rlinf.envs.realworld.rlt_client.loop import (
    EVENT_ABORT,
    EVENT_FAILURE,
    EVENT_REWIND_CREDIT,
    EVENT_REWIND_EXIT,
    EVENT_SUCCESS,
)
from rlinf.envs.realworld.rlt_client.transport import (
    ChunkExecutionResult,
    ChunkIdentity,
    OperatorEvent,
    RobotObservation,
)

logger = logging.getLogger(__name__)

COBOT_CAMERA_KEYS = ("image", "wrist_image", "side_image")
"""Cobot Magic camera keys, in the order the OpenPI Aloha transform expects.

``image`` is ``cam_high``; ``wrist_image`` and ``side_image`` become the left
and right wrist views. The server relies on this order, so changing it silently
swaps the two arms' viewpoints.
"""


class CobotTransport:
    """Drive a Cobot Magic arm pair one action chunk at a time.

    Args:
        adapter: Step-level control adapter owning the hardware.
        chunk_len: Steps per chunk, matching the server handshake.
        proprio_dim: Width of the proprioception vector.
        camera_keys: Camera keys the adapter publishes.
        rewind_history_chunks: Chunks the adapter can physically replay.
        stop_on_safety_fault: Abort the chunk as soon as the adapter reports
            ``rlt_safety_fault``.
    """

    def __init__(
        self,
        adapter: CobotControlAdapter,
        *,
        chunk_len: int,
        proprio_dim: int = 14,
        camera_keys: Sequence[str] = COBOT_CAMERA_KEYS,
        rewind_history_chunks: int = 12,
        stop_on_safety_fault: bool = True,
    ) -> None:
        self._adapter = adapter
        self._chunk_len = int(chunk_len)
        self._proprio_dim = int(proprio_dim)
        self._camera_keys = tuple(camera_keys)
        self._rewind_history_chunks = int(rewind_history_chunks)
        self._stop_on_safety_fault = bool(stop_on_safety_fault)
        self._pending_event: OperatorEvent | None = None

    # ------------------------------------------------------------- shapes

    @property
    def action_dim(self) -> int:
        """Action coordinates per step, taken from the adapter."""
        return int(self._adapter.action_dim)

    @property
    def chunk_len(self) -> int:
        """Steps per action chunk."""
        return self._chunk_len

    @property
    def proprio_dim(self) -> int:
        """Proprioception width."""
        return self._proprio_dim

    @property
    def camera_keys(self) -> tuple[str, ...]:
        """Camera keys published in each observation."""
        return self._camera_keys

    # -------------------------------------------------------- observation

    def _convert(self, observation: CobotObservation) -> RobotObservation:
        missing = [key for key in self._camera_keys if key not in observation.images]
        if missing:
            raise RuntimeError(
                f"Cobot adapter did not publish cameras {missing}; the server "
                f"expects {list(self._camera_keys)} and would otherwise see a "
                "black frame in that slot."
            )
        state = np.asarray(observation.state, dtype=np.float32).reshape(-1)
        if state.size != self._proprio_dim:
            raise RuntimeError(
                f"Cobot state has {state.size} values but the server was "
                f"configured for proprio_dim={self._proprio_dim}."
            )
        return RobotObservation(
            images={key: observation.images[key] for key in self._camera_keys},
            state=state,
            task=observation.task,
            timestamp=observation.timestamp,
        )

    def reset(self) -> RobotObservation:
        """Move the arms to the start pose and observe."""
        self._pending_event = None
        return self._convert(self._adapter.reset())

    def observe(self) -> RobotObservation:
        """Observe without moving."""
        return self._convert(self._adapter.observe())

    # ----------------------------------------------------------- chunking

    def execute_chunk(
        self, action_chunk: np.ndarray, identity: ChunkIdentity
    ) -> ChunkExecutionResult:
        """Play one chunk step by step, stopping early on a fault or takeover.

        Args:
            action_chunk: ``float32`` ``[chunk_len, action_dim]`` in robot units.
            identity: Where this chunk sits in the run.

        Returns:
            What actually ran, with the executed actions echoed back whenever
            the adapter reports a value different from what was commanded.
        """
        # Owned and writable: adapters routinely clip or filter the action in
        # place, and a chunk that came straight off the wire is read-only.
        chunk = np.array(action_chunk, dtype=np.float32).reshape(
            self._chunk_len, self.action_dim
        )
        begin = getattr(self._adapter, "on_action_chunk_begin", None)
        if callable(begin):
            begin()

        executed = np.array(chunk, dtype=np.float32, copy=True)
        rewards: list[float] = []
        terminated = False
        truncated = False
        intervention = False
        overridden = False
        info: dict[str, Any] = {}
        observation = None
        steps = 0

        committed = False
        try:
            for index in range(self._chunk_len):
                result = self._adapter.execute(chunk[index])
                steps = index + 1
                observation = result.observation
                rewards.append(float(result.reward))
                info = dict(result.info or {})

                actual = info.get("executed_action")
                if actual is not None:
                    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
                    if not np.array_equal(actual, chunk[index]):
                        executed[index] = actual
                        overridden = True
                if info.get("human_intervention") or info.get("rlt_human_takeover"):
                    intervention = True

                if result.terminated:
                    terminated = True
                if result.truncated:
                    truncated = True
                if self._stop_on_safety_fault and info.get("rlt_safety_fault"):
                    logger.warning(
                        "Safety fault during chunk %d at step %d; stopping.",
                        identity.chunk_id,
                        steps,
                    )
                    truncated = True
                    self._pending_event = OperatorEvent(
                        kind=EVENT_ABORT, note="safety_fault"
                    )
                if terminated or truncated or intervention:
                    break
            committed = steps > 0 and not truncated
        finally:
            end = getattr(self._adapter, "on_action_chunk_end", None)
            if callable(end):
                end(committed)

        if observation is None:
            observation = self._adapter.observe()

        if steps < self._chunk_len:
            # The chunk was cut short, so the tail still holds commanded
            # actions that nothing ever ran. Reporting them verbatim on an
            # intervention chunk would label them ACTION_SOURCE_HUMAN and feed
            # fabricated targets to the BC loss. Hold the last executed action
            # instead: that is physically what the arm did once it stopped.
            executed[steps:] = executed[steps - 1] if steps else chunk[0]

        return ChunkExecutionResult(
            observation=self._convert(observation),
            rewards=np.asarray(rewards, dtype=np.float32),
            steps_executed=steps,
            terminated=terminated,
            truncated=truncated,
            intervention=intervention,
            executed_chunk=executed if (overridden or intervention) else None,
            info=info,
        )

    # ------------------------------------------------------------- events

    def poll_operator_event(self) -> OperatorEvent | None:
        """Return one operator decision, preferring a locally raised abort."""
        if self._pending_event is not None:
            event, self._pending_event = self._pending_event, None
            return event
        poll = getattr(self._adapter, "poll_rewind_event", None)
        if not callable(poll):
            return None
        raw = poll()
        return None if raw is None else _to_operator_event(raw)

    def rewind_chunks(self, count: int) -> RobotObservation:
        """Reverse-play completed chunks on the hardware.

        Raises:
            NotImplementedError: The adapter has no physical rewind, so the
                loop should fall back to a credit-only correction.
        """
        rewind = getattr(self._adapter, "rewind_chunks", None)
        if not callable(rewind):
            raise NotImplementedError(
                f"{type(self._adapter).__name__} does not implement rewind_chunks"
            )
        capped = min(int(count), self._rewind_history_chunks)
        if capped < int(count):
            logger.warning(
                "Requested rewind of %d chunks but only %d are retained.",
                count,
                capped,
            )
        return self._convert(rewind(capped))

    def stop(self, reason: str) -> None:
        """Halt the arms immediately."""
        self._adapter.stop(reason)

    def close(self) -> None:
        """Release the hardware."""
        self._adapter.close()


def _to_operator_event(raw: CobotRewindEvent | OperatorEvent) -> OperatorEvent:
    """Translate a Cobot event into the transport-neutral form.

    ``CobotRewindEvent`` can only express rewinds, but a keyboard also produces
    success, failure and abort verdicts, so an adapter is allowed to emit an
    :class:`OperatorEvent` directly and it passes through untouched.
    """
    if isinstance(raw, OperatorEvent):
        return raw
    kind = EVENT_REWIND_EXIT if raw.mode == "exit" else EVENT_REWIND_CREDIT
    if getattr(raw, "rewind_exit_mode", "physical") == "credit_only":
        kind = EVENT_REWIND_CREDIT
    return OperatorEvent(
        kind=kind,
        chunks=int(raw.chunks_rewound),
        terminal_reward=float(raw.terminal_reward),
        prefix_reward=float(raw.prefix_reward),
        confidence=float(raw.confidence),
        note=getattr(raw, "safety_state", ""),
    )


def build_cobot_transport(
    cfg,
    *,
    controller_factory: Callable[..., CobotControlAdapter] | None = None,
) -> CobotTransport:
    """Build the transport described by a client config.

    Args:
        cfg: Client config with a ``transport`` section (``is_dummy``,
            ``action_dim``, ``chunk_len``, ``proprio_dim``, ``task``,
            ``rewind_history_chunks``).
        controller_factory: Builds the real adapter. Required whenever
            ``is_dummy`` is false.

    Returns:
        A configured transport.

    Raises:
        RuntimeError: A real run was requested with no controller factory.
            Silently substituting the mock would move nothing while filling
            replay with zeros, which is far worse than refusing to start.
    """
    transport_cfg = cfg.transport
    is_dummy = bool(transport_cfg.get("is_dummy", False))
    action_dim = int(transport_cfg.get("action_dim", 14))
    chunk_len = int(transport_cfg.chunk_len)

    if is_dummy:
        from rlinf.envs.realworld.cobot.control import MockRewindAdapter

        logger.warning(
            "transport.is_dummy=true: running against MockRewindAdapter. No "
            "robot will move and every camera frame is black."
        )
        adapter: CobotControlAdapter = MockRewindAdapter(
            action_dim=action_dim,
            task=str(transport_cfg.get("task", "")),
            history_size=int(transport_cfg.get("rewind_history_chunks", 12)),
        )
    elif controller_factory is None:
        raise RuntimeError(
            "transport.is_dummy=false but no controller_factory was supplied. "
            "Pass the Cobot controller factory, or set is_dummy=true to run "
            "the mock explicitly."
        )
    else:
        adapter = controller_factory(
            action_dim=action_dim,
            task=str(transport_cfg.get("task", "")),
        )

    return CobotTransport(
        adapter,
        chunk_len=chunk_len,
        proprio_dim=int(transport_cfg.get("proprio_dim", 14)),
        camera_keys=tuple(transport_cfg.get("camera_keys", COBOT_CAMERA_KEYS)),
        rewind_history_chunks=int(transport_cfg.get("rewind_history_chunks", 12)),
        stop_on_safety_fault=bool(transport_cfg.get("stop_on_safety_fault", True)),
    )


class FrankaTransport:
    """Skeleton transport for a remote Franka arm.

    Franka shares the RLT data contract with Cobot but not its control
    protocol: chunks are streamed to a separate real-time control process
    rather than stepped locally, and rewind is driven by that process. Filling
    in the methods below is enough to reuse
    :class:`~rlinf.envs.realworld.rlt_client.loop.RLTRobotLoop` unchanged.

    Args:
        chunk_len: Steps per chunk.
        action_dim: Action coordinates per step.
        proprio_dim: Proprioception width.
        camera_keys: Camera keys the Franka stack publishes.
    """

    def __init__(
        self,
        *,
        chunk_len: int,
        action_dim: int = 8,
        proprio_dim: int = 8,
        camera_keys: Sequence[str] = ("image", "wrist_image"),
    ) -> None:
        self._chunk_len = int(chunk_len)
        self._action_dim = int(action_dim)
        self._proprio_dim = int(proprio_dim)
        self._camera_keys = tuple(camera_keys)

    @property
    def action_dim(self) -> int:
        """Action coordinates per step."""
        return self._action_dim

    @property
    def chunk_len(self) -> int:
        """Steps per action chunk."""
        return self._chunk_len

    @property
    def proprio_dim(self) -> int:
        """Proprioception width."""
        return self._proprio_dim

    @property
    def camera_keys(self) -> tuple[str, ...]:
        """Camera keys published in each observation."""
        return self._camera_keys

    def reset(self) -> RobotObservation:
        """Move to the start pose and observe."""
        raise NotImplementedError(_FRANKA_TODO.format(method="reset"))

    def observe(self) -> RobotObservation:
        """Observe without moving."""
        raise NotImplementedError(_FRANKA_TODO.format(method="observe"))

    def execute_chunk(
        self, action_chunk: np.ndarray, identity: ChunkIdentity
    ) -> ChunkExecutionResult:
        """Stream one chunk to the Franka control process."""
        raise NotImplementedError(_FRANKA_TODO.format(method="execute_chunk"))

    def poll_operator_event(self) -> OperatorEvent | None:
        """Read one operator decision from the Franka teleoperation stack."""
        raise NotImplementedError(_FRANKA_TODO.format(method="poll_operator_event"))

    def rewind_chunks(self, count: int) -> RobotObservation:
        """Reverse-play chunks through the Franka control process."""
        raise NotImplementedError(_FRANKA_TODO.format(method="rewind_chunks"))

    def stop(self, reason: str) -> None:
        """Halt the arm immediately."""
        raise NotImplementedError(_FRANKA_TODO.format(method="stop"))

    def close(self) -> None:
        """Release the connection."""
        raise NotImplementedError(_FRANKA_TODO.format(method="close"))


_FRANKA_TODO = (
    "FrankaTransport.{method} is a skeleton. TODO(agent): wire it to the "
    "Franka real-time control process; the RLT loop and server need no changes."
)


_EVENT_KINDS = (EVENT_SUCCESS, EVENT_FAILURE, EVENT_REWIND_EXIT, EVENT_REWIND_CREDIT)
"""Operator event kinds a Cobot keyboard mapping is expected to produce."""
