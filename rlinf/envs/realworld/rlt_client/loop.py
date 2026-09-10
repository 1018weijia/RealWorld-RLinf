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

"""The robot-side stop-and-go loop that drives an RLT Stage 2 server.

The client owns the episode: it holds the robot, the operator interface and
the clock, and treats the server as a stateless-looking ``act`` / ``transition``
service. That split is what lets a training crash leave the robot in a safe,
operator-controlled state rather than mid-chunk.

Every chunk follows the same four steps: observe while stationary, ask the
server for a chunk, run it, report what happened. The ``transition_id`` the
server issues at ``act`` is the only thing tying the two halves together, and
the loop guarantees each one is either committed exactly once or explicitly
discarded — including on the abort path, which is why the discard lives in a
``finally``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rlinf.envs.realworld.rlt_client.transport import (
    ChunkExecutionResult,
    ChunkIdentity,
    OperatorEvent,
    RLTRobotTransport,
)
from rlinf.serving.rlt.protocol import (
    ACTION_SPACE_ROBOT,
    REQUEST_ACT,
    REQUEST_DISCARD,
    REQUEST_EPISODE_END,
    REQUEST_KEY,
    REQUEST_REWIND_CREDIT,
    REQUEST_REWIND_EXIT,
    REQUEST_TRANSITION,
    validate_server_metadata,
)
from rlinf.serving.websocket_client import RLinfWebsocketClient

logger = logging.getLogger(__name__)

EVENT_SUCCESS = "success"
EVENT_FAILURE = "failure"
EVENT_REWIND_EXIT = "rewind_exit"
EVENT_REWIND_CREDIT = "rewind_credit"
EVENT_ABORT = "abort"
# There is deliberately no "discard" event. Operator decisions are polled
# after the chunk has been committed, so by the time one arrives the replay
# row already exists; undoing it is what rewind_credit is for.


@dataclass
class EpisodeOutcome:
    """What one episode produced, as reported back to the operator.

    Attributes:
        episode_id: Episode index within the session.
        chunks: Chunks executed.
        reward: Summed reward across every chunk.
        success: Whether the operator marked the episode successful.
        aborted: Whether the operator aborted the episode.
        interventions: Chunks with an operator override.
        rewinds: Rewind events applied.
        updates_run: Gradient steps the server ran at ``episode_end``.
        server_metrics: Metrics the server returned.
    """

    episode_id: int
    chunks: int = 0
    reward: float = 0.0
    success: bool = False
    aborted: bool = False
    interventions: int = 0
    rewinds: int = 0
    updates_run: int = 0
    server_metrics: dict[str, Any] = field(default_factory=dict)


class RLTRobotLoop:
    """Run episodes against an RLT Stage 2 server.

    Args:
        transport: Robot stack implementing :class:`RLTRobotTransport`.
        client: Connected WebSocket client.
        max_episode_chunks: Local chunk budget. The server reports its own in
            the handshake; the smaller of the two wins.
        exploration_noise_sigma: Per-request exploration override, or ``None``
            to let the server use its configured value.
        success_reward: Reward credited to the final chunk on operator success.
        failure_reward: Reward credited to the final chunk on operator failure.
        rewind_terminal_reward: Penalty written to the last chunk of a
            rewound branch. Must be negative to be meaningful.
        rewind_prefix_reward: Reward written before the bad branch on a
            credit-only rewind.
        validate_handshake: Check server metadata against the transport before
            the first move.
    """

    def __init__(
        self,
        *,
        transport: RLTRobotTransport,
        client: RLinfWebsocketClient,
        max_episode_chunks: int = 150,
        exploration_noise_sigma: float | None = None,
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
        rewind_terminal_reward: float = -1.0,
        rewind_prefix_reward: float = 0.0,
        validate_handshake: bool = True,
    ) -> None:
        self.transport = transport
        self.client = client
        self.exploration_noise_sigma = exploration_noise_sigma
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.rewind_terminal_reward = float(rewind_terminal_reward)
        self.rewind_prefix_reward = float(rewind_prefix_reward)

        metadata = dict(client.server_metadata or {})
        if validate_handshake:
            validate_server_metadata(
                metadata,
                action_dim=transport.action_dim,
                chunk_length=transport.chunk_len,
                proprio_dim=transport.proprio_dim,
                camera_keys=transport.camera_keys,
            )
        self.server_metadata = metadata
        server_budget = int(metadata.get("max_episode_chunks", max_episode_chunks))
        self.max_episode_chunks = min(int(max_episode_chunks), server_budget)
        if self.max_episode_chunks != int(max_episode_chunks):
            logger.warning(
                "Using server chunk budget %d instead of local %d",
                self.max_episode_chunks,
                max_episode_chunks,
            )

        self._episode_id = 0
        self._session_id = int(metadata.get("session_id", 0))

    # ------------------------------------------------------------ requests

    def _request(self, request_type: str, **payload: Any) -> dict[str, Any]:
        return self.client.request({REQUEST_KEY: request_type, **payload})

    # -------------------------------------------------------------- episode

    def run_episode(self) -> EpisodeOutcome:
        """Run one episode start to finish.

        Returns:
            What happened, including the server's training response.
        """
        outcome = EpisodeOutcome(episode_id=self._episode_id)
        observation = self.transport.reset()
        chunk_index = 0
        last_transition_id: str | None = None

        try:
            while chunk_index < self.max_episode_chunks:
                identity = ChunkIdentity(
                    episode_id=self._episode_id,
                    session_id=self._session_id,
                    env_id=0,
                    chunk_id=chunk_index + 1,
                )
                step = self._run_chunk(observation, identity, outcome)
                if step is None:
                    outcome.aborted = True
                    break
                observation, result, last_transition_id, event = step
                chunk_index += 1
                outcome.chunks = chunk_index

                if event is not None:
                    # Handled after the commit: a rewind patches rows that
                    # must already exist, and a success verdict has already
                    # been folded into the reward above.
                    resumed = self._handle_event(event, outcome, identity)
                    if resumed is None:
                        break
                    observation = resumed
                if result.done:
                    break
            else:
                logger.warning(
                    "Episode %d hit the %d-chunk budget",
                    self._episode_id,
                    self.max_episode_chunks,
                )
        except Exception:
            # The robot must stop before the exception reaches the operator's
            # terminal, not after.
            self.transport.stop("client_loop_exception")
            raise
        finally:
            self._finalize_episode(outcome, last_transition_id)

        return outcome

    def _run_chunk(
        self,
        observation,
        identity: ChunkIdentity,
        outcome: EpisodeOutcome,
    ) -> tuple[Any, ChunkExecutionResult, str, OperatorEvent | None] | None:
        """Request, execute and commit one chunk.

        Returns:
            ``(next_observation, result, transition_id, event)``, where
            ``event`` is a pending operator decision the caller must still act
            on, or ``None`` when the chunk was discarded and the episode should
            stop.
        """
        act_payload: dict[str, Any] = {
            "observation": observation.to_payload(),
            **_identity_payload(identity),
        }
        if self.exploration_noise_sigma is not None:
            act_payload["exploration_noise_sigma"] = self.exploration_noise_sigma
        response = self._request(REQUEST_ACT, **act_payload)

        transition_id = str(response["transition_id"])
        # np.array, not np.asarray: msgpack hands back a read-only view over
        # the receive buffer, and a controller that clips or filters the chunk
        # in place would raise on it.
        action_chunk = np.array(response["actions"], dtype=np.float32)
        committed = False
        try:
            started = time.monotonic()
            result = self.transport.execute_chunk(action_chunk, identity)
            logger.info(
                "chunk %d mode=%s steps=%d reward=%.3f in %.2fs",
                identity.chunk_id,
                response.get("mode", "?"),
                result.steps_executed,
                float(np.sum(result.rewards)),
                time.monotonic() - started,
            )
            if result.steps_executed <= 0:
                # Nothing ran, so there is no transition to learn from.
                return None

            # Poll before committing: a success or failure verdict belongs on
            # this chunk's reward, and once the row is written the only way to
            # change it is a credit correction.
            event = self.transport.poll_operator_event()
            rewards = self._apply_verdict(event, result)

            outcome.reward += float(np.sum(rewards))
            if result.intervention:
                outcome.interventions += 1

            transition_payload: dict[str, Any] = {
                "transition_id": transition_id,
                "next_observation": result.observation.to_payload(),
                "rewards": rewards,
                "done": result.done,
                "bootstrap_mask": result.bootstrap_mask,
                "intervention": result.intervention,
                "info": result.info,
                **_identity_payload(identity),
            }
            if result.executed_chunk is not None:
                transition_payload["action_chunk"] = np.asarray(
                    result.executed_chunk, dtype=np.float32
                )
                transition_payload["action_chunk_space"] = ACTION_SPACE_ROBOT
            self._request(REQUEST_TRANSITION, **transition_payload)
            committed = True
            return result.observation, result, transition_id, event
        finally:
            if not committed:
                # An exception or an empty chunk leaves the server holding a
                # pending entry whose curr_obs would otherwise never be freed.
                self._safe_discard(transition_id)

    def _apply_verdict(
        self, event: OperatorEvent | None, result: ChunkExecutionResult
    ) -> np.ndarray:
        """Fold a success or failure verdict into the chunk's reward.

        The operator judges the outcome only after watching the chunk finish,
        so the verdict has to be written onto the chunk that produced it. This
        also marks the result terminated, which drops the bootstrap value —
        a solved task has no future return to estimate.

        Args:
            event: Pending operator decision, if any.
            result: Chunk result, mutated in place when a verdict lands.

        Returns:
            The reward array to send, with the verdict applied to the last
            executed step.
        """
        rewards = np.asarray(result.rewards, dtype=np.float32).reshape(-1).copy()
        if event is None or event.kind not in (EVENT_SUCCESS, EVENT_FAILURE):
            return rewards

        terminal = (
            self.success_reward
            if event.kind == EVENT_SUCCESS
            else (self.failure_reward)
        )
        if event.terminal_reward:
            terminal = float(event.terminal_reward)
        if rewards.size == 0:
            rewards = np.zeros(1, dtype=np.float32)
        rewards[-1] += float(terminal)
        result.terminated = True
        return rewards

    def _safe_discard(self, transition_id: str) -> None:
        try:
            self._request(REQUEST_DISCARD, transition_id=transition_id)
        except Exception:  # noqa: BLE001 - never mask the original failure
            logger.exception("Failed to discard transition %s", transition_id)

    # -------------------------------------------------------------- events

    def _handle_event(
        self,
        event: OperatorEvent,
        outcome: EpisodeOutcome,
        identity: ChunkIdentity,
    ):
        """Apply one operator decision.

        Returns:
            The observation to continue from, or ``None`` to stop the episode.
        """
        if event.kind == EVENT_SUCCESS:
            outcome.success = True
            logger.info("Operator marked episode %d successful", self._episode_id)
            return self.transport.observe()
        if event.kind == EVENT_FAILURE:
            logger.info("Operator marked episode %d failed", self._episode_id)
            return self.transport.observe()
        if event.kind == EVENT_ABORT:
            outcome.aborted = True
            self.transport.stop("operator_abort")
            return None
        if event.kind in (EVENT_REWIND_EXIT, EVENT_REWIND_CREDIT):
            return self._handle_rewind(event, outcome, identity)
        logger.warning("Ignoring unknown operator event %r", event.kind)
        return self.transport.observe()

    def _handle_rewind(
        self,
        event: OperatorEvent,
        outcome: EpisodeOutcome,
        identity: ChunkIdentity,
    ):
        """Apply a rewind, moving the robot only for the physical variant."""
        chunks = max(1, int(event.chunks))
        terminal = event.terminal_reward or self.rewind_terminal_reward
        confidence = float(event.confidence)
        physical = event.kind == EVENT_REWIND_EXIT

        if physical:
            try:
                observation = self.transport.rewind_chunks(chunks)
            except NotImplementedError:
                logger.warning(
                    "Transport cannot rewind physically; downgrading to a "
                    "credit-only correction."
                )
                physical = False
                observation = self.transport.observe()
        else:
            observation = self.transport.observe()

        if physical:
            self._request(
                REQUEST_REWIND_EXIT,
                terminal_reward=terminal,
                chunks_rewound=chunks,
                preference_confidence=confidence,
                **_identity_payload(identity),
            )
        else:
            self._request(
                REQUEST_REWIND_CREDIT,
                terminal_reward=terminal,
                prefix_reward=event.prefix_reward or self.rewind_prefix_reward,
                bad_chunks=chunks,
                preference_confidence=confidence,
                **_identity_payload(identity),
            )
        outcome.rewinds += 1
        logger.info(
            "Applied %s rewind over %d chunks (terminal=%.2f)",
            "physical" if physical else "credit-only",
            chunks,
            terminal,
        )
        return observation

    # ---------------------------------------------------------- finalizing

    def _finalize_episode(
        self, outcome: EpisodeOutcome, last_transition_id: str | None
    ) -> None:
        """Close the episode on the server, whatever happened locally."""
        del last_transition_id  # Already committed or discarded by _run_chunk.
        try:
            response = self._request(
                REQUEST_EPISODE_END,
                stats={
                    "success": outcome.success,
                    "aborted": outcome.aborted,
                    "chunks": outcome.chunks,
                    "reward": outcome.reward,
                    "interventions": outcome.interventions,
                    "rewinds": outcome.rewinds,
                },
                **_identity_payload(
                    ChunkIdentity(
                        episode_id=self._episode_id,
                        session_id=self._session_id,
                    )
                ),
            )
            outcome.updates_run = int(response.get("updates_run", 0))
            outcome.server_metrics = dict(response.get("metrics", {}))
            logger.info(
                "Episode %d finished: chunks=%d reward=%.3f success=%s updates=%d",
                self._episode_id,
                outcome.chunks,
                outcome.reward,
                outcome.success,
                outcome.updates_run,
            )
        except Exception:  # noqa: BLE001 - a lost episode must not kill the run
            logger.exception("episode_end failed for episode %d", self._episode_id)
        self._episode_id += 1

    def run(self, num_episodes: int) -> list[EpisodeOutcome]:
        """Run episodes back to back, always stopping the robot at the end.

        Args:
            num_episodes: Episodes to run.

        Returns:
            One outcome per completed episode.
        """
        outcomes: list[EpisodeOutcome] = []
        try:
            for _ in range(num_episodes):
                outcomes.append(self.run_episode())
        finally:
            self.transport.stop("run_complete")
            self.transport.close()
            self.client.close()
        return outcomes


def _identity_payload(identity: ChunkIdentity) -> dict[str, int]:
    return {
        "episode_id": int(identity.episode_id),
        "session_id": int(identity.session_id),
        "env_id": int(identity.env_id),
        "chunk_id": int(identity.chunk_id),
    }
