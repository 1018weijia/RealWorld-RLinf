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

"""Request router for the RLT Stage 2 online-RL server.

One lock serializes every request, so inference, replay writes and training
bursts never interleave.  That is not a throughput compromise: the robot loop
is stop-and-go by construction (the arm is stationary while the server thinks),
and a single robot cannot issue concurrent requests.

Correctness rests on the ``transition_id`` pending map.  ``act`` mints an id and
parks the observation, reference chunk and mode; ``transition`` pops it and
writes exactly one replay row; ``discard`` pops it and writes nothing.  A
prefetched-but-unused chunk therefore cannot leak into replay, and a resent
``transition`` cannot double-count, which is what the Ray path needed
trajectory-level deduplication for.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from rlinf.algorithms.rlt.learner import RLTRewindEvent
from rlinf.algorithms.rlt.transition import (
    ACTION_SOURCE_HUMAN,
    ACTION_SOURCE_POLICY,
    ACTION_SOURCE_VLA,
)
from rlinf.serving.rlt.inference import RLTStage2Inference
from rlinf.serving.rlt.protocol import (
    ACTION_SPACE_NORMALIZED,
    ACTION_SPACE_ROBOT,
    REQUEST_ACT,
    REQUEST_DISCARD,
    REQUEST_EPISODE_END,
    REQUEST_RESET,
    REQUEST_REWIND_CREDIT,
    REQUEST_REWIND_EXIT,
    REQUEST_STATUS,
    REQUEST_TRANSITION,
    ActRequest,
    ChunkIdentity,
    DiscardRequest,
    EpisodeEndRequest,
    ProtocolError,
    RewindCreditRequest,
    RewindExitRequest,
    ServerMetadata,
    TransitionRequest,
    decode_request_type,
)
from rlinf.serving.rlt.trainer import RLTStage2Trainer

logger = logging.getLogger(__name__)


@dataclass
class PendingTransition:
    """Everything ``transition`` needs that only ``act`` knows.

    Attributes:
        transition_id: Server-issued handle.
        identity: Chunk identity used for replay indexing and rewind patching.
        rlt_obs: Stage 2 observation captured before the chunk ran.
        normalized_chunk: Chunk handed out, in normalized space.
        robot_chunk: Same chunk in robot units, i.e. what the client executed
            absent an intervention.
        mode: ``"warmup"``, ``"actor"`` or ``"eval"``.
    """

    transition_id: str
    identity: ChunkIdentity
    rlt_obs: dict[str, Any]
    normalized_chunk: np.ndarray
    robot_chunk: np.ndarray
    mode: str


@dataclass
class EpisodeAccumulator:
    """Per-episode counters reported back to the client at ``episode_end``."""

    chunks: int = 0
    train_chunks: int = 0
    reward: float = 0.0
    interventions: int = 0

    def reset(self) -> None:
        """Zero every counter for the next episode."""
        self.chunks = 0
        self.train_chunks = 0
        self.reward = 0.0
        self.interventions = 0


class Stage1EvaluationState:
    """Read-only protocol counters without a Stage2 model, optimizer or replay."""

    def __init__(self, chunk_length: int, action_dim: int) -> None:
        self.replay_buffer = SimpleNamespace(total_samples=0)
        self.demo_buffer = None
        self.rewind_preference_buffer = ()
        self.update_step = 0
        self.chunk_length = int(chunk_length)
        self.action_dim = int(action_dim)

    def _chunk_shape(self) -> tuple[int, int]:
        return self.chunk_length, self.action_dim


class RLTStage2Policy:
    """Expose an :class:`RLTStage2Trainer` over the RLT WebSocket protocol.

    Args:
        trainer: Owns the Stage 2 model, buffers and update loop.
        inference: Stage 1 encode plus Stage 2 action selection.
        metadata: Handshake payload pushed to every connecting client.
        warmup_steps: Replay rows required before the Stage 2 head is allowed
            to drive the robot.
        utd_ratio: Updates run per policy-driven chunk at ``episode_end``.
        max_episode_chunks: Chunk budget per episode, reported to the client so
            both sides agree on the timeout.
        save_dir: Directory for periodic checkpoints, or ``None`` to disable.
        save_interval_episodes: Episodes between checkpoints.
        eval_only: Never train, never write replay; evaluate the loaded policy.
        store_eval_episodes: When true, in-training eval episodes still write
            replay (``eval_only`` stays a whole-run mute).
        eval_interval_episodes: Schedule a deterministic eval episode after
            every N completed training episodes. ``0`` disables.
        replay_action_space: ``"normalized"`` or ``"robot"``.
        metric_logger: Optional callable receiving each metrics dict.
    """

    def __init__(
        self,
        *,
        trainer: RLTStage2Trainer | Stage1EvaluationState,
        inference: RLTStage2Inference,
        metadata: ServerMetadata,
        warmup_steps: int,
        utd_ratio: int,
        max_episode_chunks: int,
        save_dir: str | None = None,
        save_interval_episodes: int = 10,
        eval_only: bool = False,
        store_eval_episodes: bool = False,
        eval_interval_episodes: int = 0,
        replay_action_space: str = ACTION_SPACE_NORMALIZED,
        metric_logger=None,
    ) -> None:
        if replay_action_space not in (ACTION_SPACE_NORMALIZED, ACTION_SPACE_ROBOT):
            raise ValueError(f"Unsupported replay_action_space={replay_action_space!r}")

        self.trainer = trainer
        self.inference = inference
        self._metadata = metadata
        self.warmup_steps = int(warmup_steps)
        self.utd_ratio = int(utd_ratio)
        self.max_episode_chunks = int(max_episode_chunks)
        self.save_dir = save_dir
        self.save_interval_episodes = max(1, int(save_interval_episodes))
        self.eval_only = bool(eval_only)
        self.store_eval_episodes = bool(store_eval_episodes)
        self.eval_interval_episodes = max(0, int(eval_interval_episodes))
        self._eval_pending = False
        self._eval_episode_active = False
        self._total_eval_episodes = 0
        self.vla_only = bool(metadata.vla_only)
        if self.vla_only and not self.eval_only:
            raise ValueError("VLA-only serving requires eval_only=True")
        self.replay_action_space = replay_action_space
        self.metric_logger = metric_logger

        self._lock = threading.Lock()
        self._pending: dict[str, PendingTransition] = {}
        self._next_transition_id = 0
        self._episode = EpisodeAccumulator()
        self._session_id = int(time.time()) & 0x7FFFFFFF
        self._episode_id = 0
        self._chunk_id = 0
        self._total_chunks = 0
        self._total_episodes = 0

    @property
    def metadata(self) -> dict[str, Any]:
        """Handshake payload for :class:`RLinfWebsocketPolicyServer`."""
        return self._metadata.to_payload()

    @property
    def in_warmup(self) -> bool:
        """Whether the Stage 2 head is still gated off.

        Warmup is measured in replay rows, not wall-clock or update count, so a
        resumed run that already has data does not repeat it.
        """
        if self.eval_only:
            return False
        if getattr(self.trainer, "offline_total_updates", 0) > 0:
            return False
        return self.trainer.replay_buffer.total_samples < self.warmup_steps

    # -------------------------------------------------------------- router

    def infer(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Handle one decoded request frame.

        A malformed frame or an unknown request type propagates
        :class:`~rlinf.serving.rlt.protocol.ProtocolError` from the decoder,
        which the transport turns into an error frame for the client.

        Args:
            frame: Decoded msgpack payload from the robot client.

        Returns:
            Response payload to send back.
        """
        with self._lock:
            request_type = decode_request_type(frame)
            started = time.monotonic()
            try:
                response = self._dispatch(request_type, frame)
            except Exception:
                logger.exception(
                    "RLT request %s failed after %.1f ms",
                    request_type,
                    (time.monotonic() - started) * 1000.0,
                )
                raise
            logger.info(
                "RLT request %s done in %.1f ms (replay=%d pending=%d chunk=%d)",
                request_type,
                (time.monotonic() - started) * 1000.0,
                self.trainer.replay_buffer.total_samples,
                len(self._pending),
                self._chunk_id,
            )
            return response

    def _dispatch(self, request_type: str, frame: dict[str, Any]) -> dict[str, Any]:
        if request_type == REQUEST_ACT:
            return self._act(ActRequest.from_payload(frame))
        if request_type == REQUEST_TRANSITION:
            return self._store_transition(TransitionRequest.from_payload(frame))
        if request_type == REQUEST_DISCARD:
            return self._discard(DiscardRequest.from_payload(frame))
        if request_type == REQUEST_REWIND_EXIT:
            return self._rewind(RewindExitRequest.from_payload(frame), mode="exit")
        if request_type == REQUEST_REWIND_CREDIT:
            return self._rewind(RewindCreditRequest.from_payload(frame), mode="credit")
        if request_type == REQUEST_EPISODE_END:
            return self._finish_episode(EpisodeEndRequest.from_payload(frame))
        if request_type == REQUEST_STATUS:
            return self._status()
        if request_type == REQUEST_RESET:
            self.reset()
            return {"ok": True, **self._status()}
        raise ProtocolError(f"Unhandled RLT request {request_type!r}")

    def reset(self) -> None:
        """Drop pending chunks and episode counters without touching replay."""
        logger.info("Resetting server state, dropping %d pending", len(self._pending))
        self._pending.clear()
        self._episode.reset()
        self._chunk_id = 0
        self._eval_pending = False
        self._eval_episode_active = False
        # Advance the episode as well. Chunk ids restart at 1, so reusing the
        # episode id would re-issue (session, episode, chunk) triples that are
        # already committed; the learner's dedup set would then drop the new
        # rows silently and the rewind index would point at the wrong chunks.
        self._episode_id += 1

    # ----------------------------------------------------------------- act

    def _act(self, request: ActRequest) -> dict[str, Any]:
        # Reuse the reference-only path, but never turn evaluation into warmup
        # collection or allow a random/untrained critic to select an action.
        self._maybe_begin_eval_episode()
        warmup = self.in_warmup or self.vla_only
        rlt_obs = self.inference.encode(request.observation)
        selection = self.inference.select_action(
            rlt_obs,
            warmup=warmup,
            deterministic=self.eval_only or self._eval_episode_active,
            exploration_noise_sigma=request.exploration_noise_sigma,
        )
        mode = (
            "eval" if (self.vla_only or self._eval_episode_active) else selection.mode
        )

        self._chunk_id += 1
        identity = ChunkIdentity(
            episode_id=self._episode_id,
            session_id=self._session_id,
            env_id=request.identity.env_id if request.identity else 0,
            chunk_id=self._chunk_id,
        )
        transition_id = str(self._next_transition_id)
        self._next_transition_id += 1
        self._pending[transition_id] = PendingTransition(
            transition_id=transition_id,
            identity=identity,
            rlt_obs=rlt_obs,
            normalized_chunk=selection.normalized_chunk,
            robot_chunk=selection.robot_chunk,
            mode=mode,
        )

        response = {
            "actions": selection.robot_chunk,
            "reference_actions": selection.reference_chunk,
            "transition_id": transition_id,
            "mode": mode,
            "action_chunk_space": ACTION_SPACE_ROBOT,
            "replay_action_space": self.replay_action_space,
            **identity.to_payload(),
            **selection.expo_info,
            **self._status(),
        }
        if self.replay_action_space == ACTION_SPACE_NORMALIZED:
            response["normalized_actions"] = selection.normalized_chunk
        return response

    # ---------------------------------------------------------- transition

    def _store_transition(self, request: TransitionRequest) -> dict[str, Any]:
        pending = self._pending.pop(request.transition_id, None)
        if pending is None:
            # Either a resend after the response was lost, or a client bug.
            # Refusing beats writing a row whose curr_obs we no longer have.
            raise ProtocolError(
                f"Unknown transition_id {request.transition_id!r}; it was never "
                "handed out by act, or was already committed or discarded."
            )
        if request.identity is not None and (
            request.identity.chunk_id
            and request.identity.chunk_id != pending.identity.chunk_id
        ):
            raise ProtocolError(
                f"transition {request.transition_id} carries chunk_id "
                f"{request.identity.chunk_id} but act issued "
                f"{pending.identity.chunk_id}"
            )

        chunk_len, _ = self.trainer._chunk_shape()
        rewards = self._pad_rewards(request.rewards, chunk_len)
        self._episode.chunks += 1
        self._episode.reward += float(np.sum(rewards))
        self._total_chunks += 1
        if request.intervention:
            self._episode.interventions += 1
        if pending.mode == "actor":
            self._episode.train_chunks += 1

        if self.eval_only:
            return {
                "ok": True,
                "stored": False,
                "reason": "eval_only",
                "transition_id": request.transition_id,
                **self._status(),
            }
        if pending.mode == "eval" and not self.store_eval_episodes:
            return {
                "ok": True,
                "stored": False,
                "reason": "eval_episode",
                "transition_id": request.transition_id,
                **self._status(),
            }

        replay_chunk = self._replay_chunk(request, pending)
        next_obs = self.inference.encode(request.next_observation)
        added, completed = self.trainer.add_transition(
            curr_obs=self.inference.strip_private(pending.rlt_obs),
            next_obs=self.inference.strip_private(next_obs),
            action_chunk=replay_chunk,
            rewards=rewards,
            done=request.done,
            bootstrap_mask=request.bootstrap_mask,
            intervention=request.intervention,
            identity=pending.identity,
            action_source=self._action_source(pending.mode, request.intervention),
        )
        return {
            "ok": True,
            "stored": bool(added),
            "transition_id": request.transition_id,
            "episodes_completed": completed,
            **self._status(),
        }

    @staticmethod
    def _pad_rewards(rewards: np.ndarray, chunk_len: int) -> np.ndarray:
        """Right-pad or truncate client rewards to the chunk length.

        A chunk cut short by an operator takeover carries fewer rewards than
        the chunk has steps; the discounted chunk return treats the missing
        tail as zero reward.
        """
        flat = np.asarray(rewards, dtype=np.float32).reshape(-1)
        if flat.size == chunk_len:
            return flat
        padded = np.zeros(chunk_len, dtype=np.float32)
        usable = min(flat.size, chunk_len)
        padded[:usable] = flat[:usable]
        return padded

    @staticmethod
    def _action_source(mode: str, intervention: bool) -> int:
        if intervention:
            return ACTION_SOURCE_HUMAN
        return ACTION_SOURCE_VLA if mode == "warmup" else ACTION_SOURCE_POLICY

    def _replay_chunk(
        self, request: TransitionRequest, pending: PendingTransition
    ) -> np.ndarray:
        """Resolve the executed chunk into the configured replay action space.

        Raises:
            ProtocolError: The client sent an action in a space that cannot be
                converted into the replay space.
        """
        if request.action_chunk is None:
            # Nothing was overridden, so the handed-out chunk was executed.
            return (
                pending.normalized_chunk
                if self.replay_action_space == ACTION_SPACE_NORMALIZED
                else pending.robot_chunk
            )

        chunk_len, action_dim = self.trainer._chunk_shape()
        executed = np.asarray(request.action_chunk, dtype=np.float32).reshape(
            chunk_len, action_dim
        )
        if request.action_chunk_space == self.replay_action_space:
            return executed
        if (
            request.action_chunk_space == ACTION_SPACE_ROBOT
            and self.replay_action_space == ACTION_SPACE_NORMALIZED
        ):
            return self.inference.to_normalized_space(executed, pending.rlt_obs)
        raise ProtocolError(
            f"Cannot convert an executed chunk from {request.action_chunk_space!r} "
            f"into replay space {self.replay_action_space!r}"
        )

    def _discard(self, request: DiscardRequest) -> dict[str, Any]:
        pending = self._pending.pop(request.transition_id, None)
        return {
            "ok": True,
            "discarded": pending is not None,
            "transition_id": request.transition_id,
            **self._status(),
        }

    # -------------------------------------------------------------- rewind

    def _rewind(self, request, *, mode: str) -> dict[str, Any]:
        """Apply an operator rewind decision to already-stored replay rows."""
        chunks = request.chunks_rewound if mode == "exit" else request.bad_chunks
        prefix_reward = getattr(request, "prefix_reward", 0.0)
        if self.eval_only or self._eval_episode_active:
            return {
                "ok": True,
                "applied": False,
                "reason": "eval_only" if self.eval_only else "eval_episode",
                **self._status(),
            }
        if request.terminal_reward == 0.0:
            return {
                "ok": True,
                "applied": False,
                "reason": "terminal_reward=0",
                **self._status(),
            }
        if chunks <= 0:
            return {
                "ok": True,
                "applied": False,
                "reason": "no_chunks_marked",
                **self._status(),
            }

        identity = request.identity or ChunkIdentity()
        event = RLTRewindEvent(
            mode=mode,
            chunks_rewound=int(chunks),
            terminal_reward=float(request.terminal_reward),
            prefix_reward=float(prefix_reward),
            confidence=float(request.preference_confidence),
            episode_id=self._episode_id,
            session_id=self._session_id,
            env_id=int(identity.env_id),
            chunk_id=self._chunk_id,
        )
        self.trainer._ingest_rewind_events([event])
        return {
            "ok": True,
            "applied": True,
            "mode": mode,
            "chunks": int(chunks),
            **self._status(),
        }

    # --------------------------------------------------------- episode end

    def _finish_episode(self, request: EpisodeEndRequest) -> dict[str, Any]:
        """Run the training burst for the episode that just ended.

        Updates are budgeted as ``train_chunks * utd_ratio``, counting only
        chunks the Stage 2 actor actually drove. Warmup and evaluation chunks
        collect data without earning gradient steps, which is what keeps the
        update-to-data ratio at its configured value instead of inflating it by
        the warmup episodes.
        """
        stats = dict(request.stats)
        stale = list(self._pending)
        if stale:
            logger.warning(
                "episode_end dropping %d pending transitions: %s", len(stale), stale
            )
            self._pending.clear()

        metrics: dict[str, float] = {}
        updates = 0
        was_eval_episode = self._eval_episode_active
        if not self.eval_only and not was_eval_episode:
            updates = self._episode.train_chunks * self.utd_ratio
            if updates > 0:
                metrics = self.trainer.train(updates)
            elif self.in_warmup:
                logger.info(
                    "episode_end: still warming up (%d/%d replay rows), no updates",
                    self.trainer.replay_buffer.total_samples,
                    self.warmup_steps,
                )

        self._total_episodes += 1
        if was_eval_episode:
            self._total_eval_episodes += 1
            self._eval_episode_active = False
            logger.info(
                "Finished EVAL episode %d (replay write follows store_eval_episodes)",
                self._total_eval_episodes,
            )
        elif (
            self.eval_interval_episodes > 0
            and not self.eval_only
            and not self.in_warmup
            and self._total_episodes % self.eval_interval_episodes == 0
        ):
            self._eval_pending = True
            logger.info(
                "episode_end: scheduled EVAL next (eval_interval_episodes=%d, "
                "train_episodes=%d)",
                self.eval_interval_episodes,
                self._total_episodes,
            )
        episode_metrics = {
            "env/episode_reward": self._episode.reward,
            "env/episode_chunks": float(self._episode.chunks),
            "env/episode_train_chunks": float(self._episode.train_chunks),
            "env/episode_interventions": float(self._episode.interventions),
            "env/episode_success": float(bool(stats.get("success", False))),
            "rlt/updates_this_episode": float(updates),
            "env/is_eval_episode": float(was_eval_episode),
        }
        if was_eval_episode:
            episode_metrics.update(
                {
                    "eval/episode_reward": self._episode.reward,
                    "eval/episode_chunks": float(self._episode.chunks),
                    "eval/episode_success": float(bool(stats.get("success", False))),
                    "eval/total_eval_episodes": float(self._total_eval_episodes),
                }
            )
        if self.metric_logger is not None:
            self.metric_logger({**episode_metrics, **metrics})

        saved = self._maybe_save()
        self._episode.reset()
        self._episode_id += 1
        self._chunk_id = 0
        return {
            "ok": True,
            "updates_run": updates,
            "checkpoint_saved": saved,
            "metrics": {**episode_metrics, **metrics},
            **self._status(),
        }

    def _maybe_save(self) -> bool:
        if self.save_dir is None or self.eval_only:
            return False
        if self._total_episodes % self.save_interval_episodes != 0:
            return False
        # The episode counter restarts at zero on resume, so it alone would
        # overwrite the checkpoints of the run being resumed. update_step is
        # persisted in the checkpoint and keeps increasing across restarts.
        target = os.path.join(
            self.save_dir,
            f"episode_{self._total_episodes}_step_{int(self.trainer.update_step)}",
        )
        self.trainer.save(target)
        return True

    # -------------------------------------------------------------- status

    def _status(self) -> dict[str, Any]:
        return {
            "buffer_size": int(self.trainer.replay_buffer.total_samples),
            "demo_buffer_size": int(
                0
                if self.trainer.demo_buffer is None
                else self.trainer.demo_buffer.total_samples
            ),
            "preference_buffer_size": len(self.trainer.rewind_preference_buffer),
            "warmup_steps": self.warmup_steps,
            "warmup_done": not self.in_warmup,
            "total_chunks": self._total_chunks,
            "total_episodes": self._total_episodes,
            "total_updates": int(self.trainer.update_step),
            "offline_total_updates": int(
                getattr(self.trainer, "offline_total_updates", 0)
            ),
            "offline_buffer_size": int(
                getattr(getattr(self.trainer, "offline_buffer", None), "size", 0)
            ),
            "offline_partial_data": bool(
                getattr(
                    getattr(self.trainer, "offline_buffer", None), "payload", {}
                ).get("partial_conversion", False)
            ),
            "episode_chunks": self._episode.chunks,
            "episode_train_chunks": self._episode.train_chunks,
            "max_episode_chunks": self.max_episode_chunks,
            "pending": len(self._pending),
            "eval_only": self.eval_only,
            "eval_interval_episodes": self.eval_interval_episodes,
            "eval_pending": self._eval_pending,
            "is_eval_episode": self.eval_only or self._eval_episode_active,
            "store_eval_episodes": self.store_eval_episodes,
            "total_eval_episodes": self._total_eval_episodes,
            "episode_id": self._episode_id,
            "session_id": self._session_id,
            "chunk_id": self._chunk_id,
        }

    def _maybe_begin_eval_episode(self) -> None:
        """Activate a pending eval episode on the first post-warmup act."""
        if (
            self.eval_only
            or self._eval_episode_active
            or not self._eval_pending
            or self.in_warmup
        ):
            return
        self._eval_episode_active = True
        self._eval_pending = False
        logger.info(
            "Beginning EVAL episode %d (deterministic, no UTD)",
            self._total_eval_episodes + 1,
        )
