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

"""RLT transaction state machine for the synchronous DesktopClient pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .mapping import ACTION_DIM, ACTION_SCHEMA, CAMERA_KEYS, split_ee_actions
from .protocol import (
    ACTION_SPACE_ROBOT,
    REQUEST_ACT,
    REQUEST_DISCARD,
    REQUEST_EPISODE_END,
    REQUEST_KEY,
    REQUEST_REWIND_CREDIT,
    REQUEST_REWIND_EXIT,
    REQUEST_TRANSITION,
    identity_payload,
    validate_metadata,
)


@dataclass
class PendingChunk:
    transition_id: str
    chunk_id: int
    executed_actions: np.ndarray
    mode: str
    episode_id: int = 0
    session_id: int = 0
    env_id: int = 0


@dataclass(frozen=True)
class OperatorDecision:
    kind: str
    chunks: int = 1
    terminal_reward: float = 0.0
    prefix_reward: float = 0.1
    confidence: float = 1.0


class RLTSession:
    """Bind consecutive DesktopClient observations into RLT transitions."""

    def __init__(
        self,
        request: Callable[[dict[str, Any]], dict[str, Any]],
        metadata: Mapping[str, Any],
        *,
        task: str,
        chunk_length: int,
        episode_id: int = 0,
        env_id: int = 0,
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
        gripper_limits: tuple[float, float] = (-0.1, 4.6),
    ) -> None:
        validate_metadata(
            metadata,
            action_dim=ACTION_DIM,
            chunk_length=chunk_length,
            proprio_dim=ACTION_DIM,
            camera_keys=CAMERA_KEYS,
            robot_type="x2robot",
            action_schema=ACTION_SCHEMA,
        )
        self._request = request
        self.task = str(task)
        self.chunk_length = int(chunk_length)
        self.episode_id = int(episode_id)
        self.env_id = int(env_id)
        self.session_id = int(metadata.get("session_id", 0))
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.gripper_limits = gripper_limits
        self.pending: PendingChunk | None = None
        self.next_chunk_id = 1
        self.committed_chunks = 0
        self.interventions = 0
        self.total_reward = 0.0
        self.exploration_noise_sigma: float | None = None

    def _identity(self, chunk_id: int) -> dict[str, int]:
        return identity_payload(
            episode_id=self.episode_id,
            session_id=self.session_id,
            env_id=self.env_id,
            chunk_id=chunk_id,
        )

    def request_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.pending is not None:
            raise RuntimeError("previous RLT chunk has not been committed or discarded")
        chunk_id = self.next_chunk_id
        payload = {
            REQUEST_KEY: REQUEST_ACT,
            "observation": observation,
            **self._identity(chunk_id),
        }
        if self.exploration_noise_sigma is not None:
            payload["exploration_noise_sigma"] = float(self.exploration_noise_sigma)
        response = self._request(payload)
        output, executed = split_ee_actions(
            response.get("actions"),
            chunk_length=self.chunk_length,
            gripper_limits=self.gripper_limits,
        )
        self.pending = PendingChunk(
            transition_id=str(response["transition_id"]),
            chunk_id=int(response.get("chunk_id", chunk_id)),
            executed_actions=executed,
            mode=str(response.get("mode", "unknown")),
            episode_id=int(response.get("episode_id", self.episode_id)),
            session_id=int(response.get("session_id", self.session_id)),
            env_id=int(response.get("env_id", self.env_id)),
        )
        self.episode_id = self.pending.episode_id
        self.session_id = self.pending.session_id
        return output

    @staticmethod
    def _pending_identity(pending: PendingChunk) -> dict[str, int]:
        return identity_payload(
            episode_id=pending.episode_id,
            session_id=pending.session_id,
            env_id=pending.env_id,
            chunk_id=pending.chunk_id,
        )

    def set_executed_actions(self, actions: np.ndarray) -> None:
        """Replace commanded actions with the locally observed/sent EE actions."""
        if self.pending is None:
            raise RuntimeError("there is no pending chunk")
        _, actual = split_ee_actions(
            actions,
            chunk_length=self.chunk_length,
            gripper_limits=self.gripper_limits,
        )
        self.pending.executed_actions = actual

    def commit(
        self,
        next_observation: dict[str, Any],
        decision: OperatorDecision | None = None,
        *,
        intervention: bool = False,
        score: float = 0.0,
        info: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        pending = self.pending
        if pending is None:
            raise RuntimeError("there is no pending chunk to commit")
        rewards = np.zeros(self.chunk_length, dtype=np.float32)
        done = False
        bootstrap_mask = 1.0
        if decision is not None and decision.kind in {"success", "failure"}:
            rewards[-1] = (
                self.success_reward
                if decision.kind == "success"
                else self.failure_reward
            )
            if decision.terminal_reward:
                rewards[-1] = float(decision.terminal_reward)
            done = True
            bootstrap_mask = 0.0
        elif score:
            # Mid-episode operator score (progress / regress). The episode keeps
            # running, so the chunk stays bootstrappable.
            rewards[-1] = float(score)
        payload = {
            REQUEST_KEY: REQUEST_TRANSITION,
            "transition_id": pending.transition_id,
            "next_observation": next_observation,
            "rewards": rewards,
            "done": done,
            "bootstrap_mask": bootstrap_mask,
            "intervention": bool(intervention),
            "action_chunk": pending.executed_actions,
            "action_chunk_space": ACTION_SPACE_ROBOT,
            "info": dict(info or {}),
            **self._pending_identity(pending),
        }
        response = self._request(payload)
        self.pending = None
        self.next_chunk_id += 1
        self.committed_chunks += 1
        self.interventions += int(intervention)
        self.total_reward += float(np.sum(rewards))
        if decision is not None and decision.kind.startswith("rewind_"):
            self.apply_rewind(decision, pending.chunk_id)
        return response

    def apply_rewind(
        self, decision: OperatorDecision, chunk_id: int | None = None
    ) -> None:
        identity = self._identity(
            self.next_chunk_id - 1 if chunk_id is None else chunk_id
        )
        if decision.kind == "rewind_exit":
            self._request(
                {
                    REQUEST_KEY: REQUEST_REWIND_EXIT,
                    "chunks_rewound": max(1, int(decision.chunks)),
                    "terminal_reward": float(decision.terminal_reward),
                    "preference_confidence": float(decision.confidence),
                    **identity,
                }
            )
        elif decision.kind == "rewind_credit":
            self._request(
                {
                    REQUEST_KEY: REQUEST_REWIND_CREDIT,
                    "bad_chunks": max(1, int(decision.chunks)),
                    "terminal_reward": float(decision.terminal_reward),
                    "prefix_reward": float(decision.prefix_reward),
                    "preference_confidence": float(decision.confidence),
                    **identity,
                }
            )
        else:
            raise ValueError(f"not a rewind decision: {decision.kind!r}")

    def discard(self) -> None:
        pending, self.pending = self.pending, None
        if pending is None:
            return
        self._request(
            {
                REQUEST_KEY: REQUEST_DISCARD,
                "transition_id": pending.transition_id,
                **self._pending_identity(pending),
            }
        )

    def episode_end(
        self, *, success: bool = False, aborted: bool = False
    ) -> dict[str, Any]:
        if self.pending is not None:
            raise RuntimeError("cannot end an episode with a pending chunk")
        last_chunk = max(0, self.next_chunk_id - 1)
        return self._request(
            {
                REQUEST_KEY: REQUEST_EPISODE_END,
                "stats": {
                    "chunks": self.committed_chunks,
                    "reward": self.total_reward,
                    "success": bool(success),
                    "aborted": bool(aborted),
                    "interventions": self.interventions,
                },
                **self._identity(last_chunk),
            }
        )
