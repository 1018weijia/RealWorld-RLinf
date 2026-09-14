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

"""Wire protocol for the RLT Stage 2 online-RL WebSocket server.

Request key, request-type names and payload field names are kept identical to
the ``rlt-openpi`` remote-franka protocol so a Franka client can drive an RLinf
server and vice versa.  RLinf adds four optional identity fields
(``episode_id``, ``session_id``, ``env_id``, ``chunk_id``) because
:class:`~rlinf.algorithms.rlt.learner.RLTRewindCore` indexes replay rows and
deduplicates resubmissions by that tuple rather than by ``transition_id``
alone.  A client that omits them still works; the server then fills them in
from its own episode/chunk counters.

Everything here is pure data validation with no torch or robot dependency, so
both the server and any client can import it cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

REQUEST_KEY = "rlt/request"
"""Top-level key naming the request type in every frame."""

PROTOCOL_VERSION = "rlt-online-rl/v1"
"""Handshake protocol identifier, shared with ``rlt-openpi``."""

REQUEST_ACT = "act"
REQUEST_TRANSITION = "transition"
REQUEST_DISCARD = "discard"
REQUEST_REWIND_EXIT = "rewind_exit_correction"
REQUEST_REWIND_CREDIT = "rewind_credit_correction"
REQUEST_EPISODE_END = "episode_end"
REQUEST_STATUS = "status"
REQUEST_RESET = "reset"

REQUEST_TYPES = (
    REQUEST_ACT,
    REQUEST_TRANSITION,
    REQUEST_DISCARD,
    REQUEST_REWIND_EXIT,
    REQUEST_REWIND_CREDIT,
    REQUEST_EPISODE_END,
    REQUEST_STATUS,
    REQUEST_RESET,
)

ACTION_SPACE_ROBOT = "robot"
ACTION_SPACE_NORMALIZED = "normalized"
ACTION_SPACES = (ACTION_SPACE_NORMALIZED, ACTION_SPACE_ROBOT)


class ProtocolError(ValueError):
    """A frame violated the RLT protocol and cannot be handled."""


def decode_request_type(frame: dict[str, Any]) -> str:
    """Read and validate the request type of one decoded frame.

    Args:
        frame: Decoded msgpack payload.

    Returns:
        One of :data:`REQUEST_TYPES`. Frames without :data:`REQUEST_KEY`
        default to ``"act"``, matching ``rlt-openpi``.

    Raises:
        ProtocolError: The request type is not recognized.
    """
    request_type = str(frame.get(REQUEST_KEY, REQUEST_ACT))
    if request_type not in REQUEST_TYPES:
        raise ProtocolError(
            f"Unknown RLT websocket request {request_type!r}; "
            f"expected one of {list(REQUEST_TYPES)}"
        )
    return request_type


def _require(frame: dict[str, Any], key: str) -> Any:
    if key not in frame:
        raise ProtocolError(
            f"RLT request {frame.get(REQUEST_KEY, REQUEST_ACT)!r} is missing "
            f"required field {key!r}"
        )
    return frame[key]


def _float_array(value: Any, name: str) -> np.ndarray:
    # np.array, not np.asarray: msgpack decodes arrays as read-only views over
    # the received byte buffer. Passing one on would make torch.as_tensor alias
    # that buffer, so a later in-place write would corrupt a replay row instead
    # of failing, and any consumer that mutates the array would raise instead.
    array = np.array(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ProtocolError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class ChunkIdentity:
    """Robot-side identity of one action chunk.

    ``RLTRewindCore`` keys replay rows, duplicate suppression and rewind
    patching on this tuple, so it has to survive the wire.

    Attributes:
        episode_id: Monotonic episode counter on the client.
        session_id: Client session counter, bumped on reconnect so a restarted
            client cannot collide with rows from the previous run.
        env_id: Index of the robot within one client, ``0`` for single-arm.
        chunk_id: Monotonic chunk counter within the episode.
    """

    episode_id: int = 0
    session_id: int = 0
    env_id: int = 0
    chunk_id: int = 0

    @classmethod
    def from_payload(cls, frame: dict[str, Any]) -> ChunkIdentity:
        """Read identity fields from a frame, defaulting missing ones to ``0``."""
        return cls(
            episode_id=int(frame.get("episode_id", 0)),
            session_id=int(frame.get("session_id", 0)),
            env_id=int(frame.get("env_id", 0)),
            chunk_id=int(frame.get("chunk_id", 0)),
        )

    def to_payload(self) -> dict[str, int]:
        """Render the identity as wire fields."""
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "env_id": self.env_id,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class ActRequest:
    """``act``: ask the server for the next action chunk.

    Attributes:
        observation: Raw robot observation. Keys follow the OpenPI convention
            (``observation/state``, ``observation/<name>_image``, ``prompt``);
            the server owns all preprocessing.
        exploration_noise_sigma: Client override for actor noise. ``None``
            leaves the server's configured sigma in force.
        identity: Optional client-side chunk identity.
    """

    observation: dict[str, Any]
    exploration_noise_sigma: float | None = None
    identity: ChunkIdentity | None = None

    @classmethod
    def from_payload(cls, frame: dict[str, Any]) -> ActRequest:
        """Validate and decode an ``act`` frame."""
        observation = frame.get("observation", frame)
        if not isinstance(observation, dict):
            raise ProtocolError("act.observation must be a mapping")
        sigma = frame.get("exploration_noise_sigma")
        if sigma is not None:
            sigma = float(sigma)
            if sigma < 0.0:
                sigma = None
        identity = (
            ChunkIdentity.from_payload(frame)
            if any(
                key in frame
                for key in ("episode_id", "session_id", "env_id", "chunk_id")
            )
            else None
        )
        return cls(
            observation=observation,
            exploration_noise_sigma=sigma,
            identity=identity,
        )


@dataclass(frozen=True)
class TransitionRequest:
    """``transition``: commit the chunk previously handed out by ``act``.

    Attributes:
        transition_id: Server-issued handle from the matching ``act`` response.
        next_observation: Observation after the chunk finished executing.
        rewards: Per-step rewards for the chunk, length ``chunk_length``.
        done: Whether the episode terminated on this chunk.
        bootstrap_mask: ``1.0`` to bootstrap from ``next_observation``, ``0.0``
            for a hard terminal such as a robot fault.
        info: Free-form client diagnostics; also carries ``intervention``.
        intervention: Whether a human drove any step of this chunk.
        action_chunk: Actions actually executed, which may differ from the
            handed-out chunk after an intervention.
        action_chunk_space: Space of ``action_chunk``, ``"robot"`` or
            ``"normalized"``.
        identity: Optional client-side chunk identity.
    """

    transition_id: str
    next_observation: dict[str, Any]
    rewards: np.ndarray
    done: bool = False
    bootstrap_mask: float = 1.0
    info: dict[str, Any] = field(default_factory=dict)
    intervention: bool = False
    action_chunk: np.ndarray | None = None
    action_chunk_space: str = ACTION_SPACE_ROBOT
    identity: ChunkIdentity | None = None

    @classmethod
    def from_payload(cls, frame: dict[str, Any]) -> TransitionRequest:
        """Validate and decode a ``transition`` frame."""
        next_observation = _require(frame, "next_observation")
        if not isinstance(next_observation, dict):
            raise ProtocolError("transition.next_observation must be a mapping")

        rewards = _float_array(frame.get("rewards", 0.0), "transition.rewards")
        rewards = np.atleast_1d(rewards).reshape(-1)

        bootstrap_mask = float(frame.get("bootstrap_mask", 1.0))
        if bootstrap_mask not in (0.0, 1.0):
            raise ProtocolError(
                f"transition.bootstrap_mask must be 0.0 or 1.0, got {bootstrap_mask}"
            )

        info = frame.get("info") or {}
        if not isinstance(info, dict):
            raise ProtocolError("transition.info must be a mapping")

        action_chunk = frame.get("action_chunk")
        if action_chunk is not None:
            action_chunk = _float_array(action_chunk, "transition.action_chunk")
        space = str(frame.get("action_chunk_space", ACTION_SPACE_ROBOT)).strip().lower()
        if space not in ACTION_SPACES:
            raise ProtocolError(
                f"transition.action_chunk_space must be one of {list(ACTION_SPACES)}, "
                f"got {space!r}"
            )

        identity = (
            ChunkIdentity.from_payload(frame)
            if any(
                key in frame
                for key in ("episode_id", "session_id", "env_id", "chunk_id")
            )
            else None
        )
        return cls(
            transition_id=str(_require(frame, "transition_id")),
            next_observation=next_observation,
            rewards=rewards,
            done=bool(frame.get("done", False)),
            bootstrap_mask=bootstrap_mask,
            info=info,
            intervention=bool(frame.get("intervention", False))
            or bool(info.get("intervention", False)),
            action_chunk=action_chunk,
            action_chunk_space=space,
            identity=identity,
        )


@dataclass(frozen=True)
class DiscardRequest:
    """``discard``: drop a prefetched chunk the robot never executed."""

    transition_id: str

    @classmethod
    def from_payload(cls, frame: dict[str, Any]) -> DiscardRequest:
        """Validate and decode a ``discard`` frame."""
        return cls(transition_id=str(_require(frame, "transition_id")))


@dataclass(frozen=True)
class RewindExitRequest:
    """``rewind_exit_correction``: the operator physically rewound the robot.

    Attributes:
        terminal_reward: Negative reward written to the last bad chunk. ``0.0``
            makes the request a no-op, matching ``rlt-openpi``.
        chunks_rewound: Number of trailing chunks on the discarded branch.
        preference_confidence: Weight of the resulting preference pair, in
            ``(0, 1]``.
        identity: Session identity of the fork point.
    """

    terminal_reward: float = 0.0
    chunks_rewound: int = 0
    preference_confidence: float = 1.0
    identity: ChunkIdentity | None = None

    @classmethod
    def from_payload(cls, frame: dict[str, Any]) -> RewindExitRequest:
        """Validate and decode a ``rewind_exit_correction`` frame."""
        confidence = float(frame.get("preference_confidence", 1.0))
        if not 0.0 < confidence <= 1.0:
            raise ProtocolError(
                f"preference_confidence must be in (0, 1], got {confidence}"
            )
        return cls(
            terminal_reward=float(frame.get("terminal_reward", 0.0)),
            chunks_rewound=max(0, int(frame.get("chunks_rewound", 0))),
            preference_confidence=confidence,
            identity=ChunkIdentity.from_payload(frame),
        )


@dataclass(frozen=True)
class RewindCreditRequest:
    """``rewind_credit_correction``: replay-only correction, no robot motion.

    Attributes:
        terminal_reward: Negative reward written to the last bad chunk.
        prefix_reward: Reward written to the chunk just before the bad branch.
        bad_chunks: Number of trailing chunks to mark bad.
        preference_confidence: Weight of the resulting preference pair.
        identity: Session identity of the fork point.
    """

    terminal_reward: float = 0.0
    prefix_reward: float = 0.1
    bad_chunks: int = 0
    preference_confidence: float = 1.0
    identity: ChunkIdentity | None = None

    @classmethod
    def from_payload(cls, frame: dict[str, Any]) -> RewindCreditRequest:
        """Validate and decode a ``rewind_credit_correction`` frame."""
        confidence = float(frame.get("preference_confidence", 1.0))
        if not 0.0 < confidence <= 1.0:
            raise ProtocolError(
                f"preference_confidence must be in (0, 1], got {confidence}"
            )
        return cls(
            terminal_reward=float(frame.get("terminal_reward", 0.0)),
            prefix_reward=float(frame.get("prefix_reward", 0.1)),
            bad_chunks=max(0, int(frame.get("bad_chunks", 0))),
            preference_confidence=confidence,
            identity=ChunkIdentity.from_payload(frame),
        )


@dataclass(frozen=True)
class EpisodeEndRequest:
    """``episode_end``: run the training burst and checkpoint.

    Attributes:
        stats: Client-side episode summary (reward, chunk/step counts,
            intervention counts, success flag).
        identity: Session identity of the episode that just ended.
    """

    stats: dict[str, Any] = field(default_factory=dict)
    identity: ChunkIdentity | None = None

    @classmethod
    def from_payload(cls, frame: dict[str, Any]) -> EpisodeEndRequest:
        """Validate and decode an ``episode_end`` frame."""
        stats = frame.get("stats") or {}
        if not isinstance(stats, dict):
            raise ProtocolError("episode_end.stats must be a mapping")
        return cls(stats=stats, identity=ChunkIdentity.from_payload(frame))


@dataclass(frozen=True)
class ServerMetadata:
    """Handshake frame the server pushes before any request.

    The client validates its own assumptions against these values and refuses
    to move the robot on a mismatch, which is what turns a silent shape or
    action-space error into a startup failure.
    """

    action_dim: int
    chunk_length: int
    proprio_dim: int
    warmup_steps: int
    run_name: str
    action_space: str
    replay_action_space: str
    action_selection_mode: str
    edit_scale: float
    expo_num_base_samples: int
    expo_num_edit_samples: int
    actor_action_clip_min: float
    actor_action_clip_max: float
    max_episode_chunks: int
    camera_keys: tuple[str, ...]
    task_prompt: str
    eval_only: bool = False
    vla_only: bool = False
    use_preference_loss: bool = False
    # Disambiguate embodiments that happen to share tensor sizes.
    robot_type: str = ""
    action_schema: str = ""
    protocol: str = PROTOCOL_VERSION
    transition_action_spaces: tuple[str, ...] = ACTION_SPACES

    def to_payload(self) -> dict[str, Any]:
        """Render the handshake as a msgpack-friendly mapping."""
        return {
            "protocol": self.protocol,
            "action_dim": int(self.action_dim),
            "chunk_length": int(self.chunk_length),
            "proprio_dim": int(self.proprio_dim),
            "warmup_steps": int(self.warmup_steps),
            "run_name": str(self.run_name),
            "action_space": str(self.action_space),
            "replay_action_space": str(self.replay_action_space),
            "transition_action_spaces": list(self.transition_action_spaces),
            "action_selection_mode": str(self.action_selection_mode),
            "edit_scale": float(self.edit_scale),
            "expo_num_base_samples": int(self.expo_num_base_samples),
            "expo_num_edit_samples": int(self.expo_num_edit_samples),
            "actor_action_clip_min": float(self.actor_action_clip_min),
            "actor_action_clip_max": float(self.actor_action_clip_max),
            "max_episode_chunks": int(self.max_episode_chunks),
            "camera_keys": list(self.camera_keys),
            "task_prompt": str(self.task_prompt),
            "eval_only": bool(self.eval_only),
            "vla_only": bool(self.vla_only),
            "use_preference_loss": bool(self.use_preference_loss),
            "robot_type": str(self.robot_type),
            "action_schema": str(self.action_schema),
        }


def validate_server_metadata(
    metadata: dict[str, Any],
    *,
    action_dim: int | None = None,
    chunk_length: int | None = None,
    proprio_dim: int | None = None,
    camera_keys: tuple[str, ...] | None = None,
) -> None:
    """Fail fast when the client and server disagree about the contract.

    Args:
        metadata: Handshake payload received from the server.
        action_dim: Client action dimension, when known.
        chunk_length: Client chunk length, when known.
        proprio_dim: Client proprioception dimension, when known.
        camera_keys: Camera names the client can actually publish.

    Raises:
        ProtocolError: The protocol version or any provided expectation does
            not match the server.
    """
    protocol = str(metadata.get("protocol", ""))
    if protocol != PROTOCOL_VERSION:
        raise ProtocolError(
            f"Server speaks protocol {protocol!r} but this client requires "
            f"{PROTOCOL_VERSION!r}. Restart both sides on the same revision."
        )

    for name, expected in (
        ("action_dim", action_dim),
        ("chunk_length", chunk_length),
        ("proprio_dim", proprio_dim),
    ):
        if expected is None:
            continue
        actual = metadata.get(name)
        if actual is None:
            raise ProtocolError(f"Server handshake is missing {name!r}")
        if int(actual) != int(expected):
            raise ProtocolError(
                f"Server {name}={int(actual)} but client expects {int(expected)}. "
                "Check that both sides load the same Stage 1 checkpoint config."
            )

    if camera_keys is not None:
        required = set(metadata.get("camera_keys") or ())
        missing = sorted(required - set(camera_keys))
        if missing:
            raise ProtocolError(
                f"Server requires camera keys {missing} that this client cannot "
                f"publish (available: {sorted(camera_keys)})."
            )
