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

"""Small robot-side subset of the RLinf RLT online protocol."""

from __future__ import annotations

from typing import Any, Mapping

PROTOCOL_VERSION = "rlt-online-rl/v1"
REQUEST_KEY = "rlt/request"
REQUEST_ACT = "act"
REQUEST_TRANSITION = "transition"
REQUEST_DISCARD = "discard"
REQUEST_REWIND_EXIT = "rewind_exit_correction"
REQUEST_REWIND_CREDIT = "rewind_credit_correction"
REQUEST_EPISODE_END = "episode_end"
ACTION_SPACE_ROBOT = "robot"


class ProtocolError(RuntimeError):
    """Raised before motion when the server/client contracts disagree."""


def validate_metadata(
    metadata: Mapping[str, Any],
    *,
    action_dim: int,
    chunk_length: int,
    proprio_dim: int,
    camera_keys: tuple[str, ...],
    robot_type: str,
    action_schema: str,
) -> None:
    """Fail closed on any field that changes physical action interpretation."""
    expected = {
        "protocol": PROTOCOL_VERSION,
        "action_dim": int(action_dim),
        "chunk_length": int(chunk_length),
        "proprio_dim": int(proprio_dim),
        "camera_keys": list(camera_keys),
        "action_space": ACTION_SPACE_ROBOT,
        "robot_type": robot_type,
        "action_schema": action_schema,
    }
    for key, wanted in expected.items():
        if key not in metadata:
            raise ProtocolError(f"server metadata is missing required field {key!r}")
        actual = metadata[key]
        if key == "camera_keys":
            actual = list(actual)
        if actual != wanted:
            raise ProtocolError(
                f"server metadata {key}={actual!r}, expected {wanted!r}"
            )


def identity_payload(
    *, episode_id: int, session_id: int, env_id: int, chunk_id: int
) -> dict[str, int]:
    return {
        "episode_id": int(episode_id),
        "session_id": int(session_id),
        "env_id": int(env_id),
        "chunk_id": int(chunk_id),
    }
