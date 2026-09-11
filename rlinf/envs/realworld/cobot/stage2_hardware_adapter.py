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

"""Site-owned Cobot adapter for RLT Stage 2. Fill in the hardware calls.

The WebSocket client never talks to ROS or the arm SDK. It only calls this
adapter. Point ``transport.controller_factory`` at
``rlinf.envs.realworld.cobot.stage2_hardware_adapter:create_adapter``.

``execute`` receives **robot-space** actions (already de-normalized by the
server), not ``[-1, 1]`` model units. The Protocol docstring on
:class:`CobotControlAdapter` still mentions normalized actions; ignore that
for this path.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from typing import Sequence

import numpy as np

from rlinf.envs.realworld.cobot.control import CobotObservation, CobotStepResult
from rlinf.envs.realworld.rlt_client.loop import (
    EVENT_ABORT,
    EVENT_FAILURE,
    EVENT_REWIND_CREDIT,
    EVENT_REWIND_EXIT,
    EVENT_SUCCESS,
)
from rlinf.envs.realworld.rlt_client.transport import OperatorEvent

logger = logging.getLogger(__name__)

CAMERA_KEYS = ("image", "wrist_image", "side_image")
"""Must match the server handshake. Do not swap the two wrist views."""


def create_adapter(*, action_dim: int, task: str) -> "Stage2HardwareAdapter":
    """Factory imported by ``transport.controller_factory``."""
    return Stage2HardwareAdapter(action_dim=action_dim, task=task)


class Stage2HardwareAdapter:
    """Replace every ``NotImplementedError`` with the cell's control stack."""

    def __init__(self, *, action_dim: int, task: str) -> None:
        self.action_dim = int(action_dim)
        self._task = str(task)
        self._events: deque[OperatorEvent] = deque()
        self._lock = threading.Lock()
        # The client process does not listen for keys. Uncomment when a tty is
        # available, or call enqueue_key() from your own pedal / ROS callback.
        # self.start_stdin_keyboard_listener()

    def reset(self) -> CobotObservation:
        """Move to the start pose and return the first observation."""
        raise NotImplementedError("connect the arm reset / home here")

    def observe(self) -> CobotObservation:
        """Capture cameras + state without moving."""
        raise NotImplementedError("read the three cameras and 14-D state here")

    def execute(self, action: Sequence[float]) -> CobotStepResult:
        """Send one robot-space step and report what the arm actually did.

        Fill-in outline::

            commanded = np.asarray(action, dtype=np.float32).reshape(-1)
            sent = locally_clipped(commanded)  # keep the value you send
            intervention = human_is_driving()
            if intervention:
                sent = human_action()
            send_to_arm(sent)
            info = {"executed_action": sent.copy()}
            if intervention:
                info["human_intervention"] = True
            if safety_fault:
                info["rlt_safety_fault"] = True
            return CobotStepResult(self.observe(), reward=0.0, info=info)
        """
        del action
        raise NotImplementedError("send the robot-space action to the arm")

    def stop(self, reason: str) -> None:
        """Halt immediately. Must work when the server is unreachable."""
        logger.warning("stop requested: %s", reason)
        raise NotImplementedError("engage the local e-stop / hold here")

    def close(self) -> None:
        """Release drivers and camera handles."""
        raise NotImplementedError("tear down the control stack here")

    def poll_rewind_event(self) -> OperatorEvent | None:
        """Return one operator decision, consuming it."""
        with self._lock:
            return self._events.popleft() if self._events else None

    def rewind_chunks(self, count: int) -> CobotObservation:
        """Physically reverse at most ``count`` committed chunks.

        Omit this method to force credit-only rewind (replay patch, no motion).
        """
        raise NotImplementedError(
            "play back saved chunk targets, or delete this method"
        )

    def on_action_chunk_begin(self) -> None:
        """Optional. Called before the first ``execute`` of a chunk.

        Snapshot the current joint/gripper state here if ``rewind_chunks``
        needs a target to play back to.
        """

    def on_action_chunk_end(self, committed: bool) -> None:
        """Optional. Called after the chunk is done.

        Keep at most ``rewind_history_chunks`` (default 12) start-of-chunk
        snapshots when ``committed`` is true.
        """
        del committed

    def start_stdin_keyboard_listener(self) -> None:
        """Map the documented s/f/b/q/escape keys from a POSIX tty.

        Call this from ``__init__`` only when stdin is an interactive terminal.
        ROS / camera nodes that steal stdin should use ``enqueue_key`` from
        their own callback instead.
        """
        if not sys.stdin.isatty():
            logger.warning("stdin is not a tty; keyboard listener not started")
            return

        def _loop() -> None:
            try:
                import termios
                import tty
            except ImportError:
                logger.warning("stdin keyboard listener needs a POSIX tty")
                return
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while True:
                    key = sys.stdin.read(1)
                    if key == "\x1b":
                        self.enqueue_key("escape")
                    else:
                        self.enqueue_key(key)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        thread = threading.Thread(
            target=_loop, name="stage2-operator-keys", daemon=True
        )
        thread.start()
        logger.info("stdin keyboard listener started (s/f/b/q/escape)")

    def enqueue_operator_event(self, event: OperatorEvent) -> None:
        """Call this from the keyboard / pedal / space-mouse callback."""
        with self._lock:
            self._events.append(event)

    def enqueue_key(self, key: str) -> None:
        """Map the documented keys onto :class:`OperatorEvent`."""
        mapping = {
            "s": OperatorEvent(kind=EVENT_SUCCESS),
            "f": OperatorEvent(kind=EVENT_FAILURE),
            "b": OperatorEvent(kind=EVENT_REWIND_EXIT, chunks=1, terminal_reward=-1.0),
            "q": OperatorEvent(
                kind=EVENT_REWIND_CREDIT, chunks=1, terminal_reward=-1.0
            ),
            "escape": OperatorEvent(kind=EVENT_ABORT),
        }
        event = mapping.get(key)
        if event is None:
            logger.warning("unbound operator key %r", key)
            return
        self.enqueue_operator_event(event)

    def _observation(
        self, images: dict[str, np.ndarray], state: np.ndarray
    ) -> CobotObservation:
        missing = [key for key in CAMERA_KEYS if key not in images]
        if missing:
            raise RuntimeError(f"missing cameras {missing}; need {list(CAMERA_KEYS)}")
        for key, frame in images.items():
            array = np.asarray(frame)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                raise RuntimeError(
                    f"{key} must be uint8 [H,W,3] RGB, got {array.dtype} {array.shape}"
                )
        vector = np.asarray(state, dtype=np.float32).reshape(-1)
        if vector.size != self.action_dim:
            raise RuntimeError(
                f"state has {vector.size} values, expected {self.action_dim}"
            )
        return CobotObservation(images=images, state=vector, task=self._task)
