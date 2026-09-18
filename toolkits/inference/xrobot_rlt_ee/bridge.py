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

"""Bridge the installed X2Robot DesktopClient to an RLT WebSocket server.

The downstream wire remains the vendor client's msgpack-numpy protocol. The
upstream wire is RLinf's RLT protocol. This module never publishes ROS topics
or talks to a controller directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import msgpack
import msgpack_numpy
import numpy as np
import websockets

from .mapping import (
    ACTION_DIM,
    ACTION_SCHEMA,
    CAMERA_KEYS,
    execution_receipt,
    observation_to_rlt,
)
from .session import OperatorDecision, RLTSession
from .websocket_client import RLTWebSocketClient

LOG = logging.getLogger("x2robot_rlt.bridge")


def _legacy_packb(value: Any, **kwargs: Any) -> bytes:
    return msgpack.packb(value, default=msgpack_numpy.encode, **kwargs)


def _legacy_unpackb(value: bytes, **kwargs: Any) -> Any:
    return msgpack.unpackb(value, object_hook=msgpack_numpy.decode, **kwargs)


class BridgeError(RuntimeError):
    """A request was rejected before an action could be sent to the robot."""


@dataclass(frozen=True)
class QueuedDecision:
    decision: OperatorDecision | None = None
    intervention: bool = False
    abort: bool = False
    executed_actions: Any | None = None
    terminal_state: Any | None = None


class DecisionInbox:
    """One-shot operator decision consumed on the next completed chunk."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: QueuedDecision | None = None
        self._pending_started_wall_ns: int | None = None
        self._pending_transition_id: str | None = None
        self._exploration_noise_sigma: float | None = None

    def submit(self, value: QueuedDecision) -> None:
        with self._lock:
            current = self._value or QueuedDecision()
            if current.decision is not None and value.decision is not None:
                raise BridgeError("an operator decision is already queued")
            if (
                current.executed_actions is not None
                and value.executed_actions is not None
            ):
                raise BridgeError("an execution receipt is already queued")
            if current.terminal_state is not None and value.terminal_state is not None:
                raise BridgeError("a terminal state is already queued")
            self._value = QueuedDecision(
                decision=value.decision or current.decision,
                intervention=current.intervention or value.intervention,
                abort=current.abort or value.abort,
                executed_actions=(
                    value.executed_actions
                    if value.executed_actions is not None
                    else current.executed_actions
                ),
                terminal_state=(
                    value.terminal_state
                    if value.terminal_state is not None
                    else current.terminal_state
                ),
            )

    def pop(self) -> QueuedDecision:
        with self._lock:
            value, self._value = self._value, None
        return value or QueuedDecision()

    def pop_terminal(self) -> QueuedDecision | None:
        with self._lock:
            value = self._value
            is_terminal = value is not None and (
                value.abort
                or (
                    value.decision is not None
                    and value.decision.kind in {"success", "failure"}
                )
            )
            if not is_terminal:
                return None
            self._value = None
            return value

    def mark_pending(self, transition_id: str) -> None:
        with self._lock:
            self._pending_started_wall_ns = time.time_ns()
            self._pending_transition_id = str(transition_id)

    def clear_pending(self) -> None:
        with self._lock:
            self._pending_started_wall_ns = None
            self._pending_transition_id = None

    def set_exploration_noise_sigma(self, value: Any) -> float | None:
        """Override actor noise on the next ``act``. ``None`` restores server default."""
        if value is None or value == "" or str(value).strip().lower() == "default":
            sigma: float | None = None
        else:
            sigma = float(value)
            if sigma < 0.0:
                raise BridgeError("exploration_noise_sigma must be >= 0")
        with self._lock:
            self._exploration_noise_sigma = sigma
        return sigma

    def exploration_noise_sigma(self) -> float | None:
        with self._lock:
            return self._exploration_noise_sigma

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pending": self._pending_transition_id is not None,
                "pending_transition_id": self._pending_transition_id,
                "pending_started_wall_ns": self._pending_started_wall_ns,
                "decision_queued": self._value is not None,
                "exploration_noise_sigma": self._exploration_noise_sigma,
            }


def parse_operator_command(payload: Mapping[str, Any]) -> QueuedDecision:
    command = str(payload.get("command", "")).strip().lower()
    if command == "intervention":
        return QueuedDecision(
            intervention=True, executed_actions=payload.get("action_chunk")
        )
    if command == "abort":
        return QueuedDecision(abort=True)
    if command in {"success", "failure", "rewind_exit", "rewind_credit"}:
        return QueuedDecision(
            decision=OperatorDecision(
                kind=command,
                chunks=int(payload.get("chunks", 1)),
                terminal_reward=float(payload.get("terminal_reward", 0.0)),
                prefix_reward=float(payload.get("prefix_reward", 0.1)),
                confidence=float(payload.get("confidence", 1.0)),
            ),
            intervention=bool(payload.get("intervention", False)),
            terminal_state=payload.get("terminal_state"),
        )
    raise BridgeError(
        "command must be success, failure, intervention, abort, "
        "rewind_exit, or rewind_credit"
    )


class RLTBridgeCore:
    """Testable transaction coordinator, independent of WebSocket serving."""

    def __init__(
        self,
        upstream: RLTWebSocketClient,
        *,
        task: str,
        chunk_length: int,
        probe_only: bool,
        inbox: DecisionInbox,
    ) -> None:
        self.upstream = upstream
        self.task = task
        self.probe_only = bool(probe_only)
        self.inbox = inbox
        self.session = RLTSession(
            upstream.request,
            upstream.metadata,
            task=task,
            chunk_length=chunk_length,
        )
        self.closed = False
        self._last_rlt_observation: dict[str, Any] | None = None

    def process_observation(
        self, observation: Mapping[str | bytes, Any]
    ) -> tuple[dict[str, Any] | None, bool]:
        """Commit the preceding chunk and optionally return the next chunk."""
        if self.closed:
            raise BridgeError("episode is already closed")
        receipt_actions, receipt_intervention = execution_receipt(
            observation, chunk_length=self.session.chunk_length
        )
        rlt_observation = observation_to_rlt(observation, task=self.task)
        self._last_rlt_observation = rlt_observation
        self.session.exploration_noise_sigma = self.inbox.exploration_noise_sigma()
        if self.session.pending is not None:
            queued = self.inbox.pop()
            if queued.abort:
                self.session.discard()
                self.session.episode_end(aborted=True)
                self.inbox.clear_pending()
                self.closed = True
                return None, True
            if queued.executed_actions is not None:
                if receipt_actions is not None:
                    raise BridgeError("execution receipt supplied by two sources")
                self.session.set_executed_actions(queued.executed_actions)
                receipt_actions = self.session.pending.executed_actions
                receipt_intervention = queued.intervention
            if receipt_actions is not None:
                self.session.set_executed_actions(receipt_actions)
            if queued.intervention and not receipt_intervention:
                raise BridgeError(
                    "intervention was marked but the client supplied no actual "
                    "EE14 execution receipt"
                )
            next_observation = rlt_observation
            has_rewind_state = (
                queued.decision is not None
                and queued.decision.kind == "rewind_exit"
                and queued.terminal_state is not None
            )
            if has_rewind_state:
                state = np.asarray(queued.terminal_state, dtype=np.float32).reshape(-1)
                if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
                    raise BridgeError(
                        "physical rewind terminal_state must be finite EE14"
                    )
                next_observation = dict(rlt_observation)
                next_observation["state"] = state
            self.session.commit(
                next_observation,
                queued.decision,
                intervention=receipt_intervention,
                info={
                    "transport": "x2robot-desktop-client",
                    "execution_receipt": receipt_actions is not None,
                    "physical_rewind_state_receipt": has_rewind_state,
                },
            )
            self.inbox.clear_pending()
            if queued.decision and queued.decision.kind in {"success", "failure"}:
                self.session.episode_end(
                    success=queued.decision.kind == "success", aborted=False
                )
                self.closed = True
                return None, True
        actions = self.session.request_action(rlt_observation)
        assert self.session.pending is not None
        self.inbox.mark_pending(self.session.pending.transition_id)
        if self.probe_only:
            # Shape-check upstream output, but never place it on the robot socket.
            self.session.discard()
            self.session.episode_end(aborted=True)
            self.inbox.clear_pending()
            self.closed = True
            return None, True
        return actions, False

    def finish_queued_terminal(self) -> bool:
        """Commit success/failure without waiting for a new DesktopClient request."""
        queued = self.inbox.pop_terminal()
        if queued is None:
            return False
        if queued.abort:
            self.abort()
            return True
        if self.session.pending is None:
            raise BridgeError("terminal decision has no pending RLT chunk")
        if self._last_rlt_observation is None:
            raise BridgeError("terminal decision has no cached observation")
        if queued.executed_actions is not None:
            self.session.set_executed_actions(queued.executed_actions)
        if queued.intervention and queued.executed_actions is None:
            raise BridgeError("terminal intervention has no actual EE14 receipt")
        next_observation = dict(self._last_rlt_observation)
        has_terminal_state = queued.terminal_state is not None
        if has_terminal_state:
            state = np.asarray(queued.terminal_state, dtype=np.float32).reshape(-1)
            if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
                raise BridgeError("terminal_state must be finite EE14")
            next_observation["state"] = state
        assert queued.decision is not None
        self.session.commit(
            next_observation,
            queued.decision,
            intervention=queued.intervention,
            info={
                "transport": "x2robot-v2-terminal-event",
                "execution_receipt": queued.executed_actions is not None,
                "terminal_state_receipt": has_terminal_state,
                "terminal_images": "last_observation",
            },
        )
        self.inbox.clear_pending()
        self.session.episode_end(
            success=queued.decision.kind == "success", aborted=False
        )
        self.closed = True
        return True

    def abort(self) -> None:
        if self.closed:
            return
        self.session.discard()
        self.session.episode_end(aborted=True)
        self.inbox.clear_pending()
        self.closed = True


async def _operator_handler(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    inbox: DecisionInbox,
) -> None:
    try:
        raw = await reader.readline()
        if not raw or len(raw) > 262144:
            raise BridgeError("operator command must be one JSON line under 256 KiB")
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, Mapping):
            raise BridgeError("operator command must be a JSON object")
        command = str(decoded.get("command", "")).strip().lower()
        if command == "status":
            reply = {"ok": True, **inbox.status()}
        elif command == "sigma":
            inbox.set_exploration_noise_sigma(
                decoded.get("sigma", decoded.get("value"))
            )
            reply = {"ok": True, **inbox.status()}
        else:
            inbox.submit(parse_operator_command(decoded))
            reply = {"ok": True, "queued": decoded.get("command")}
    except Exception as exc:
        reply = {"ok": False, "error": str(exc)}
    writer.write((json.dumps(reply) + "\n").encode("utf-8"))
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def _serve_robot(
    websocket: Any,
    *,
    args: argparse.Namespace,
    inbox: DecisionInbox,
) -> None:
    peer = getattr(websocket, "remote_address", "unknown")
    LOG.info("DesktopClient connected from %s", peer)
    upstream = await asyncio.to_thread(
        RLTWebSocketClient,
        args.upstream_uri,
        connect_timeout=args.connect_timeout,
        recv_timeout=args.recv_timeout,
    )
    core: RLTBridgeCore | None = None
    try:
        core = RLTBridgeCore(
            upstream,
            task=args.task,
            chunk_length=args.chunk_length,
            probe_only=args.probe_only,
            inbox=inbox,
        )
        await websocket.send(
            _legacy_packb(
                {
                    "protocol": "x2robot-rlt-ee-bridge/v1",
                    "robot_type": "x2robot",
                    "control_mode": "end_pose",
                    "action_schema": ACTION_SCHEMA,
                    "action_dim": ACTION_DIM,
                    "chunk_length": args.chunk_length,
                    "camera_keys": list(CAMERA_KEYS),
                    "probe_only": args.probe_only,
                },
                use_bin_type=True,
            )
        )
        while True:
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=0.2)
            except asyncio.TimeoutError:
                terminal = await asyncio.to_thread(core.finish_queued_terminal)
                if terminal:
                    await websocket.close(code=1000, reason="RLT episode ended")
                    return
                continue
            if not isinstance(raw, bytes):
                raise BridgeError("DesktopClient request must be binary MessagePack")
            observation = _legacy_unpackb(raw, raw=False)
            if not isinstance(observation, Mapping):
                raise BridgeError("DesktopClient observation must be a mapping")
            actions, terminal = await asyncio.to_thread(
                core.process_observation, observation
            )
            if terminal:
                await websocket.close(code=1000, reason="RLT episode ended")
                return
            await websocket.send(_legacy_packb(actions, use_bin_type=True))
    except websockets.ConnectionClosed:
        LOG.info("DesktopClient disconnected from %s", peer)
    except Exception as exc:
        LOG.exception("RLT bridge rejected request")
        await websocket.close(code=1011, reason=str(exc)[:120])
    finally:
        if core is not None and not core.closed:
            try:
                await asyncio.to_thread(core.abort)
            except Exception:
                LOG.exception("could not abort incomplete RLT episode")
        upstream.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-uri", required=True, help="ws://server:port")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=33057)
    parser.add_argument("--operator-host", default="127.0.0.1")
    parser.add_argument("--operator-port", type=int, default=33058)
    parser.add_argument("--task", default="put ring on the rod")
    parser.add_argument("--chunk-length", type=int, default=50)
    parser.add_argument("--connect-timeout", type=float, default=600.0)
    parser.add_argument("--recv-timeout", type=float, default=900.0)
    parser.add_argument(
        "--exploration-noise-sigma",
        type=float,
        default=None,
        help="per-act actor noise override; omit to use the server default",
    )
    parser.add_argument(
        "--probe-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="validate one round trip without forwarding actions",
    )
    args = parser.parse_args()
    if not args.upstream_uri.startswith(("ws://", "wss://")):
        parser.error("--upstream-uri must start with ws:// or wss://")
    for name in ("listen_port", "operator_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be in 1..65535")
    if args.chunk_length <= 0:
        parser.error("--chunk-length must be positive")
    return args


async def _main_async(args: argparse.Namespace) -> None:
    inbox = DecisionInbox()
    if args.exploration_noise_sigma is not None:
        inbox.set_exploration_noise_sigma(args.exploration_noise_sigma)
    operator_server = await asyncio.start_server(
        lambda reader, writer: _operator_handler(reader, writer, inbox),
        args.operator_host,
        args.operator_port,
    )
    async with (
        operator_server,
        websockets.serve(
            lambda websocket: _serve_robot(websocket, args=args, inbox=inbox),
            args.listen_host,
            args.listen_port,
            compression=None,
            max_size=None,
            ping_interval=20.0,
            ping_timeout=600.0,
        ),
    ):
        LOG.info(
            "bridge ready: downstream ws://%s:%d, upstream %s, probe_only=%s",
            args.listen_host,
            args.listen_port,
            args.upstream_uri,
            args.probe_only,
        )
        LOG.info(
            "operator control ready on tcp://%s:%d",
            args.operator_host,
            args.operator_port,
        )
        await asyncio.Future()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_main_async(args))
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
