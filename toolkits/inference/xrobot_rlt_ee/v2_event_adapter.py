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

"""Translate V2 takeover events into pending RLT operator transactions."""

from __future__ import annotations

import argparse
import json
import logging
import math
import socket
from typing import Any, Mapping

from .v2_receipt import receipt_from_v2_history, state_from_v2_sample

LOG = logging.getLogger("x2robot_rlt.v2_adapter")


class BridgeControlClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        wire = (json.dumps(dict(payload), separators=(",", ":")) + "\n").encode()
        with socket.create_connection((self.host, self.port), timeout=5.0) as stream:
            stream.sendall(wire)
            reply = stream.makefile("rb").readline(262144)
        decoded = json.loads(reply.decode("utf-8"))
        if not isinstance(decoded, dict) or not decoded.get("ok"):
            raise RuntimeError(f"RLT bridge rejected V2 event: {decoded!r}")
        return decoded


_HEALTHY_ROLLBACK_OUTCOMES = {
    "operator_stop",
    "history_exhausted",
    "max_duration",
}


# V2 scoring keys, mirroring the rlt-openpi remote-franka client p / o / x.
_SCORE_EVENTS = {
    "session_progress": "progress",
    "session_small_progress": "small_progress",
    "session_regress": "regress",
}


def _terminal_ee14(detail: Any, sample_key: str, event_name: str) -> Any:
    sample = detail.get(sample_key) if isinstance(detail, Mapping) else None
    if not isinstance(sample, Mapping):
        raise ValueError(f"{event_name} has no {sample_key} sample")
    return state_from_v2_sample(sample)


def _physical_rewind_completion(detail: Any) -> dict[str, Any]:
    """Validate a completed V2 physical rollback and extract its RLT receipt."""
    if not isinstance(detail, Mapping):
        raise ValueError("rollback_control_stopped has no detail mapping")
    runtime_stop = detail.get("runtime_stop")
    if not isinstance(runtime_stop, Mapping):
        raise ValueError("rollback completion has no runtime_stop")
    if runtime_stop.get("ok") is not True:
        raise ValueError(
            "rollback runtime failed: {}".format(runtime_stop.get("outcome", "unknown"))
        )
    outcome = str(runtime_stop.get("outcome", ""))
    if outcome not in _HEALTHY_ROLLBACK_OUTCOMES:
        raise ValueError("rollback stopped with unsafe outcome {!r}".format(outcome))

    quality = detail.get("quality")
    if not isinstance(quality, Mapping) or quality.get("settled") is not True:
        raise ValueError("rollback did not settle on its logical forward-path anchor")
    alignment = detail.get("path_alignment")
    if (
        isinstance(alignment, Mapping)
        and alignment.get("required") is True
        and alignment.get("success") is not True
    ):
        raise ValueError(
            "required path alignment failed: {}".format(
                alignment.get("error", "unknown")
            )
        )

    logical = detail.get("logical_source")
    if not isinstance(logical, Mapping):
        raise ValueError("rollback completion has no logical_source sample")
    logical_sample = dict(logical)
    # The V2 ring is end_pose-only. Older events omitted this redundant field.
    logical_sample.setdefault("source_kind", "end_pose")
    terminal_state = state_from_v2_sample(logical_sample)

    raw_runtime_detail = runtime_stop.get("detail")
    duration_value = detail.get("duration_s", runtime_stop.get("duration_s", 0.0))
    if (
        not duration_value
        and isinstance(raw_runtime_detail, Mapping)
        and raw_runtime_detail.get("duration_s") is not None
    ):
        duration_value = raw_runtime_detail["duration_s"]
    duration = float(duration_value or 0.0)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("rollback duration is invalid: {!r}".format(duration_value))
    return {
        "duration_s": duration,
        "terminal_state": terminal_state,
        "rollback_id": str(detail.get("rollback_id", "")),
    }


class V2EventAdapter:
    """Convert authoritative V2 physical rollback completion events.

    V2 owns all physical rollback motion. This adapter only reports a
    rewind_exit after V2 has aligned to a real forward-path sample and
    declared the result healthy. It never publishes arm commands.
    """

    def __init__(
        self,
        control: BridgeControlClient,
        *,
        chunk_length: int = 50,
        control_hz: float = 30.0,
        rewind_mode: str = "rewind_exit",
    ) -> None:
        if rewind_mode not in {"rewind_credit", "rewind_exit"}:
            raise ValueError("rewind_mode must be rewind_credit or rewind_exit")
        self.control = control
        self.chunk_length = int(chunk_length)
        self.chunk_duration_s = self.chunk_length / float(control_hz)
        self.rewind_mode = rewind_mode
        self._seen: set[str] = set()

    def _pending_status(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        status = self.control.request({"command": "status"})
        if not status.get("pending"):
            return None
        pending_start = status.get("pending_started_wall_ns")
        event_wall = event.get("t_wall_ns")
        if (
            isinstance(pending_start, int)
            and isinstance(event_wall, int)
            and event_wall < pending_start
        ):
            return None
        return status

    def handle(self, event: Mapping[str, Any]) -> dict[str, Any]:
        event_id = str(event.get("event_id", ""))
        if not event_id:
            raise ValueError("V2 event_id is required")
        if event_id in self._seen:
            return {"ok": True, "ignored": "duplicate", "event_id": event_id}
        name = str(event.get("event", ""))
        result: dict[str, Any]
        pending_status = None
        if name in {
            "policy_control_stopped",
            "rollback_control_stopped",
            "session_succeeded",
            "session_failed",
            "session_aborted",
        }:
            pending_status = self._pending_status(event)
            if pending_status is None:
                self._seen.add(event_id)
                return {"ok": True, "ignored": "no_matching_pending_rlt_chunk"}
        if name == "policy_control_stopped":
            status = pending_status
            assert status is not None
            if not status.get("pending"):
                result = {"ok": True, "ignored": "no_pending_rlt_chunk"}
            else:
                detail = event.get("detail")
                history = (
                    detail.get("frozen_history")
                    if isinstance(detail, Mapping)
                    else None
                )
                if not isinstance(history, list):
                    raise ValueError(
                        "policy_control_stopped has no frozen_history list"
                    )
                receipt = receipt_from_v2_history(
                    history,
                    chunk_length=self.chunk_length,
                    start_wall_ns=status.get("pending_started_wall_ns"),
                )
                result = self.control.request(
                    {
                        "command": "intervention",
                        "action_chunk": receipt.action_chunk.tolist(),
                        # Last measured pose of the takeover. The normal commit
                        # path uses the fresh observation and ignores this; it
                        # is what lets an operator `submit` close the chunk
                        # while the policy is still paused.
                        "terminal_state": receipt.action_chunk[-1].tolist(),
                        "source_samples": receipt.source_samples,
                        "source_start_wall_ns": receipt.source_start_wall_ns,
                        "source_end_wall_ns": receipt.source_end_wall_ns,
                        "v2_event_id": event_id,
                    }
                )
        elif name == "rollback_control_stopped":
            detail = event.get("detail")
            try:
                completion = _physical_rewind_completion(detail)
            except (TypeError, ValueError, RuntimeError) as error:
                LOG.error("V2 physical rollback %s is invalid: %s", event_id, error)
                result = self.control.request(
                    {
                        "command": "abort",
                        "reason": "v2_physical_rewind_failed",
                        "detail": str(error),
                        "v2_event_id": event_id,
                    }
                )
            else:
                chunks = max(
                    1,
                    int(math.ceil(completion["duration_s"] / self.chunk_duration_s)),
                )
                result = self.control.request(
                    {
                        "command": self.rewind_mode,
                        "chunks": chunks,
                        "terminal_reward": -1.0,
                        "terminal_state": completion["terminal_state"].tolist(),
                        "physical_rewind_completed": True,
                        "rollback_id": completion["rollback_id"],
                        "v2_event_id": event_id,
                    }
                )
        elif name == "session_succeeded":
            result = self.control.request(
                {
                    "command": "success",
                    "terminal_state": _terminal_ee14(
                        event.get("detail"), "success_end", "session_succeeded"
                    ).tolist(),
                    "v2_event_id": event_id,
                }
            )
        elif name == "session_failed":
            result = self.control.request(
                {
                    "command": "failure",
                    "terminal_state": _terminal_ee14(
                        event.get("detail"), "failure_end", "session_failed"
                    ).tolist(),
                    "v2_event_id": event_id,
                }
            )
        elif name == "session_aborted":
            result = self.control.request({"command": "abort", "v2_event_id": event_id})
        elif name in _SCORE_EVENTS:
            # Mid-episode operator score; the rollout continues. Deliberately
            # not gated on a pending chunk: a press between two chunks waits in
            # the inbox and lands on whichever chunk commits next.
            payload = {"command": _SCORE_EVENTS[name], "v2_event_id": event_id}
            detail = event.get("detail")
            if isinstance(detail, Mapping) and detail.get("reward") is not None:
                payload["reward"] = float(detail["reward"])
            result = self.control.request(payload)
        elif name == "session_submit":
            result = self.control.request(
                {"command": "submit", "v2_event_id": event_id}
            )
        else:
            result = {"ok": True, "ignored": "unmapped_event", "event": name}
        self._seen.add(event_id)
        return result


def _run_ros(args: argparse.Namespace, adapter: V2EventAdapter) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=50,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    class AdapterNode(Node):
        def __init__(self) -> None:
            super().__init__("x2robot_rlt_v2_event_adapter")
            self.create_subscription(String, args.ros_topic, self._event, qos)

        def _event(self, message: String) -> None:
            try:
                payload = json.loads(message.data)
                if not isinstance(payload, Mapping):
                    raise ValueError("V2 ROS event must be a JSON object")
                result = adapter.handle(payload)
                LOG.info("event %s: %s", payload.get("event"), result)
            except Exception:
                LOG.exception("failed to adapt V2 event")

    rclpy.init()
    node = AdapterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=33058)
    parser.add_argument("--ros-topic", default="/take_over_data")
    parser.add_argument("--chunk-length", type=int, default=50)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--rewind-mode", choices=("rewind_credit", "rewind_exit"), default="rewind_exit"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    adapter = V2EventAdapter(
        BridgeControlClient(args.bridge_host, args.bridge_port),
        chunk_length=args.chunk_length,
        control_hz=args.control_hz,
        rewind_mode=args.rewind_mode,
    )
    return _run_ros(args, adapter)


if __name__ == "__main__":
    raise SystemExit(main())
