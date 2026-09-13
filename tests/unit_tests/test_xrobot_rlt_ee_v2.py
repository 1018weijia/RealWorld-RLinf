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

#!/usr/bin/env python3
"""Offline tests for the X2Robot EE14 RLT adapter (never touches ROS)."""

from __future__ import annotations

import base64
import sys
import types
import unittest

import cv2
import numpy as np

try:
    import websockets  # noqa: F401
except ImportError:
    sys.modules["websockets"] = types.SimpleNamespace(ConnectionClosed=Exception)

from toolkits.inference.xrobot_rlt_ee import codec
from toolkits.inference.xrobot_rlt_ee.bridge import (
    DecisionInbox,
    QueuedDecision,
    RLTBridgeCore,
)
from toolkits.inference.xrobot_rlt_ee.mapping import (
    execution_receipt,
    observation_to_rlt,
    split_ee_actions,
)
from toolkits.inference.xrobot_rlt_ee.protocol import PROTOCOL_VERSION, ProtocolError
from toolkits.inference.xrobot_rlt_ee.session import OperatorDecision, RLTSession
from toolkits.inference.xrobot_rlt_ee.v2_event_adapter import V2EventAdapter
from toolkits.inference.xrobot_rlt_ee.v2_receipt import receipt_from_v2_history


def _jpeg(value: int) -> str:
    image = np.full((12, 20, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def desktop_observation() -> dict:
    return {
        "state": {
            "follow_left_position": [0, 1, 2],
            "follow_left_rotation": [3, 4, 5],
            "follow_left_gripper": [6],
            "follow_right_position": [7, 8, 9],
            "follow_right_rotation": [10, 11, 12],
            "follow_right_gripper": [13],
        },
        "views": {
            "camera_front": _jpeg(10),
            "camera_left": _jpeg(20),
            "camera_right": _jpeg(30),
        },
    }


def metadata(**overrides) -> dict:
    result = {
        "protocol": PROTOCOL_VERSION,
        "action_dim": 14,
        "chunk_length": 2,
        "proprio_dim": 14,
        "camera_keys": ["image", "wrist_image", "side_image"],
        "action_space": "robot",
        "robot_type": "x2robot",
        "action_schema": "x2robot-ee14-v1",
    }
    result.update(overrides)
    return result


class MappingTest(unittest.TestCase):
    def test_observation_order_and_images(self) -> None:
        converted = observation_to_rlt(desktop_observation(), task="task")
        np.testing.assert_array_equal(converted["state"], np.arange(14))
        self.assertEqual(converted["image"].shape, (224, 224, 3))
        self.assertEqual(converted["prompt"], "task")

    def test_actions_are_split_and_grippers_clipped(self) -> None:
        actions = np.arange(28, dtype=np.float32).reshape(2, 14)
        output, actual = split_ee_actions(actions, chunk_length=2)
        self.assertEqual(output["follow_left_position"].shape, (2, 3))
        np.testing.assert_array_equal(output["follow_right_rotation"], actual[:, 10:13])
        self.assertTrue(np.all(actual[:, [6, 13]] <= 4.6))

    def test_intervention_requires_actual_action_receipt(self) -> None:
        observation = desktop_observation()
        observation["rlt_execution"] = {"intervention": True}
        with self.assertRaisesRegex(Exception, "missing action_chunk"):
            execution_receipt(observation, chunk_length=2)


class CodecTest(unittest.TestCase):
    def test_openpi_numpy_round_trip(self) -> None:
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        decoded = codec.unpackb(codec.packb({"array": source}))
        np.testing.assert_array_equal(decoded["array"], source)


class SessionTest(unittest.TestCase):
    def test_server_identity_and_executed_chunk_are_committed(self) -> None:
        requests = []

        def request(payload):
            requests.append(payload)
            if payload["rlt/request"] == "act":
                actions = np.zeros((2, 14), dtype=np.float32)
                actions[:, 6] = 99
                return {
                    "actions": actions,
                    "transition_id": "t0",
                    "episode_id": 7,
                    "session_id": 9,
                    "env_id": 0,
                    "chunk_id": 42,
                }
            return {"ok": True}

        session = RLTSession(request, metadata(), task="task", chunk_length=2)
        session.request_action({"state": np.zeros(14)})
        session.commit({"state": np.ones(14)})
        transition = requests[1]
        self.assertEqual(
            (
                transition["episode_id"],
                transition["session_id"],
                transition["chunk_id"],
            ),
            (7, 9, 42),
        )
        self.assertAlmostEqual(float(transition["action_chunk"][0, 6]), 4.6, places=5)

    def test_wrong_action_schema_fails_closed(self) -> None:
        with self.assertRaises(ProtocolError):
            RLTSession(
                lambda _: {},
                metadata(action_schema="cobot-joint14"),
                task="x",
                chunk_length=2,
            )


class V2ReceiptTest(unittest.TestCase):
    @staticmethod
    def _sample(index: int) -> dict:
        return {
            "source_kind": "end_pose",
            "t_wall_ns": 1_000_000_000 + index * 20_000_000,
            "left": {"xyz": [index * 0.001, 0.1, 0.2], "quat_xyzw": [0, 0, 0, 1]},
            "right": {"xyz": [-index * 0.001, -0.1, 0.3], "quat_xyzw": [0, 0, 0, 1]},
            "left_gripper": [1.0 + index * 0.01],
            "right_gripper": [2.0 + index * 0.01],
        }

    def test_resamples_measured_end_pose_to_ee14(self) -> None:
        history = [self._sample(index) for index in range(10)]
        receipt = receipt_from_v2_history(
            history, chunk_length=50, start_wall_ns=history[3]["t_wall_ns"]
        )
        self.assertEqual(receipt.action_chunk.shape, (50, 14))
        self.assertEqual(receipt.source_samples, 7)
        self.assertAlmostEqual(float(receipt.action_chunk[0, 0]), 0.003)
        self.assertAlmostEqual(float(receipt.action_chunk[-1, 7]), -0.009)
        self.assertAlmostEqual(float(receipt.action_chunk[-1, 13]), 2.09, places=5)

    def test_rejects_pose_command_instead_of_feedback(self) -> None:
        sample = self._sample(0)
        sample["source_kind"] = "pose_cmd"
        with self.assertRaisesRegex(Exception, "source_kind=end_pose"):
            receipt_from_v2_history([sample], chunk_length=50)


class FakeUpstream:
    def __init__(self) -> None:
        self.metadata = metadata()
        self.requests = []

    def request(self, payload):
        self.requests.append(payload)
        if payload["rlt/request"] == "act":
            return {
                "actions": np.zeros((2, 14), dtype=np.float32),
                "transition_id": "bridge-t0",
                "episode_id": 3,
                "session_id": 4,
                "chunk_id": 5,
            }
        return {"ok": True}


class BridgeTerminalTest(unittest.TestCase):
    def test_v2_success_finishes_without_next_desktop_observation(self) -> None:
        upstream = FakeUpstream()
        inbox = DecisionInbox()
        core = RLTBridgeCore(
            upstream,
            task="task",
            chunk_length=2,
            probe_only=False,
            inbox=inbox,
        )
        core.process_observation(desktop_observation())
        terminal_state = np.arange(14, dtype=np.float32)
        inbox.submit(
            QueuedDecision(
                decision=OperatorDecision(kind="success"),
                terminal_state=terminal_state,
            )
        )
        self.assertTrue(core.finish_queued_terminal())
        self.assertTrue(core.closed)
        self.assertEqual(
            [item["rlt/request"] for item in upstream.requests],
            ["act", "transition", "episode_end"],
        )
        transition = upstream.requests[1]
        self.assertTrue(transition["done"])
        np.testing.assert_array_equal(
            transition["next_observation"]["state"], terminal_state
        )

    def test_physical_rewind_uses_v2_logical_anchor_as_next_state(self) -> None:
        upstream = FakeUpstream()
        inbox = DecisionInbox()
        core = RLTBridgeCore(
            upstream,
            task="task",
            chunk_length=2,
            probe_only=False,
            inbox=inbox,
        )
        core.process_observation(desktop_observation())
        rewind_state = np.arange(14, dtype=np.float32) + 100.0
        inbox.submit(
            QueuedDecision(
                decision=OperatorDecision(kind="rewind_exit", chunks=2),
                terminal_state=rewind_state,
            )
        )

        core.process_observation(desktop_observation())

        transition = upstream.requests[1]
        np.testing.assert_array_equal(
            transition["next_observation"]["state"], rewind_state
        )
        self.assertTrue(transition["info"]["physical_rewind_state_receipt"])
        self.assertEqual(upstream.requests[2]["rlt/request"], "rewind_exit_correction")


class FakeBridgeControl:
    def __init__(self, start_wall_ns: int, *, pending: bool = True) -> None:
        self.start_wall_ns = start_wall_ns
        self.pending = pending
        self.payloads = []

    def request(self, payload):
        self.payloads.append(payload)
        if payload["command"] == "status":
            return {
                "ok": True,
                "pending": self.pending,
                "pending_started_wall_ns": self.start_wall_ns,
            }
        return {"ok": True}


class V2EventAdapterTest(unittest.TestCase):
    def test_policy_stop_and_rollback_merge_inputs(self) -> None:
        history = [V2ReceiptTest._sample(index) for index in range(10)]
        control = FakeBridgeControl(history[3]["t_wall_ns"])
        adapter = V2EventAdapter(control, chunk_length=50, control_hz=30)
        adapter.handle(
            {
                "event_id": "event-stop",
                "event": "policy_control_stopped",
                "detail": {"frozen_history": history},
            }
        )
        self.assertEqual(control.payloads[-1]["command"], "intervention")
        self.assertEqual(
            np.asarray(control.payloads[-1]["action_chunk"]).shape, (50, 14)
        )
        logical_source = V2ReceiptTest._sample(4)
        adapter.handle(
            {
                "event_id": "event-rewind",
                "event": "rollback_control_stopped",
                "detail": {
                    "rollback_id": "I001-R001",
                    "duration_s": 2.0,
                    "logical_source": logical_source,
                    "quality": {"settled": True},
                    "path_alignment": {"required": True, "success": True},
                    "runtime_stop": {"ok": True, "outcome": "operator_stop"},
                },
            }
        )
        self.assertEqual(control.payloads[-1]["command"], "rewind_exit")
        self.assertEqual(control.payloads[-1]["chunks"], 2)
        self.assertTrue(control.payloads[-1]["physical_rewind_completed"])
        self.assertEqual(len(control.payloads[-1]["terminal_state"]), 14)

    def test_physical_rewind_failure_aborts_pending_rlt_chunk(self) -> None:
        sample = V2ReceiptTest._sample(0)
        control = FakeBridgeControl(sample["t_wall_ns"])
        adapter = V2EventAdapter(control)
        adapter.handle(
            {
                "event_id": "event-rewind-fault",
                "event": "rollback_control_stopped",
                "detail": {
                    "rollback_id": "I001-R001",
                    "logical_source": sample,
                    "quality": {"settled": False},
                    "runtime_stop": {"ok": False, "outcome": "safety_abort"},
                },
            }
        )
        self.assertEqual(control.payloads[-1]["command"], "abort")
        self.assertEqual(control.payloads[-1]["reason"], "v2_physical_rewind_failed")

    def test_success_carries_terminal_ee14(self) -> None:
        sample = V2ReceiptTest._sample(0)
        control = FakeBridgeControl(sample["t_wall_ns"])
        adapter = V2EventAdapter(control, chunk_length=50, control_hz=30)
        adapter.handle(
            {
                "event_id": "event-success",
                "event": "session_succeeded",
                "detail": {"success_end": sample},
            }
        )
        self.assertEqual(control.payloads[-1]["command"], "success")
        self.assertEqual(len(control.payloads[-1]["terminal_state"]), 14)

    def test_stale_terminal_is_ignored_and_current_abort_is_forwarded(self) -> None:
        start_wall_ns = 2_000_000_000
        control = FakeBridgeControl(start_wall_ns)
        adapter = V2EventAdapter(control)
        stale = adapter.handle(
            {
                "event_id": "old-success",
                "event": "session_succeeded",
                "t_wall_ns": start_wall_ns - 1,
                "detail": {},
            }
        )
        self.assertEqual(stale["ignored"], "no_matching_pending_rlt_chunk")
        self.assertEqual(control.payloads[-1]["command"], "status")
        adapter.handle(
            {
                "event_id": "current-abort",
                "event": "session_aborted",
                "t_wall_ns": start_wall_ns + 1,
            }
        )
        self.assertEqual(control.payloads[-1]["command"], "abort")


if __name__ == "__main__":
    unittest.main()()()
