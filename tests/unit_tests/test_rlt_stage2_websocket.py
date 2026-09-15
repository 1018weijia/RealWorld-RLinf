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

"""CPU tests for the RLT Stage 2 WebSocket architecture.

Everything here runs without a GPU, a robot or a Stage 1 checkpoint. The
Stage 1 encoder and the Stage 2 trainer are replaced by fakes that record what
they were asked to do, which is enough to pin the properties that actually
break real runs: a chunk must never enter replay twice, a discarded chunk must
never enter at all, the update budget must count only policy-driven chunks,
and the normalized/robot action conversion must round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.serving.rlt.inference import (
    ActionSelection,
    CameraLayout,
    RLTObservationRepacker,
)
from rlinf.serving.rlt.policy import RLTStage2Policy
from rlinf.serving.rlt.preflight import (
    PreflightError,
    check_camera_layout,
    check_norm_stats_configured,
    check_stage2_algorithm,
    check_task_prompt,
    read_rlt_prefix_seq_len,
    resolve_stage1_weights,
)
from rlinf.serving.rlt.protocol import (
    ACTION_SPACE_NORMALIZED,
    ACTION_SPACE_ROBOT,
    REQUEST_ACT,
    REQUEST_DISCARD,
    REQUEST_EPISODE_END,
    REQUEST_KEY,
    REQUEST_RESET,
    REQUEST_REWIND_CREDIT,
    REQUEST_REWIND_EXIT,
    REQUEST_STATUS,
    REQUEST_TRANSITION,
    ActRequest,
    ChunkIdentity,
    ProtocolError,
    ServerMetadata,
    TransitionRequest,
    decode_request_type,
    validate_server_metadata,
)

CHUNK_LEN = 4
ACTION_DIM = 3
PROPRIO_DIM = 5
CAMERA_KEYS = ("image", "wrist_image", "side_image")


# --------------------------------------------------------------------- fakes


class FakeInference:
    """Stand-in for :class:`RLTStage2Inference` with an affine action space.

    Robot space is ``2 * normalized + 1``, so the round trip has a non-trivial
    scale and offset and a sign error cannot pass unnoticed.
    """

    SCALE = 2.0
    OFFSET = 1.0

    def __init__(self) -> None:
        self.encode_calls = 0
        self.select_calls: list[dict] = []

    def encode(self, observation):
        self.encode_calls += 1
        return {
            "z_rl": np.zeros(8, dtype=np.float32),
            "proprio": np.asarray(observation["state"], dtype=np.float32),
            "ref_chunk": np.full((CHUNK_LEN, ACTION_DIM), 0.25, dtype=np.float32),
            "_scratch": "dropped by strip_private",
        }

    @staticmethod
    def strip_private(rlt_obs):
        return {k: v for k, v in rlt_obs.items() if not k.startswith("_")}

    def select_action(self, rlt_obs, *, warmup, deterministic, **_):
        self.select_calls.append({"warmup": warmup, "deterministic": deterministic})
        reference = rlt_obs["ref_chunk"]
        normalized = reference if warmup else reference + 0.1
        mode = "warmup" if warmup else ("eval" if deterministic else "actor")
        return ActionSelection(
            normalized_chunk=normalized.astype(np.float32),
            reference_chunk=reference.astype(np.float32),
            robot_chunk=self.to_robot_space(normalized, rlt_obs),
            mode=mode,
            expo_info={"expo_selected_index": 0},
        )

    def to_robot_space(self, normalized_chunk, rlt_obs):
        del rlt_obs
        return (np.asarray(normalized_chunk) * self.SCALE + self.OFFSET).astype(
            np.float32
        )

    def to_normalized_space(self, robot_chunk, rlt_obs):
        del rlt_obs
        return ((np.asarray(robot_chunk) - self.OFFSET) / self.SCALE).astype(np.float32)


class FakeBuffer:
    """Minimal replay buffer exposing only what the policy reads."""

    def __init__(self) -> None:
        self.total_samples = 0


class FakeTrainer:
    """Records ingestion and update calls instead of running them."""

    def __init__(self) -> None:
        self.replay_buffer = FakeBuffer()
        self.demo_buffer = None
        self.rewind_preference_buffer: list = []
        self.update_step = 0
        self.transitions: list[dict] = []
        self.rewind_events: list = []
        self.train_calls: list[int] = []
        self.saves: list[str] = []

    def _chunk_shape(self):
        return CHUNK_LEN, ACTION_DIM

    def add_transition(self, **kwargs):
        self.transitions.append(kwargs)
        self.replay_buffer.total_samples += 1
        return True, 1 if kwargs["done"] else 0

    def _ingest_rewind_events(self, events):
        self.rewind_events.extend(events)

    def train(self, num_updates):
        self.train_calls.append(num_updates)
        self.update_step += num_updates
        return {"train/critic_loss": 0.5}

    def save(self, path):
        self.saves.append(path)


def build_policy(**overrides) -> tuple[RLTStage2Policy, FakeTrainer, FakeInference]:
    trainer = FakeTrainer()
    inference = FakeInference()
    metadata = ServerMetadata(
        action_dim=ACTION_DIM,
        chunk_length=CHUNK_LEN,
        proprio_dim=PROPRIO_DIM,
        warmup_steps=overrides.get("warmup_steps", 2),
        run_name="test",
        action_space=ACTION_SPACE_ROBOT,
        replay_action_space=ACTION_SPACE_NORMALIZED,
        action_selection_mode="expo",
        edit_scale=0.2,
        expo_num_base_samples=4,
        expo_num_edit_samples=4,
        actor_action_clip_min=-1.4,
        actor_action_clip_max=1.4,
        max_episode_chunks=150,
        camera_keys=CAMERA_KEYS,
        task_prompt="put cube in drawer",
    )
    kwargs = {
        "warmup_steps": 2,
        "utd_ratio": 3,
        "max_episode_chunks": 150,
        "save_dir": None,
        "save_interval_episodes": 10,
        "eval_only": False,
        "replay_action_space": ACTION_SPACE_NORMALIZED,
    }
    kwargs.update(overrides)
    policy = RLTStage2Policy(
        trainer=trainer, inference=inference, metadata=metadata, **kwargs
    )
    return policy, trainer, inference


def observation_payload() -> dict:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    return {
        "image": frame,
        "wrist_image": frame,
        "side_image": frame,
        "state": np.arange(PROPRIO_DIM, dtype=np.float32),
        "prompt": "put cube in drawer",
    }


def act(policy) -> dict:
    return policy.infer(
        {REQUEST_KEY: REQUEST_ACT, "observation": observation_payload()}
    )


def commit(policy, transition_id, *, done=False, **extra) -> dict:
    payload = {
        REQUEST_KEY: REQUEST_TRANSITION,
        "transition_id": transition_id,
        "next_observation": observation_payload(),
        "rewards": np.zeros(CHUNK_LEN, dtype=np.float32),
        "done": done,
        "bootstrap_mask": 0.0 if done else 1.0,
    }
    payload.update(extra)
    return policy.infer(payload)


# ------------------------------------------------------------------ protocol


def test_request_type_defaults_to_act_and_rejects_unknown():
    assert decode_request_type({"observation": {}}) == REQUEST_ACT
    assert decode_request_type({REQUEST_KEY: REQUEST_STATUS}) == REQUEST_STATUS
    with pytest.raises(ProtocolError):
        decode_request_type({REQUEST_KEY: "definitely_not_a_request"})


def test_act_and_transition_payloads_round_trip():
    identity = ChunkIdentity(episode_id=3, session_id=7, env_id=1, chunk_id=11)
    payload = {
        REQUEST_KEY: REQUEST_ACT,
        "observation": observation_payload(),
        "exploration_noise_sigma": 0.05,
        **identity.to_payload(),
    }
    request = ActRequest.from_payload(payload)
    assert request.exploration_noise_sigma == pytest.approx(0.05)
    assert request.identity == identity

    transition = TransitionRequest.from_payload(
        {
            REQUEST_KEY: REQUEST_TRANSITION,
            "transition_id": "42",
            "next_observation": observation_payload(),
            "rewards": [0.0, 0.0, 0.0, 1.0],
            "done": True,
            "bootstrap_mask": 0.0,
            "intervention": True,
            "action_chunk": np.zeros((CHUNK_LEN, ACTION_DIM), dtype=np.float32),
            "action_chunk_space": ACTION_SPACE_ROBOT,
            **identity.to_payload(),
        }
    )
    assert transition.transition_id == "42"
    assert transition.done is True
    assert transition.bootstrap_mask == pytest.approx(0.0)
    assert transition.intervention is True
    assert transition.action_chunk.shape == (CHUNK_LEN, ACTION_DIM)
    assert transition.identity == identity


def test_handshake_validation_rejects_every_shape_mismatch():
    metadata = build_policy()[0].metadata
    validate_server_metadata(
        metadata,
        action_dim=ACTION_DIM,
        chunk_length=CHUNK_LEN,
        proprio_dim=PROPRIO_DIM,
        camera_keys=CAMERA_KEYS,
    )
    for field, value in (
        ("action_dim", ACTION_DIM + 1),
        ("chunk_length", CHUNK_LEN + 1),
        ("proprio_dim", PROPRIO_DIM + 1),
    ):
        with pytest.raises(ValueError):
            validate_server_metadata(metadata, **{field: value})
    with pytest.raises(ValueError):
        validate_server_metadata(metadata, camera_keys=("image", "wrist_image"))


def test_xrobot_handshake_metadata_identifies_the_embodiment_and_action_schema():
    import dataclasses

    base = build_policy()[0]._metadata
    metadata = dataclasses.replace(
        base, robot_type="x2robot", action_schema="x2robot-ee14-v1"
    ).to_payload()

    assert metadata["robot_type"] == "x2robot"
    assert metadata["action_schema"] == "x2robot-ee14-v1"


def test_xrobot_stage2_server_reads_its_assets_from_the_environment(
    monkeypatch, server_config
):
    # The embodiment contract itself is pinned in test_rlt_embodiment.py; what
    # is specific to this config is that a shared server takes the checkpoint
    # and the norm statistics from the environment rather than a baked path.
    monkeypatch.setenv("XROBOT_RLT_STAGE1_CHECKPOINT", "/weights/global_step_30000")
    monkeypatch.setenv("XROBOT_NORM_STATS", "/weights/assets/norm_stats.json")
    config = server_config("xrobot_ee_rlt_stage2_ws_server")

    resolved = OmegaConf.to_container(config, resolve=True)
    feature_model = resolved["rlt_feature_model"]
    assert feature_model["model_path"] == "/weights/global_step_30000"
    assert feature_model["openpi_data"]["norm_stats_path"] == (
        "/weights/assets/norm_stats.json"
    )


def test_xrobot_usb_overlay_reads_its_assets_from_the_environment(
    monkeypatch, server_config
):
    monkeypatch.setenv("XROBOT_USB_STAGE1_CHECKPOINT", "/weights/usb/global_step_20000")
    monkeypatch.setenv(
        "XROBOT_USB_NORM_STATS", "/weights/assets/usb_plug/norm_stats.json"
    )
    config = server_config("xrobot_usb_plug_rlt_stage2_ws_server")

    resolved = OmegaConf.to_container(config, resolve=True)
    feature_model = resolved["rlt_feature_model"]
    assert feature_model["model_path"] == "/weights/usb/global_step_20000"
    assert feature_model["openpi_data"]["norm_stats_path"] == (
        "/weights/assets/usb_plug/norm_stats.json"
    )
    assert resolved["server"]["task_prompt"] == "Bimanual usb pick and insert"


# -------------------------------------------------------------- pending map


def test_transition_commits_exactly_once():
    policy, trainer, _ = build_policy()
    response = act(policy)
    transition_id = response["transition_id"]

    assert commit(policy, transition_id)["stored"] is True
    assert len(trainer.transitions) == 1

    # A resent transition must not produce a second replay row. Without the
    # pending map this is the duplicate the Ray path needed dedup sets for.
    with pytest.raises(ProtocolError):
        commit(policy, transition_id)
    assert len(trainer.transitions) == 1


def test_discarded_chunk_never_reaches_replay():
    policy, trainer, _ = build_policy()
    transition_id = act(policy)["transition_id"]

    discarded = policy.infer(
        {REQUEST_KEY: REQUEST_DISCARD, "transition_id": transition_id}
    )
    assert discarded["discarded"] is True
    assert trainer.transitions == []

    with pytest.raises(ProtocolError):
        commit(policy, transition_id)
    assert trainer.transitions == []


def test_transition_rejects_identity_that_contradicts_act():
    policy, _, _ = build_policy()
    response = act(policy)
    with pytest.raises(ProtocolError, match="chunk_id"):
        commit(policy, response["transition_id"], chunk_id=response["chunk_id"] + 5)


def test_reset_drops_pending_without_touching_replay():
    policy, trainer, _ = build_policy()
    transition_id = act(policy)["transition_id"]
    assert policy.infer({REQUEST_KEY: REQUEST_RESET})["pending"] == 0
    with pytest.raises(ProtocolError):
        commit(policy, transition_id)
    assert trainer.transitions == []


# ------------------------------------------------------------- action space


def test_executed_robot_chunk_is_normalized_before_replay():
    policy, trainer, inference = build_policy()
    response = act(policy)
    executed_robot = np.asarray(response["actions"], dtype=np.float32) + 0.5

    commit(
        policy,
        response["transition_id"],
        action_chunk=executed_robot,
        action_chunk_space=ACTION_SPACE_ROBOT,
        intervention=True,
    )
    stored = trainer.transitions[0]["action_chunk"]
    np.testing.assert_allclose(
        stored, inference.to_normalized_space(executed_robot, {}), rtol=1e-6
    )
    # Round-tripping back must recover exactly what the robot executed.
    np.testing.assert_allclose(
        inference.to_robot_space(stored, {}), executed_robot, rtol=1e-6
    )


def test_replay_stores_the_reference_expo_actually_edited():
    """The stored ``ref_chunk`` must be the candidate the residual edited.

    EXPO ranks several VLA samples and edits the winner, which is usually not
    the one ``encode`` parked in ``ref_chunk``. Storing the wrong one makes the
    actor loss anchor the residual on a chunk the policy never edited, and the
    BC term regress toward an unrelated sample - silently, with no shape error.
    """
    policy, trainer, _ = build_policy(warmup_steps=0)
    response = act(policy)
    commit(policy, response["transition_id"])

    stored_ref = np.asarray(trainer.transitions[0]["curr_obs"]["ref_chunk"])
    np.testing.assert_allclose(
        stored_ref.reshape(CHUNK_LEN, ACTION_DIM),
        np.asarray(response["reference_actions"], dtype=np.float32),
        rtol=1e-6,
    )


def test_uneventful_chunk_stores_the_handed_out_normalized_chunk():
    policy, trainer, _ = build_policy()
    response = act(policy)
    commit(policy, response["transition_id"])
    np.testing.assert_allclose(
        trainer.transitions[0]["action_chunk"],
        np.asarray(response["normalized_actions"], dtype=np.float32),
    )


def test_short_reward_array_is_zero_padded_to_the_chunk():
    policy, trainer, _ = build_policy()
    response = act(policy)
    commit(policy, response["transition_id"], rewards=np.array([1.0], dtype=np.float32))
    rewards = trainer.transitions[0]["rewards"]
    assert rewards.shape == (CHUNK_LEN,)
    np.testing.assert_allclose(rewards, [1.0, 0.0, 0.0, 0.0])


# ------------------------------------------------------------- warmup / UTD


def test_warmup_gates_the_stage2_head_on_replay_size():
    policy, trainer, inference = build_policy(warmup_steps=2, utd_ratio=1)
    assert policy.in_warmup is True

    for _ in range(2):
        response = act(policy)
        assert response["mode"] == "warmup"
        commit(policy, response["transition_id"])

    # Two rows collected, so the gate opens without any explicit signal.
    assert policy.in_warmup is False
    assert act(policy)["mode"] == "actor"
    assert [call["warmup"] for call in inference.select_calls] == [True, True, False]
    assert trainer.replay_buffer.total_samples == 2


def test_update_budget_counts_only_policy_driven_chunks():
    policy, trainer, _ = build_policy(warmup_steps=2, utd_ratio=3)

    # Two warmup chunks earn no gradient steps.
    for _ in range(2):
        commit(policy, act(policy)["transition_id"])
    policy.infer({REQUEST_KEY: REQUEST_EPISODE_END, "stats": {}})
    assert trainer.train_calls == []

    # Three actor chunks at utd_ratio=3 earn nine.
    for _ in range(3):
        commit(policy, act(policy)["transition_id"])
    response = policy.infer({REQUEST_KEY: REQUEST_EPISODE_END, "stats": {}})
    assert trainer.train_calls == [9]
    assert response["updates_run"] == 9
    assert response["episode_chunks"] == 0, "counters reset for the next episode"


def test_eval_only_never_writes_replay_or_trains():
    policy, trainer, _ = build_policy(eval_only=True)
    assert policy.in_warmup is False
    response = act(policy)
    assert response["mode"] == "eval"
    assert commit(policy, response["transition_id"])["stored"] is False
    policy.infer({REQUEST_KEY: REQUEST_EPISODE_END, "stats": {"success": True}})
    assert trainer.transitions == []
    assert trainer.train_calls == []


def test_eval_episode_writes_replay_without_updates():
    policy, trainer, inference = build_policy(
        warmup_steps=0,
        eval_interval_episodes=1,
        store_eval_episodes=True,
    )
    commit(policy, act(policy)["transition_id"])
    end = policy.infer({REQUEST_KEY: REQUEST_EPISODE_END, "stats": {}})
    assert end["eval_pending"] is True
    assert trainer.train_calls == [3]

    response = act(policy)
    assert response["mode"] == "eval"
    assert response["is_eval_episode"] is True
    stored = commit(policy, response["transition_id"])
    assert stored["stored"] is True
    assert len(trainer.transitions) == 2
    end_eval = policy.infer(
        {REQUEST_KEY: REQUEST_EPISODE_END, "stats": {"success": True}}
    )
    assert end_eval["updates_run"] == 0
    assert trainer.train_calls == [3]
    assert end_eval["total_eval_episodes"] == 1
    assert inference.select_calls[-1]["deterministic"] is True


def test_eval_episode_skips_replay_when_store_disabled():
    policy, trainer, _ = build_policy(
        warmup_steps=0,
        eval_interval_episodes=1,
        store_eval_episodes=False,
    )
    commit(policy, act(policy)["transition_id"])
    policy.infer({REQUEST_KEY: REQUEST_EPISODE_END, "stats": {}})
    stored = commit(policy, act(policy)["transition_id"])
    assert stored["stored"] is False
    assert stored["reason"] == "eval_episode"
    assert len(trainer.transitions) == 1


def test_checkpoints_land_on_the_configured_episode_interval(tmp_path):
    policy, trainer, _ = build_policy(
        warmup_steps=0, save_dir=str(tmp_path), save_interval_episodes=2
    )
    for _ in range(4):
        commit(policy, act(policy)["transition_id"])
        policy.infer({REQUEST_KEY: REQUEST_EPISODE_END, "stats": {}})
    names = [p.split("/")[-1] for p in trainer.saves]
    assert len(names) == 2
    assert names[0].startswith("episode_2_") and names[1].startswith("episode_4_")

    # A resumed run restarts the episode counter, so the name must also carry
    # update_step or the second run would overwrite the first run's
    # checkpoints.
    resumed, resumed_trainer, _ = build_policy(
        warmup_steps=0, save_dir=str(tmp_path), save_interval_episodes=2
    )
    resumed_trainer.update_step = 500
    for _ in range(2):
        commit(resumed, act(resumed)["transition_id"])
        resumed.infer({REQUEST_KEY: REQUEST_EPISODE_END, "stats": {}})
    assert resumed_trainer.saves[0].split("/")[-1] not in names


# ------------------------------------------------------------------ rewinds


def test_rewind_exit_and_credit_reach_the_learner_with_their_mode():
    policy, trainer, _ = build_policy(warmup_steps=0)
    commit(policy, act(policy)["transition_id"])

    policy.infer(
        {
            REQUEST_KEY: REQUEST_REWIND_EXIT,
            "terminal_reward": -1.0,
            "chunks_rewound": 2,
            "preference_confidence": 0.8,
        }
    )
    policy.infer(
        {
            REQUEST_KEY: REQUEST_REWIND_CREDIT,
            "terminal_reward": -1.0,
            "prefix_reward": 0.1,
            "bad_chunks": 3,
            "preference_confidence": 0.5,
        }
    )

    assert [event.mode for event in trainer.rewind_events] == ["exit", "credit"]
    exit_event, credit_event = trainer.rewind_events
    assert exit_event.chunks_rewound == 2
    assert exit_event.confidence == pytest.approx(0.8)
    assert credit_event.prefix_reward == pytest.approx(0.1)
    assert credit_event.chunks_rewound == 3


def test_rewind_without_marked_chunks_is_a_no_op():
    policy, trainer, _ = build_policy(warmup_steps=0)
    response = policy.infer(
        {
            REQUEST_KEY: REQUEST_REWIND_EXIT,
            "terminal_reward": -1.0,
            "chunks_rewound": 0,
        }
    )
    assert response["applied"] is False
    assert trainer.rewind_events == []


# ------------------------------------------------------------- observations


def test_repacker_maps_three_cameras_into_the_aloha_wrist_stack():
    repacker = RLTObservationRepacker(
        camera_layout=CameraLayout(main="image", wrist=("wrist_image", "side_image")),
        proprio_dim=PROPRIO_DIM,
        default_prompt="put cube in drawer",
    )
    env_obs = repacker.to_env_obs(observation_payload())

    # AlohaInputs indexes wrist_image[0] as cam_left_wrist and [1] as
    # cam_right_wrist, so the two views must arrive stacked, not as separate
    # keys the transform would ignore.
    assert env_obs["wrist_images"].shape[:2] == (1, 2)
    assert env_obs["main_images"].shape[0] == 1
    assert env_obs["states"].shape == (1, PROPRIO_DIM)
    assert env_obs["task_descriptions"] == ["put cube in drawer"]


def test_repacker_rejects_a_missing_camera():
    repacker = RLTObservationRepacker(
        camera_layout=CameraLayout(main="image", wrist=("wrist_image", "side_image")),
        proprio_dim=PROPRIO_DIM,
        default_prompt="put cube in drawer",
    )
    payload = observation_payload()
    del payload["side_image"]
    # Failing loudly matters more here than elsewhere: OpenPI would otherwise
    # substitute a black frame and a false image mask, and the policy would
    # keep producing confident actions off two cameras.
    with pytest.raises(ValueError, match="missing camera"):
        repacker.to_env_obs(payload)

    short_state = observation_payload()
    short_state["state"] = np.zeros(PROPRIO_DIM - 1, dtype=np.float32)
    with pytest.raises(ValueError, match="proprio_dim"):
        repacker.to_env_obs(short_state)


# ------------------------------------------------------------------ preflight


def test_preflight_rejects_placeholder_prompts_and_camera_mismatch():
    check_task_prompt("put cube in drawer")
    for placeholder in ("", "   ", "TODO", "prompt"):
        with pytest.raises(PreflightError):
            check_task_prompt(placeholder)

    check_camera_layout(CAMERA_KEYS, 3)
    with pytest.raises(PreflightError, match="num_images_in_input"):
        check_camera_layout(CAMERA_KEYS, 2)
    with pytest.raises(PreflightError, match="Duplicate"):
        check_camera_layout(("image", "image", "side_image"), 3)


def test_preflight_rejects_expo_decoupled_on_twin_q():
    cfg = OmegaConf.create(
        {
            "algorithm": {"rl_algo_td_backup": "expo_decoupled", "rl_algo_act": "expo"},
            "actor": {
                "model": {
                    "action_selection_mode": "expo",
                    "critic_num_qs": 2,
                    "critic_num_min_qs": 2,
                }
            },
        }
    )
    with pytest.raises(PreflightError, match="2 \\* critic_num_min_qs"):
        check_stage2_algorithm(cfg)
    cfg.actor.model.critic_num_qs = 10
    check_stage2_algorithm(cfg)


# --------------------------------------------------------------- client loop


class LoopbackClient:
    """Wires :class:`RLTRobotLoop` straight into a policy, without a socket."""

    def __init__(self, policy: RLTStage2Policy) -> None:
        self._policy = policy
        self.server_metadata = policy.metadata
        self.requests: list[str] = []
        self.closed = False
        self.fail_on: str | None = None

    @property
    def uri(self) -> str:
        return "loopback://"

    def request(self, payload):
        request_type = payload.get(REQUEST_KEY, REQUEST_ACT)
        self.requests.append(request_type)
        if request_type == self.fail_on:
            raise ConnectionError(f"simulated disconnect on {request_type}")
        return self._policy.infer(payload)

    def close(self):
        self.closed = True


def build_loop(policy, **overrides):
    from rlinf.envs.realworld.cobot.control import MockRewindAdapter
    from rlinf.envs.realworld.rlt_client.cobot import CobotTransport
    from rlinf.envs.realworld.rlt_client.loop import RLTRobotLoop

    adapter = MockRewindAdapter(action_dim=ACTION_DIM, task="put cube in drawer")
    transport = CobotTransport(
        adapter,
        chunk_len=CHUNK_LEN,
        proprio_dim=ACTION_DIM,
        camera_keys=CAMERA_KEYS,
    )
    client = LoopbackClient(policy)
    kwargs = {"max_episode_chunks": 3, "validate_handshake": False}
    kwargs.update(overrides)
    loop = RLTRobotLoop(transport=transport, client=client, **kwargs)
    return loop, client, adapter


def test_client_loop_commits_every_chunk_and_ends_the_episode():
    policy, trainer, _ = build_policy(warmup_steps=0, utd_ratio=2)
    loop, client, _ = build_loop(policy)

    outcome = loop.run_episode()

    assert outcome.chunks == 3
    assert len(trainer.transitions) == 3
    assert client.requests.count(REQUEST_ACT) == 3
    assert client.requests.count(REQUEST_TRANSITION) == 3
    assert client.requests[-1] == REQUEST_EPISODE_END
    assert REQUEST_DISCARD not in client.requests
    assert trainer.train_calls == [6], "3 actor chunks x utd_ratio 2"


def test_client_loop_discards_the_pending_chunk_when_execution_fails():
    policy, trainer, _ = build_policy(warmup_steps=0)
    loop, client, adapter = build_loop(policy)

    def explode(action):
        raise RuntimeError("gripper fault")

    adapter.execute = explode
    with pytest.raises(RuntimeError, match="gripper fault"):
        loop.run_episode()

    # The chunk the server handed out must not stay pending, and the arm must
    # be stopped before the exception surfaces.
    assert REQUEST_DISCARD in client.requests
    assert trainer.transitions == []
    assert policy.infer({REQUEST_KEY: REQUEST_STATUS})["pending"] == 0
    assert adapter._stopped is True


def test_client_loop_folds_the_success_verdict_into_the_last_chunk():
    policy, trainer, _ = build_policy(warmup_steps=0)
    loop, _, adapter = build_loop(policy, success_reward=1.0)

    from rlinf.envs.realworld.rlt_client.transport import OperatorEvent

    events = [None, OperatorEvent(kind="success")]
    adapter.poll_rewind_event = lambda: events.pop(0) if events else None

    outcome = loop.run_episode()

    assert outcome.success is True
    assert outcome.chunks == 2, "the verdict ends the episode"
    last = trainer.transitions[-1]
    assert last["rewards"][-1] == pytest.approx(1.0)
    assert last["done"] is True
    assert last["bootstrap_mask"] == pytest.approx(0.0), (
        "a solved task has no future return to bootstrap"
    )


def test_client_loop_always_stops_and_closes_the_robot():
    policy, _, _ = build_policy(warmup_steps=0)
    loop, client, adapter = build_loop(policy)
    loop.run(1)
    assert adapter._stopped is True
    assert client.closed is True


def test_client_loop_survives_an_episode_end_disconnect():
    policy, trainer, _ = build_policy(warmup_steps=0)
    loop, client, _ = build_loop(policy)
    client.fail_on = REQUEST_EPISODE_END

    # Losing the final request must not lose the chunks already committed, and
    # must not raise into the operator's terminal.
    outcome = loop.run_episode()
    assert outcome.chunks == 3
    assert len(trainer.transitions) == 3
    assert outcome.updates_run == 0


# ------------------------------------------------------- msgpack read-only


def msgpack_round_trip(payload: dict) -> dict:
    """Send a payload through the real wire codec both server and client use."""
    from openpi_client import msgpack_numpy

    return msgpack_numpy.unpackb(msgpack_numpy.Packer().pack(payload))


def test_decoded_arrays_are_read_only_views():
    """Pin the upstream behaviour the copies below defend against."""
    decoded = msgpack_round_trip({"a": np.zeros((2, 3), dtype=np.float32)})["a"]
    assert decoded.flags.writeable is False
    assert np.asarray(decoded, dtype=np.float32).flags.writeable is False


def test_protocol_owns_its_arrays_after_decoding():
    """A decoded chunk must not alias the receive buffer.

    torch.as_tensor on a read-only array produces a tensor sharing that
    buffer, so an in-place write would silently corrupt a stored replay row
    rather than fail.
    """
    frame = msgpack_round_trip(
        {
            REQUEST_KEY: REQUEST_TRANSITION,
            "transition_id": "0",
            "next_observation": observation_payload(),
            "rewards": np.zeros(CHUNK_LEN, dtype=np.float32),
            "done": False,
            "bootstrap_mask": 1.0,
            "action_chunk": np.zeros((CHUNK_LEN, ACTION_DIM), dtype=np.float32),
            "action_chunk_space": ACTION_SPACE_ROBOT,
        }
    )
    request = TransitionRequest.from_payload(frame)
    assert request.rewards.flags.writeable is True
    assert request.action_chunk.flags.writeable is True

    repacker = RLTObservationRepacker(
        camera_layout=CameraLayout(main="image", wrist=("wrist_image", "side_image")),
        proprio_dim=PROPRIO_DIM,
        default_prompt="put cube in drawer",
    )
    env_obs = repacker.to_env_obs(msgpack_round_trip(observation_payload()))
    assert env_obs["states"].flags.writeable is True
    assert env_obs["main_images"].flags.writeable is True


def test_transport_accepts_a_read_only_action_chunk():
    """Controllers clip and filter actions in place; the wire hands back views."""
    from rlinf.envs.realworld.cobot.control import MockRewindAdapter
    from rlinf.envs.realworld.rlt_client.cobot import CobotTransport
    from rlinf.envs.realworld.rlt_client.transport import ChunkIdentity as TId

    adapter = MockRewindAdapter(action_dim=ACTION_DIM, task="t")
    transport = CobotTransport(
        adapter, chunk_len=CHUNK_LEN, proprio_dim=ACTION_DIM, camera_keys=CAMERA_KEYS
    )
    transport.reset()
    chunk = msgpack_round_trip(
        {"actions": np.ones((CHUNK_LEN, ACTION_DIM), dtype=np.float32)}
    )["actions"]
    assert chunk.flags.writeable is False

    transport.execute_chunk(chunk, TId(chunk_id=1))
    # The adapter stored the last action as its own state; resetting it must
    # not fail on a read-only buffer inherited from the socket.
    transport.reset()


def test_truncated_chunk_does_not_report_unexecuted_commands():
    """A chunk cut short must not present its unrun tail as executed actions.

    Those steps get labelled ACTION_SOURCE_HUMAN on an intervention chunk and
    become BC targets, so reporting the commanded values would train the
    policy toward motions the operator never made.
    """
    from rlinf.envs.realworld.cobot.control import CobotStepResult, MockRewindAdapter
    from rlinf.envs.realworld.rlt_client.cobot import CobotTransport
    from rlinf.envs.realworld.rlt_client.transport import ChunkIdentity as TId

    adapter = MockRewindAdapter(action_dim=ACTION_DIM, task="t")
    transport = CobotTransport(
        adapter, chunk_len=CHUNK_LEN, proprio_dim=ACTION_DIM, camera_keys=CAMERA_KEYS
    )
    transport.reset()

    real_execute = adapter.execute
    calls = {"n": 0}

    def take_over_on_second_step(action):
        calls["n"] += 1
        result = real_execute(action)
        if calls["n"] == 2:
            override = np.full(ACTION_DIM, 99.0, dtype=np.float32)
            return CobotStepResult(
                result.observation,
                result.reward,
                info={
                    **result.info,
                    "executed_action": override,
                    "human_intervention": True,
                },
            )
        return result

    adapter.execute = take_over_on_second_step
    commanded = np.arange(CHUNK_LEN * ACTION_DIM, dtype=np.float32).reshape(
        CHUNK_LEN, ACTION_DIM
    )
    result = transport.execute_chunk(commanded, TId(chunk_id=1))

    assert result.intervention is True
    assert result.steps_executed == 2
    tail = result.executed_chunk[2:]
    # Held at the last executed action, which is what the arm physically did.
    np.testing.assert_allclose(
        tail, np.broadcast_to(np.full(ACTION_DIM, 99.0, dtype=np.float32), tail.shape)
    )
    assert not np.any(tail == commanded[2:]), "must not echo unrun commands"


# ------------------------------------------------------ real socket round trip


def _free_port() -> int:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_full_episode_over_a_real_websocket():
    """Drive a real server with a real client over a real socket.

    The loopback tests above stub the transport out, so they cannot catch
    anything the wire codec does to the data. This one can, and did: msgpack
    returns read-only arrays.
    """
    import dataclasses
    import threading

    from rlinf.envs.realworld.cobot.control import MockRewindAdapter
    from rlinf.envs.realworld.rlt_client.cobot import CobotTransport
    from rlinf.envs.realworld.rlt_client.loop import RLTRobotLoop
    from rlinf.serving.websocket_client import RLinfWebsocketClient
    from rlinf.serving.websocket_server import RLinfWebsocketPolicyServer

    policy, trainer, _ = build_policy(warmup_steps=2, utd_ratio=2)
    # MockRewindAdapter ties state width to action_dim.
    policy._metadata = dataclasses.replace(policy._metadata, proprio_dim=ACTION_DIM)

    port = _free_port()
    server = RLinfWebsocketPolicyServer(
        policy=policy, host="127.0.0.1", port=port, metadata=policy.metadata
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()

    client = RLinfWebsocketClient(host="127.0.0.1", port=port, connect_timeout=30.0)
    try:
        assert client.server_metadata["protocol"] == "rlt-online-rl/v1"
        adapter = MockRewindAdapter(action_dim=ACTION_DIM, task="put cube in drawer")
        transport = CobotTransport(
            adapter,
            chunk_len=CHUNK_LEN,
            proprio_dim=ACTION_DIM,
            camera_keys=CAMERA_KEYS,
        )
        # Handshake validation runs against metadata that crossed the wire.
        loop = RLTRobotLoop(
            transport=transport,
            client=client,
            max_episode_chunks=3,
            validate_handshake=True,
        )

        first = loop.run_episode()
        # Two chunks spent on warmup, so only the third earns gradient steps.
        assert first.chunks == 3
        assert first.updates_run == 1 * 2

        second = loop.run_episode()
        assert second.updates_run == 3 * 2

        status = client.request({REQUEST_KEY: REQUEST_STATUS})
        assert status["pending"] == 0
        assert status["buffer_size"] == 6
        assert len(trainer.transitions) == 6
    finally:
        client.close()


# ------------------------------------------------------------------ preflight


def test_preflight_reads_the_rlt_prefix_length_from_a_checkpoint(tmp_path):
    path = tmp_path / "full_weights.pt"
    torch.save({"rlt_module.encoder.prefix_pos_enc": torch.zeros(968, 8)}, path)
    assert read_rlt_prefix_seq_len(str(path)) == 968

    plain = tmp_path / "sft.pt"
    torch.save({"paligemma.embed": torch.zeros(4)}, plain)
    assert read_rlt_prefix_seq_len(str(plain)) is None

    with pytest.raises(PreflightError, match="not found"):
        read_rlt_prefix_seq_len(str(tmp_path / "missing.pt"))


def test_preflight_resolves_both_stage1_checkpoint_layouts(tmp_path):
    """The loader accepts two layouts; preflight must agree on both."""
    for parts in (("model_state_dict",), ("actor", "model_state_dict")):
        root = tmp_path / "_".join(parts)
        target = root.joinpath(*parts, "full_weights.pt")
        target.parent.mkdir(parents=True)
        torch.save({"rlt_module.encoder.prefix_pos_enc": torch.zeros(968, 8)}, target)
        assert resolve_stage1_weights(str(root)) == str(target)


def test_preflight_rejects_a_model_path_pointing_at_model_state_dict(tmp_path):
    """The old guide documented this path shape; it dies inside safetensors.

    Pointing model_path at the directory that holds full_weights.pt matches
    neither loader layout, so the run would fail well after startup looked
    healthy. Preflight must say which way to correct it.
    """
    deep = tmp_path / "global_step_25000" / "actor" / "model_state_dict"
    deep.mkdir(parents=True)
    torch.save(
        {"rlt_module.encoder.prefix_pos_enc": torch.zeros(968, 8)},
        deep / "full_weights.pt",
    )

    with pytest.raises(PreflightError, match="one level too deep"):
        resolve_stage1_weights(str(deep))

    # ...and the correct parent resolves.
    assert resolve_stage1_weights(str(deep.parent.parent)).endswith("full_weights.pt")


def test_preflight_requires_norm_stats_to_be_configured_explicitly():
    """An unset norm_stats_path silently selects another task's quantiles."""
    assert (
        check_norm_stats_configured(OmegaConf.create({"norm_stats_path": "/a/b.json"}))
        == "/a/b.json"
    )
    for bad in ({}, {"norm_stats_path": ""}, {"norm_stats_path": None}):
        with pytest.raises(PreflightError, match="norm_stats_path is not set"):
            check_norm_stats_configured(OmegaConf.create(bad))


def test_preflight_rejects_a_prompt_that_disagrees_with_stage1():
    """server.task_prompt and openpi_data.default_prompt feed different paths."""
    check_task_prompt("assemble parts", stage1_prompt="assemble parts")
    check_task_prompt("Assemble Parts", stage1_prompt="assemble parts")

    with pytest.raises(PreflightError, match="disagrees with"):
        check_task_prompt("put cube in drawer", stage1_prompt="assemble parts")


def test_build_transport_loads_controller_factory_from_config():
    """The yaml string is what the robot-side launch command actually sets."""
    from rlinf.envs.realworld.cobot.stage2_hardware_adapter import (
        Stage2HardwareAdapter,
    )
    from rlinf.envs.realworld.rlt_client.cobot import build_cobot_transport

    cfg = OmegaConf.create(
        {
            "transport": {
                "is_dummy": False,
                "controller_factory": (
                    "rlinf.envs.realworld.cobot.stage2_hardware_adapter:create_adapter"
                ),
                "action_dim": 14,
                "chunk_len": 16,
                "proprio_dim": 14,
                "task": "assemble parts",
                "camera_keys": ["image", "wrist_image", "side_image"],
            }
        }
    )
    transport = build_cobot_transport(cfg)
    assert isinstance(transport._adapter, Stage2HardwareAdapter)
    assert transport.action_dim == 14


def test_stage2_hardware_adapter_maps_keys_and_checks_observation():
    from rlinf.envs.realworld.cobot.stage2_hardware_adapter import create_adapter
    from rlinf.envs.realworld.rlt_client.loop import EVENT_SUCCESS

    adapter = create_adapter(action_dim=14, task="assemble parts")
    adapter.enqueue_key("s")
    event = adapter.poll_rewind_event()
    assert event is not None
    assert event.kind == EVENT_SUCCESS
    assert adapter.poll_rewind_event() is None

    images = {
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "side_image": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    obs = adapter._observation(images, np.zeros(14, dtype=np.float32))
    assert obs.task == "assemble parts"
    assert obs.state.shape == (14,)

    with pytest.raises(RuntimeError, match="missing cameras"):
        adapter._observation({"image": images["image"]}, np.zeros(14))
    with pytest.raises(RuntimeError, match="expected 14"):
        adapter._observation(images, np.zeros(7))


def test_build_transport_refuses_a_real_run_without_a_factory():
    from rlinf.envs.realworld.rlt_client.cobot import build_cobot_transport

    cfg = OmegaConf.create(
        {
            "transport": {
                "is_dummy": False,
                "controller_factory": None,
                "action_dim": 14,
                "chunk_len": 16,
            }
        }
    )
    with pytest.raises(RuntimeError, match="no controller_factory"):
        build_cobot_transport(cfg)
