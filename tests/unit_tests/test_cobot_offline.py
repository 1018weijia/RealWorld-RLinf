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

"""Offline conversion, Cal-QL gradients, native checkpoint and online handoff."""

import copy
import json

import numpy as np
import pytest
import torch

from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import RLTTD3MLPPolicy
from rlinf.serving.rlt.cobot_offline_data import (
    CAMERAS,
    FORMAT,
    JOINT_NAMES,
    CobotLeRobotV3,
    OfflineBuffer,
    concatenate,
    contract,
    convert_episode,
)
from rlinf.serving.rlt.cobot_offline_trainer import (
    CobotOfflineTrainer,
    conservative_gap,
)
from rlinf.serving.rlt.protocol import ChunkIdentity


@pytest.fixture
def dataset(tmp_path):
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "lerobot"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    names = [
        n.replace(".pos", suffix)
        for n in JOINT_NAMES
        for suffix in (".pos", ".vel", ".torque")
    ]
    features = {"observation.state": {"names": names}, "action": {"names": JOINT_NAMES}}
    features.update({k: {"dtype": "video"} for k in CAMERAS})
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": 2,
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    episodes = []
    for episode in range(2):
        row = {
            "episode_index": episode,
            "length": 61,
            "tasks": ["assemble parts"],
            "episode_success": "success" if episode == 0 else "failure",
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": episode * 61,
            "dataset_to_index": (episode + 1) * 61,
        }
        for camera in CAMERAS:
            row.update(
                {
                    f"videos/{camera}/chunk_index": 0,
                    f"videos/{camera}/file_index": 0,
                    f"videos/{camera}/from_timestamp": episode * 61 / 30,
                    f"videos/{camera}/to_timestamp": (episode + 1) * 61 / 30,
                }
            )
        episodes.append(row)
    pq.write_table(
        pa.Table.from_pylist(episodes),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    rows = []
    for i in range(122):
        state = np.arange(42, dtype=np.float32) + i
        rows.append(
            {
                "episode_index": i // 61,
                "frame_index": i % 61,
                "index": i,
                "timestamp": (i % 61) / 30,
                "observation.state": state.tolist(),
                "action": np.full(14, 1.4, dtype=np.float32).tolist(),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
    for camera in CAMERAS:
        path = root / f"videos/{camera}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            for i in range(122):
                frame = av.VideoFrame.from_ndarray(
                    np.full((16, 16, 3), i, dtype=np.uint8), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    return CobotLeRobotV3(root, "assemble parts")


class FeatureModel:
    chunk_len = 30
    device = torch.device("cpu")

    def encode(self, observation):
        return {
            "z_rl": torch.zeros(1, 8),
            "proprio": torch.tensor(observation["state"])[None],
            "ref_chunk": torch.zeros(1, 50, 14),
            "ref_candidates": torch.zeros(1, 4, 50, 14),
        }

    def strip_private(self, obs):
        return obs

    def to_normalized_space(self, action, obs):
        return (action - 1) / 2

    def to_robot_space(self, action, obs):
        return action.numpy() * 2 + 1


def test_v3_offsets_names_rewards_and_boundary(dataset):
    state, _ = dataset.table(dataset.episodes[1])
    np.testing.assert_array_equal(state[0], np.arange(0, 42, 3) + 61)
    frames = dataset.frames(dataset.episodes[1], next(iter(CAMERAS)), [0, 60])
    assert abs(float(frames[0].mean()) - 61) < 4
    assert abs(float(frames[60].mean()) - 121) < 4
    rows = [
        convert_episode(dataset, ep, FeatureModel(), 0.99) for ep in dataset.episodes
    ]
    assert rows[0]["rewards"].sum() == 1
    assert rows[1]["rewards"].sum() == 0
    assert rows[0]["mc_returns"][0].item() == pytest.approx(0.99**59)
    assert rows[0]["mc_returns"][1].item() == pytest.approx(0.99**29)
    for key in rows[0]["curr_obs"]:
        torch.testing.assert_close(
            rows[0]["next_obs"][key][0], rows[0]["curr_obs"][key][1]
        )
    assert rows[0]["actions"].shape == (2, 420)
    assert rows[0]["curr_obs"]["ref_chunk"].shape == (2, 50, 14)


def test_calql_only_calibrates_policy_and_has_finite_gradients():
    policy = torch.full((1, 2, 2), -2.0, requires_grad=True)
    other = torch.full((1, 2, 2), -3.0, requires_grad=True)
    data = torch.zeros(1, 2, requires_grad=True)
    loss = conservative_gap(policy, other, data, torch.ones(1, 1))
    expected = torch.log(
        (torch.exp(torch.tensor(1.0)) + torch.exp(torch.tensor(-3.0))) / 2
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert policy.grad.abs().sum() == 0
    assert other.grad.abs().sum() > 0
    assert data.grad.sum() == -1


@pytest.mark.parametrize("reconfigure", [False, True])
def test_native_offline_train_resume_and_online_update(
    dataset, tmp_path, reconfigure, server_config
):
    torch.set_num_threads(1)
    cfg = server_config()
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture")
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    cfg.rlt_feature_model.model_path = str(weights)
    cfg.rlt_feature_model.openpi_data.norm_stats_path = str(stats)
    cfg.rlt_feature_model.num_action_chunks = 50
    cfg.actor.model.num_action_chunks = 30
    cfg.actor.model.ref_num_action_chunks = 50
    cfg.actor.model.z_dim = 8
    cfg.actor.global_batch_size = 4
    cfg.runner.logger.log_path = str(tmp_path / "logs")
    original_contract = contract(cfg)
    if reconfigure:
        cfg.actor.model.actor_noise_sigma = 0.1
        cfg.actor.model.residual_scale = 0.3
    model = RLTTD3MLPPolicy(
        z_dim=8,
        proprio_dim=14,
        action_dim=14,
        num_action_chunks=30,
        ref_num_action_chunks=50,
        mlp_hidden_dim=32,
        actor_noise_sigma=cfg.actor.model.actor_noise_sigma,
        residual_scale=cfg.actor.model.residual_scale,
    )

    def make():
        return CobotOfflineTrainer(
            cfg,
            model=copy.deepcopy(model),
            target_model=copy.deepcopy(model),
            device=torch.device("cpu"),
        )

    rows = [
        convert_episode(dataset, ep, FeatureModel(), 0.99) for ep in dataset.episodes
    ]
    payload = {
        "format": FORMAT,
        "contract": original_contract,
        "rows": concatenate(rows),
        "validation_episodes": [1],
    }
    buffer = OfflineBuffer(payload)
    assert set(buffer.sample(20, torch.device("cpu"))["episode_index"].tolist()) == {0}
    assert set(
        buffer.sample(20, torch.device("cpu"), validation=True)[
            "episode_index"
        ].tolist()
    ) == {1}
    trainer = make()
    trainer.offline_mode = True
    assert trainer.model.actor.sigma == cfg.actor.model.actor_noise_sigma
    assert trainer.model.actor.edit_scale == cfg.actor.model.residual_scale
    if reconfigure:
        with pytest.raises(ValueError, match="differs"):
            trainer.attach_offline_buffer(payload)
    trainer.attach_offline_buffer(payload, allow_actor_reconfiguration=reconfigure)
    assert payload["contract"] == original_contract
    assert trainer.offline_buffer.payload["contract"] == contract(cfg)
    if reconfigure:
        assert (
            trainer.offline_buffer.payload["conversion_contract"] == original_contract
        )
        assert trainer.offline_buffer.payload["rows"] is payload["rows"]
    before = copy.deepcopy(trainer.model.state_dict())
    metrics = trainer.update_once(train_actor=True)
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(
        not torch.equal(before[k], v)
        for k, v in trainer.model.state_dict().items()
        if k.startswith("actor.")
    )
    assert any(
        not torch.equal(before[k], v)
        for k, v in trainer.model.state_dict().items()
        if k.startswith("q_head.")
    )
    trainer.offline_total_updates = 1
    destination = tmp_path / "offline_step_1"
    trainer.save(str(destination))
    restored = make()
    restored.load(str(destination))
    assert restored.offline_total_updates == 1 and restored.update_step == 0
    assert restored.offline_buffer.size == 4
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[key], value)
    if reconfigure:
        cfg.actor.model.residual_scale = 0.2
        with pytest.raises(ValueError, match="differs"):
            make().load(str(destination))
        cfg.actor.model.residual_scale = 0.3
    sample = buffer.sample(1, torch.device("cpu"))
    restored.add_transition(
        curr_obs=sample["curr_obs"],
        next_obs=sample["next_obs"],
        action_chunk=sample["actions"][0].numpy().reshape(30, 14),
        rewards=np.zeros(30, np.float32),
        done=True,
        bootstrap_mask=0,
        intervention=False,
        identity=ChunkIdentity(episode_id=0, chunk_id=0),
        action_source=0,
    )
    cfg.algorithm.offline_sample_ratio = 0.5
    online = restored.train(1)
    assert all(np.isfinite(v) for v in online.values())
    assert restored.update_step == 1
    assert restored.offline_buffer.size == 4
    bad = copy.deepcopy(payload)
    bad["contract"]["task"] = "wrong task"
    with pytest.raises(ValueError, match="differs"):
        make().attach_offline_buffer(bad)
    for field in ("task", "gamma", "norm_sha256", "feature_model", "actor_model"):
        bad = copy.deepcopy(payload)
        if field == "actor_model":
            bad["contract"][field]["num_action_chunks"] = 8
        else:
            bad["contract"][field] = "incompatible"
        fresh = make()
        fresh.offline_mode = True
        with pytest.raises(ValueError, match="differs"):
            fresh.attach_offline_buffer(bad, allow_actor_reconfiguration=True)
