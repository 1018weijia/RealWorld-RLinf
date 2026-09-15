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

"""XRobot USB LeRobot v3 reader, 50-step chunks, and USB overlay A–E."""

import json

import numpy as np
import pytest
import torch

from rlinf.serving.rlt.cobot_offline_data import (
    XROBOT_CAMERAS,
    XROBOT_EE_NAMES,
    XRobotLeRobotV3,
    chunk_starts,
    convert_episode,
    episode_source,
    flatten_feature_names,
)
from rlinf.serving.rlt.preflight import check_stage2_algorithm


PROMPT = "Bimanual usb pick and insert"


@pytest.fixture
def xrobot_dataset(tmp_path):
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "lerobot"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    length = 151
    features = {
        "observation.state": {"names": [XROBOT_EE_NAMES], "dtype": "float32"},
        "action": {"names": [XROBOT_EE_NAMES], "dtype": "float32"},
    }
    features.update({key: {"dtype": "video"} for key in XROBOT_CAMERAS})
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
            "length": length,
            "tasks": [PROMPT],
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": episode * length,
            "dataset_to_index": (episode + 1) * length,
        }
        for camera in XROBOT_CAMERAS:
            row.update(
                {
                    f"videos/{camera}/chunk_index": 0,
                    f"videos/{camera}/file_index": 0,
                    f"videos/{camera}/from_timestamp": episode * length / 30,
                    f"videos/{camera}/to_timestamp": (episode + 1) * length / 30,
                }
            )
        episodes.append(row)
    pq.write_table(
        pa.Table.from_pylist(episodes),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    rows = []
    for index in range(2 * length):
        rows.append(
            {
                "episode_index": index // length,
                "frame_index": index % length,
                "index": index,
                "timestamp": (index % length) / 30,
                "observation.state": (
                    np.arange(14, dtype=np.float32) + (index % length)
                ).tolist(),
                "action": np.full(14, 1.4, dtype=np.float32).tolist(),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
    for camera in XROBOT_CAMERAS:
        path = root / f"videos/{camera}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            for frame_index in range(2 * length):
                frame = av.VideoFrame.from_ndarray(
                    np.full((16, 16, 3), frame_index % 256, dtype=np.uint8),
                    format="rgb24",
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    return XRobotLeRobotV3(root, PROMPT)


class FeatureModel:
    chunk_len = 50
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


def test_flatten_nested_lerobot_names():
    assert flatten_feature_names([XROBOT_EE_NAMES]) == XROBOT_EE_NAMES
    assert flatten_feature_names(XROBOT_EE_NAMES) == XROBOT_EE_NAMES


def test_chunk_starts_requires_horizon_and_uses_fifty_for_xrobot():
    with pytest.raises(TypeError):
        chunk_starts(151)
    assert chunk_starts(151, 50) == [0, 50, 100]
    assert chunk_starts(50, 50) == []


def test_xrobot_reader_ee_names_unlabeled_success_and_chunk50(xrobot_dataset):
    assert xrobot_dataset.cameras == XROBOT_CAMERAS
    assert {row["episode_success"] for row in xrobot_dataset.episodes} == {"success"}
    state, actions = xrobot_dataset.table(xrobot_dataset.episodes[1])
    np.testing.assert_array_equal(state[0], np.arange(14, dtype=np.float32))
    assert actions.shape == (151, 14)
    frames = xrobot_dataset.frames(
        xrobot_dataset.episodes[1], next(iter(XROBOT_CAMERAS)), [0, 150]
    )
    assert frames[0].shape == (16, 16, 3)
    rows = [
        convert_episode(xrobot_dataset, episode, FeatureModel(), 0.99)
        for episode in xrobot_dataset.episodes
    ]
    assert rows[0]["rewards"].shape == (3, 50)
    assert rows[0]["rewards"].sum() == 1
    assert rows[0]["actions"].shape == (3, 700)
    assert rows[0]["curr_obs"]["ref_chunk"].shape == (3, 50, 14)


def test_xrobot_reader_rejects_joint_names(tmp_path, xrobot_dataset):
    info = json.loads((xrobot_dataset.root / "meta/info.json").read_text())
    info["features"]["observation.state"]["names"] = [
        [f"left_joint_{i}.pos" for i in range(1, 15)]
    ]
    (xrobot_dataset.root / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="14 EE pose"):
        XRobotLeRobotV3(xrobot_dataset.root, PROMPT)


def test_episode_source_selects_xrobot_reader(xrobot_dataset, server_config):
    cfg = server_config("xrobot_usb_plug_rlt_stage2_ws_server")
    cfg.offline = {"dataset_root": str(xrobot_dataset.root)}
    data = episode_source(cfg)
    assert isinstance(data, XRobotLeRobotV3)
    assert data.prompt == PROMPT


def test_usb_overlay_matches_ae_algorithm(server_config):
    cfg = server_config("xrobot_usb_plug_rlt_stage2_ws_server")
    check_stage2_algorithm(cfg)
    assert cfg.algorithm.rl_algo_td_backup == "expo_decoupled"
    assert cfg.actor.model.critic_num_qs == 10
    assert cfg.actor.model.residual_scale == 0.4
    assert cfg.actor.model.gripper_absolute_output is True
    assert cfg.actor.model.num_action_chunks == 50
    assert cfg.server.task_prompt == PROMPT
    assert cfg.rlt_feature_model.openpi_data.default_prompt == PROMPT
    assert cfg.rlt_feature_model.openpi_data.repo_id == "xrobot/usb_plug"
