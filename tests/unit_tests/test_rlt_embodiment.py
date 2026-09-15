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

"""The per-robot contract is pinned, self-consistent and enforced.

An embodiment mismatch does not raise at runtime: a Stage 2 head built for one
action width on top of a Stage 1 pipeline configured for another mostly runs,
and sends wrong units to a real arm. These tests cover the two things that
prevent that -- the shipped profiles stay what the robots were run with, and a
config that drifts away from its profile fails at startup.
"""

import dataclasses

import pytest

from rlinf.serving.rlt.embodiment import EmbodimentError, EmbodimentProfile

CAMERAS = ("image", "wrist_image", "side_image")

# Golden contracts. x2robot is the combination validated end-to-end on the real
# robot; changing any number here changes what a deployed client must send.
SHIPPED = {
    "xrobot_usb_plug_rlt_stage2_ws_server": {
        "name": "x2robot",
        "robot_type": "x2robot",
        "action_schema": "x2robot-ee14-v1",
        "action_dim": 14,
        "proprio_dim": 14,
        "chunk_length": 50,
        "ref_chunk_length": 50,
        "camera_keys": CAMERAS,
        "openpi_config_name": "pi05_xrobot",
    },
    "xrobot_ee_rlt_stage2_ws_server": {
        "name": "x2robot",
        "robot_type": "x2robot",
        "action_schema": "x2robot-ee14-v1",
        "action_dim": 14,
        "proprio_dim": 14,
        "chunk_length": 50,
        "ref_chunk_length": 50,
        "camera_keys": CAMERAS,
        "openpi_config_name": "pi05_xrobot",
    },
    "cobot_rlt_stage2_ws_server": {
        "name": "cobot_magic",
        "robot_type": "cobot_magic",
        "action_schema": "cobot-joint14-v1",
        "action_dim": 14,
        "proprio_dim": 14,
        # Stage 1 proposes 20 actions, the robot executes the first 16.
        "chunk_length": 16,
        "ref_chunk_length": 20,
        "camera_keys": CAMERAS,
        "openpi_config_name": "pi05_cobot_magic",
    },
}


def profile(**overrides) -> EmbodimentProfile:
    """A valid profile, with fields replaced for the invalid cases."""
    return dataclasses.replace(
        EmbodimentProfile(**SHIPPED["xrobot_ee_rlt_stage2_ws_server"]), **overrides
    )


@pytest.mark.parametrize("config_name", sorted(SHIPPED))
def test_shipped_config_matches_its_golden_contract(config_name, server_config):
    cfg = server_config(config_name)
    parsed = EmbodimentProfile.from_config(cfg)

    assert dataclasses.asdict(parsed) == SHIPPED[config_name]


@pytest.mark.parametrize("config_name", sorted(SHIPPED))
def test_shipped_config_agrees_with_its_profile(config_name, server_config):
    cfg = server_config(config_name)

    # Every section that carries an embodiment number interpolates from the
    # group, so this is what proves the interpolations are actually wired.
    EmbodimentProfile.from_config(cfg).check_config(cfg)


@pytest.mark.parametrize(
    "path",
    [
        "actor.model.action_dim",
        "actor.model.proprio_dim",
        "actor.model.num_action_chunks",
        "actor.model.ref_num_action_chunks",
        "rlt_feature_model.action_dim",
        "rlt_feature_model.num_action_chunks",
        "rlt_feature_model.openpi.num_images_in_input",
    ],
)
def test_config_drifting_from_its_profile_is_rejected(path, server_config):
    cfg = server_config("xrobot_ee_rlt_stage2_ws_server")
    parsed = EmbodimentProfile.from_config(cfg)

    section = cfg
    *parents, leaf = path.split(".")
    for part in parents:
        section = section[part]
    section[leaf] = int(section[leaf]) + 1

    with pytest.raises(EmbodimentError, match=path.replace(".", r"\.")):
        parsed.check_config(cfg)


def test_openpi_pipeline_drifting_from_the_profile_is_rejected(server_config):
    cfg = server_config("xrobot_ee_rlt_stage2_ws_server")
    parsed = EmbodimentProfile.from_config(cfg)

    # This single key decides joint-versus-end-effector decoding.
    cfg.rlt_feature_model.openpi.config_name = "pi05_cobot_magic"

    with pytest.raises(EmbodimentError, match="config_name"):
        parsed.check_config(cfg)


def test_missing_embodiment_section_names_the_fix():
    with pytest.raises(EmbodimentError, match="defaults"):
        EmbodimentProfile.from_config({"server": {}})


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"action_dim": 0}, "action_dim"),
        ({"proprio_dim": -1}, "proprio_dim"),
        ({"chunk_length": 0}, "chunk_length"),
        # A robot cannot execute more actions than Stage 1 proposes.
        ({"chunk_length": 50, "ref_chunk_length": 20}, "ref_chunk_length"),
        ({"camera_keys": ()}, "camera_keys"),
        ({"camera_keys": ("image", "image")}, "Duplicate"),
    ],
)
def test_impossible_profile_is_rejected(overrides, expected):
    with pytest.raises(EmbodimentError, match=expected):
        profile(**overrides)._check_self()
