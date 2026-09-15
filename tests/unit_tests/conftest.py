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

"""Shared fixtures for the unit tests."""

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"

# XRobot overlays resolve checkpoint/norm paths from the environment at compose
# time. Tests that do not care about the site paths still need the keys set.
os.environ.setdefault("XROBOT_RLT_STAGE1_CHECKPOINT", "/tmp/xrobot_stage1")
os.environ.setdefault("XROBOT_NORM_STATS", "/tmp/xrobot_norm_stats.json")
os.environ.setdefault("XROBOT_USB_STAGE1_CHECKPOINT", "/tmp/xrobot_usb_stage1")
os.environ.setdefault("XROBOT_USB_NORM_STATS", "/tmp/xrobot_usb_norm_stats.json")


@pytest.fixture
def server_config():
    """Compose a Stage 2 server config the way the entry point does.

    Reading the YAML with ``OmegaConf.load`` skips Hydra's defaults list, so the
    ``embodiment`` config group is never merged and every
    ``${embodiment.*}`` interpolation is left dangling.
    """

    def load(name: str = "cobot_rlt_stage2_ws_server") -> DictConfig:
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
            cfg = compose(config_name=name)
        # Struct mode is on after compose. Tests inject keys the YAML omits,
        # which the launchers pass as Hydra's `+server.vla_only=True`.
        OmegaConf.set_struct(cfg, False)
        return cfg

    return load
