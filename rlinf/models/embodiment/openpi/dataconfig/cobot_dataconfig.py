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

"""LeRobot data config for Cobot Magic dual-arm joint control."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import aloha_policy

# Ensure PyAV video decode is available in this process (and forked workers
# that already imported this module before spawning DataLoader workers).
from rlinf.data.datasets.openpi_rlinf.pyav_video_patch import (
    apply_pyav_video_decode_patch,
)

apply_pyav_video_decode_patch()

# Cobot Magic ``observation.state`` layout (from meta/info.json): for each of
# left/right 6 joints + gripper the fields are (pos, vel, torque). EE pose
# follows after index 42 and is dropped. Joint/gripper positions are every
# third index in the first 42 dims.
_COBOT_STATE_POS_INDICES: tuple[int, ...] = tuple(range(0, 42, 3))
_COBOT_JOINT_ACTION_DIM = 14
_COBOT_RAW_ACTION_DIM = 28
_COBOT_RAW_STATE_DIM = 56


@dataclasses.dataclass(frozen=True)
class SliceCobotToJoint14(_transforms.DataTransformFn):
    """Slice Cobot Magic 28-D actions / 56-D state down to 14-D joint+gripper."""

    state_pos_indices: tuple[int, ...] = _COBOT_STATE_POS_INDICES
    joint_action_dim: int = _COBOT_JOINT_ACTION_DIM
    raw_action_dim: int = _COBOT_RAW_ACTION_DIM
    raw_state_dim: int = _COBOT_RAW_STATE_DIM

    def __call__(self, data: dict) -> dict:
        data = dict(data)

        if "state" in data:
            state = np.asarray(data["state"])
            if state.shape[-1] != self.raw_state_dim:
                raise ValueError(
                    "Cobot state last dim must be "
                    f"{self.raw_state_dim}, got {state.shape[-1]}."
                )
            data["state"] = state[..., list(self.state_pos_indices)]
            if data["state"].shape[-1] != self.joint_action_dim:
                raise ValueError(
                    "Sliced Cobot state must have dim "
                    f"{self.joint_action_dim}, got {data['state'].shape[-1]}."
                )

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != self.raw_action_dim:
                raise ValueError(
                    "Cobot action last dim must be "
                    f"{self.raw_action_dim}, got {actions.shape[-1]}."
                )
            data["actions"] = actions[..., : self.joint_action_dim]

        return data


@dataclasses.dataclass(frozen=True)
class LeRobotCobotDataConfig(DataConfigFactory):
    """Data configuration for Cobot Magic LeRobot datasets (joint 14-D)."""

    default_prompt: str | None = None
    extra_delta_transform: bool = True
    adapt_to_pi: bool = False

    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        # Prompt is injected later via ModelTransformFactory(default_prompt=...).
                    }
                )
            ]
        )
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[
                SliceCobotToJoint14(),
                aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi),
            ],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        if self.extra_delta_transform:
            delta_action_mask = np.array(
                [True] * 6 + [False] + [True] * 6 + [False],
                dtype=bool,
            )
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
