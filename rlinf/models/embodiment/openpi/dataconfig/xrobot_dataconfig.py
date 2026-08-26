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

"""LeRobot data config for XRobot dual-arm end-effector control (14-D).

Robot type ``x2robot_dual_arm_ee``: left/right xyz + rpy + gripper.
Camera keys are remapped to the Aloha-style names expected by ``AlohaInputs``.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.data.datasets.openpi_rlinf.pyav_video_patch import (
    apply_pyav_video_decode_patch,
)
from rlinf.models.embodiment.openpi.policies import aloha_policy

apply_pyav_video_decode_patch()


@dataclasses.dataclass(frozen=True)
class LeRobotXRobotDataConfig(DataConfigFactory):
    """Data configuration for XRobot LeRobot datasets (14-D EE pose + gripper)."""

    default_prompt: str | None = None
    extra_delta_transform: bool = True
    adapt_to_pi: bool = False

    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.head",
                            "cam_left_wrist": "observation.images.left_arm",
                            "cam_right_wrist": "observation.images.right_arm",
                        },
                        "state": "observation.state",
                        "actions": "action",
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
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        if self.extra_delta_transform:
            # Delta on pose dims; keep gripper absolute (same mask layout as Aloha/Cobot).
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
