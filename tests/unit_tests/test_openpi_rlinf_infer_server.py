# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the YAML-driven OpenPI RLinf inference server."""

import base64
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODULE_PATH = (
    Path(__file__).parents[2] / "toolkits/inference/openpi_rlinf_infer_server.py"
)
SPEC = importlib.util.spec_from_file_location("openpi_rlinf_infer_server", MODULE_PATH)
assert SPEC and SPEC.loader
infer_server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = infer_server
SPEC.loader.exec_module(infer_server)


class _RecordingModel:
    def predict_action_batch(self, env_obs):
        self.env_obs = env_obs
        return torch.zeros((1, 2, 14)), None


def _encode_image() -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_infer_ignores_payload_prompt_and_uses_yaml_default():
    server = object.__new__(infer_server.OpenPiRlinfInferServer)
    server.device = torch.device("cpu")
    server.robot = "dobot"
    server.default_prompt = "yaml default prompt"
    server.model = _RecordingModel()
    server.num_action_chunks = 2
    server.action_dim = 14
    image = _encode_image()

    response = server.infer_from_payload(
        {
            "prompt": "client override prompt",
            "state": [0.0] * 14,
            "cam_high_b64": image,
            "cam_left_wrist_b64": image,
            "cam_right_wrist_b64": image,
        }
    )

    assert server.model.env_obs["task_descriptions"] == ["yaml default prompt"]
    assert response["prompt"] == "yaml default prompt"
