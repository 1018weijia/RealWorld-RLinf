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

"""Pure VLA evaluation must never instantiate or execute Stage2 learning."""

import runpy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.serving.rlt.inference import RLTStage2Inference
from rlinf.serving.rlt.policy import RLTStage2Policy, Stage1EvaluationState

ROOT = Path(__file__).resolve().parents[2]


def test_stage1_reference_bypasses_actor_critic_and_never_trains():
    helpers = runpy.run_path(
        str(Path(__file__).with_name("test_rlt_stage2_websocket.py"))
    )
    base, _, _ = helpers["build_policy"]()
    metadata = replace(
        base._metadata, vla_only=True, eval_only=True, action_selection_mode="vla"
    )
    inference = RLTStage2Inference(
        feature_model=None,
        policy_model=None,
        repacker=None,
        chunk_len=4,
        action_dim=3,
        num_ref_candidates=1,
        device=torch.device("cpu"),
    )
    reference = torch.linspace(-2, 2, 12).reshape(1, 4, 3)
    inference.encode = Mock(side_effect=lambda _: {"ref_chunk": reference.clone()})
    inference.to_robot_space = lambda actions, _: actions.numpy()
    inference._policy_chunk = Mock(
        side_effect=AssertionError("Stage2 must not execute")
    )
    state = Stage1EvaluationState(4, 3)
    policy = RLTStage2Policy(
        trainer=state,
        inference=inference,
        metadata=metadata,
        warmup_steps=0,
        utd_ratio=4,
        max_episode_chunks=150,
        eval_only=True,
    )
    for _ in range(3):
        response = helpers["act"](policy)
        np.testing.assert_array_equal(response["actions"], reference.numpy()[0])
        assert response["mode"] == "eval" and not response["expo_active"]
        committed = helpers["commit"](policy, response["transition_id"])
        assert not committed["stored"] and committed["buffer_size"] == 0
    result = policy.infer({"rlt/request": "episode_end", "stats": {"success": True}})
    assert result["updates_run"] == 0 and not result["checkpoint_saved"]
    assert result["offline_total_updates"] == result["offline_buffer_size"] == 0
    inference._policy_chunk.assert_not_called()
    with pytest.raises(ValueError, match="eval_only"):
        RLTStage2Policy(
            trainer=state,
            inference=inference,
            metadata=metadata,
            warmup_steps=0,
            utd_ratio=4,
            max_episode_chunks=150,
            eval_only=False,
        )


def test_build_stage1_loads_only_feature_model():
    entry = runpy.run_path(str(ROOT / "examples/embodiment/rlt_stage2_server.py"))
    cfg = OmegaConf.load(
        ROOT / "examples/embodiment/config/cobot_rlt_stage2_ws_server.yaml"
    )
    cfg.server.vla_only = True
    cfg.server.eval_only = True
    cfg.runner.resume_dir = None
    factory = Mock(return_value=torch.nn.Linear(1, 1))
    trainer = Mock(side_effect=AssertionError("Must not build Stage2 trainer"))
    build = entry["build_policy"]
    with (
        patch.dict(build.__globals__, get_model=factory, RLTStage2Trainer=trainer),
        patch("torch.cuda.is_available", return_value=False),
    ):
        policy = build(cfg)
    factory.assert_called_once_with(cfg.rlt_feature_model)
    trainer.assert_not_called()
    assert policy.inference.policy_model is None
    assert policy.inference.num_ref_candidates == 1
    assert policy.metadata["vla_only"] and policy.metadata["eval_only"]
    assert policy.metadata["edit_scale"] == 0


@pytest.mark.parametrize("resume,eval_only", [("checkpoint", True), (None, False)])
def test_stage1_rejects_resume_or_training_before_loading(resume, eval_only):
    entry = runpy.run_path(str(ROOT / "examples/embodiment/rlt_stage2_server.py"))
    cfg = OmegaConf.create(
        {
            "server": {"vla_only": True, "eval_only": eval_only},
            "runner": {"resume_dir": resume},
        }
    )
    factory = Mock()
    build = entry["build_policy"]
    with (
        patch.dict(build.__globals__, get_model=factory),
        pytest.raises(ValueError, match="VLA-only"),
    ):
        build(cfg)
    factory.assert_not_called()
