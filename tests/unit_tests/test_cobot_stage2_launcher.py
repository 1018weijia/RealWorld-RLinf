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

"""Project launchers isolate credentials and select task-specific assets."""

import os
import runpy
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "task,prompt,port,model",
    [
        ("assemble_parts", "assemble parts", "8000", "assemble"),
        ("cube_into_drawer", "put cube in drawer", "8001", "mixed"),
        ("cook_vegetable", "cook vegetable", "8002", "mixed"),
        (
            "pack_and_pour_fruit",
            "pack fruit into a container and pour it out",
            "8003",
            "mixed",
        ),
    ],
)
@pytest.mark.parametrize("mode", ["preflight", "train", "eval", "offline"])
def test_task_launcher_argv_and_private_key(tmp_path, task, prompt, port, model, mode):
    scripts = tmp_path / "examples/embodiment"
    scripts.mkdir(parents=True)
    launcher = scripts / "start_cobot_stage2.sh"
    shutil.copyfile(ROOT / "examples/embodiment/start_cobot_stage2.sh", launcher)
    private = tmp_path / ".private-cobot-stage2"
    private.mkdir()
    (private / "wandb_api_key").write_text("test-private-key")
    variables = []
    for name in ("assemble", "mixed"):
        checkpoint = tmp_path / name
        (checkpoint / "actor/model_state_dict").mkdir(parents=True)
        (checkpoint / "actor/model_state_dict/full_weights.pt").touch()
        stats = checkpoint / "norm_stats.json"
        stats.write_text("{}")
        variables.extend(
            [
                f"COBOT_{name.upper()}_CHECKPOINT={checkpoint}",
                f"COBOT_{name.upper()}_STATS={stats}",
            ]
        )
    (private / "paths.env").write_text("\n".join(variables))
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/bin/sh\ntest "$WANDB_API_KEY" = test-private-key || exit 8\n'
        'test -z "${WANDB_ENTITY:-}" || exit 9\nprintf "%s\\n" "$@"\n'
    )
    python.chmod(0o700)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RLT_", "STAGE2_"))}
    env["WANDB_ENTITY"] = "another-shared-user-project"
    env["RLT_COBOT_ACTOR_NOISE_SIGMA"] = "0.1"
    env["RLT_COBOT_RESIDUAL_SCALE"] = "0.3"
    if mode == "eval":
        resume = tmp_path / "resume"
        resume.mkdir()
        (resume / "stage2_state.pt").touch()
        env["STAGE2_RESUME_DIR"] = str(resume)
    result = subprocess.run(
        ["bash", str(launcher), task, mode],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    args = result.stdout.splitlines()
    assert f"server.task_prompt={prompt}" in args
    assert f"server.port={port}" in args
    assert f"rlt_feature_model.model_path={tmp_path / model}" in args
    assert "actor.model.num_action_chunks=30" in args
    assert "rlt_feature_model.num_action_chunks=50" in args
    assert "test-private-key" not in result.stdout
    assert "actor.model.actor_noise_sigma=0.1" in args
    assert "actor.model.residual_scale=0.3" in args
    if mode == "offline":
        assert "+offline.allow_actor_reconfiguration=false" in args
        assert not any(
            arg.startswith("+algorithm.offline_sample_ratio=") for arg in args
        )
    if mode in ("train", "eval"):
        assert "+algorithm.offline_sample_ratio=0.1" in args


def test_build_policy_uses_registry_signature():
    main = runpy.run_path(str(ROOT / "examples/embodiment/rlt_stage2_server.py"))
    cfg = OmegaConf.load(
        ROOT / "examples/embodiment/config/cobot_rlt_stage2_ws_server.yaml"
    )
    built = []

    def factory(config):
        built.append(config)
        return torch.nn.Linear(1, 1)

    build = main["build_policy"]
    with (
        patch.dict(
            build.__globals__,
            get_model=factory,
            RLTStage2Trainer=MagicMock(),
            RLTStage2Inference=MagicMock(),
        ),
        patch("torch.cuda.is_available", return_value=False),
    ):
        policy = build(cfg)
    assert [config.precision for config in built] == ["bf16", "fp32"]
    assert policy.metadata["action_space"] == "robot"


def test_real_trainer_updates_with_horizon50_execute30(tmp_path):
    import copy

    import numpy as np

    from rlinf.models import get_model
    from rlinf.serving.rlt.protocol import ChunkIdentity
    from rlinf.serving.rlt.trainer import RLTStage2Trainer

    cfg = OmegaConf.load(
        ROOT / "examples/embodiment/config/cobot_rlt_stage2_ws_server.yaml"
    )
    cfg.actor.model.num_action_chunks = 30
    cfg.actor.model.ref_num_action_chunks = 50
    cfg.actor.global_batch_size = 2
    cfg.runner.logger.log_path = str(tmp_path)
    model = get_model(cfg.actor.model).cpu()
    trainer = RLTStage2Trainer(
        cfg, model=model, target_model=copy.deepcopy(model), device=torch.device("cpu")
    )
    observation = {
        "z_rl": torch.zeros(1, 2048),
        "proprio": torch.zeros(1, 14),
        "ref_chunk": torch.zeros(1, 50, 14),
    }
    before = {key: value.clone() for key, value in model.state_dict().items()}
    for index in range(2):
        trainer.add_transition(
            curr_obs=observation,
            next_obs=observation,
            action_chunk=np.zeros((30, 14), np.float32),
            rewards=np.full(30, 0.1, np.float32),
            done=index == 1,
            bootstrap_mask=0.0 if index == 1 else 1.0,
            intervention=False,
            identity=ChunkIdentity(chunk_id=index + 1),
            action_source=0,
        )
    metrics = trainer.train(2)
    assert trainer.update_step == 2
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before[key], value) for key, value in model.state_dict().items()
    )
