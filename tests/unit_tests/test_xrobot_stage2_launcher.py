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

"""XRobot USB launcher keeps A–E geometry and isolates site paths."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROMPT = "Bimanual usb pick and insert"


@pytest.mark.parametrize(
    "mode", ["preflight", "train", "eval", "eval-stage1", "offline"]
)
def test_usb_launcher_argv_does_not_override_actor_geometry(tmp_path, mode):
    scripts = tmp_path / "examples/embodiment"
    scripts.mkdir(parents=True)
    launcher = scripts / "start_xrobot_stage2.sh"
    shutil.copyfile(ROOT / "examples/embodiment/start_xrobot_stage2.sh", launcher)
    private = tmp_path / ".private-xrobot-stage2"
    private.mkdir()
    checkpoint = tmp_path / "usb_stage1"
    (checkpoint / "actor/model_state_dict").mkdir(parents=True)
    (checkpoint / "actor/model_state_dict/full_weights.pt").touch()
    stats = checkpoint / "norm_stats.json"
    stats.write_text("{}")
    dataset = tmp_path / "XRobot_USB"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text("{}")
    (private / "paths.env").write_text(
        "\n".join(
            [
                f"XROBOT_USB_STAGE1_CHECKPOINT={checkpoint}",
                f"XROBOT_USB_NORM_STATS={stats}",
                f"XROBOT_USB_DATASET={dataset}",
            ]
        )
    )
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o700)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RLT_", "STAGE2_"))}
    env.pop("WANDB_PROJECT", None)
    env.pop("OFFLINE_BUFFER", None)
    env.pop("XROBOT_USB_OFFLINE_BUFFER", None)
    env.pop("XROBOT_OFFLINE_BUFFER_ROOT", None)
    env["XROBOT_OFFLINE_BUFFER_ROOT"] = str(tmp_path / "offline_rl_buffers")
    if mode == "eval":
        resume = tmp_path / "resume"
        resume.mkdir()
        (resume / "stage2_state.pt").touch()
        env["STAGE2_RESUME_DIR"] = str(resume)
    result = subprocess.run(
        ["bash", str(launcher), "usb_plug", mode],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    args = result.stdout.splitlines()
    assert "--config-name" in args
    assert "xrobot_usb_plug_rlt_stage2_ws_server" in args
    assert f"server.task_prompt={PROMPT}" in args
    assert "server.port=8016" in args
    assert f"rlt_feature_model.model_path={checkpoint}" in args
    assert not any(arg.startswith("actor.model.residual_scale=") for arg in args)
    assert not any(arg.startswith("actor.model.actor_noise_sigma=") for arg in args)
    assert not any(arg.startswith("actor.model.num_action_chunks=") for arg in args)
    assert "runner.logger.project_name=xrobot-usb-offline" in args
    if mode == "offline":
        assert "runner.logger.experiment_name=usb_plug-offline-calql" in args
        assert "+offline.steps=40000" in args
        assert (
            f"+offline.buffer={tmp_path / 'offline_rl_buffers' / 'xrobot_usb_plug' / 'offline_buffer.pt'}"
            in args
        )
        assert not any(
            arg.startswith("+algorithm.offline_sample_ratio=") for arg in args
        )
    else:
        assert "runner.logger.experiment_name=xrobot-stage2-usb_plug" in args
    if mode in ("train", "eval"):
        assert "+algorithm.offline_sample_ratio=0.1" in args
    if mode == "eval-stage1":
        assert "server.eval_only=True" in args
        assert "+server.vla_only=True" in args


def test_usb_offline_refuses_existing_run_directory(tmp_path):
    scripts = tmp_path / "examples/embodiment"
    scripts.mkdir(parents=True)
    launcher = scripts / "start_xrobot_stage2.sh"
    shutil.copyfile(ROOT / "examples/embodiment/start_xrobot_stage2.sh", launcher)
    private = tmp_path / ".private-xrobot-stage2"
    private.mkdir()
    checkpoint = tmp_path / "usb_stage1"
    (checkpoint / "actor/model_state_dict").mkdir(parents=True)
    (checkpoint / "actor/model_state_dict/full_weights.pt").touch()
    stats = checkpoint / "norm_stats.json"
    stats.write_text("{}")
    (private / "paths.env").write_text(
        "\n".join(
            [
                f"XROBOT_USB_STAGE1_CHECKPOINT={checkpoint}",
                f"XROBOT_USB_NORM_STATS={stats}",
            ]
        )
    )
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o700)
    existing = tmp_path / "existing_run"
    existing.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RLT_", "STAGE2_"))}
    env.pop("WANDB_PROJECT", None)
    env.pop("OFFLINE_BUFFER", None)
    env.pop("XROBOT_USB_OFFLINE_BUFFER", None)
    env.pop("XROBOT_OFFLINE_BUFFER_ROOT", None)
    env["XROBOT_OFFLINE_BUFFER_ROOT"] = str(tmp_path / "offline_rl_buffers")
    env["RLT_RUN_DIR"] = str(existing)
    result = subprocess.run(
        ["bash", str(launcher), "usb_plug", "offline"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Refusing to reuse" in result.stderr
