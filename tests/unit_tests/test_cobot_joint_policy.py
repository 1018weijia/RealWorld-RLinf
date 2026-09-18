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


"""Native actor/critic, Cal-QL, replay and checkpoint tests for joint motion v2."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.mlp_policy import get_model
from rlinf.models.embodiment.mlp_policy.cobot_joint_motion import ARM
from rlinf.serving.rlt.cobot_offline_data import FORMAT, contract
from rlinf.serving.rlt.cobot_offline_trainer import (
    CobotOfflineTrainer,
    conservative_gap,
    joint_budget_bc,
    joint_loss_options,
)
from rlinf.serving.rlt.protocol import ChunkIdentity


def configuration(tmp_path):
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.merge(
        OmegaConf.load(
            root / "examples/embodiment/config/cobot_rlt_stage2_ws_server.yaml"
        ),
        OmegaConf.load(root / "examples/embodiment/config/cobot_joint_motion_v2.yaml"),
    )
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
    cfg.actor.model.mlp_hidden_dim = 32
    cfg.actor.global_batch_size = 4
    cfg.runner.logger.log_path = str(tmp_path / "logs")
    cfg.offline = {"warmup_steps": 1, "steps": 20}
    return cfg


def observation(count=8):
    torch.manual_seed(25)
    scale = torch.linspace(0.1, 1.6, 14).repeat(count, 1)
    scale[:, 1::2] *= -1
    return {
        "z_rl": torch.randn(count, 8),
        "proprio": torch.zeros(count, 14),
        "ref_chunk": torch.randn(count, 50, 14),
        "ref_candidates": torch.randn(count, 4, 50, 14),
        "motion_context": torch.cat((scale, torch.zeros(count, 28)), dim=-1),
    }


def assert_feasible(model, obs, normalized):
    motion = model.joint_motion
    scale, offset, anchor = motion.context(model._state(obs))
    physical = normalized.reshape(-1, 30, 14) * scale[:, None] + offset[:, None]
    velocity, acceleration = motion.differences(physical[:, :, ARM], anchor[:, ARM])
    assert torch.all(velocity.abs() <= motion.velocity_rad_s / 30 + 1e-5)
    assert torch.all(acceleration.abs() <= motion.acceleration_rad_s2 / 900 + 1e-5)
    assert velocity[:, -1].abs().max() < 1e-5


@pytest.mark.parametrize("objective_version", ["legacy", "joint-loss-v3"])
def test_candidates_gradients_offline_online_resume(tmp_path, objective_version):
    torch.set_num_threads(1)
    cfg = configuration(tmp_path)
    cfg.offline.update(
        {
            "objective_version": objective_version,
            "q_warmup_steps": 1,
            "q_ramp_steps": 1,
            "mc_warmup_steps": 1,
            "gradient_every": 1,
        }
    )
    cfg.algorithm.q_weight = 0.01
    cfg.algorithm.offline_sample_ratio = 0.5
    model = get_model(cfg.actor.model, torch.float32)
    obs = observation()
    for noise in (False, True):
        candidates, _ = model.build_expo_candidates(obs, exploration=noise)
        for i in range(8):
            assert_feasible(model, obs, candidates[:, i])
    trainer = CobotOfflineTrainer(
        cfg, model=model, target_model=copy.deepcopy(model), device=torch.device("cpu")
    )
    rows = {
        "curr_obs": obs,
        "next_obs": copy.deepcopy(obs),
        "actions": model.actor.mean(
            model._state(obs), model._get_ref_chunk(obs)
        ).detach()
        + 0.001,
        "rewards": torch.zeros(8, 30),
        "dones": torch.ones(8, 1, dtype=torch.bool),
        "terminations": torch.ones(8, 1, dtype=torch.bool),
        "bootstrap_mask": torch.zeros(8, 1),
        "mc_returns": torch.ones(8, 1),
        "success": torch.ones(8, 1, dtype=torch.bool),
        "episode_index": torch.tensor([0] * 4 + [1] * 4),
    }
    rows["rewards"][:, -1] = 1
    payload = {
        "format": FORMAT,
        "contract": contract(cfg),
        "rows": rows,
        "validation_episodes": [1],
    }
    trainer.offline_mode = True
    trainer.attach_offline_buffer(payload)
    if objective_version == "joint-loss-v3":
        cached = trainer.offline_buffer.rows["curr_obs"]
        torch.testing.assert_close(
            model._get_ref_chunk(cached), model._get_ref_chunk(obs)
        )
        torch.testing.assert_close(
            model._get_ref_candidates(cached), model._get_ref_candidates(obs)
        )
        assert "joint_projected_ref" not in payload["rows"]["curr_obs"]
        assert "joint_projected_candidates" not in payload["rows"]["next_obs"]
    before = [p.detach().clone() for p in model.actor.parameters()]
    for step in range(3):
        metrics = trainer.update_once(train_actor=True)
        trainer.offline_total_updates += 1
        assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.actor.parameters()))
    assert_feasible(
        model, obs, model.actor.mean(model._state(obs), model._get_ref_chunk(obs))
    )
    rng = torch.get_rng_state()
    validation = trainer.validate_offline()
    if objective_version == "joint-loss-v3":
        assert torch.equal(rng, torch.get_rng_state())
        assert validation == trainer.validate_offline()
        assert not trainer.alpha_optimizer.state
        assert metrics["actor/q_weight"] == pytest.approx(0.01)
        assert "actor/bc_grad_norm" in metrics
    assert validation["validation/max_acceleration_rad_s2"] <= 4.001
    directory = tmp_path / "checkpoint"
    trainer.save(str(directory))
    restored_model = get_model(cfg.actor.model, torch.float32)
    restored = CobotOfflineTrainer(
        cfg,
        model=restored_model,
        target_model=copy.deepcopy(restored_model),
        device=torch.device("cpu"),
    )
    restored.load(str(directory))
    assert restored.offline_total_updates == 3
    torch.testing.assert_close(
        restored_model.actor.mean(
            restored_model._state(obs), restored_model._get_ref_chunk(obs)
        ),
        model.actor.mean(model._state(obs), model._get_ref_chunk(obs)),
    )
    sample = trainer.offline_buffer.sample(1, torch.device("cpu"))
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
    result = restored.train(1)
    assert all(np.isfinite(v) for v in result.values())
    assert restored.update_step == 1
    if objective_version == "joint-loss-v3":
        assert "actor/offline_demo_bc" in result
        assert restored.objective == trainer.objective
        offline_cfg = copy.deepcopy(cfg)
        offline_cfg.offline.bc_weight = 0.5
        changed_model = get_model(offline_cfg.actor.model, torch.float32)
        changed_trainer = CobotOfflineTrainer(
            offline_cfg,
            model=changed_model,
            target_model=copy.deepcopy(changed_model),
            device=torch.device("cpu"),
        )
        changed_trainer.offline_mode = True
        with pytest.raises(ValueError, match="loss recipe differs"):
            changed_trainer.load(str(directory))
    assert_feasible(
        restored_model,
        obs,
        restored_model.actor.mean(
            restored_model._state(obs), restored_model._get_ref_chunk(obs)
        ),
    )
    bad = copy.deepcopy(payload)
    del bad["contract"]["actor_model"]["joint_motion"]
    with pytest.raises(ValueError, match="differs"):
        restored.attach_offline_buffer(bad)
    changed = copy.deepcopy(cfg.actor.model)
    changed.joint_motion.residual_rad[0] = 0.05
    with pytest.raises(ValueError, match="limits differ"):
        get_model(changed, torch.float32).load_state_dict(model.state_dict())


def test_fixed_conservative_weight_corrects_overestimated_data_q():
    options = joint_loss_options({"objective_version": "joint-loss-v3"})
    data = torch.full((1, 2), 10.0, requires_grad=True)
    policy = torch.full((1, 4, 2), -1.0)
    other = torch.full((1, 8, 2), -1.0)
    td = (
        torch.nn.functional.huber_loss(
            data, torch.zeros_like(data), delta=0.5, reduction="none"
        )
        .sum(-1)
        .mean()
    )
    gap = conservative_gap(policy, other, data, torch.full((1, 1), 0.1))
    (td + options["calql_alpha"] * (gap - 0.05)).backward()
    assert (data.grad > 0).all(), (
        "Gradient descent must reduce already overestimated data Q"
    )
    with pytest.raises(ValueError, match="calql_alpha"):
        joint_loss_options({"objective_version": "joint-loss-v3", "calql_alpha": 2.718})


def test_family_calibration_stops_penalizing_low_q_without_clipping_network_values():
    policy = torch.full((1, 4, 2), -2.0, requires_grad=True)
    other = torch.full((1, 8, 2), -3.0, requires_grad=True)
    data = torch.zeros(1, 2, requires_grad=True)
    conservative_gap(
        policy, other, data, torch.full((1, 1), 0.1), calibrate_other=True
    ).backward()
    assert policy.grad.abs().sum() == 0
    assert other.grad.abs().sum() == 0
    assert data.grad.sum() == -1
    assert (policy == -2).all() and (other == -3).all()


def test_budget_bc_uses_physical_units_success_mask_and_ignores_grippers(tmp_path):
    cfg = configuration(tmp_path)
    model = get_model(cfg.actor.model, torch.float32)
    obs = observation(2)
    scale = obs["motion_context"][:, :14]
    target = torch.zeros(2, 30, 14)
    prediction = target.clone()
    prediction[:, :, ARM] = 0.5 * model.joint_motion.residual_rad / scale[:, None, ARM]
    prediction[:, :, [6, 13]] = 100
    prediction[1] *= 100
    loss = joint_budget_bc(
        model,
        obs,
        prediction.flatten(1),
        target.flatten(1),
        torch.tensor([True, False]),
    )
    assert loss == pytest.approx(0.25)
    zero = joint_budget_bc(
        model,
        obs,
        prediction.flatten(1),
        target.flatten(1),
        torch.tensor([False, False]),
    )
    assert zero == 0
