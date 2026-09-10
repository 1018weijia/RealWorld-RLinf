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

"""Entry point for the RLT Stage 2 online-RL WebSocket server.

One process, no Ray: it loads the frozen Stage 1 VLA and the Stage 2
actor/critic, owns the replay buffers and the optimizers, and answers robot
requests over WebSocket. Training happens inline at ``episode_end``, so the
weights that act are always the weights that were just updated.
"""

from __future__ import annotations

import copy
import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.models import get_model
from rlinf.serving.rlt.inference import (
    CameraLayout,
    RLTObservationRepacker,
    RLTStage2Inference,
)
from rlinf.serving.rlt.policy import RLTStage2Policy
from rlinf.serving.rlt.preflight import (
    check_camera_layout,
    check_stage1_checkpoint,
    check_task_prompt,
    report_reference_deviations,
)
from rlinf.serving.rlt.protocol import ACTION_SPACE_ROBOT, ServerMetadata
from rlinf.serving.rlt.trainer import RLTStage2Trainer
from rlinf.serving.websocket_server import RLinfWebsocketPolicyServer

logger = logging.getLogger(__name__)

REFERENCE_HYPERPARAMETERS = {
    "actor_lr": 3e-5,
    "critic_lr": 3e-4,
    "mlp_num_hidden_layers": 3,
    "critic_use_layer_norm": True,
    "residual_scale": 0.2,
    "action_clip_min": -1.4,
    "action_clip_max": 1.4,
    "warmup_steps": 250,
    "utd_ratio": 5,
    "max_episode_chunks": 150,
    "gamma": 0.99,
    "tau": 0.005,
    "replay_action_space": "normalized",
}
"""Values used by the rlt-openpi remote-franka deployment.

Deviations are warnings, not errors — a deliberate sweep must stay possible.
But an accidental deviation is nearly invisible in a multi-hour robot run, so
every difference is printed once at startup.
"""


def _effective_hyperparameters(cfg: DictConfig) -> dict[str, object]:
    model_cfg = cfg.actor.model
    return {
        "actor_lr": float(cfg.actor.optim.lr),
        "critic_lr": float(cfg.actor.critic_optim.lr),
        "mlp_num_hidden_layers": int(model_cfg.get("mlp_num_hidden_layers", 2)),
        "critic_use_layer_norm": bool(model_cfg.get("critic_use_layer_norm", False)),
        "residual_scale": float(model_cfg.residual_scale),
        "action_clip_min": float(model_cfg.get("action_clip_min", -1.4)),
        "action_clip_max": float(model_cfg.get("action_clip_max", 1.4)),
        "warmup_steps": int(cfg.server.warmup_steps),
        "utd_ratio": int(cfg.server.utd_ratio),
        "max_episode_chunks": int(cfg.server.max_episode_chunks),
        "gamma": float(cfg.algorithm.gamma),
        "tau": float(cfg.algorithm.tau),
        "replay_action_space": str(cfg.server.replay_action_space),
    }


def run_preflight(cfg: DictConfig) -> None:
    """Fail fast on the misconfigurations that only surface mid-episode.

    Any failing check propagates
    :class:`~rlinf.serving.rlt.preflight.PreflightError`, which must abort
    startup: the robot may not be armed against a misconfigured server.

    Args:
        cfg: Full server config.
    """
    feature_cfg = cfg.rlt_feature_model
    weights_path = os.path.join(
        feature_cfg.model_path, "actor", "model_state_dict", "full_weights.pt"
    )
    if not os.path.exists(weights_path):
        weights_path = feature_cfg.model_path

    check_stage1_checkpoint(
        weights_path,
        configured_prefix_seq_len=int(feature_cfg.openpi.rlt_prefix_seq_len),
        require_rlt=bool(feature_cfg.get("require_rlt_checkpoint", True)),
    )
    check_task_prompt(cfg.server.task_prompt)
    check_camera_layout(
        tuple(cfg.server.camera_keys),
        int(feature_cfg.openpi.num_images_in_input),
    )

    values = _effective_hyperparameters(cfg)
    logger.info("Effective Stage 2 hyperparameters:")
    for key in sorted(values):
        logger.info("  %-24s %s", key, values[key])
    if bool(cfg.server.get("check_reference_hyperparameters", True)):
        report_reference_deviations(values, REFERENCE_HYPERPARAMETERS)


def build_policy(cfg: DictConfig) -> RLTStage2Policy:
    """Load both stages and assemble the request router.

    Args:
        cfg: Full server config.

    Returns:
        A policy ready to serve.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading Stage 1 from %s", cfg.rlt_feature_model.model_path)
    feature_model = get_model(cfg.rlt_feature_model, torch_dtype=torch.bfloat16)
    feature_model.to(device).eval()
    for parameter in feature_model.parameters():
        parameter.requires_grad_(False)

    logger.info("Building Stage 2 head (%s)", cfg.actor.model.model_type)
    model = get_model(cfg.actor.model, torch_dtype=torch.float32).to(device)
    target_model = copy.deepcopy(model).to(device)
    target_model.requires_grad_(False)

    trainer = RLTStage2Trainer(
        cfg=cfg,
        model=model,
        target_model=target_model,
        device=device,
        torch_dtype=torch.float32,
    )
    if cfg.runner.get("resume_dir"):
        logger.info("Resuming Stage 2 state from %s", cfg.runner.resume_dir)
        trainer.load(cfg.runner.resume_dir)

    camera_keys = tuple(cfg.server.camera_keys)
    repacker = RLTObservationRepacker(
        camera_layout=CameraLayout(main=camera_keys[0], wrist=camera_keys[1:]),
        proprio_dim=int(cfg.actor.model.proprio_dim),
        default_prompt=str(cfg.server.task_prompt),
    )
    inference = RLTStage2Inference(
        feature_model=feature_model,
        policy_model=model,
        repacker=repacker,
        chunk_len=int(cfg.actor.model.num_action_chunks),
        action_dim=int(cfg.actor.model.action_dim),
        num_ref_candidates=int(cfg.actor.model.expo_num_base_samples),
        device=device,
    )

    metadata = ServerMetadata(
        action_dim=int(cfg.actor.model.action_dim),
        chunk_length=int(cfg.actor.model.num_action_chunks),
        proprio_dim=int(cfg.actor.model.proprio_dim),
        warmup_steps=int(cfg.server.warmup_steps),
        run_name=str(cfg.runner.logger.experiment_name),
        # Chunks always leave the server in robot units; `replay_action_space`
        # only describes what the buffer stores.
        action_space=ACTION_SPACE_ROBOT,
        replay_action_space=str(cfg.server.replay_action_space),
        action_selection_mode=str(cfg.actor.model.action_selection_mode),
        edit_scale=float(cfg.actor.model.residual_scale),
        expo_num_base_samples=int(cfg.actor.model.expo_num_base_samples),
        expo_num_edit_samples=int(cfg.actor.model.expo_num_edit_samples),
        actor_action_clip_min=float(cfg.actor.model.get("action_clip_min", -1.4)),
        actor_action_clip_max=float(cfg.actor.model.get("action_clip_max", 1.4)),
        max_episode_chunks=int(cfg.server.max_episode_chunks),
        camera_keys=camera_keys,
        task_prompt=str(cfg.server.task_prompt),
        eval_only=bool(cfg.server.eval_only),
        use_preference_loss=bool(cfg.algorithm.rewind_preference.enable),
    )

    return RLTStage2Policy(
        trainer=trainer,
        inference=inference,
        metadata=metadata,
        warmup_steps=int(cfg.server.warmup_steps),
        utd_ratio=int(cfg.server.utd_ratio),
        max_episode_chunks=int(cfg.server.max_episode_chunks),
        save_dir=cfg.server.get("save_dir"),
        save_interval_episodes=int(cfg.server.get("save_interval_episodes", 10)),
        eval_only=bool(cfg.server.eval_only),
        replay_action_space=str(cfg.server.replay_action_space),
    )


@hydra.main(version_base="1.1", config_path="config", config_name=None)
def main(cfg: DictConfig) -> None:
    """Run preflight, load both stages and serve until interrupted."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Server config:\n%s", OmegaConf.to_yaml(cfg))

    run_preflight(cfg)
    policy = build_policy(cfg)

    server = RLinfWebsocketPolicyServer(
        policy=policy,
        host=str(cfg.server.host),
        port=int(cfg.server.port),
        metadata=policy.metadata,
        ping_interval=float(cfg.server.get("ping_interval", 20.0)),
        ping_timeout=cfg.server.get("ping_timeout"),
    )
    logger.info(
        "RLT Stage 2 server ready on %s:%s (warmup %d rows, utd %d)",
        cfg.server.host,
        cfg.server.port,
        policy.warmup_steps,
        policy.utd_ratio,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
