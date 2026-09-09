# Copyright 2026 The RLinf Authors.

"""Gym registration for the generic Cobot adapter."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.common.wrappers.rlt_intervention_metadata import (
    RLTInterventionMetadataWrapper,
)

from .cobot_env import CobotEnv


def create_cobot_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    """Build a Cobot environment and opt into RLT replay metadata."""

    env = CobotEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
        env_cfg=env_cfg,
    )
    if env_cfg.get("rlt_intervention_metadata", False):
        env = RLTInterventionMetadataWrapper(env)
    return env


register(
    id="CobotEnv-v1", entry_point="rlinf.envs.realworld.cobot.tasks:create_cobot_env"
)
