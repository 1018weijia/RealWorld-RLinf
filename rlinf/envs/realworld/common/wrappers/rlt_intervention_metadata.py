# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Normalize real-world intervention and terminal metadata for RLT replay."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType


class RLTInterventionMetadataWrapper(gym.Wrapper):
    """Attach device-independent intervention metadata to environment steps.

    A teleoperation wrapper may set ``info['intervene_action']`` to the action
    actually sent to the robot and optionally ``info['intervene_flag']``. This
    wrapper validates that contract and emits fields consumed by the RLT
    transition pipeline. Hardware drivers remain responsible for local safety
    and emergency-stop behavior.
    """

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        executed_action = np.asarray(
            info.get("executed_action", info.get("intervene_action", action)),
            dtype=np.float32,
        )
        expected_shape = np.asarray(action).shape
        if executed_action.shape != expected_shape:
            raise ValueError(
                "intervene_action must have the same shape as the proposed action: "
                f"expected {expected_shape}, got {executed_action.shape}."
            )

        intervened = bool(
            np.asarray(info.get("intervene_flag", "intervene_action" in info)).any()
        )
        if intervened:
            info["intervene_action"] = executed_action
        else:
            info.pop("intervene_action", None)
        info["executed_action"] = executed_action
        info["intervene_flag"] = intervened
        info["rlt_action_source"] = "human" if intervened else "policy"

        terminal_reason = str(info.get("rlt_terminal_reason", ""))
        safety_terminal = bool(info.get("rlt_safety_fault", False))
        operator_abort = bool(info.get("rlt_operator_abort", False))
        if safety_terminal or operator_abort:
            terminated = True
            terminal_reason = terminal_reason or (
                "safety" if safety_terminal else "operator_abort"
            )

        info["rlt_terminal_reason"] = terminal_reason
        info["rlt_bootstrap_mask"] = 0.0 if safety_terminal or operator_abort else 1.0
        return obs, reward, terminated, truncated, info
