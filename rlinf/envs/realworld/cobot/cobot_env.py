# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generic Gym adapter for a hardware-specific Cobot controller."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .control import (
    CobotControlAdapter,
    CobotObservation,
    MockRewindAdapter,
    validate_action,
)


class CobotEnv(gym.Env):
    """Expose a chunk-oriented Cobot controller through the RLinf env contract.

    ``override_cfg.controller_factory`` may name a callable
    ``"package.module:function"``. The callable receives ``override_cfg`` and
    ``hardware_info`` and returns a :class:`CobotControlAdapter`. A dummy
    controller is used when ``is_dummy=True`` so configuration and replay tests
    do not require hardware.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Any = None,
        hardware_info: Any = None,
        env_idx: int = 0,
        env_cfg: Any = None,
    ):
        del worker_info, env_idx, env_cfg
        config_values = dict(override_cfg)
        config_values.setdefault("is_dummy", True)
        config_values.setdefault("end_effector_type", "cobot")
        self.config = SimpleNamespace(**config_values)
        self._config_values = config_values
        self._task_description = str(
            config_values.get("task_description", "complete the Cobot task")
        )
        self.action_dim = int(config_values.get("action_dim", 14))
        if self.action_dim <= 0:
            raise ValueError("Cobot action_dim must be positive")
        self.is_dummy = bool(config_values.get("is_dummy", True))
        self._at_chunk_boundary = False
        self._recovery_next_chunk = False
        self._recovery_current_chunk = False
        self._adapter = self._build_adapter(hardware_info)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        image_shape = tuple(self._config_values.get("image_shape", (224, 224, 3)))
        if len(image_shape) != 3 or any(int(value) <= 0 for value in image_shape):
            raise ValueError("Cobot image_shape must contain three positive dimensions")
        self._image_keys = tuple(
            self._config_values.get(
                "image_keys", ("image", "wrist_image", "side_image")
            )
        )
        if not self._image_keys or len(set(self._image_keys)) != len(self._image_keys):
            raise ValueError("Cobot image_keys must be non-empty and unique")
        state_dim = int(self._config_values.get("state_dim", 14))
        if state_dim <= 0:
            raise ValueError("Cobot state_dim must be positive")
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Dict(
                    {
                        "proprio": spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(state_dim,),
                            dtype=np.float32,
                        )
                    }
                ),
                "frames": spaces.Dict(
                    {
                        key: spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
                        for key in self._image_keys
                    }
                ),
            }
        )

    def _build_adapter(self, hardware_info: Any) -> CobotControlAdapter:
        factory_path = self._config_values.get("controller_factory")
        if self.is_dummy or not factory_path:
            return MockRewindAdapter(
                self.action_dim,
                self._task_description,
                history_size=max(
                    1, int(self._config_values.get("rewind_history_chunks", 12))
                ),
                session_id=self._config_values.get("session_id"),
            )
        module_name, function_name = str(factory_path).rsplit(":", 1)
        factory = getattr(importlib.import_module(module_name), function_name)
        adapter = factory(self._config_values, hardware_info)
        # Protocols are structural and intentionally not runtime-checkable.
        required = ("action_dim", "reset", "observe", "execute", "stop", "close")
        if any(not hasattr(adapter, name) for name in required):
            raise TypeError(f"Cobot controller must implement {required}.")
        if int(adapter.action_dim) != self.action_dim:
            raise ValueError(
                f"Controller action_dim={adapter.action_dim} does not match config action_dim={self.action_dim}."
            )
        if self._config_values.get("rewind_history_chunks", 0) and any(
            not hasattr(adapter, method)
            for method in ("rewind_chunks", "poll_rewind_event")
        ):
            raise TypeError(
                "Cobot physical rewind requires controller methods "
                "rewind_chunks() and poll_rewind_event()."
            )
        return adapter

    @property
    def task_description(self) -> str:
        return self._task_description

    def _to_raw_obs(self, observation: CobotObservation) -> dict[str, Any]:
        missing_images = set(self._image_keys) - set(observation.images)
        if missing_images:
            raise ValueError(
                f"Cobot adapter observation is missing image keys: {sorted(missing_images)}"
            )
        frames = {key: np.asarray(observation.images[key]) for key in self._image_keys}
        raw_obs = {
            "state": {"proprio": np.asarray(observation.state, dtype=np.float32)},
            "frames": frames,
        }
        if not self.observation_space.contains(raw_obs):
            raise ValueError(
                "Cobot adapter observation does not match the configured state/image schema."
            )
        return raw_obs

    def on_action_chunk_begin(self) -> None:
        """Open one policy chunk and expose a boundary-only rewind poll."""

        self._at_chunk_boundary = True
        self._recovery_current_chunk = self._recovery_next_chunk
        self._recovery_next_chunk = False
        callback = getattr(self._adapter, "on_action_chunk_begin", None)
        if callable(callback):
            callback()

    def on_action_chunk_end(self, committed: bool) -> None:
        """Close one policy chunk after the vector environment aggregates it."""

        self._at_chunk_boundary = False
        callback = getattr(self._adapter, "on_action_chunk_end", None)
        if callable(callback):
            callback(bool(committed))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._at_chunk_boundary = False
        self._recovery_next_chunk = False
        self._recovery_current_chunk = False
        return self._to_raw_obs(self._adapter.reset()), {}

    def step(self, action):
        action = validate_action(action, self.action_dim)
        rewind_event = None
        if self._at_chunk_boundary and hasattr(self._adapter, "poll_rewind_event"):
            rewind_event = self._adapter.poll_rewind_event()
            if rewind_event is not None:
                if rewind_event.mode == "exit":
                    observation = self._adapter.rewind_chunks(
                        rewind_event.chunks_rewound
                    )
                else:
                    observation = self._adapter.observe()
                self._recovery_next_chunk = True
                self._recovery_current_chunk = False
                return (
                    self._to_raw_obs(observation),
                    0.0,
                    False,
                    False,
                    {
                        "rlt_rewind_event": rewind_event,
                        "rlt_event_only": True,
                        "record_transition": False,
                        "rlt_bootstrap_mask": 1.0,
                        "executed_action": np.zeros_like(action),
                        "intervene_flag": False,
                    },
                )
        self._at_chunk_boundary = False
        result = self._adapter.execute(action)
        info = dict(result.info)
        if self._recovery_current_chunk:
            info["rlt_recovery_root"] = True
        info.setdefault("intervene_flag", False)
        executed_action = validate_action(
            info.get("executed_action", info.get("intervene_action", action)),
            self.action_dim,
        )
        if np.any(executed_action < -1.0) or np.any(executed_action > 1.0):
            raise ValueError(
                "Cobot adapter executed_action must be in normalized [-1, 1] space."
            )
        info["executed_action"] = executed_action
        if info["intervene_flag"]:
            info.setdefault("intervene_action", executed_action)
        safety_fault = bool(info.get("rlt_safety_fault", False))
        info.setdefault("record_transition", not safety_fault)
        info.setdefault("rlt_bootstrap_mask", 0.0 if safety_fault else 1.0)
        return (
            self._to_raw_obs(result.observation),
            float(result.reward),
            bool(result.terminated),
            bool(result.truncated),
            info,
        )

    def close(self):
        self._adapter.close()

    def controller_state_dict(self) -> dict[str, Any]:
        """Expose optional adapter state for robot-side checkpointing."""

        state_dict = getattr(self._adapter, "state_dict", None)
        if not callable(state_dict):
            raise NotImplementedError(
                "The configured Cobot adapter is not checkpointable"
            )
        return state_dict()

    def load_controller_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optional adapter state before collecting more chunks."""

        load_state_dict = getattr(self._adapter, "load_state_dict", None)
        if not callable(load_state_dict):
            raise NotImplementedError(
                "The configured Cobot adapter is not checkpointable"
            )
        load_state_dict(state)
