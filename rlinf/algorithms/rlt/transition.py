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

from typing import Any

import torch

from rlinf.envs import SupportedEnvType
from rlinf.utils.nested_dict_process import copy_dict_tensor

RLT_OBS_KEYS = ("z_rl", "proprio", "ref_chunk")
RLT_OPTIONAL_OBS_KEYS = ("ref_candidates",)
RLT_TRANSITION_PREFIX = "rlt_transition_"

# Optional progress-aware / branch fields (defaults applied by losses/smoke).
# Values may live on Trajectory tensor fields or inside forward_inputs.
RLT_BRANCH_FIELDS = (
    "bootstrap_mask",
    "branch_id",
    "terminal_type",
    "action_source",
    "progress_label",
    "progress_mask",
    "auto_trigger",
    "anchor_id",
    "rollback_confirmed",
    "rewind_mode",
    "rewind_chunks",
    "rewind_terminal_reward",
    "rewind_prefix_reward",
    "rewind_confidence",
    "recovery_root",
    "rewind_episode_id",
    "rewind_session_id",
    "rewind_env_id",
    "rewind_chunk_id",
    "next_action_override",
    "next_action_override_mask",
    "record_transition",
)

# action_source integer encoding for replay diagnostics
ACTION_SOURCE_VLA = 0
ACTION_SOURCE_POLICY = 1
ACTION_SOURCE_HUMAN = 2

# terminal_type integer encoding
TERMINAL_NONE = 0
TERMINAL_SUCCESS = 1
TERMINAL_FAILURE = 2
TERMINAL_ROLLBACK = 3
TERMINAL_SAFETY = 4

# branch_id integer encoding
BRANCH_MAIN = 0
BRANCH_D2 = 2
BRANCH_D3 = 3
BRANCH_D4 = 4
BRANCH_D5 = 5
BRANCH_SAFETY = 6


def annotate_rlt_branch_fields(
    *,
    batch_size: int,
    device: torch.device | None = None,
    bootstrap_mask: float | torch.Tensor = 1.0,
    branch_id: int | torch.Tensor = BRANCH_MAIN,
    terminal_type: int | torch.Tensor = TERMINAL_NONE,
    action_source: int | torch.Tensor = ACTION_SOURCE_POLICY,
    progress_label: float | torch.Tensor | None = None,
    progress_mask: bool | torch.Tensor = False,
    auto_trigger: bool | torch.Tensor = False,
    rollback_confirmed: float | torch.Tensor | None = None,
    rewind_mode: int | torch.Tensor = 0,
    rewind_chunks: int | torch.Tensor = 0,
    rewind_terminal_reward: float | torch.Tensor = 0.0,
    rewind_prefix_reward: float | torch.Tensor = 0.0,
    rewind_confidence: float | torch.Tensor = 1.0,
    recovery_root: bool | torch.Tensor = False,
    rewind_episode_id: int | torch.Tensor = 0,
    rewind_session_id: int | torch.Tensor = 0,
    rewind_env_id: int | torch.Tensor = 0,
    rewind_chunk_id: int | torch.Tensor = 0,
    next_action_override: torch.Tensor | None = None,
    next_action_override_mask: bool | torch.Tensor = False,
    record_transition: bool | torch.Tensor = True,
) -> dict[str, torch.Tensor]:
    """Build per-transition branch tensors with shape ``[B, 1]`` (or ``[B]``)."""

    def _as_col(
        value: float | int | bool | torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=dtype)
        else:
            tensor = torch.full((batch_size, 1), value, device=device, dtype=dtype)
        if tensor.numel() == 1:
            tensor = tensor.reshape(1, 1).expand(batch_size, 1)
        elif tensor.shape[0] != batch_size:
            raise ValueError(
                f"RLT metadata batch mismatch: expected {batch_size}, got {tensor.shape[0]}"
            )
        return tensor.reshape(batch_size, -1)[:, :1]

    fields = {
        "bootstrap_mask": _as_col(bootstrap_mask, torch.float32),
        "branch_id": _as_col(branch_id, torch.int64),
        "terminal_type": _as_col(terminal_type, torch.int64),
        "action_source": _as_col(action_source, torch.int64),
        "progress_mask": _as_col(progress_mask, torch.bool),
        "auto_trigger": _as_col(auto_trigger, torch.bool),
        "rewind_mode": _as_col(rewind_mode, torch.int64),
        "rewind_chunks": _as_col(rewind_chunks, torch.int64),
        "rewind_terminal_reward": _as_col(rewind_terminal_reward, torch.float32),
        "rewind_prefix_reward": _as_col(rewind_prefix_reward, torch.float32),
        "rewind_confidence": _as_col(rewind_confidence, torch.float32),
        "recovery_root": _as_col(recovery_root, torch.bool),
        "rewind_episode_id": _as_col(rewind_episode_id, torch.int64),
        "rewind_session_id": _as_col(rewind_session_id, torch.int64),
        "rewind_env_id": _as_col(rewind_env_id, torch.int64),
        "rewind_chunk_id": _as_col(rewind_chunk_id, torch.int64),
        "next_action_override_mask": _as_col(next_action_override_mask, torch.bool),
        "record_transition": _as_col(record_transition, torch.bool),
    }
    if next_action_override is not None:
        fields["next_action_override"] = next_action_override

    if progress_label is not None:
        fields["progress_label"] = _as_col(progress_label, torch.float32)
    if rollback_confirmed is not None:
        fields["rollback_confirmed"] = _as_col(rollback_confirmed, torch.float32)
    return fields


def branch_fields_from_env_info(
    env_info: dict[str, Any] | None,
    *,
    batch_size: int,
    device: torch.device | None,
    intervene_flags: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Convert normalized real-world environment metadata into replay fields."""

    info = env_info or {}
    if isinstance(info.get("final_info"), dict):
        info = {**info, **info["final_info"]}

    def _as_batch(value: Any, *, dtype: torch.dtype, default: float | int):
        if value is None:
            return torch.full((batch_size, 1), default, dtype=dtype, device=device)
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
        if tensor.numel() == 1:
            return tensor.reshape(1, 1).expand(batch_size, 1)
        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"RLT env metadata batch mismatch: expected {batch_size}, got {tensor.shape[0]}"
            )
        return tensor.reshape(batch_size, -1)[:, -1:]

    def _first_event(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return next((item for item in value if item is not None), None)
        if hasattr(value, "dtype") and getattr(value.dtype, "kind", None) == "O":
            return next((item for item in value.reshape(-1) if item is not None), None)
        return value

    bootstrap_mask = _as_batch(
        info.get("rlt_bootstrap_mask"), dtype=torch.float32, default=1.0
    )
    if intervene_flags is None:
        human_action = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
    else:
        human_action = (
            torch.as_tensor(intervene_flags, device=device)
            .reshape(batch_size, -1)
            .any(dim=1, keepdim=True)
        )
    event = _first_event(info.get("rlt_rewind_event"))
    mode_name = getattr(event, "mode", "") if event is not None else ""
    rewind_mode = 1 if mode_name == "exit" else 2 if mode_name == "credit" else 0
    safety_or_abort = bootstrap_mask <= 0
    event_bool = event is not None
    record_transition = torch.as_tensor(
        info.get("record_transition", True), device=device, dtype=torch.bool
    )
    if record_transition.numel() == 1:
        record_transition = record_transition.reshape(1, 1).expand(batch_size, 1)
    else:
        if record_transition.shape[0] != batch_size:
            raise ValueError("record_transition must have one row per environment")
        record_transition = record_transition.reshape(batch_size, -1).all(
            dim=1, keepdim=True
        )
    event_only = bool(
        torch.as_tensor(info.get("rlt_event_only", False)).reshape(-1).any()
    )
    if event_bool and event_only:
        record_transition = torch.zeros_like(record_transition)
    next_action_override = info.get("next_action_override")
    next_action_override_mask = info.get("next_action_override_mask", False)
    return annotate_rlt_branch_fields(
        batch_size=batch_size,
        device=device,
        bootstrap_mask=bootstrap_mask,
        branch_id=torch.where(
            safety_or_abort,
            torch.full_like(bootstrap_mask, BRANCH_SAFETY, dtype=torch.int64),
            torch.full_like(bootstrap_mask, BRANCH_MAIN, dtype=torch.int64),
        ),
        terminal_type=torch.where(
            safety_or_abort,
            torch.full_like(bootstrap_mask, TERMINAL_SAFETY, dtype=torch.int64),
            torch.full_like(bootstrap_mask, TERMINAL_NONE, dtype=torch.int64),
        ),
        action_source=torch.where(
            human_action,
            torch.full_like(bootstrap_mask, ACTION_SOURCE_HUMAN, dtype=torch.int64),
            torch.full_like(bootstrap_mask, ACTION_SOURCE_POLICY, dtype=torch.int64),
        ),
        rewind_mode=rewind_mode,
        rewind_chunks=getattr(event, "chunks_rewound", 0),
        rewind_terminal_reward=getattr(event, "terminal_reward", 0.0),
        rewind_prefix_reward=getattr(event, "prefix_reward", 0.0),
        rewind_confidence=getattr(event, "confidence", 1.0),
        recovery_root=bool(info.get("rlt_recovery_root", False)),
        rewind_episode_id=getattr(event, "episode_id", info.get("rlt_episode_id", 0)),
        rewind_session_id=getattr(event, "session_id", info.get("rlt_session_id", 0)),
        rewind_env_id=getattr(event, "env_id", info.get("rlt_env_id", 0)),
        rewind_chunk_id=getattr(event, "chunk_id", info.get("rlt_chunk_id", 0)),
        record_transition=record_transition,
        next_action_override=next_action_override,
        next_action_override_mask=next_action_override_mask,
    )


def use_simulator_transition_replay(cfg: Any) -> bool:
    """Return True for envs that store one replay row per env step."""
    # The WebSocket server config has no `env` section at all: the robot loop
    # lives in a separate process, so there is nothing simulator-like here.
    env_cfg = cfg.get("env", None)
    if env_cfg is None:
        return False
    train_env_cfg = env_cfg.get("train", None)
    if train_env_cfg is None:
        return False
    try:
        return (
            SupportedEnvType(train_env_cfg.get("env_type", ""))
            == SupportedEnvType.MANISKILL_RLT
        )
    except ValueError:
        return False


def extract_rlt_obs_from_forward_inputs(
    forward_inputs: dict[str, Any],
    *,
    transition: bool = False,
) -> dict[str, Any]:
    prefix = RLT_TRANSITION_PREFIX if transition else ""
    missing = [
        f"{prefix}{key}"
        for key in RLT_OBS_KEYS
        if f"{prefix}{key}" not in forward_inputs
    ]
    if missing:
        raise ValueError(
            f"Missing RLT forward_inputs keys: {missing}. Ensure "
            "rollout.rlt_feature_model is configured and the rollout worker "
            "populates RLT features."
        )
    obs = {key: forward_inputs[f"{prefix}{key}"] for key in RLT_OBS_KEYS}
    for key in RLT_OPTIONAL_OBS_KEYS:
        full_key = f"{prefix}{key}"
        if full_key in forward_inputs:
            obs[key] = forward_inputs[full_key]
    return copy_dict_tensor(obs)


def update_rlt_transitions(
    stage_id: int,
    pending_obs: list[dict[str, Any] | None],
    trajectory_builders: list[Any],
    policy_output: Any,
    *,
    cache_current: bool,
    intervene_actions: torch.Tensor | None = None,
    intervene_flags: torch.Tensor | None = None,
    branch_fields: dict[str, Any] | None = None,
) -> None:
    if pending_obs[stage_id] is not None:
        if branch_fields is None:
            branch_fields = annotate_rlt_branch_fields(
                batch_size=pending_obs[stage_id]["ref_chunk"].shape[0],
                device=pending_obs[stage_id]["ref_chunk"].device,
            )
        branch_fields = dict(branch_fields)
        policy_actions = getattr(policy_output, "actions", None)
        if policy_actions is None:
            policy_actions = policy_output.forward_inputs.get("action")
        if policy_actions is None:
            policy_actions = pending_obs[stage_id]["ref_chunk"].reshape(
                pending_obs[stage_id]["ref_chunk"].shape[0], -1
            )
        action_shape = policy_actions.shape
        branch_fields.setdefault(
            "next_action_override", torch.zeros_like(policy_actions)
        )
        branch_fields.setdefault(
            "next_action_override_mask",
            torch.zeros(
                (action_shape[0], 1),
                dtype=torch.bool,
                device=policy_actions.device,
            ),
        )
        branch_fields.setdefault(
            "record_transition",
            torch.ones(
                (action_shape[0], 1),
                dtype=torch.bool,
                device=policy_actions.device,
            ),
        )
        # Human takeover replaces the replay action through the trajectory
        # builder. The frozen VLA reference remains unchanged for residual actor
        # conditioning, EXPO, and preference likelihoods.
        next_obs = extract_rlt_obs_from_forward_inputs(
            policy_output.forward_inputs,
            transition=True,
        )
        trajectory_builders[stage_id].append_transitions(
            pending_obs[stage_id],
            next_obs,
            branch_fields=branch_fields,
        )
        pending_obs[stage_id] = None

    if cache_current:
        pending_obs[stage_id] = extract_rlt_obs_from_forward_inputs(
            policy_output.forward_inputs
        )
