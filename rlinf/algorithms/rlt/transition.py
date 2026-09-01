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
) -> dict[str, torch.Tensor]:
    """Build per-transition branch tensors with shape ``[B, 1]`` (or ``[B]``)."""

    def _as_col(value: float | int | bool | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=dtype)
        else:
            tensor = torch.full(
                (batch_size, 1), value, device=device, dtype=dtype
            )
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1)
        return tensor.reshape(batch_size, -1)[:, :1]

    fields = {
        "bootstrap_mask": _as_col(bootstrap_mask, torch.float32),
        "branch_id": _as_col(branch_id, torch.int64),
        "terminal_type": _as_col(terminal_type, torch.int64),
        "action_source": _as_col(action_source, torch.int64),
        "progress_mask": _as_col(progress_mask, torch.bool),
        "auto_trigger": _as_col(auto_trigger, torch.bool),
    }
    if progress_label is not None:
        fields["progress_label"] = _as_col(progress_label, torch.float32)
    if rollback_confirmed is not None:
        fields["rollback_confirmed"] = _as_col(rollback_confirmed, torch.float32)
    return fields


def use_simulator_transition_replay(cfg: Any) -> bool:
    """Return True for envs that store one replay row per env step."""
    train_env_cfg = cfg.env.get("train", None)
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
    return copy_dict_tensor(
        {key: forward_inputs[f"{prefix}{key}"] for key in RLT_OBS_KEYS}
    )


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
        if intervene_actions is not None and intervene_flags is not None:
            current_obs = pending_obs[stage_id]
            ref_chunk = current_obs["ref_chunk"]
            batch_size = ref_chunk.shape[0]
            flags = intervene_flags.reshape(batch_size, -1, 1).to(
                device=ref_chunk.device, dtype=torch.bool
            )
            human_actions = intervene_actions.reshape(batch_size, flags.shape[1], -1)
            action_dim = human_actions.shape[-1]
            ref_actions = ref_chunk.reshape(batch_size, -1, action_dim).clone()
            ref_actions[:, : flags.shape[1]] = torch.where(
                flags,
                human_actions.to(device=ref_chunk.device, dtype=ref_chunk.dtype),
                ref_actions[:, : flags.shape[1]],
            )
            current_obs["ref_chunk"] = ref_actions.reshape_as(ref_chunk)
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
