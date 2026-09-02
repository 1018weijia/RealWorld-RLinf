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

"""Build Stage-2 RLT transition batches from offline demo tensors."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rlinf.algorithms.rlt.progress_head import voc_progress_labels
from rlinf.algorithms.rlt.transition import (
    ACTION_SOURCE_VLA,
    BRANCH_MAIN,
    TERMINAL_FAILURE,
    TERMINAL_NONE,
    TERMINAL_SUCCESS,
    annotate_rlt_branch_fields,
)


def build_offline_demo_batch(
    *,
    z_rl: Tensor,
    proprio: Tensor,
    ref_chunk: Tensor,
    actions: Tensor,
    success: bool,
    chunk_length: int | None = None,
) -> dict[str, Any]:
    """Convert a single-episode chunk sequence into a Stage-2 training batch.

    Args:
        z_rl: ``[T, Z]`` RL-token features (one row per action chunk).
        proprio: ``[T, P]`` proprioception.
        ref_chunk: ``[T, C, A]`` VLA reference chunks (or longer horizon truncated).
        actions: ``[T, C, A]`` executed / demo action chunks.
        success: Whether the episode succeeded (terminal reward +1 vs 0).
        chunk_length: Optional truncate length ``C`` for RL.

    Returns:
        A flat batch dict suitable for ``compute_rlt_critic_loss`` /
        ``compute_rlt_actor_loss``.
    """
    if z_rl.dim() != 2:
        raise ValueError(f"z_rl must be [T, Z], got {tuple(z_rl.shape)}")
    t = z_rl.shape[0]
    if proprio.shape[0] != t or ref_chunk.shape[0] != t or actions.shape[0] != t:
        raise ValueError(
            "z_rl/proprio/ref_chunk/actions must share leading T, got "
            f"{tuple(z_rl.shape)}, {tuple(proprio.shape)}, "
            f"{tuple(ref_chunk.shape)}, {tuple(actions.shape)}"
        )

    if ref_chunk.dim() == 2:
        # Flat [T, C*A] -> infer C from actions if possible.
        if actions.dim() != 3:
            raise ValueError("actions must be [T, C, A] when ref_chunk is flat")
        c, a = actions.shape[1], actions.shape[2]
        ref_chunk = ref_chunk.reshape(t, -1, a)[:, :c]
    if actions.dim() != 3 or ref_chunk.dim() != 3:
        raise ValueError("actions and ref_chunk must be [T, C, A]")

    c = int(actions.shape[1] if chunk_length is None else chunk_length)
    a = int(actions.shape[2])
    ref_chunk = ref_chunk[:, :c, :a]
    actions = actions[:, :c, :a]

    if t < 2:
        # Duplicate the only frame so curr/next are defined.
        z_rl = z_rl.repeat(2, 1)
        proprio = proprio.repeat(2, 1)
        ref_chunk = ref_chunk.repeat(2, 1, 1)
        actions = actions.repeat(2, 1, 1)
        t = 2

    # Transitions use consecutive chunk indices 0..T-2.
    n = t - 1
    curr_obs = {
        "z_rl": z_rl[:-1],
        "proprio": proprio[:-1],
        "ref_chunk": ref_chunk[:-1],
    }
    next_obs = {
        "z_rl": z_rl[1:],
        "proprio": proprio[1:],
        "ref_chunk": ref_chunk[1:],
    }
    step_actions = actions[:-1]
    rewards = torch.zeros(n, c, dtype=torch.float32)
    terminations = torch.zeros(n, 1, dtype=torch.bool)
    bootstrap = torch.ones(n, 1, dtype=torch.float32)
    progress = voc_progress_labels(t)[:-1]
    progress_mask = torch.ones(n, 1, dtype=torch.bool)

    # Terminal credit on the last transition of the episode.
    if success:
        rewards[-1, -1] = 1.0
        terminal_type = TERMINAL_SUCCESS
    else:
        terminal_type = TERMINAL_FAILURE
    terminations[-1] = True
    bootstrap[-1] = 0.0

    branch = annotate_rlt_branch_fields(
        batch_size=n,
        bootstrap_mask=bootstrap,
        branch_id=BRANCH_MAIN,
        terminal_type=torch.full((n, 1), TERMINAL_NONE, dtype=torch.int64),
        action_source=ACTION_SOURCE_VLA,
        progress_label=progress,
        progress_mask=progress_mask,
        auto_trigger=False,
    )
    branch["terminal_type"][-1] = terminal_type

    return {
        "curr_obs": curr_obs,
        "next_obs": next_obs,
        "actions": step_actions,
        "rewards": rewards,
        "terminations": terminations,
        "intervene_flags": torch.zeros(n, c, dtype=torch.bool),
        **branch,
    }


def make_synthetic_offline_batch(
    *,
    num_chunks: int = 5,
    z_dim: int = 16,
    proprio_dim: int = 4,
    action_dim: int = 4,
    chunk_length: int = 3,
    success: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Random-tensor offline batch for unit tests (no dataset / feature model)."""
    g = torch.Generator().manual_seed(seed)
    z_rl = torch.randn(num_chunks, z_dim, generator=g)
    proprio = torch.randn(num_chunks, proprio_dim, generator=g)
    ref_chunk = torch.randn(num_chunks, chunk_length, action_dim, generator=g)
    actions = torch.randn(num_chunks, chunk_length, action_dim, generator=g)
    return build_offline_demo_batch(
        z_rl=z_rl,
        proprio=proprio,
        ref_chunk=ref_chunk,
        actions=actions,
        success=success,
        chunk_length=chunk_length,
    )
