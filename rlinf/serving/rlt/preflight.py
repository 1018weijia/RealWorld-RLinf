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

"""Startup checks that must pass before a robot is allowed to move.

Every failure mode covered here is one that otherwise surfaces late and
misleadingly: a Stage 1 checkpoint whose RLT positional encoding does not match
``rlt_prefix_seq_len`` fails deep inside ``load_state_dict``, missing norm stats
fail on the first inference request, and a leftover placeholder prompt produces
plausible-looking but wrong actions with no error at all.

These run on the config and the checkpoint header only, so they are cheap
enough to run unconditionally at startup and testable without a GPU.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

RLT_POSITION_KEYS = (
    "rlt_module.encoder.prefix_pos_enc",
    "rlt_module.decoder.decoder_pos_enc",
)
"""Checkpoint tensors whose leading dimension is ``rlt_prefix_seq_len``."""

PLACEHOLDER_PROMPTS = ("", "xxxx", "todo", "task", "prompt", "/path/to/task")
"""Prompt values that indicate an unconfigured run rather than a real task."""


class PreflightError(RuntimeError):
    """A startup check failed; the server must not begin serving."""


def read_rlt_prefix_seq_len(weights_path: str) -> int | None:
    """Read the RLT prefix length baked into a Stage 1 checkpoint.

    Args:
        weights_path: Path to the ``full_weights.pt`` produced by Stage 1.

    Returns:
        The positional-encoding length, or ``None`` when the checkpoint has no
        RLT module (a plain SFT warm-start).

    Raises:
        PreflightError: The file is missing or unreadable.
    """
    import torch

    if not os.path.exists(weights_path):
        raise PreflightError(f"Stage 1 checkpoint not found: {weights_path}")
    try:
        # ``mmap`` keeps this cheap: the assemble checkpoint is over 16 GB, and
        # only two tensor headers are needed.
        state = torch.load(
            weights_path, map_location="cpu", mmap=True, weights_only=True
        )
    except Exception as error:  # noqa: BLE001 - reported as a preflight failure
        raise PreflightError(
            f"Could not read Stage 1 checkpoint {weights_path}: {error}"
        ) from error

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    for key in RLT_POSITION_KEYS:
        tensor = state.get(key) if isinstance(state, dict) else None
        if tensor is not None and hasattr(tensor, "shape"):
            return int(tensor.shape[0])
    return None


def check_stage1_checkpoint(
    weights_path: str,
    *,
    configured_prefix_seq_len: int,
    require_rlt: bool = True,
) -> None:
    """Verify the Stage 1 checkpoint matches the configured RLT geometry.

    Args:
        weights_path: Path to the Stage 1 ``full_weights.pt``.
        configured_prefix_seq_len: ``openpi.rlt_prefix_seq_len`` from the config.
        require_rlt: Whether the checkpoint must carry ``rlt_module.*`` weights.

    Raises:
        PreflightError: The checkpoint has no RLT module while one is required,
            or its positional encoding length disagrees with the config.
    """
    actual = read_rlt_prefix_seq_len(weights_path)
    if actual is None:
        if require_rlt:
            raise PreflightError(
                f"Stage 1 checkpoint {weights_path} has no {RLT_POSITION_KEYS[0]}, "
                "so it was not trained with openpi.use_rlt=True. Stage 2 cannot "
                "run on a plain SFT checkpoint."
            )
        logger.warning(
            "Stage 1 checkpoint %s carries no RLT module; rlt_module stays "
            "randomly initialized.",
            weights_path,
        )
        return

    if actual != int(configured_prefix_seq_len):
        raise PreflightError(
            f"openpi.rlt_prefix_seq_len={configured_prefix_seq_len} does not match "
            f"the Stage 1 checkpoint, which was trained with {actual}. The RLT "
            "positional encodings are sized by this value, so loading would fail "
            f"or silently drop weights. Set rlt_prefix_seq_len: {actual}."
        )
    logger.info(
        "Preflight: Stage 1 RLT prefix_seq_len=%d matches %s", actual, weights_path
    )


def check_norm_stats(norm_stats_path: str) -> None:
    """Verify the OpenPI norm stats used for (un)normalization exist.

    Args:
        norm_stats_path: Path to ``norm_stats.json`` or its parent directory.

    Raises:
        PreflightError: The file is absent or does not parse as JSON with the
            ``norm_stats`` payload OpenPI expects.
    """
    import json

    path = norm_stats_path
    if os.path.isdir(path):
        path = os.path.join(path, "norm_stats.json")
    if not os.path.exists(path):
        raise PreflightError(
            f"norm_stats.json not found at {path}. Actions cannot be "
            "de-normalized into robot units without it, and the robot would "
            "receive model-space numbers."
        )
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except Exception as error:  # noqa: BLE001 - reported as a preflight failure
        raise PreflightError(f"Could not parse {path}: {error}") from error

    stats = payload.get("norm_stats", payload)
    if not isinstance(stats, dict) or "actions" not in stats:
        raise PreflightError(
            f"{path} has no 'actions' entry; it was not produced by the OpenPI "
            "norm-stats tool for this dataset."
        )
    logger.info("Preflight: norm stats loaded from %s", path)


def check_task_prompt(prompt: str, *, stage1_prompt: str | None = None) -> None:
    """Reject an unconfigured or inconsistent task prompt.

    Args:
        prompt: The prompt the server will send to the frozen VLA.
        stage1_prompt: ``openpi_data.default_prompt``, i.e. the prompt the
            Stage 1 checkpoint was trained with. When given it must match
            ``prompt``; the two feed different code paths (the observation
            repacker and the OpenPI data config) and disagreement means one of
            them conditions the VLA on a task it never learned.

    Raises:
        PreflightError: The prompt is empty, a known placeholder, or disagrees
            with ``stage1_prompt``.
    """
    normalized = str(prompt).strip().lower()
    if normalized in PLACEHOLDER_PROMPTS:
        raise PreflightError(
            f"Task prompt {prompt!r} is a placeholder. The frozen Stage 1 VLA is "
            "prompt-conditioned, so a wrong prompt yields confident but wrong "
            "actions with no error. Set the prompt the Stage 1 checkpoint was "
            "trained with."
        )
    if stage1_prompt is not None and normalized != str(stage1_prompt).strip().lower():
        raise PreflightError(
            f"server.task_prompt={prompt!r} disagrees with "
            f"openpi_data.default_prompt={stage1_prompt!r}. Set both to the "
            "prompt used by the Stage 1 SFT config that produced this "
            "checkpoint."
        )
    logger.info("Preflight: task prompt = %r", prompt)


def resolve_stage1_weights(model_path: str) -> str:
    """Locate the Stage 1 weights file the OpenPI loader will actually read.

    The loader accepts a checkpoint directory in either of two layouts and
    otherwise falls back to globbing ``*.safetensors``. Pointing ``model_path``
    one level too deep -- at the ``model_state_dict`` directory that holds
    ``full_weights.pt`` -- therefore matches neither layout and dies inside the
    safetensors reader long after startup looked healthy. Resolving the path
    the same way here turns that into an immediate, explicit failure.

    Args:
        model_path: ``rlt_feature_model.model_path`` from the config.

    Returns:
        Absolute path to the ``full_weights.pt`` that will be loaded.

    Raises:
        PreflightError: No known layout matches ``model_path``.
    """
    candidates = (
        os.path.join(model_path, "model_state_dict", "full_weights.pt"),
        os.path.join(model_path, "actor", "model_state_dict", "full_weights.pt"),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    if os.path.isfile(model_path):
        return model_path

    direct = os.path.join(model_path, "full_weights.pt")
    hint = ""
    if os.path.exists(direct):
        hint = (
            " It does contain full_weights.pt directly, so model_path is one "
            "level too deep: point it at the global_step_* directory instead "
            "of its actor/model_state_dict subdirectory."
        )
    raise PreflightError(
        f"No Stage 1 weights under model_path={model_path}. Expected "
        f"model_state_dict/full_weights.pt or "
        f"actor/model_state_dict/full_weights.pt beneath it.{hint}"
    )


def check_norm_stats_configured(openpi_data: Any) -> str:
    """Require an explicit norm-stats path and return it.

    Leaving ``norm_stats_path`` unset is not a benign omission. OpenPI then
    resolves stats relative to the Stage 1 checkpoint directory, which holds
    only ``full_weights.pt``, and falls back to the ``asset_id`` baked into the
    named OpenPI config -- a *different* task's statistics. The load either
    fails or, worse, succeeds with the wrong quantiles and de-normalizes every
    action into wrong robot units.

    Args:
        openpi_data: The ``rlt_feature_model.openpi_data`` config section.

    Returns:
        The configured norm-stats path.

    Raises:
        PreflightError: The path is missing or empty.
    """
    path = None
    if openpi_data is not None:
        path = openpi_data.get("norm_stats_path")
    if not path:
        raise PreflightError(
            "rlt_feature_model.openpi_data.norm_stats_path is not set. Without "
            "it OpenPI looks for norm stats inside the Stage 1 checkpoint "
            "directory and falls back to the named config's default asset_id, "
            "so actions get de-normalized with another task's statistics. "
            "Point it at the norm_stats.json used by the Stage 1 SFT run."
        )
    return str(path)


def check_camera_layout(camera_keys: tuple[str, ...], num_images: int) -> None:
    """Verify the camera layout matches what the Stage 1 transform consumes.

    Args:
        camera_keys: Client camera keys the server will request, main first.
        num_images: ``openpi.num_images_in_input`` from the config.

    Raises:
        PreflightError: The counts disagree, or a key is duplicated.
    """
    if len(camera_keys) != int(num_images):
        raise PreflightError(
            f"Server is configured for cameras {list(camera_keys)} "
            f"({len(camera_keys)}) but openpi.num_images_in_input="
            f"{num_images}. A missing view is filled with a black frame and a "
            "false image mask, which silently degrades the policy."
        )
    if len(set(camera_keys)) != len(camera_keys):
        raise PreflightError(
            f"Duplicate camera keys {list(camera_keys)}; each OpenPI image slot "
            "must come from a distinct camera."
        )
    logger.info("Preflight: camera layout = %s", list(camera_keys))


def report_reference_deviations(
    values: dict[str, Any],
    reference: dict[str, Any],
) -> list[str]:
    """Warn about hyperparameters that deviate from the reference deployment.

    These are not errors: a deliberate sweep should be possible. But an
    accidental deviation (an actor learning rate an order of magnitude off, a
    shallower MLP) is very hard to spot in a long real-robot run, so every
    difference is surfaced once at startup.

    Args:
        values: Effective values on this server.
        reference: Values from the reference remote-franka deployment.

    Returns:
        Human-readable deviation messages, also emitted as warnings.
    """
    messages = []
    for key, expected in reference.items():
        actual = values.get(key)
        if actual is None or actual == expected:
            continue
        messages.append(f"{key}={actual!r} deviates from reference {expected!r}")
    for message in messages:
        logger.warning("Preflight: %s", message)
    return messages
