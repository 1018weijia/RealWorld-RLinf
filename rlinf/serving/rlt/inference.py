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

"""Server-side Stage 1 encode and Stage 2 action selection.

The robot client sends raw cameras, joint state and a prompt.  Everything from
there is owned by this module: the OpenPI observation repack, the frozen Stage 1
prefix cache, the RLT token features, the Stage 2 EXPO selection over VLA
candidates, and the de-normalization back to robot space.

Keeping the whole pipeline on one side of the socket is what makes the client
thin, and it removes the ``RealWorldEnv``/``EmbodiedTrajectoryBuilder`` hop where
the wrist cameras used to be dropped (they were published as
``extra_view_images``, a key the Aloha input transform ignores, while
``observation/wrist_image`` - which it requires - was never produced).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraLayout:
    """Mapping from client camera names to OpenPI observation slots.

    Attributes:
        main: Client camera feeding ``observation/image`` (Aloha ``cam_high``).
        wrist: Client cameras feeding ``observation/wrist_image``, stacked on a
            new leading axis. The Aloha input transform reads index ``0`` as
            ``cam_left_wrist`` and index ``1`` as ``cam_right_wrist``, so order
            matters and a dual-arm setup must supply exactly two.
    """

    main: str = "image"
    wrist: tuple[str, ...] = ("wrist_image", "side_image")

    @property
    def required_keys(self) -> tuple[str, ...]:
        """Every client camera key this layout consumes."""
        return (self.main, *self.wrist)


class RLTObservationRepacker:
    """Turn a client observation frame into RLinf standardized ``env_obs``.

    Args:
        camera_layout: Client-camera to OpenPI-slot mapping.
        proprio_dim: Expected joint-state dimension.
        default_prompt: Prompt used when the client sends none.
        image_size: Expected ``(height, width)``, or ``None`` to accept any.
    """

    def __init__(
        self,
        camera_layout: CameraLayout,
        proprio_dim: int,
        default_prompt: str,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.camera_layout = camera_layout
        self.proprio_dim = int(proprio_dim)
        self.default_prompt = str(default_prompt)
        self.image_size = image_size

    @staticmethod
    def _image(value: Any, name: str) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"Camera {name!r} must be [H, W, 3], got shape {image.shape}"
            )
        if np.issubdtype(image.dtype, np.floating):
            image = (255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
        return np.ascontiguousarray(image.astype(np.uint8))

    def _lookup_camera(self, observation: dict[str, Any], name: str) -> np.ndarray:
        # Accept both the flat client key and the OpenPI-prefixed spelling so a
        # stock openpi_client robot node works without changes.
        for key in (name, f"observation/{name}", f"observation/{name}_image"):
            if key in observation:
                return self._image(observation[key], name)
        raise ValueError(
            f"Observation is missing camera {name!r}; got keys {sorted(observation)}"
        )

    def to_env_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Repack one client frame into a batch-of-one ``env_obs``.

        Returns:
            Dict with ``states``, ``main_images``, ``wrist_images`` and
            ``task_descriptions``, all carrying a leading batch axis of 1.

        Raises:
            ValueError: A camera or the state is missing or misshaped.
        """
        state = None
        for key in ("state", "observation/state", "states"):
            if key in observation:
                # np.array, not np.asarray: a msgpack-decoded array is a
                # read-only view over the receive buffer, and torch.as_tensor
                # downstream would alias it.
                state = np.array(observation[key], dtype=np.float32).reshape(-1)
                break
        if state is None:
            raise ValueError(
                f"Observation is missing joint state; got keys {sorted(observation)}"
            )
        if state.size != self.proprio_dim:
            raise ValueError(
                f"Observation state dim {state.size} != configured proprio_dim "
                f"{self.proprio_dim}"
            )
        if not np.isfinite(state).all():
            raise ValueError("Observation state contains non-finite values")

        main = self._lookup_camera(observation, self.camera_layout.main)
        wrists = [
            self._lookup_camera(observation, name) for name in self.camera_layout.wrist
        ]
        if self.image_size is not None:
            expected = tuple(self.image_size)
            for name, image in zip(self.camera_layout.required_keys, [main, *wrists]):
                if tuple(image.shape[:2]) != expected:
                    raise ValueError(
                        f"Camera {name!r} has size {image.shape[:2]}, expected "
                        f"{expected}"
                    )

        prompt = observation.get("prompt") or self.default_prompt
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

        return {
            "states": state[None, ...],
            "main_images": main[None, ...],
            # Aloha input transform slices index 0/1 as left/right wrist.
            "wrist_images": np.stack(wrists, axis=0)[None, ...],
            "task_descriptions": [str(prompt)],
        }


@dataclass
class ActionSelection:
    """One Stage 2 decision, in both model and robot action space.

    Attributes:
        normalized_chunk: Chosen action chunk in OpenPI normalized space,
            shaped ``[chunk_len, action_dim]``. This is what the replay buffer
            stores.
        reference_chunk: The Stage 1 VLA candidate the residual edited, same
            shape and space. Needed as the BC target.
        robot_chunk: ``normalized_chunk`` de-normalized into robot units, which
            is what the client executes.
        mode: ``"warmup"``, ``"actor"``, or ``"eval"``.
        expo_info: EXPO diagnostics (candidate counts, chosen index).
    """

    normalized_chunk: np.ndarray
    reference_chunk: np.ndarray
    robot_chunk: np.ndarray
    mode: str
    expo_info: dict[str, Any] = field(default_factory=dict)


class RLTStage2Inference:
    """Frozen Stage 1 encode plus Stage 2 residual action selection.

    Args:
        feature_model: Frozen Stage 1 OpenPI wrapper with the RLT token module.
        policy_model: Stage 2 ``RLTTD3MLPPolicy``.
        repacker: Client-frame to ``env_obs`` adapter.
        chunk_len: Number of chunk steps Stage 2 controls.
        action_dim: Robot action dimension.
        num_ref_candidates: Stage 1 reference chunks sampled per decision, i.e.
            the EXPO base candidate count.
        device: Device both models live on.
    """

    def __init__(
        self,
        *,
        feature_model,
        policy_model,
        repacker: RLTObservationRepacker,
        chunk_len: int,
        action_dim: int,
        num_ref_candidates: int,
        device: torch.device,
    ) -> None:
        self.feature_model = feature_model
        self.policy_model = policy_model
        self.repacker = repacker
        self.chunk_len = int(chunk_len)
        self.action_dim = int(action_dim)
        self.num_ref_candidates = max(1, int(num_ref_candidates))
        self.device = device

    # ------------------------------------------------------------- Stage 1

    @torch.no_grad()
    def encode(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Run the frozen Stage 1 encode for one client frame.

        This mirrors ``OpenPiPytorchEvalActionModel.extract_rlt_obs`` but also
        keeps the raw model-space actions and the normalized OpenPI state, both
        of which are required to de-normalize a Stage 2 edit back to robot
        units.

        Returns:
            Dict with the Stage 2 observation tensors (``z_rl``, ``proprio``,
            ``ref_chunk``, ``ref_candidates``) plus ``_model_actions`` and
            ``_openpi_state`` used only for de-normalization.
        """
        from rlinf.models.embodiment.openpi_rlinf.pi0_model import (
            model as pi0_model_module,
        )

        model = self.feature_model
        model._require_rlt()

        env_obs = self.repacker.to_env_obs(observation)
        repacked = {
            "observation/image": env_obs["main_images"],
            "observation/wrist_image": env_obs["wrist_images"],
            "observation/state": env_obs["states"],
            "prompt": env_obs["task_descriptions"],
        }
        processed = model.input_transform(repacked, transpose=False)
        openpi_obs = model._observation_dict_to_device(processed)

        prepared = pi0_model_module.preprocess_observation(openpi_obs, train=False)
        prefix_output, prefix_mask, kv_cache = model.model.build_prefix_cache(prepared)
        rlt_prefix_output, rlt_prefix_mask = model._select_rlt_prefix_embeddings(
            prefix_output, prefix_mask, prepared.tokenized_prompt
        )
        z_rl = model._encode_rlt_flat(rlt_prefix_output, rlt_prefix_mask).to(
            dtype=torch.float32
        )

        model_actions = [
            model._sample_actions_from_prefix_cache(prepared, prefix_mask, kv_cache)
            for _ in range(self.num_ref_candidates)
        ]
        candidates = [actions[..., : self.action_dim] for actions in model_actions]

        proprio = torch.as_tensor(env_obs["states"])
        return {
            "z_rl": z_rl,
            "proprio": proprio.to(device=z_rl.device, dtype=torch.float32),
            "ref_chunk": candidates[0].to(device=z_rl.device, dtype=torch.float32),
            "ref_candidates": torch.stack(candidates, dim=1).to(
                device=z_rl.device, dtype=torch.float32
            ),
            # De-normalization inputs, stripped before the replay write.
            "_model_actions": model_actions[0],
            "_openpi_state": openpi_obs.state,
        }

    @staticmethod
    def strip_private(rlt_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Drop the de-normalization scratch keys before replay ingestion."""
        return {
            key: value
            for key, value in rlt_obs.items()
            if not key.startswith("_") and torch.is_tensor(value)
        }

    # ------------------------------------------------------------- Stage 2

    @torch.no_grad()
    def select_action(
        self,
        rlt_obs: dict[str, Any],
        *,
        warmup: bool,
        deterministic: bool,
        exploration_noise_sigma: float | None = None,
    ) -> ActionSelection:
        """Pick the chunk to execute for one Stage 2 decision.

        During warmup the Stage 2 head is bypassed entirely and the Stage 1
        reference is executed as-is, so the replay buffer fills with on-manifold
        VLA behaviour before an untrained critic can steer the robot.

        Args:
            rlt_obs: Output of :meth:`encode`.
            warmup: Whether the warmup gate is still active.
            deterministic: Suppress actor noise (evaluation episodes).
            exploration_noise_sigma: Per-request actor-noise override.

        Returns:
            The chosen chunk in normalized and robot space.
        """
        obs = {key: value for key, value in rlt_obs.items() if torch.is_tensor(value)}
        reference = self._reference_chunk(obs)

        if warmup:
            selected = reference
            mode = "warmup"
            expo_info: dict[str, Any] = {"expo_active": False}
        else:
            selected, base, expo_info = self._policy_chunk(
                obs,
                deterministic=deterministic,
                exploration_noise_sigma=exploration_noise_sigma,
            )
            reference = base
            mode = "eval" if deterministic else "actor"

        normalized_chunk = selected.reshape(self.chunk_len, self.action_dim)
        reference_chunk = reference.reshape(self.chunk_len, self.action_dim)

        # EXPO edits whichever of the sampled VLA candidates scored best, not
        # necessarily candidates[0] that encode() parked in ``ref_chunk``. The
        # replay row must carry the candidate the residual was actually
        # measured against, or the actor loss anchors the residual on a chunk
        # the policy never edited and the BC term regresses toward an
        # unrelated sample. This mirrors the write-back the Ray rollout path
        # does in RLTTD3MLPPolicy.predict_action_batch: only the first
        # ``chunk_len`` steps are replaced, the reference tail is preserved.
        stored_ref = rlt_obs["ref_chunk"].reshape(1, -1, self.action_dim).clone()
        stored_ref[:, : self.chunk_len] = reference_chunk.reshape(
            1, self.chunk_len, self.action_dim
        )
        rlt_obs["ref_chunk"] = stored_ref.reshape(rlt_obs["ref_chunk"].shape)

        robot_chunk = self.to_robot_space(normalized_chunk, rlt_obs)
        return ActionSelection(
            normalized_chunk=normalized_chunk.detach().float().cpu().numpy(),
            reference_chunk=reference_chunk.detach().float().cpu().numpy(),
            robot_chunk=robot_chunk,
            mode=mode,
            expo_info=expo_info,
        )

    def _reference_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        ref = obs["ref_chunk"].reshape(1, -1, self.action_dim)[:, : self.chunk_len]
        return ref.reshape(1, -1)

    def _policy_chunk(
        self,
        obs: dict[str, torch.Tensor],
        *,
        deterministic: bool,
        exploration_noise_sigma: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        policy = self.policy_model
        actor = policy.actor
        original_sigma = float(actor.sigma)
        if exploration_noise_sigma is not None:
            actor.sigma = max(0.0, float(exploration_noise_sigma))
        try:
            if policy.action_selection_mode == "expo":
                action, base = policy.select_expo_action(
                    obs, exploration=not deterministic
                )
                candidates, _ = policy.build_expo_candidates(
                    obs, exploration=not deterministic
                )
                info = {
                    "expo_active": True,
                    "expo_num_candidates": int(candidates.shape[1]),
                    "expo_num_base_samples": int(policy.expo_num_base_samples),
                    "expo_num_edit_samples": int(policy.expo_num_edit_samples),
                }
            else:
                action, _, _ = policy.sac_forward(
                    obs,
                    deterministic=deterministic,
                    apply_action_noise=not deterministic,
                )
                base = self._reference_chunk(obs)
                info = {"expo_active": False}
        finally:
            actor.sigma = original_sigma
        return action, base, info

    # -------------------------------------------------------- action space

    @torch.no_grad()
    def to_robot_space(
        self,
        normalized_chunk: torch.Tensor,
        rlt_obs: dict[str, Any],
    ) -> np.ndarray:
        """De-normalize a Stage 2 chunk into robot units.

        The Stage 2 head only edits the first ``action_dim`` model dimensions,
        so the edit is written back into the Stage 1 model action before the
        OpenPI output pipeline (model outputs, ``Unnormalize``, then the
        dataset-specific absolute/Aloha decode) runs unchanged.

        Args:
            normalized_chunk: ``[chunk_len, action_dim]`` in normalized space.
            rlt_obs: Output of :meth:`encode`, carrying the scratch keys.

        Returns:
            ``[chunk_len, action_dim]`` float32 array in robot units.
        """
        model_actions = rlt_obs["_model_actions"]
        openpi_state = rlt_obs["_openpi_state"]
        edited = model_actions.clone()
        chunk = normalized_chunk.to(device=edited.device, dtype=edited.dtype)
        edited[:, : self.chunk_len, : self.action_dim] = chunk
        env_outputs = self.feature_model.output_transform(
            {"actions": edited, "state": openpi_state}
        )
        robot = env_outputs["actions"].detach().float().cpu().numpy()
        return np.ascontiguousarray(
            robot[0, : self.chunk_len, : self.action_dim].astype(np.float32)
        )

    @torch.no_grad()
    def to_normalized_space(
        self,
        robot_chunk: np.ndarray,
        rlt_obs: dict[str, Any],
    ) -> np.ndarray:
        """Map an executed robot-space chunk back to normalized space.

        Human interventions arrive in robot units, but replay rows are stored
        in normalized space so the critic, the BC target and the actor output
        all share one frame.

        :meth:`_affine_to_robot` raises ``RuntimeError`` when the OpenPI output
        pipeline turns out not to be affine in the normalized action, since
        this inverse would then silently corrupt every stored replay action.

        Args:
            robot_chunk: Executed actions in robot units.
            rlt_obs: Output of :meth:`encode`.

        Returns:
            ``[chunk_len, action_dim]`` float32 array in normalized space.
        """
        scale, offset = self._affine_to_robot(rlt_obs)
        robot = np.asarray(robot_chunk, dtype=np.float32).reshape(
            self.chunk_len, self.action_dim
        )
        return np.ascontiguousarray(((robot - offset) / scale).astype(np.float32))

    def _affine_to_robot(
        self, rlt_obs: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and cache ``robot = scale * normalized + offset`` for this frame.

        Every stage of the OpenPI output pipeline used here is affine in the
        normalized action: quantile ``Unnormalize`` scales and shifts,
        ``AbsoluteActions`` adds the current state on the masked dimensions, and
        the Aloha decode is a per-dimension sign/offset map. Two probes through
        the real pipeline therefore recover the map exactly rather than
        approximately, and a third probe checks that assumption instead of
        trusting it.
        """
        cached = rlt_obs.get("_denorm_affine")
        if cached is not None:
            return cached

        zeros = torch.zeros(
            self.chunk_len, self.action_dim, dtype=torch.float32, device=self.device
        )
        offset = self.to_robot_space(zeros, rlt_obs)
        scale = self.to_robot_space(zeros + 1.0, rlt_obs) - offset
        if np.any(np.abs(scale) < 1e-8):
            raise RuntimeError(
                "OpenPI output pipeline has a zero-slope action dimension; "
                "cannot recover normalized actions from robot-space actions."
            )

        probe = zeros + 0.5
        predicted = probe.cpu().numpy() * scale + offset
        actual = self.to_robot_space(probe, rlt_obs)
        max_error = float(np.max(np.abs(predicted - actual)))
        tolerance = 1e-3 * float(np.max(np.abs(scale)))
        if max_error > max(tolerance, 1e-5):
            raise RuntimeError(
                "OpenPI output pipeline is not affine in the normalized action "
                f"(max probe error {max_error:.3e}). Storing normalized replay "
                "actions would corrupt intervention data; set the server's "
                "replay_action_space to 'robot' instead."
            )

        rlt_obs["_denorm_affine"] = (scale, offset)
        return scale, offset
