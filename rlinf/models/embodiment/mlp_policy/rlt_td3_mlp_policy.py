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

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.utils import make_mlp


def _make_td3_mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
    use_layer_norm: bool = False,
) -> nn.Sequential:
    layers = make_mlp(
        in_channels=input_dim,
        mlp_channels=[
            *[hidden_dim for _ in range(num_hidden_layers)],
            output_dim,
        ],
        act_builder=nn.ReLU,
        last_act=False,
        use_layer_norm=use_layer_norm,
    )
    # Keep the historical ablation MLP state_dict shape: mlp.net.*
    return nn.Sequential(OrderedDict([("net", nn.Sequential(*layers))]))


DEFAULT_ACTION_CLIP_MIN = -1.4
DEFAULT_ACTION_CLIP_MAX = 1.4
"""Default residual-action bounds in OpenPI normalized space.

OpenPI quantile normalization maps ``q01``/``q99`` to ``-1``/``+1``, so a
legitimate demonstration action can sit outside ``[-1, 1]``. Clipping the
residual output to ``[-1, 1]`` would also clip the frozen VLA reference
``a_tilde`` it is added to, silently deleting reachable actions. ``+/-1.4``
matches the remote-franka reference deployment.
"""


class DirectGaussianActor(nn.Module):
    """EXPO residual actor conditioned on RLT state and a VLA reference.

    The class name is retained for checkpoint and import compatibility; Stage 2
    actions are residual edits of ``a_tilde``, not unconstrained direct actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        sigma: float = 0.1,
        ref_dropout: float = 0.0,
        edit_scale: float = 0.2,
        action_clip_min: float = DEFAULT_ACTION_CLIP_MIN,
        action_clip_max: float = DEFAULT_ACTION_CLIP_MAX,
    ) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.ref_dropout = float(ref_dropout)
        self.edit_scale = float(edit_scale)
        self.action_clip_min = float(action_clip_min)
        self.action_clip_max = float(action_clip_max)
        if self.action_clip_min >= self.action_clip_max:
            raise ValueError(
                "action_clip_min must be < action_clip_max, got "
                f"[{self.action_clip_min}, {self.action_clip_max}]"
            )
        if self.edit_scale <= 0.0:
            raise ValueError("edit_scale must be positive")
        self.mlp = _make_td3_mlp(
            input_dim=int(state_dim) + int(action_chunk_dim),
            output_dim=int(action_chunk_dim),
            hidden_dim=int(hidden_dim),
            num_hidden_layers=int(num_hidden_layers),
        )
        last_linear = [
            module for module in self.mlp.modules() if isinstance(module, nn.Linear)
        ][-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def _drop_reference(
        self,
        a_tilde: torch.Tensor,
        ref_dropout: float | None = None,
    ) -> torch.Tensor:
        dropout = self.ref_dropout if ref_dropout is None else float(ref_dropout)
        if dropout <= 0.0:
            return a_tilde
        keep_mask = torch.rand(a_tilde.shape[0], 1, device=a_tilde.device) >= dropout
        return a_tilde * keep_mask.to(dtype=a_tilde.dtype)

    def forward(
        self,
        x: torch.Tensor,
        a_tilde: torch.Tensor,
        *,
        deterministic: bool = False,
        apply_ref_dropout: bool | None = None,
        apply_action_noise: bool | None = None,
        ref_dropout: float | None = None,
    ) -> torch.Tensor:
        if apply_ref_dropout is None:
            apply_ref_dropout = False
        if apply_action_noise is None:
            apply_action_noise = not deterministic

        reference = (
            self._drop_reference(a_tilde, ref_dropout=ref_dropout)
            if apply_ref_dropout
            else a_tilde
        )
        residual = self.mlp(torch.cat([x, reference], dim=-1))
        if apply_action_noise and self.sigma > 0.0:
            residual = residual + torch.randn_like(residual) * self.sigma
        action = a_tilde + self.edit_scale * torch.tanh(residual)
        return action.clamp(self.action_clip_min, self.action_clip_max)

    def mean(self, x: torch.Tensor, a_tilde: torch.Tensor) -> torch.Tensor:
        return self.forward(
            x,
            a_tilde,
            deterministic=True,
            apply_ref_dropout=False,
            apply_action_noise=False,
        )


class QNetwork(nn.Module):
    """Single TD3 Q network: (x, action_chunk) -> scalar."""

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.mlp = _make_td3_mlp(
            input_dim=int(state_dim) + int(action_chunk_dim),
            output_dim=1,
            hidden_dim=int(hidden_dim),
            num_hidden_layers=int(num_hidden_layers),
            use_layer_norm=bool(use_layer_norm),
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x, action], dim=-1))


class TwinQCritic(nn.Module):
    """Twin-Q critic matching the ablation TD3 MLP structure."""

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(
            state_dim=state_dim,
            action_chunk_dim=action_chunk_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=use_layer_norm,
        )
        self.q2 = QNetwork(
            state_dim=state_dim,
            action_chunk_dim=action_chunk_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.q1(x, action), self.q2(x, action)], dim=-1)


class RLTTD3MLPPolicy(nn.Module, BasePolicy):
    """Residual TD3 MLP policy over cached RLT features.

    The rollout feature path is unchanged: OpenPI still produces ``z_rl``,
    ``proprio`` and ``ref_chunk``. The actor predicts a bounded edit of each
    VLA reference chunk, while the twin-Q critic ranks EXPO base and edited
    candidates.
    """

    ACTOR_SEMANTIC_VERSION = 2

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_action_chunks: int,
        ref_num_action_chunks: int | None = None,
        add_q_head: bool = True,
        q_head_type: str = "default",
        mlp_hidden_dim: int = 256,
        mlp_num_hidden_layers: int = 2,
        actor_noise_sigma: float = 0.1,
        ref_action_dropout: float = 0.0,
        residual_scale: float = 0.2,
        action_selection_mode: str = "expo",
        expo_num_base_samples: int = 4,
        expo_num_edit_samples: int = 4,
        action_clip_min: float = DEFAULT_ACTION_CLIP_MIN,
        action_clip_max: float = DEFAULT_ACTION_CLIP_MAX,
        critic_use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if not add_q_head:
            raise ValueError("RLTTD3MLPPolicy requires add_q_head=True.")
        if q_head_type != "default":
            raise ValueError(
                "RLTTD3MLPPolicy only supports q_head_type='default', got "
                f"{q_head_type!r}."
            )

        self.z_dim = int(z_dim)
        self.proprio_dim = int(proprio_dim)
        self.step_action_dim = int(action_dim)
        self.chunk_len = int(num_action_chunks)
        self.ref_chunk_len = (
            self.chunk_len
            if ref_num_action_chunks is None
            else int(ref_num_action_chunks)
        )
        if self.ref_chunk_len < self.chunk_len:
            raise ValueError(
                "ref_num_action_chunks must be >= num_action_chunks, got "
                f"{self.ref_chunk_len} < {self.chunk_len}."
            )

        self.action_dim = self.step_action_dim
        self.num_action_chunks = self.chunk_len
        self.flat_action_dim = self.chunk_len * self.step_action_dim
        self.state_dim = self.z_dim + self.proprio_dim
        self.torch_compile_enabled = False
        self.action_selection_mode = str(action_selection_mode).lower()
        if self.action_selection_mode not in ("original", "expo"):
            raise ValueError("action_selection_mode must be 'original' or 'expo'")
        self.expo_num_base_samples = int(expo_num_base_samples)
        self.expo_num_edit_samples = int(expo_num_edit_samples)
        if self.expo_num_base_samples <= 0:
            raise ValueError("expo_num_base_samples must be positive")
        if not 0 <= self.expo_num_edit_samples <= self.expo_num_base_samples:
            raise ValueError(
                "expo_num_edit_samples must be in [0, expo_num_base_samples]"
            )

        self.action_clip_min = float(action_clip_min)
        self.action_clip_max = float(action_clip_max)
        self.actor = DirectGaussianActor(
            state_dim=self.state_dim,
            action_chunk_dim=self.flat_action_dim,
            hidden_dim=mlp_hidden_dim,
            num_hidden_layers=mlp_num_hidden_layers,
            sigma=actor_noise_sigma,
            ref_dropout=ref_action_dropout,
            edit_scale=residual_scale,
            action_clip_min=action_clip_min,
            action_clip_max=action_clip_max,
        )
        # Name this q_head so existing SAC/RLT optimizer filtering keeps actor
        # and critic optimizers separate.
        self.q_head = TwinQCritic(
            state_dim=self.state_dim,
            action_chunk_dim=self.flat_action_dim,
            hidden_dim=mlp_hidden_dim,
            num_hidden_layers=mlp_num_hidden_layers,
            use_layer_norm=critic_use_layer_norm,
        )
        # Rollout workers need the target critic for EXPO selection. This frozen
        # shadow is refreshed by the learner before normal RLinf weight sync.
        self.selection_critic = copy.deepcopy(self.q_head)
        self.selection_critic.requires_grad_(False)
        self.register_buffer(
            "actor_semantic_version",
            torch.tensor(self.ACTOR_SEMANTIC_VERSION, dtype=torch.int32),
            persistent=True,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        semantic_key = f"{prefix}actor_semantic_version"
        if semantic_key not in state_dict:
            error_msgs.append(
                "RLT TD3 checkpoint predates residual actor semantic version 2; "
                "direct-action checkpoints require an explicit conversion."
            )
        elif (
            int(torch.as_tensor(state_dict[semantic_key]).item())
            != self.ACTOR_SEMANTIC_VERSION
        ):
            error_msgs.append(
                f"Unsupported RLT TD3 actor semantic version in {semantic_key}."
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def preprocess_env_obs(self, env_obs: dict) -> dict:
        device = next(self.parameters()).device
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in env_obs.items()
        }

    @staticmethod
    def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _get_z(self, obs: dict) -> torch.Tensor:
        return self._flatten_batch(obs["z_rl"])

    def _get_proprio(self, obs: dict) -> torch.Tensor:
        return self._flatten_batch(obs["proprio"])

    def _get_ref_chunk(self, obs: dict) -> torch.Tensor:
        ref_chunk = self._flatten_batch(obs["ref_chunk"]).reshape(
            obs["ref_chunk"].shape[0], -1, self.step_action_dim
        )
        ref_chunk = ref_chunk[:, : self.chunk_len]
        return ref_chunk.reshape(ref_chunk.shape[0], -1)

    def _get_ref_candidates(self, obs: dict) -> torch.Tensor:
        candidates = obs.get("ref_candidates")
        if candidates is None:
            return self._get_ref_chunk(obs).unsqueeze(1)
        candidates = candidates.reshape(
            candidates.shape[0], candidates.shape[1], -1, self.step_action_dim
        )
        if candidates.shape[1] < self.expo_num_base_samples:
            raise ValueError(
                "ref_candidates has fewer samples than expo_num_base_samples"
            )
        candidates = candidates[:, : self.expo_num_base_samples, : self.chunk_len]
        return candidates.reshape(candidates.shape[0], candidates.shape[1], -1)

    def _state(self, obs: dict) -> torch.Tensor:
        return torch.cat([self._get_z(obs), self._get_proprio(obs)], dim=-1)

    def _format_chunk_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.reshape(-1, self.chunk_len, self.step_action_dim)

    def build_expo_candidates(
        self,
        obs: dict,
        *,
        exploration: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return N VLA bases plus M residual edits and their base indices."""

        bases = self._get_ref_candidates(obs)
        num_edits = min(self.expo_num_edit_samples, bases.shape[1])
        candidates = bases
        base_indices = torch.arange(bases.shape[1], device=bases.device)
        if num_edits:
            state = self._state(obs)[:, None, :].expand(-1, num_edits, -1)
            edit_refs = bases[:, :num_edits, :]
            edited = self.actor(
                state.reshape(-1, state.shape[-1]),
                edit_refs.reshape(-1, edit_refs.shape[-1]),
                deterministic=not exploration,
                apply_action_noise=exploration,
            ).reshape(bases.shape[0], num_edits, -1)
            candidates = torch.cat([bases, edited], dim=1)
            base_indices = torch.cat(
                [base_indices, torch.arange(num_edits, device=bases.device)]
            )
        return candidates, base_indices

    def select_expo_action(
        self,
        obs: dict,
        *,
        exploration: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidates, base_indices = self.build_expo_candidates(
            obs, exploration=exploration
        )
        state = self._state(obs)
        candidate_state = state[:, None, :].expand(-1, candidates.shape[1], -1)
        q_values = self.selection_critic(
            candidate_state.reshape(-1, state.shape[-1]),
            candidates.reshape(-1, self.flat_action_dim),
        ).reshape(state.shape[0], candidates.shape[1], -1)
        best = torch.min(q_values[..., 0], q_values[..., 1]).argmax(dim=1)
        batch = torch.arange(state.shape[0], device=state.device)
        action = candidates[batch, best]
        selected_base = self._get_ref_candidates(obs)[batch, base_indices[best]]
        return action, selected_base

    @torch.no_grad()
    def sync_selection_critic_from(self, source: nn.Module) -> None:
        self.selection_critic.load_state_dict(source.state_dict())

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "RLTTD3MLPPolicy does not use PPO-style default_forward."
        )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs")
        if obs is not None:
            kwargs["obs"] = self.preprocess_env_obs(obs)
        next_obs = kwargs.get("next_obs")
        if next_obs is not None:
            kwargs["next_obs"] = self.preprocess_env_obs(next_obs)

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type == ForwardType.CROSSQ:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.CROSSQ_Q:
            return self.crossq_q_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.SFT:
            raise NotImplementedError("RLTTD3MLPPolicy does not implement SFT.")
        raise NotImplementedError(f"Unsupported forward_type: {forward_type}")

    def sac_forward(
        self,
        obs: dict,
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float | None = None,
        deterministic: bool = False,
        apply_action_noise: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        del kwargs
        action = self.actor(
            self._state(obs),
            self._get_ref_chunk(obs),
            deterministic=deterministic,
            apply_ref_dropout=apply_reference_dropout,
            apply_action_noise=apply_action_noise,
            ref_dropout=reference_dropout_prob,
        )
        return action, torch.zeros_like(action), None

    def sac_q_forward(
        self,
        obs: dict,
        actions: torch.Tensor,
        shared_feature=None,
        detach_encoder: bool = False,
    ) -> torch.Tensor:
        del shared_feature
        state = self._state(obs)
        if detach_encoder:
            state = state.detach()
        return self.q_head(state, self._flatten_batch(actions))

    def crossq_q_forward(
        self,
        obs: dict,
        actions: torch.Tensor,
        next_obs: dict | None = None,
        next_actions: torch.Tensor | None = None,
        shared_feature=None,
        detach_encoder: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data_q = self.sac_q_forward(
            obs=obs,
            actions=actions,
            shared_feature=shared_feature,
            detach_encoder=detach_encoder,
        )
        if next_obs is None or next_actions is None:
            return data_q, data_q.new_zeros(data_q.shape)
        next_q = self.sac_q_forward(
            obs=next_obs,
            actions=next_actions,
            shared_feature=None,
            detach_encoder=detach_encoder,
        )
        return data_q, next_q

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        del calculate_logprobs, calculate_values, kwargs
        obs = self.preprocess_env_obs(env_obs=env_obs)
        if self.action_selection_mode == "expo":
            action, selected_base = self.select_expo_action(
                obs, exploration=(mode != "eval")
            )
            chunk_logprobs = torch.zeros_like(action)
        else:
            action, chunk_logprobs, _ = self.sac_forward(
                obs,
                deterministic=(mode == "eval"),
                apply_action_noise=(mode != "eval"),
            )
            selected_base = self._get_ref_chunk(obs)
        chunk_actions = self._format_chunk_actions(action)

        forward_inputs = {"action": action, "model_action": action}
        if return_obs:
            forward_inputs.update(obs)
            ref_chunk = (
                obs["ref_chunk"]
                .reshape(obs["ref_chunk"].shape[0], -1, self.step_action_dim)
                .clone()
            )
            ref_chunk[:, : self.chunk_len] = selected_base.reshape(
                -1, self.chunk_len, self.step_action_dim
            )
            forward_inputs["ref_chunk"] = ref_chunk

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": torch.zeros_like(action[..., :1]),
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

    def set_critic_requires_grad(self, requires_grad: bool) -> None:
        for param in self.q_head.parameters():
            param.requires_grad_(requires_grad)
