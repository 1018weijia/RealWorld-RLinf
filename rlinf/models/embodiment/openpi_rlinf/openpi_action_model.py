# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn

from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    OpenPiPytorchRLTConfig,
)


class OpenPiPytorchActionModel(nn.Module):
    """Abstract base wrapper around the vendored ``Pi0`` model.

    Concrete subclasses must provide their own ``predict_action_batch`` and
    ``forward`` (if training is needed). This base only wires up the Pi0
    model, optional RLT-token module, the device shortcut, and the
    gradient-checkpointing pass-through.
    """

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        rlt_cfg: OpenPiPytorchRLTConfig | None = None,
    ):
        super().__init__()
        self.model = pi0_model
        self.num_steps = num_steps
        self.action_env_dim = action_env_dim
        self.rlt_cfg = rlt_cfg or OpenPiPytorchRLTConfig()
        if self.rlt_cfg.use_rlt:
            from rlinf.models.embodiment.modules.rlt_token_transformer import (
                get_rlt_token_transformer_class,
            )

            rlt_class = get_rlt_token_transformer_class(self.rlt_cfg.rlt_architecture)
            rlt_module = rlt_class(
                input_dim=self.rlt_cfg.rlt_input_dim,
                embed_dim=self.rlt_cfg.rlt_embed_dim,
                prefix_seq_len=self.rlt_cfg.rlt_prefix_seq_len,
                num_layers=self.rlt_cfg.rlt_num_layers,
                num_heads=self.rlt_cfg.rlt_num_heads,
                mlp_ratio=self.rlt_cfg.rlt_mlp_ratio,
                dropout_rate=self.rlt_cfg.rlt_dropout,
            )
            if self.rlt_cfg.rlt_architecture == "openpi":
                rlt_module = rlt_module.float()
            else:
                rlt_module = rlt_module.to(dtype=next(self.model.parameters()).dtype)
            self.rlt_module = rlt_module

        self._mark_fsdp_wrap_names()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def _no_split_modules(self) -> list[str] | None:
        if not self.rlt_cfg.use_rlt:
            return None
        modules = ["Block", "Encoder1DBlock"]
        if self.rlt_cfg.rlt_architecture == "legacy":
            modules.append("RLTSelfAttentionLayer")
        return modules

    @property
    def _no_split_names(self) -> list[str] | None:
        if not self.rlt_cfg.use_rlt:
            return None
        names = [
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            "time_mlp_in",
            "time_mlp_out",
        ]
        if self.rlt_cfg.rlt_architecture == "openpi":
            # Keep the standard Transformer encoder/decoder as one FSDP leaf.
            names.append("rlt_module")
        return names

    def _mark_fsdp_wrap_names(self) -> None:
        """Mark modules so RLinf's FSDP lambda policy can find leaf projects."""
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    # --- Gradient checkpointing pass-through (used by the FSDP training path) ---
    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict | None = None, **kwargs
    ) -> None:
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self, **kwargs) -> None:
        self.model.gradient_checkpointing_disable()

    def _require_rlt(self) -> None:
        if not self.rlt_cfg.use_rlt or not hasattr(self, "rlt_module"):
            raise ValueError("RLT operation requires actor.model.openpi.use_rlt=True.")

    def set_vla_trainable_scope(self, scope: str) -> int:
        """Freeze/select VLA parameters using rlt-openpi Stage-1 semantics."""
        scope = str(scope).strip().lower().replace("-", "_")
        if scope not in {"off", "action_expert", "full"}:
            raise ValueError(
                "vla_finetune_scope must be one of: off, action_expert, full"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(scope == "full")
        if scope == "action_expert":
            expert_layer_markers = (
                ".attn.q_proj.1.",
                ".attn.k_proj.1.",
                ".attn.v_proj.1.",
                ".attn.o_proj.1.",
                ".pre_attention_norms.1.",
                ".pre_ffw_norms.1.",
                ".mlps.1.",
            )
            prefixes = (
                "action_in_proj.",
                "action_out_proj.",
                "time_mlp_in.",
                "time_mlp_out.",
                "action_time_mlp_in.",
                "action_time_mlp_out.",
                "state_proj.",
            )
            for name, parameter in self.model.named_parameters():
                is_expert_block = name.startswith("llm.layers.") and any(
                    marker in name for marker in expert_layer_markers
                )
                is_expert_norm = name.startswith("llm.final_norms.1.")
                if is_expert_block or is_expert_norm or name.startswith(prefixes):
                    parameter.requires_grad_(True)
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if scope == "action_expert" and count == 0:
            raise RuntimeError(
                "action_expert scope selected no trainable VLA parameters"
            )
        return count

    def _select_rlt_prefix_embeddings(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
        lang_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rlt_cfg.rlt_image_only and lang_tokens is not None:
            num_image_tokens = prefix_output.shape[1] - lang_tokens.shape[1]
            prefix_output = prefix_output[:, :num_image_tokens]
            prefix_mask = prefix_mask[:, :num_image_tokens]
        return prefix_output, prefix_mask

    def _rlt_forward(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._require_rlt()
        rlt_param = next(self.rlt_module.parameters())
        # Match the live dtype of the independent RLT FSDP leaf. Its master
        # parameters are initialized in FP32, while FSDP may expose BF16 views
        # during the forward; the loss reduction remains FP32 internally.
        prefix_output = prefix_output.to(device=rlt_param.device, dtype=rlt_param.dtype)
        rlt_mask = prefix_mask if self.rlt_cfg.rlt_use_mask else None
        return self.rlt_module(prefix_output, rlt_mask)

    def _encode_rlt_flat(
        self,
        prefix_output: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._require_rlt()
        rlt_param = next(self.rlt_module.parameters())
        prefix_output = prefix_output.to(device=rlt_param.device, dtype=rlt_param.dtype)
        rlt_mask = prefix_mask if self.rlt_cfg.rlt_use_mask else None
        return self.rlt_module.encode_flat(prefix_output, rlt_mask)
