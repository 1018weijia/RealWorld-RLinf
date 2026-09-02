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

"""Progress prediction heads on top of frozen RL-token features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ProgressHeadOutput:
    """Batched progress ensemble output."""

    logits: Tensor  # [B, H, K]
    alpha: Tensor  # [B, H]
    beta: Tensor  # [B, H]


class _ProgressHeadMember(nn.Module):
    """One bootstrap ensemble member with shared trunk and two output heads."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_bins: int,
        *,
        mlp_layers: int,
        dropout: float,
        min_beta_concentration: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = embedding_dim
        for _ in range(mlp_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_bins)
        self.beta_params = nn.Linear(hidden_dim, 2)
        self.min_beta_concentration = float(min_beta_concentration)

    def forward(self, z_rl: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.trunk(z_rl)
        logits = self.classifier(h)
        alpha_beta = F.softplus(self.beta_params(h)) + self.min_beta_concentration
        alpha, beta = alpha_beta.unbind(dim=-1)
        return logits, alpha, beta


class ProgressHeadEnsemble(nn.Module):
    """Bootstrap progress ensemble.

    Each member has its own MLP trunk. Within a member, the classification
    logits and Beta scalar regression parameters share that trunk.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 2048,
        num_heads: int = 5,
        num_bins: int = 20,
        hidden_dim: int = 512,
        mlp_layers: int = 2,
        dropout: float = 0.0,
        min_beta_concentration: float = 1e-4,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        self.num_heads = int(num_heads)
        self.num_bins = int(num_bins)
        self.embedding_dim = int(embedding_dim)
        self.members = nn.ModuleList(
            [
                _ProgressHeadMember(
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_bins=num_bins,
                    mlp_layers=mlp_layers,
                    dropout=dropout,
                    min_beta_concentration=min_beta_concentration,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, z_rl: Tensor) -> ProgressHeadOutput:
        logits_rows: list[Tensor] = []
        alpha_rows: list[Tensor] = []
        beta_rows: list[Tensor] = []
        for member in self.members:
            logits, alpha, beta = member(z_rl)
            logits_rows.append(logits)
            alpha_rows.append(alpha)
            beta_rows.append(beta)
        return ProgressHeadOutput(
            logits=torch.stack(logits_rows, dim=1),
            alpha=torch.stack(alpha_rows, dim=1),
            beta=torch.stack(beta_rows, dim=1),
        )

    def bin_centers(
        self, *, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> Tensor:
        return (
            torch.arange(self.num_bins, device=device, dtype=dtype or torch.float32)
            + 0.5
        ) / self.num_bins


def soft_progress_targets(
    progress: Tensor,
    *,
    num_bins: int,
    sigma_bins: float = 1.0,
) -> Tensor:
    """Return Gaussian soft labels over progress bins."""
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    if sigma_bins <= 0:
        raise ValueError("sigma_bins must be positive")
    progress = progress.float().clamp(0.0, 1.0)
    centers = (
        torch.arange(num_bins, device=progress.device, dtype=progress.dtype) + 0.5
    ) / num_bins
    sigma = float(sigma_bins) / num_bins
    dist = torch.exp(-0.5 * ((progress[:, None] - centers[None, :]) / sigma) ** 2)
    return dist / dist.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def progress_head_loss(
    output: ProgressHeadOutput,
    progress: Tensor,
    head_mask: Tensor,
    *,
    bin_loss_weight: float = 1.0,
    beta_loss_weight: float = 1.0,
    soft_label_sigma_bins: float = 1.0,
    beta_target_eps: float = 1e-4,
) -> tuple[Tensor, dict[str, float]]:
    """Compute masked soft-bin CE and Beta NLL losses."""
    logits = output.logits.float()
    alpha = output.alpha.float()
    beta = output.beta.float()
    progress = progress.float().reshape(-1)
    weights = head_mask.float()
    if weights.shape != alpha.shape:
        raise ValueError(
            f"head_mask shape {tuple(weights.shape)} does not match "
            f"alpha shape {tuple(alpha.shape)}"
        )

    target_dist = soft_progress_targets(
        progress,
        num_bins=logits.shape[-1],
        sigma_bins=soft_label_sigma_bins,
    )
    log_probs = F.log_softmax(logits, dim=-1)
    ce_per_head = -(target_dist[:, None, :] * log_probs).sum(dim=-1)

    y = progress[:, None].clamp(beta_target_eps, 1.0 - beta_target_eps)
    beta_dist = torch.distributions.Beta(alpha, beta)
    beta_nll_per_head = -beta_dist.log_prob(y)

    denom = weights.sum().clamp_min(1.0)
    ce_loss = (ce_per_head * weights).sum() / denom
    beta_nll = (beta_nll_per_head * weights).sum() / denom
    loss = bin_loss_weight * ce_loss + beta_loss_weight * beta_nll

    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        centers = (
            torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
            + 0.5
        ) / logits.shape[-1]
        p_cls = (probs * centers).sum(dim=-1)
        p_beta = alpha / (alpha + beta).clamp_min(1e-12)
        p_cls_mean = p_cls.mean(dim=1)
        p_beta_mean = p_beta.mean(dim=1)
        entropy = -(probs * log_probs).sum(dim=-1) / torch.log(
            torch.tensor(float(logits.shape[-1]), device=logits.device)
        )
        hard_target = torch.clamp(
            (progress * logits.shape[-1]).long(), max=logits.shape[-1] - 1
        )
        hard_pred = logits.argmax(dim=-1)
        metrics = {
            "loss": float(loss.detach().item()),
            "progress_ce_loss": float(ce_loss.detach().item()),
            "progress_beta_nll": float(beta_nll.detach().item()),
            "progress_cls_mae": float((p_cls_mean - progress).abs().mean().item()),
            "progress_beta_mae": float((p_beta_mean - progress).abs().mean().item()),
            "progress_cls_beta_gap": float(
                (p_cls_mean - p_beta_mean).abs().mean().item()
            ),
            "progress_head_cls_std": float(
                p_cls.std(dim=1, unbiased=False).mean().item()
            ),
            "progress_head_beta_std": float(
                p_beta.std(dim=1, unbiased=False).mean().item()
            ),
            "progress_entropy": float(entropy.mean().item()),
            "progress_bin_acc": float(
                (hard_pred == hard_target[:, None]).float().mean().item()
            ),
            "progress_head_mask_fraction": float(weights.mean().item()),
            "progress_alpha_mean": float(alpha.mean().item()),
            "progress_beta_mean": float(beta.mean().item()),
        }
    return loss, metrics


def voc_progress_labels(
    num_steps: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """VOC labels ``p*_t = t / T`` for a success trajectory of length ``T``."""
    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative, got {num_steps}")
    if num_steps == 0:
        return torch.empty(0, device=device, dtype=dtype)
    t = torch.arange(num_steps, device=device, dtype=dtype)
    return t / float(num_steps)


def progress_voc_loss(
    output: ProgressHeadOutput,
    progress: Tensor,
    head_mask: Tensor,
    **loss_kwargs,
) -> tuple[Tensor, dict[str, float]]:
    """Supervise progress head with VOC-style scalar targets."""
    loss, metrics = progress_head_loss(output, progress, head_mask, **loss_kwargs)
    return loss, {**metrics, "progress_supervision": 1.0}


def progress_d2_ranking_loss(
    p_t: Tensor,
    p_checkpoint: Tensor,
    p_next: Tensor,
    distinguishable_mask: Tensor,
    *,
    margin: float = 0.05,
) -> tuple[Tensor, dict[str, float]]:
    """D2 relative ranking + local non-increasing constraints."""
    p_t = p_t.float().reshape(-1)
    p_checkpoint = p_checkpoint.float().reshape(-1)
    p_next = p_next.float().reshape(-1)
    mask = distinguishable_mask.float().reshape(-1)
    if not (p_t.shape == p_checkpoint.shape == p_next.shape == mask.shape):
        raise ValueError(
            "p_t, p_checkpoint, p_next, and distinguishable_mask must share shape, "
            f"got {tuple(p_t.shape)}, {tuple(p_checkpoint.shape)}, "
            f"{tuple(p_next.shape)}, {tuple(mask.shape)}"
        )

    rank_hinge = torch.relu(p_t - p_checkpoint + float(margin))
    mask_sum = mask.sum().clamp_min(1.0)
    rank_term = (rank_hinge * mask).sum() / mask_sum
    noninc_term = torch.relu(p_next - p_t).mean()
    loss = rank_term + noninc_term
    return loss, {
        "progress_d2_rank_loss": float(rank_term.detach().item()),
        "progress_d2_noninc_loss": float(noninc_term.detach().item()),
        "progress_d2_loss": float(loss.detach().item()),
        "progress_d2_mask_fraction": float(mask.mean().item()),
    }
