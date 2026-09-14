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
"""Check that a saved RLT Stage 2 checkpoint can still drive gradient updates.

The replay buffer stores encoded features rather than raw camera frames, so
this never loads the frozen Stage 1 VLA and finishes in seconds. It is also
safe to run against a checkpoint of a live server: the checkpoint is opened
read-only and training happens on a private copy of the weights.

Usage::

    python toolkits/realworld_check/verify_rlt_stage2_training.py \\
        --checkpoint .../checkpoints/episode_10_step_0 \\
        --config-name xrobot_ee_rlt_stage2_ws_server \\
        --num-updates 25

Exits non-zero when the buffer is empty, a loss is not finite, or no weight
moved, which is what distinguishes "training ran" from "training was a no-op".
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models import get_model
from rlinf.serving.rlt.offline_trainer import RLTOfflineTrainer

DEFAULT_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"
)


def summarize(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Compare weights around a training burst and judge whether it did work.

    Args:
        before: Model ``state_dict`` snapshot taken before the burst.
        after: Model ``state_dict`` after the burst.
        metrics: Metrics returned by the trainer for the burst.

    Returns:
        A mapping with ``tensors_changed``, ``tensors_total``, ``max_delta``,
        ``losses_finite`` and the overall ``ok`` verdict.
    """
    changed = sum(
        1 for key, value in after.items() if not torch.equal(value, before[key])
    )
    floating = [key for key in after if after[key].dtype.is_floating_point]
    max_delta = max(
        (
            (after[key].float() - before[key].float()).abs().max().item()
            for key in floating
        ),
        default=0.0,
    )
    losses = [float(value) for name, value in metrics.items() if "loss" in name]
    # NaN compares unequal to itself, which is how a diverged burst shows up.
    losses_finite = all(
        value == value and abs(value) != float("inf") for value in losses
    )
    return {
        "tensors_changed": changed,
        "tensors_total": len(after),
        "max_delta": max_delta,
        "losses_finite": losses_finite,
        "ok": bool(losses_finite and changed > 0 and max_delta > 0.0),
    }


def build_trainer(config_dir: Path, config_name: str) -> RLTOfflineTrainer:
    """Build a Stage 2 trainer without loading the frozen Stage 1 VLA."""
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg.actor.model).to(device)
    target_model = copy.deepcopy(model).to(device)
    target_model.requires_grad_(False)
    return RLTOfflineTrainer(
        cfg=cfg,
        model=model,
        target_model=target_model,
        device=device,
        torch_dtype=torch.float32,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config-name", default="xrobot_ee_rlt_stage2_ws_server")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR, type=Path)
    parser.add_argument("--num-updates", default=25, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trainer = build_trainer(args.config_dir, args.config_name)

    trainer.load(str(args.checkpoint))
    rows = trainer.replay_buffer.total_samples
    print(f"replay rows restored: {rows}")
    if rows == 0:
        print("FAIL: replay buffer is empty, there is nothing to train on")
        return 1

    before = {
        key: value.detach().clone() for key, value in trainer.model.state_dict().items()
    }
    print(f"running {args.num_updates} critic updates on stored data ...")
    metrics = trainer.train(args.num_updates)
    after = {key: value.detach() for key, value in trainer.model.state_dict().items()}

    verdict = summarize(before, after, metrics)
    for name in sorted(metrics):
        if any(token in name for token in ("loss", "grad_norm", "updates_run")):
            print(f"  {name:40} {metrics[name]:.6g}")
    print(
        f"\n  replay rows used   {rows}\n"
        f"  tensors changed    {verdict['tensors_changed']}/{verdict['tensors_total']}\n"
        f"  max |weight delta| {verdict['max_delta']:.6g}\n"
        f"  losses finite      {verdict['losses_finite']}"
    )
    print("PASS: stored data produced gradient updates" if verdict["ok"] else "FAIL")
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
