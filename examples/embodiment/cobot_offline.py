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

"""Audit/convert LeRobot v3, then pretrain the native Cobot Stage2 head."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.models import get_model
from rlinf.serving.rlt.cobot_offline_data import (
    FORMAT,
    CobotLeRobotV3,
    atomic_save,
    chunk_starts,
    concatenate,
    contract,
    convert_episode,
)
from rlinf.serving.rlt.cobot_offline_trainer import CobotOfflineTrainer

logger = logging.getLogger(__name__)


def convert(cfg: DictConfig) -> None:
    """Resume completed episode shards only when source/config identity matches."""
    from rlt_stage2_server import build_policy, run_preflight

    data = CobotLeRobotV3(cfg.offline.dataset_root, str(cfg.server.task_prompt))
    count = int(cfg.offline.get("max_episodes", 0))
    rows = data.episodes[:count] if count else data.episodes
    if len(rows) < 2:
        raise ValueError("Select at least two episodes for disjoint train/validation")
    counts = {
        label: sum(r["episode_success"] == label for r in rows)
        for label in ("success", "failure")
    }
    logger.info(
        "Dataset: %s; episodes=%d labels=%s transitions=%d",
        data.root,
        len(rows),
        counts,
        sum(len(chunk_starts(r["length"])) for r in rows),
    )
    if cfg.offline.mode == "audit":
        for row in rows:
            data.table(row)
            for camera in data.info["features"]:
                if camera.startswith("observation.images."):
                    data.frames(row, camera, [0, int(row["length"]) - 1])
        logger.info("All selected episode tables and video endpoints passed")
        return
    run_preflight(cfg)
    identity = contract(cfg)
    signature = {
        "contract": identity,
        "dataset_sha256": data.fingerprint(),
        "seed": int(cfg.actor.seed),
    }
    output = Path(cfg.offline.buffer)
    shards = output.parent / f"{output.stem}_shards"
    shards.mkdir(parents=True, exist_ok=True)
    policy = None
    converted = []
    for row in rows:
        episode = row["episode_index"]
        path = shards / f"episode_{episode:06d}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["signature"] != signature:
                raise ValueError(
                    f"Stale feature shard {path}; use a new output directory"
                )
            result = payload["rows"]
        else:
            if policy is None:
                policy = build_policy(cfg)
            torch.manual_seed(int(cfg.actor.seed) + episode)
            result = convert_episode(
                data, row, policy.inference, float(cfg.algorithm.gamma)
            )
            atomic_save({"signature": signature, "rows": result}, path)
        converted.append(result)
        logger.info(
            "Encoded episode %d/%d: %d transitions",
            episode + 1,
            len(rows),
            len(result["actions"]),
        )
    # Stratify by outcome where possible; split whole episodes, never overlapping chunks.
    rng = np.random.default_rng(int(cfg.actor.seed))
    validation = []
    for label in ("success", "failure"):
        ids = [r["episode_index"] for r in rows if r["episode_success"] == label]
        rng.shuffle(ids)
        if len(ids) > 1:
            validation.extend(ids[: max(1, round(len(ids) * 0.1))])
    if not validation:
        validation = [rows[-1]["episode_index"]]
    payload = {
        "format": FORMAT,
        "contract": identity,
        "source": signature,
        "rows": concatenate(converted),
        "validation_episodes": validation,
        "partial_conversion": len(rows) != len(data.episodes),
        "reward_rule": "success=1/failure=0 at final observed transition; both are terminal",
        "tail_rule": "terminal-aligned full chunks, drop prefix remainder and final action without next observation",
    }
    atomic_save(payload, output)
    logger.info("Saved %d transitions to %s", len(payload["rows"]["actions"]), output)


def train(cfg: DictConfig) -> None:
    """Train only cached features; Stage1 never loads during offline updates."""
    from rlinf.utils.metric_logger import MetricLogger

    torch.manual_seed(int(cfg.actor.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg.actor.model).to(device)
    trainer = CobotOfflineTrainer(
        cfg, model=model, target_model=copy.deepcopy(model), device=device
    )
    trainer.offline_mode = True
    if cfg.runner.get("resume_dir"):
        trainer.load(str(cfg.runner.resume_dir))
    else:
        trainer.attach_offline_buffer(
            torch.load(cfg.offline.buffer, map_location="cpu", weights_only=False)
        )
    if trainer.offline_buffer.payload.get("partial_conversion") and not cfg.offline.get(
        "allow_partial", False
    ):
        raise ValueError(
            "Partial dataset: set offline.allow_partial=true only for smoke tests"
        )
    if not len(trainer.offline_buffer.validation_indices):
        raise ValueError("Offline pretraining requires held-out episodes")
    total = int(cfg.offline.steps)
    if total <= trainer.offline_total_updates:
        raise ValueError(
            "Requested steps already completed; select a new run or larger step budget"
        )
    logger.info(
        "Offline training: %d rows; train=%d validation=%d; device=%s",
        trainer.offline_buffer.size,
        len(trainer.offline_buffer.train_indices),
        len(trainer.offline_buffer.validation_indices),
        device,
    )
    metrics_logger = MetricLogger(cfg)
    try:
        for step in range(trainer.offline_total_updates + 1, total + 1):
            # Use offline counter for TD3 actor/target cadence, but preserve online counters at handoff.
            trainer.update_step = step - 1
            metrics = trainer.update_once(
                train_actor=(step - 1) % trainer.critic_actor_ratio == 0
            )
            trainer.offline_total_updates = step
            trainer.update_step = 0
            if (
                step == 1
                or step % int(cfg.offline.get("validation_every", 500)) == 0
                or step == total
            ):
                metrics.update(trainer.validate_offline())
            if step == 1 or step % 100 == 0 or step == total:
                logger.info("Offline update %d/%d: %s", step, total, metrics)
            metrics_logger.log(
                {f"offline/{k}": v for k, v in metrics.items()}, step=step
            )
            if step % int(cfg.offline.get("save_every", 5000)) == 0 or step == total:
                destination = Path(cfg.server.save_dir) / f"offline_step_{step}"
                trainer.save(str(destination))
                logger.info("Online handoff ready: STAGE2_RESUME_DIR=%s", destination)
    finally:
        metrics_logger.finish()


@hydra.main(version_base="1.1", config_path="config", config_name=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.info(
        "Offline options: %s", OmegaConf.to_container(cfg.offline, resolve=True)
    )
    if cfg.offline.mode in ("audit", "convert"):
        convert(cfg)
    elif cfg.offline.mode == "train":
        train(cfg)
    else:
        raise ValueError("Expected offline.mode=audit|convert|train")


if __name__ == "__main__":
    main()
