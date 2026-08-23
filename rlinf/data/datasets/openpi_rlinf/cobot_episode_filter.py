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

"""Episode filters for Cobot Magic LeRobot v3 datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_SUCCESS_VALUE = "success"


def _normalize_success_label(value: object) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip().lower()


def list_success_episode_indices_from_meta(
    repo_id: str,
) -> list[int] | None:
    """Return success episode indices from LeRobot metadata, or None if unmarked.

    Cobot Magic stores per-episode labels in ``episode_success`` (``success`` /
    ``failure``). When the column is missing, returns ``None`` so callers keep
    the full dataset.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    episodes = meta.episodes
    column_names = getattr(episodes, "column_names", None)
    if column_names is not None and "episode_success" not in column_names:
        logger.warning(
            "Cobot dataset %s has no episode_success column; keeping all episodes.",
            repo_id,
        )
        return None

    success: list[int] = []
    failure: list[int] = []
    for ex in episodes:
        ep_idx = int(ex["episode_index"])
        label = _normalize_success_label(ex.get("episode_success", _SUCCESS_VALUE))
        if label == _SUCCESS_VALUE:
            success.append(ep_idx)
        else:
            failure.append(ep_idx)

    success = sorted(success)
    logger.info(
        "Cobot episode filter on %s: keep %d success, drop %d failure %s",
        repo_id,
        len(success),
        len(failure),
        failure,
    )
    if not success:
        raise ValueError(
            f"Cobot dataset {repo_id!r} has no episodes with "
            f"episode_success={_SUCCESS_VALUE!r}."
        )
    return success


def list_success_episode_indices_from_root(dataset_root: str | Path) -> set[int]:
    """Read success episode indices directly from ``meta/episodes`` parquet."""
    root = Path(dataset_root)
    ep_root = root / "meta" / "episodes"
    if not ep_root.is_dir():
        raise FileNotFoundError(f"Missing episode meta under {ep_root}")

    success: set[int] = set()
    failure: list[int] = []
    for parquet_path in sorted(ep_root.glob("chunk-*/file-*.parquet")):
        table = pq.read_table(
            parquet_path, columns=["episode_index", "episode_success"]
        )
        for ep_idx, label in zip(
            table.column("episode_index").to_pylist(),
            table.column("episode_success").to_pylist(),
            strict=True,
        ):
            if _normalize_success_label(label) == _SUCCESS_VALUE:
                success.add(int(ep_idx))
            else:
                failure.append(int(ep_idx))

    logger.info(
        "Cobot episode filter on %s: keep %d success, drop %d failure %s",
        root,
        len(success),
        len(failure),
        sorted(failure),
    )
    if not success:
        raise ValueError(
            f"No success episodes under {root} "
            f"(episode_success={_SUCCESS_VALUE!r})."
        )
    return success
