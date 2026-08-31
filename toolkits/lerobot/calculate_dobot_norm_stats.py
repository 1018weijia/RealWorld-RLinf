#!/usr/bin/env python3
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

"""Fast Dobot norm_stats from parquet only (no video decode).

Applies the same 14-D delta-action transform used by ``pi05_dobot``.
Uses all episodes (Dobot NormalData has no episode_success column).
"""

from __future__ import annotations

import pathlib

import numpy as np
import openpi.shared.normalize as normalize
import pyarrow.parquet as pq
import tqdm
import tyro

_DOBOT_JOINT_ACTION_DIM = 14
_DELTA_MASK = np.asarray(
    [True] * 6 + [False] + [True] * 6 + [False],
    dtype=bool,
)


def _iter_parquet_tables(data_root: pathlib.Path):
    files = sorted(data_root.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_root}")
    for path in files:
        yield pq.read_table(path)


def _episode_arrays(table) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Group rows by episode_index -> (states[T,14], actions[T,14])."""
    ep = np.asarray(table.column("episode_index").to_pylist())
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    out: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for i, e in enumerate(ep):
        out.setdefault(int(e), []).append((states[i], actions[i]))
    packed = {}
    for e, rows in out.items():
        s = np.stack([r[0] for r in rows], axis=0)
        a = np.stack([r[1] for r in rows], axis=0)
        packed[e] = (s, a)
    return packed


def _delta_actions(state_14: np.ndarray, actions_chunk: np.ndarray) -> np.ndarray:
    """Match openpi ``DeltaActions`` for the 14-D joint mask."""
    actions = actions_chunk.copy()
    dims = _DELTA_MASK.shape[-1]
    actions[..., :dims] -= np.expand_dims(
        np.where(_DELTA_MASK, state_14[..., :dims], 0), axis=-2
    )
    return actions


def main(
    dataset_root: str = (
        "/data/gxy/realworldRL/datasets/Dobot/NormalData/dobot_cook_vegetable_fullV30"
    ),
    output_dir: str = ("/data/gxy/realworldRL/checkpoints/assets/dobot/cook_vegetable"),
    action_horizon: int = 50,
) -> None:
    root = pathlib.Path(dataset_root)
    data_root = root / "data"
    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    tables = list(_iter_parquet_tables(data_root))
    episode_count = 0
    for table in tqdm.tqdm(tables, desc="Dobot parquet files"):
        for states, actions in _episode_arrays(table).values():
            episode_count += 1
            if states.shape[-1] != _DOBOT_JOINT_ACTION_DIM:
                raise SystemExit(
                    f"state dim != {_DOBOT_JOINT_ACTION_DIM}: {states.shape}"
                )
            if actions.shape[-1] != _DOBOT_JOINT_ACTION_DIM:
                raise SystemExit(
                    f"action dim != {_DOBOT_JOINT_ACTION_DIM}: {actions.shape}"
                )
            t = states.shape[0]
            if t == 0:
                continue
            stats["state"].update(states)
            for t0 in range(t):
                idxs = np.clip(np.arange(t0, t0 + action_horizon), 0, t - 1)
                chunk = actions[idxs]
                chunk = _delta_actions(states[t0], chunk)
                stats["actions"].update(chunk[None, ...])

    print(f"Used {episode_count} episodes from {root}")

    norm_stats = {k: v.get_statistics() for k, v in stats.items()}
    for key, ns in norm_stats.items():
        mean = np.asarray(ns.mean)
        print(f"{key}: dim={mean.shape[-1]} mean[:4]={mean[:4]}")
        if mean.shape[-1] != _DOBOT_JOINT_ACTION_DIM:
            raise SystemExit(f"{key} dim != {_DOBOT_JOINT_ACTION_DIM}")

    out = pathlib.Path(output_dir)
    normalize.save(out, norm_stats)
    print(f"Wrote {out / 'norm_stats.json'}")


if __name__ == "__main__":
    tyro.cli(main)
