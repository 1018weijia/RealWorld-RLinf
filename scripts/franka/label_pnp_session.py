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

"""Label one saved pnp rollout session with success/failure metadata."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "success", "s"}:
        return True
    if normalized in {"0", "false", "no", "n", "failure", "fail", "f"}:
        return False
    raise argparse.ArgumentTypeError(
        "--success must be one of: 1/0, true/false, success/failure"
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _upsert_csv(csv_path: Path, row: dict[str, str]) -> None:
    fieldnames = [
        "session",
        "session_dir",
        "task",
        "task_id",
        "prompt",
        "trial",
        "is_success",
        "checkpoint_profile",
        "checkpoint_path",
        "chunks",
        "video_dir",
        "notes",
        "labeled_at",
    ]
    rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    rows = [r for r in rows if r.get("session_dir") != row["session_dir"]]
    rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a saved pnp logs/session_async_* rollout."
    )
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--success", required=True, type=_parse_bool)
    parser.add_argument("--trial", type=int, default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--checkpoint-profile", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Defaults to <session_dir>/../recap_episode_labels.csv.",
    )
    args = parser.parse_args()

    session_dir = args.session_dir.expanduser().resolve()
    actions_path = session_dir / "actions.json"
    if not actions_path.exists():
        raise FileNotFoundError(f"actions.json not found: {actions_path}")

    actions = _load_json(actions_path)
    labeled_at = datetime.now().isoformat(timespec="seconds")
    task_from_file = str(actions.get("task", ""))
    prompt = args.prompt or task_from_file

    label = {
        "is_success": bool(args.success),
        "trial": args.trial,
        "task_id": args.task_id or task_from_file,
        "prompt": prompt,
        "checkpoint_profile": args.checkpoint_profile,
        "checkpoint_path": args.checkpoint_path,
        "notes": args.notes,
        "labeled_at": labeled_at,
    }

    actions["recap_label"] = label
    actions["is_success"] = bool(args.success)
    _write_json(actions_path, actions)

    label_path = session_dir / "episode_label.json"
    _write_json(label_path, label)

    labels_csv = args.labels_csv or (session_dir.parent / "recap_episode_labels.csv")
    row = {
        "session": str(actions.get("session", session_dir.name)),
        "session_dir": str(session_dir),
        "task": task_from_file,
        "task_id": str(label["task_id"] or ""),
        "prompt": str(label["prompt"] or ""),
        "trial": "" if args.trial is None else str(args.trial),
        "is_success": "1" if args.success else "0",
        "checkpoint_profile": str(args.checkpoint_profile or ""),
        "checkpoint_path": str(args.checkpoint_path or ""),
        "chunks": str(len(actions.get("chunks", []))),
        "video_dir": str(session_dir / "videos"),
        "notes": args.notes,
        "labeled_at": labeled_at,
    }
    _upsert_csv(labels_csv, row)

    print(f"labeled actions: {actions_path}")
    print(f"wrote label:     {label_path}")
    print(f"updated csv:     {labels_csv}")


if __name__ == "__main__":
    main()
