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

"""Example client for :mod:`openpi_rlinf_cobot_infer_server`.

Sends one observation bundle and prints the returned action chunk. Use
``--smoke-test`` for a connectivity check with random images (no real robot).

Examples:

    # Random images — verify server loads and responds:
    python toolkits/inference/openpi_rlinf_cobot_infer_client.py \\
        --server-url http://127.0.0.1:8000 --smoke-test

    # Real camera files from the robot:
    python toolkits/inference/openpi_rlinf_cobot_infer_client.py \\
        --server-url http://127.0.0.1:8000 \\
        --cam-high /path/to/top.jpg \\
        --cam-left-wrist /path/to/left.jpg \\
        --cam-right-wrist /path/to/right.jpg \\
        --state-json '[0.0, ...]' \\
        --prompt "put cube in drawer"
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

DEFAULT_PROMPT = "put cube in drawer"
DEFAULT_SERVER = "http://127.0.0.1:8000"


def encode_image(path: Path) -> str:
    """Read an image file and return base64-encoded JPEG bytes."""
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def encode_random_image(height: int, width: int) -> str:
    """Build a random RGB image payload for smoke tests."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = __import__("io").BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_state(state_json: str | None, state_dim: int) -> list[float]:
    """Parse CLI state JSON or return zeros."""
    if state_json is None:
        return [0.0] * state_dim
    parsed = json.loads(state_json)
    if not isinstance(parsed, list):
        raise ValueError("state-json must be a JSON array")
    if len(parsed) != state_dim:
        raise ValueError(f"state-json length must be {state_dim}, got {len(parsed)}")
    return [float(x) for x in parsed]


def execute_action_chunk(
    actions: list[list[float]],
    *,
    dry_run: bool = True,
) -> None:
    """Demonstrate how a robot client would consume the chunk step by step."""
    print(f"Received action chunk: {len(actions)} steps x {len(actions[0])} dims")
    for step_idx, action in enumerate(actions):
        if dry_run:
            preview = ", ".join(f"{v:.4f}" for v in action[:6])
            print(f"  step {step_idx:02d}: [{preview}, ...]")
        else:
            # Replace with your robot API, e.g. send_joint_command(action)
            pass
    print("Chunk consumed. Request a new chunk when the queue is empty.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cobot openpi_rlinf infer client.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--state-json",
        default=None,
        help="JSON array of 14 joint states (default: zeros).",
    )
    parser.add_argument("--cam-high", type=Path, default=None)
    parser.add_argument("--cam-left-wrist", type=Path, default=None)
    parser.add_argument("--cam-right-wrist", type=Path, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use random 480x640 images (no files required).",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--save-npy",
        type=Path,
        default=None,
        help="Optional path to save actions as .npy",
    )
    args = parser.parse_args()

    state = parse_state(args.state_json, state_dim=14)

    if args.smoke_test:
        height, width = 480, 640
        cam_high_b64 = encode_random_image(height, width)
        cam_left_b64 = encode_random_image(height, width)
        cam_right_b64 = encode_random_image(height, width)
        print(f"Smoke test: random {height}x{width} images")
    else:
        if not args.cam_high or not args.cam_left_wrist or not args.cam_right_wrist:
            print(
                "Provide --cam-high, --cam-left-wrist, --cam-right-wrist "
                "or use --smoke-test.",
                file=sys.stderr,
            )
            sys.exit(1)
        for p in (args.cam_high, args.cam_left_wrist, args.cam_right_wrist):
            if not p.is_file():
                print(f"image not found: {p}", file=sys.stderr)
                sys.exit(1)
        cam_high_b64 = encode_image(args.cam_high)
        cam_left_b64 = encode_image(args.cam_left_wrist)
        cam_right_b64 = encode_image(args.cam_right_wrist)

    payload = {
        "prompt": args.prompt,
        "state": state,
        "cam_high_b64": cam_high_b64,
        "cam_left_wrist_b64": cam_left_b64,
        "cam_right_wrist_b64": cam_right_b64,
    }

    health_url = f"{args.server_url.rstrip('/')}/health"
    infer_url = f"{args.server_url.rstrip('/')}/infer"

    print(f"Health check: {health_url}")
    health = requests.get(health_url, timeout=10)
    print(f"  -> {health.status_code} {health.json()}")

    print(f"POST infer: {infer_url}")
    response = requests.post(infer_url, json=payload, timeout=args.timeout)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}", file=sys.stderr)
        sys.exit(1)

    data = response.json()
    actions = data["actions"]
    print(
        f"OK prompt={data['prompt']} "
        f"shape=({data['num_action_chunks']}, {data['action_dim']})"
    )

    if args.save_npy:
        np.save(args.save_npy, np.asarray(actions, dtype=np.float32))
        print(f"Saved {args.save_npy}")

    execute_action_chunk(actions, dry_run=True)


if __name__ == "__main__":
    main()
