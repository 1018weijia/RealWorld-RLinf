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

"""Flask inference server for Cobot Magic openpi_rlinf checkpoints.

Loads ``full_weights.pt`` directly (no checkpoint conversion). Clients POST
observations and receive an action chunk ``[num_action_chunks, action_dim]``.

Example:

    export REPO_PATH=/data/gxy/realworldRL/RLinf
    export PYTHONPATH=${REPO_PATH}:${PYTHONPATH}
    python toolkits/inference/openpi_rlinf_cobot_infer_server.py --device cuda:0

    # or
    bash toolkits/inference/run_cobot_cube_into_drawer_infer_server.sh 0
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import Flask, jsonify, request
from omegaconf import OmegaConf
from PIL import Image

from rlinf.models.embodiment.openpi_rlinf import get_model

# Cobot Magic 56-D raw state -> 14-D joint+gripper positions (every 3rd in first 42).
_COBOT_RAW_STATE_DIM = 56
_COBOT_JOINT_STATE_DIM = 14
_COBOT_STATE_POS_INDICES = tuple(range(0, 42, 3))

DEFAULT_CKPT = (
    "/data/gxy/realworldRL/RLinf/logs/"
    "20260822-05:24:30-cobot_sft_openpi_rlinf_pi05/"
    "cobot_pi05_openpi_rlinf_sft/checkpoints/global_step_20000/"
    "actor/model_state_dict"
)
DEFAULT_NORM_STATS = (
    "/data/gxy/realworldRL/checkpoints/assets/"
    "cobot_magic/cube_into_drawer/norm_stats.json"
)
DEFAULT_PROMPT = "put cube in drawer"

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_repo_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def slice_cobot_state(state: np.ndarray) -> np.ndarray:
    """Accept 14-D joint state or 56-D raw Cobot state."""
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 1:
        raise ValueError(f"state must be 1-D, got shape {state.shape}")
    if state.shape[0] == _COBOT_JOINT_STATE_DIM:
        return state
    if state.shape[0] == _COBOT_RAW_STATE_DIM:
        return state[list(_COBOT_STATE_POS_INDICES)]
    raise ValueError(
        f"state length must be {_COBOT_JOINT_STATE_DIM} or "
        f"{_COBOT_RAW_STATE_DIM}, got {state.shape[0]}"
    )


def decode_image_b64(b64: str) -> np.ndarray:
    """Decode base64 JPEG/PNG to uint8 ``[H, W, 3]``."""
    try:
        raw = base64.b64decode(b64, validate=True)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"invalid image payload: {exc}") from exc
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"image must be [H,W,3], got {arr.shape}")
    return arr


def build_model_config(
    ckpt_path: str,
    norm_stats_path: str,
    precision: str = "bf16",
) -> Any:
    """Build ``actor.model``-style OmegaConf for eval inference."""
    return OmegaConf.create(
        {
            "precision": precision,
            "pi05": True,
            "model_path": ckpt_path,
            "num_action_chunks": 50,
            "action_dim": 14,
            "num_steps": 5,
            "add_value_head": False,
            "openpi": {
                "task": "eval",
                "config_name": "pi05_cobot_magic",
                "num_images_in_input": 3,
                "discrete_state_input": True,
                "max_token_len": 200,
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "action_chunk": 50,
                "action_env_dim": 14,
            },
            "openpi_data": {"norm_stats_path": norm_stats_path},
        }
    )


class CobotInferServer:
    """Wraps openpi_rlinf eval model and observation preprocessing."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        self.device = device
        self.cfg = cfg
        logger.info("Loading openpi_rlinf model from %s ...", cfg.model_path)
        self.model = get_model(cfg).to(device).eval()
        self.num_action_chunks = int(cfg.num_action_chunks)
        self.action_dim = int(cfg.action_dim)
        logger.info(
            "Model ready on %s (chunks=%d, dim=%d).",
            device,
            self.num_action_chunks,
            self.action_dim,
        )

    def infer_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run inference from a plain JSON dict."""
        prompt = str(payload.get("prompt", DEFAULT_PROMPT))
        state_raw = payload.get("state")
        if state_raw is None:
            raise ValueError("missing field: state")

        cam_high_b64 = payload.get("cam_high_b64")
        cam_left_b64 = payload.get("cam_left_wrist_b64")
        cam_right_b64 = payload.get("cam_right_wrist_b64")
        if not cam_high_b64 or not cam_left_b64 or not cam_right_b64:
            raise ValueError(
                "missing image fields: cam_high_b64, cam_left_wrist_b64, "
                "cam_right_wrist_b64"
            )

        state_np = slice_cobot_state(np.asarray(state_raw, dtype=np.float32))
        high = decode_image_b64(cam_high_b64)
        left = decode_image_b64(cam_left_b64)
        right = decode_image_b64(cam_right_b64)

        if left.shape != high.shape or right.shape != high.shape:
            raise ValueError(
                f"all cameras must share shape; high={high.shape}, "
                f"left={left.shape}, right={right.shape}"
            )

        wrist = np.stack([left, right], axis=0)
        env_obs = {
            "states": torch.from_numpy(state_np).reshape(1, -1).to(self.device),
            "main_images": torch.from_numpy(high).reshape(1, *high.shape).to(self.device),
            "wrist_images": torch.from_numpy(wrist).reshape(
                1, 2, *left.shape
            ).to(self.device),
            "task_descriptions": [prompt],
        }

        with torch.no_grad():
            actions, _ = self.model.predict_action_batch(env_obs)

        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        if actions_np.ndim == 3:
            actions_np = actions_np[0]

        return {
            "actions": actions_np.tolist(),
            "num_action_chunks": self.num_action_chunks,
            "action_dim": self.action_dim,
            "prompt": prompt,
        }


def create_app(server: CobotInferServer) -> Flask:
    """Build Flask app bound to a loaded model."""
    app = Flask(__name__)

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.post("/infer")
    def infer() -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "expected JSON object"}), 400
        try:
            return jsonify(server.infer_from_payload(payload))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logger.exception("inference failed")
            return jsonify({"error": str(exc)}), 500

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="openpi_rlinf Cobot Magic HTTP inference server."
    )
    parser.add_argument(
        "--ckpt",
        default=DEFAULT_CKPT,
        help="Checkpoint dir or full_weights.pt path.",
    )
    parser.add_argument(
        "--norm-stats",
        default=DEFAULT_NORM_STATS,
        help="norm_stats.json for cube_into_drawer.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu (slow).",
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        choices=["bf16", "fp32"],
        help="Inference compute precision.",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_repo_on_path()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    ckpt = Path(args.ckpt).expanduser()
    norm_stats = Path(args.norm_stats).expanduser()
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not norm_stats.is_file():
        raise FileNotFoundError(f"norm_stats not found: {norm_stats}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; use --device cpu")

    cfg = build_model_config(str(ckpt), str(norm_stats), precision=args.precision)
    server = CobotInferServer(cfg, device)
    app = create_app(server)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
