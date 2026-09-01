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

"""YAML-driven Flask inference server for openpi_rlinf Cobot / Dobot checkpoints.

Loads ``full_weights.pt`` directly (no checkpoint conversion). The HTTP API is
unchanged: clients POST observations and receive an action chunk
``[num_action_chunks, action_dim]``.

Usage:

    python toolkits/inference/openpi_rlinf_infer_server.py \\
        --config toolkits/inference/config/cobot_cube_into_drawer.yaml \\
        --device cuda:0

    bash toolkits/inference/run_infer_server.sh cobot_cube_into_drawer 0 8000
    bash toolkits/inference/run_infer_server.sh dobot_cook_vegetable 7 8001
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import Flask, jsonify, request
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from rlinf.models.embodiment.openpi_rlinf import get_model

# Cobot Magic 56-D raw state -> 14-D joint+gripper positions (every 3rd in first 42).
_COBOT_RAW_STATE_DIM = 56
_JOINT_STATE_DIM = 14
_COBOT_STATE_POS_INDICES = tuple(range(0, 42, 3))

_SUPPORTED_ROBOTS = ("cobot", "dobot", "xrobot")

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _config_dir() -> Path:
    return Path(__file__).resolve().parent / "config"


def _ensure_repo_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_config_path(config_arg: str) -> Path:
    """Resolve ``--config`` to a YAML path (name under config/ or absolute path)."""
    raw = Path(config_arg).expanduser()
    candidates = [
        raw,
        _config_dir() / raw.name,
        _config_dir() / f"{raw.name}.yaml",
        _config_dir() / f"{config_arg}.yaml",
    ]
    if not raw.suffix:
        candidates.insert(0, _config_dir() / f"{raw.name}.yaml")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"inference config not found: {config_arg!r}; looked under "
        f"{_config_dir()} and as a filesystem path."
    )


def load_infer_yaml(config_path: Path) -> DictConfig:
    """Load and validate an inference YAML config."""
    cfg = OmegaConf.load(str(config_path))
    required = ("robot", "default_prompt", "ckpt", "norm_stats", "model")
    missing = [k for k in required if OmegaConf.select(cfg, k) is None]
    if missing:
        raise ValueError(f"{config_path}: missing required keys {missing}")
    robot = str(cfg.robot).lower()
    if robot not in _SUPPORTED_ROBOTS:
        raise ValueError(f"{config_path}: robot={robot!r} not in {_SUPPORTED_ROBOTS}")
    cfg.robot = robot
    if OmegaConf.select(cfg, "model.openpi.task") is None:
        raise ValueError(f"{config_path}: model.openpi.task is required (use 'eval')")
    if OmegaConf.select(cfg, "model.openpi.config_name") is None:
        raise ValueError(f"{config_path}: model.openpi.config_name is required")
    return cfg


def slice_robot_state(state: np.ndarray, robot: str) -> np.ndarray:
    """Normalize client state to 14-D joint+gripper for Cobot or Dobot."""
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 1:
        raise ValueError(f"state must be 1-D, got shape {state.shape}")
    if state.shape[0] == _JOINT_STATE_DIM:
        return state
    if robot == "cobot" and state.shape[0] == _COBOT_RAW_STATE_DIM:
        return state[list(_COBOT_STATE_POS_INDICES)]
    if robot == "cobot":
        raise ValueError(
            f"cobot state length must be {_JOINT_STATE_DIM} or "
            f"{_COBOT_RAW_STATE_DIM}, got {state.shape[0]}"
        )
    # dobot / xrobot: 14-D joint/EE pose, passed through directly.
    if robot in ("dobot", "xrobot"):
        raise ValueError(
            f"{robot} state length must be {_JOINT_STATE_DIM}, got {state.shape[0]}"
        )
    raise ValueError(f"unsupported robot={robot!r}")


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


def _channel_means(image: np.ndarray) -> dict[str, float]:
    means = image.reshape(-1, 3).mean(axis=0)
    return {
        "R": float(means[0]),
        "G": float(means[1]),
        "B": float(means[2]),
    }


def dump_client_observation(
    dump_dir: Path,
    *,
    request_idx: int,
    robot: str,
    prompt: str,
    state: np.ndarray,
    high: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> Path:
    """Save one decoded client observation for visual debugging."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = dump_dir / f"req_{request_idx:06d}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    Image.fromarray(high, mode="RGB").save(out / "cam_high_rgb.png")
    Image.fromarray(left, mode="RGB").save(out / "cam_left_wrist_rgb.png")
    Image.fromarray(right, mode="RGB").save(out / "cam_right_wrist_rgb.png")
    # BGR-looking copy helps catch channel-order mistakes by eye.
    Image.fromarray(high[:, :, ::-1], mode="RGB").save(
        out / "cam_high_as_bgr_swapped.png"
    )

    collage = np.concatenate([high, left, right], axis=1)
    Image.fromarray(collage, mode="RGB").save(out / "collage_high_left_right.png")

    meta = {
        "request_idx": request_idx,
        "robot": robot,
        "prompt": prompt,
        "state": [float(x) for x in np.asarray(state, dtype=np.float64).reshape(-1)],
        "shapes": {
            "cam_high": list(high.shape),
            "cam_left_wrist": list(left.shape),
            "cam_right_wrist": list(right.shape),
        },
        "dtypes": {
            "cam_high": str(high.dtype),
            "cam_left_wrist": str(left.dtype),
            "cam_right_wrist": str(right.dtype),
        },
        "channel_means_rgb": {
            "cam_high": _channel_means(high),
            "cam_left_wrist": _channel_means(left),
            "cam_right_wrist": _channel_means(right),
        },
        "value_range": {
            "cam_high": [int(high.min()), int(high.max())],
            "cam_left_wrist": [int(left.min()), int(left.max())],
            "cam_right_wrist": [int(right.min()), int(right.max())],
        },
        "note": (
            "Images are server-decoded RGB (PIL). "
            "cam_high_as_bgr_swapped.png is R/B swapped for litmus comparison. "
            "Model later resize_with_pad to 224x224; dump keeps client resolution."
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (dump_dir / "LATEST").write_text(str(out), encoding="utf-8")
    return out


def build_model_config(infer_cfg: DictConfig) -> Any:
    """Build ``actor.model``-style OmegaConf for eval inference from YAML."""
    model = infer_cfg.model
    openpi = model.openpi
    num_chunks = int(model.num_action_chunks)
    action_dim = int(model.action_dim)
    return OmegaConf.create(
        {
            "precision": str(infer_cfg.get("precision", "bf16")),
            "pi05": bool(model.get("pi05", True)),
            "model_path": str(infer_cfg.ckpt),
            "num_action_chunks": num_chunks,
            "action_dim": action_dim,
            "num_steps": int(model.get("num_steps", 5)),
            "add_value_head": bool(model.get("add_value_head", False)),
            "openpi": {
                "task": str(openpi.task),
                "config_name": str(openpi.config_name),
                "num_images_in_input": int(openpi.get("num_images_in_input", 3)),
                "discrete_state_input": bool(openpi.get("discrete_state_input", True)),
                "max_token_len": int(openpi.get("max_token_len", 200)),
                "model_action_dim": int(openpi.get("model_action_dim", 32)),
                "paligemma_variant": str(openpi.get("paligemma_variant", "gemma_2b")),
                "action_expert_variant": str(
                    openpi.get("action_expert_variant", "gemma_300m")
                ),
                "action_chunk": num_chunks,
                "action_env_dim": action_dim,
            },
            "openpi_data": {"norm_stats_path": str(infer_cfg.norm_stats)},
        }
    )


class OpenPiRlinfInferServer:
    """Wraps openpi_rlinf eval model and robot-specific observation prep."""

    def __init__(
        self,
        infer_cfg: DictConfig,
        device: torch.device,
        dump_obs_dir: Path | None = None,
    ) -> None:
        self.infer_cfg = infer_cfg
        self.device = device
        self.robot = str(infer_cfg.robot)
        self.default_prompt = str(infer_cfg.default_prompt)
        self.dump_obs_dir = dump_obs_dir
        self._request_idx = 0
        if self.dump_obs_dir is not None:
            self.dump_obs_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Dumping decoded client observations to %s", self.dump_obs_dir)
        model_cfg = build_model_config(infer_cfg)
        logger.info(
            "Loading openpi_rlinf robot=%s config_name=%s from %s ...",
            self.robot,
            model_cfg.openpi.config_name,
            model_cfg.model_path,
        )
        self.model = get_model(model_cfg).to(device).eval()
        self.num_action_chunks = int(model_cfg.num_action_chunks)
        self.action_dim = int(model_cfg.action_dim)
        logger.info(
            "Model ready on %s (robot=%s chunks=%d dim=%d prompt=%r).",
            device,
            self.robot,
            self.num_action_chunks,
            self.action_dim,
            self.default_prompt,
        )

    def infer_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run inference from a plain JSON dict (client-compatible schema)."""
        prompt = self.default_prompt
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

        state_np = slice_robot_state(
            np.asarray(state_raw, dtype=np.float32), self.robot
        )
        high = decode_image_b64(cam_high_b64)
        left = decode_image_b64(cam_left_b64)
        right = decode_image_b64(cam_right_b64)

        if left.shape != high.shape or right.shape != high.shape:
            raise ValueError(
                f"all cameras must share shape; high={high.shape}, "
                f"left={left.shape}, right={right.shape}"
            )

        if self.dump_obs_dir is not None:
            self._request_idx += 1
            out = dump_client_observation(
                self.dump_obs_dir,
                request_idx=self._request_idx,
                robot=self.robot,
                prompt=prompt,
                state=state_np,
                high=high,
                left=left,
                right=right,
            )
            logger.info(
                "dumped client obs #%d shape=%s means_high=%s -> %s",
                self._request_idx,
                high.shape,
                _channel_means(high),
                out,
            )

        wrist = np.stack([left, right], axis=0)
        env_obs = {
            "states": torch.from_numpy(state_np.copy()).reshape(1, -1).to(self.device),
            "main_images": torch.from_numpy(high.copy())
            .reshape(1, *high.shape)
            .to(self.device),
            "wrist_images": torch.from_numpy(wrist.copy())
            .reshape(1, 2, *left.shape)
            .to(self.device),
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


def create_app(server: OpenPiRlinfInferServer) -> Flask:
    """Build Flask app bound to a loaded model."""
    app = Flask(__name__)

    @app.get("/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "robot": server.robot,
                "default_prompt": server.default_prompt,
            }
        )

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
        description="YAML-driven openpi_rlinf Cobot/Dobot HTTP inference server."
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Inference YAML name under toolkits/inference/config/ "
            "(e.g. cobot_cube_into_drawer) or a filesystem path."
        ),
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Optional override for YAML ckpt path.",
    )
    parser.add_argument(
        "--norm-stats",
        default=None,
        help="Optional override for YAML norm_stats path.",
    )
    parser.add_argument("--host", default=None, help="Override YAML host.")
    parser.add_argument("--port", type=int, default=None, help="Override YAML port.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu (slow).",
    )
    parser.add_argument(
        "--precision",
        default=None,
        choices=["bf16", "fp32"],
        help="Override YAML inference compute precision.",
    )
    parser.add_argument(
        "--dump-obs-dir",
        default=None,
        help=(
            "If set, save every decoded client observation (RGB PNGs + meta.json) "
            "under this directory for camera debugging."
        ),
    )
    return parser.parse_args()


def main() -> None:
    _ensure_repo_on_path()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    config_path = resolve_config_path(args.config)
    infer_cfg = load_infer_yaml(config_path)
    if args.ckpt is not None:
        infer_cfg.ckpt = args.ckpt
    if args.norm_stats is not None:
        infer_cfg.norm_stats = args.norm_stats
    if args.precision is not None:
        infer_cfg.precision = args.precision

    host = args.host if args.host is not None else str(infer_cfg.get("host", "0.0.0.0"))
    port = int(args.port) if args.port is not None else int(infer_cfg.get("port", 8000))

    ckpt = Path(str(infer_cfg.ckpt)).expanduser()
    norm_stats = Path(str(infer_cfg.norm_stats)).expanduser()
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not norm_stats.is_file():
        raise FileNotFoundError(f"norm_stats not found: {norm_stats}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; use --device cpu")

    dump_obs_dir = (
        Path(args.dump_obs_dir).expanduser().resolve() if args.dump_obs_dir else None
    )

    logger.info("Using inference config %s", config_path)
    server = OpenPiRlinfInferServer(infer_cfg, device, dump_obs_dir=dump_obs_dir)
    app = create_app(server)
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
