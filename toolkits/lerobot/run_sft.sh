#!/usr/bin/env bash
# Generic VLA SFT launcher (cobot / dobot / xrobot / hf_v21 / usb, ...).
#
# Usage:
#   bash toolkits/lerobot/run_sft.sh <config_name> <physical_gpu_id> [resume_dir]
#
# Examples:
#   bash toolkits/lerobot/run_sft.sh dobot_sft_openpi_rlinf_pi05_cook_vegetable_hf_v21 0
#   bash toolkits/lerobot/run_sft.sh xrobot_sft_openpi_rlinf_pi05_usb_plug 7
#   bash toolkits/lerobot/run_sft.sh cobot_sft_openpi_rlinf_pi05_cook_vegetable 2
#
# GPU selection uses COBOT_GPU_ID -> cluster.component_placement in YAML.
# Prefers repo .venv/bin/python so tmux/non-interactive shells work without activate.

set -euo pipefail
CONFIG_NAME="${1:?config name required}"
GPU_ID="${2:?physical GPU id required}"
# Optional: path to checkpoints/global_step_N (must contain actor/)
RESUME_DIR="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/env_ffmpeg_torchcodec.sh"
unset CUDA_VISIBLE_DEVICES

export COBOT_GPU_ID="${GPU_ID}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/gxy/realworldRL/checkpoints/lerobot_home}"
export EMBODIED_PATH="${REPO_PATH}/examples/sft"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export RAY_TMPDIR="${RAY_TMPDIR:-/data/gxy/realworldRL/ray_tmp}"
export TMPDIR="${TMPDIR:-/data/gxy/realworldRL/tmp}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "${REPO_PATH}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_PATH}/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

LOG_DIR="${REPO_PATH}/logs/$(date +%Y%m%d-%H%M%S)-${CONFIG_NAME}-gpu${GPU_ID}"
mkdir -p "${LOG_DIR}"

CMD=(
  "${PYTHON_BIN}" "${REPO_PATH}/examples/sft/train_vla_sft.py"
  --config-path "${EMBODIED_PATH}/config"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
)
if [[ -n "${RESUME_DIR}" ]]; then
  # Struct config: resume_dir is optional, so use + to insert the key.
  CMD+=("+runner.resume_dir=${RESUME_DIR}")
fi

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "GPU placement: COBOT_GPU_ID=${COBOT_GPU_ID} (physical GPU ${GPU_ID})"
echo "RAY_ADDRESS=${RAY_ADDRESS} RESUME_DIR=${RESUME_DIR:-<none>}"
echo "${CMD[*]}" | tee "${LOG_DIR}/run.log"
"${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/run.log"
