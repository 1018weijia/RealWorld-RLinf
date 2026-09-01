#!/usr/bin/env bash
# Generic VLA SFT launcher (cobot / dobot / xrobot / hf_v21 / usb, ...).
#
# Usage:
#   bash toolkits/lerobot/run_sft.sh <config_name> <physical_gpu_id> [resume_dir] [hydra_overrides...]
#
# Examples:
#   bash toolkits/lerobot/run_sft.sh cobot_rlt_stage1_sft_openpi_pi05_cube_into_drawer 6
#   bash toolkits/lerobot/run_sft.sh cobot_rlt_stage1_sft_openpi_pi05_cube_into_drawer 6 "" \
#     actor.micro_batch_size=32 actor.global_batch_size=32

set -euo pipefail
CONFIG_NAME="${1:?config name required}"
GPU_ID="${2:?physical GPU id required}"
shift 2

RESUME_DIR=""
EXTRA_OVERRIDES=()
if [[ $# -gt 0 ]]; then
  if [[ "${1}" == *=* ]] || [[ "${1}" == +* ]]; then
    EXTRA_OVERRIDES=("$@")
  else
    RESUME_DIR="${1}"
    shift
    EXTRA_OVERRIDES=("$@")
  fi
fi

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
  CMD+=("+runner.resume_dir=${RESUME_DIR}")
fi
if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_OVERRIDES[@]}")
fi

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "GPU placement: COBOT_GPU_ID=${COBOT_GPU_ID} (physical GPU ${GPU_ID})"
echo "RAY_ADDRESS=${RAY_ADDRESS} RESUME_DIR=${RESUME_DIR:-<none>}"
echo "${CMD[*]}" | tee "${LOG_DIR}/run.log"
"${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/run.log"
