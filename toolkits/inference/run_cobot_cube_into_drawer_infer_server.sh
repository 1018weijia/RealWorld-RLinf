#!/usr/bin/env bash
# Start Cobot cube_into_drawer openpi_rlinf inference server (global_step_20000).
#
# Usage:
#   bash toolkits/inference/run_cobot_cube_into_drawer_infer_server.sh <gpu_id> [port]
#
# Example:
#   bash toolkits/inference/run_cobot_cube_into_drawer_infer_server.sh 0 8000

set -euo pipefail

GPU_ID="${1:?physical GPU id required}"
PORT="${2:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES

CKPT="${CKPT:-${REPO_PATH}/logs/20260822-05:24:30-cobot_sft_openpi_rlinf_pi05/cobot_pi05_openpi_rlinf_sft/checkpoints/global_step_20000/actor/model_state_dict}"
NORM_STATS="${NORM_STATS:-/data/gxy/realworldRL/checkpoints/assets/cobot_magic/cube_into_drawer/norm_stats.json}"

echo "GPU: cuda:${GPU_ID}  port: ${PORT}"
echo "ckpt: ${CKPT}"
echo "norm_stats: ${NORM_STATS}"

python "${SCRIPT_DIR}/openpi_rlinf_cobot_infer_server.py" \
  --ckpt "${CKPT}" \
  --norm-stats "${NORM_STATS}" \
  --device "cuda:${GPU_ID}" \
  --host 0.0.0.0 \
  --port "${PORT}"
