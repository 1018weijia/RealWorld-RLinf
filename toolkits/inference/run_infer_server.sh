#!/usr/bin/env bash
# Start an openpi_rlinf HTTP inference server from a YAML under
# toolkits/inference/config/.
#
# Usage:
#   bash toolkits/inference/run_infer_server.sh <config_name> <gpu_id> [port]
#
# Examples:
#   bash toolkits/inference/run_infer_server.sh cobot_cube_into_drawer 0 8000
#   bash toolkits/inference/run_infer_server.sh dobot_cook_vegetable 7 8001
#
# Available configs (name without .yaml):
#   cobot_cube_into_drawer
#   cobot_cook_vegetable
#   cobot_assemble_parts
#   dobot_cook_vegetable
#   dobot_pour_water
#   dobot_tidy_up_the_desk
#   dobot_towel

set -euo pipefail

CONFIG_NAME="${1:?config name required (e.g. cobot_cube_into_drawer)}"
GPU_ID="${2:?physical GPU id required}"
PORT="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/config"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES

if [[ -f "${CONFIG_DIR}/${CONFIG_NAME}.yaml" ]]; then
  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}.yaml"
elif [[ -f "${CONFIG_NAME}" ]]; then
  CONFIG_PATH="${CONFIG_NAME}"
elif [[ -f "${CONFIG_DIR}/${CONFIG_NAME}" ]]; then
  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}"
else
  echo "ERROR: config not found: ${CONFIG_NAME}" >&2
  echo "Available:" >&2
  ls -1 "${CONFIG_DIR}"/*.yaml 2>/dev/null | xargs -n1 basename >&2
  exit 1
fi

CMD=(
  python "${SCRIPT_DIR}/openpi_rlinf_infer_server.py"
  --config "${CONFIG_PATH}"
  --device "cuda:${GPU_ID}"
  --host 0.0.0.0
)
if [[ -n "${PORT}" ]]; then
  CMD+=(--port "${PORT}")
fi

echo "GPU: cuda:${GPU_ID}"
echo "config: ${CONFIG_PATH}"
echo "${CMD[*]}"
"${CMD[@]}"
