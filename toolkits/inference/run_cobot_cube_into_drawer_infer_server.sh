#!/usr/bin/env bash
# Backward-compatible launcher for Cobot cube_into_drawer.
#
# Usage:
#   bash toolkits/inference/run_cobot_cube_into_drawer_infer_server.sh <gpu_id> [port]
#
# Example:
#   bash toolkits/inference/run_cobot_cube_into_drawer_infer_server.sh 0 8000
#
# Prefer the generic launcher for other tasks:
#   bash toolkits/inference/run_infer_server.sh dobot_cook_vegetable 7 8001

set -euo pipefail

GPU_ID="${1:?physical GPU id required}"
PORT="${2:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_infer_server.sh" cobot_cube_into_drawer "${GPU_ID}" "${PORT}"
