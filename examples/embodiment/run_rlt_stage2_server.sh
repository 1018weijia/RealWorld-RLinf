#! /bin/bash
# Launch the RLT Stage 2 online-RL WebSocket server (training side).
#
# Run this on the GPU machine. It holds the frozen Stage 1 VLA, the Stage 2
# actor/critic, replay and the optimizers. Ray is not used and must not be
# started. Start this before the robot client; the client retries until the
# Stage 1 checkpoint finishes loading.
#
# Usage: bash examples/embodiment/run_rlt_stage2_server.sh [config_name] [hydra overrides...]

set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/rlt_stage2_server.py"
# RLinf is used from the source tree, not pip-installed, and running the script
# by path puts only its own directory on sys.path.
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

CONFIG_NAME="${1:-cobot_rlt_stage2_ws_server}"
shift || true

# OpenPI renders nothing, but MuJoCo-backed utilities in the import path pick
# up this variable and fail without a display otherwise.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Keep the Stage 1 forward pass deterministic across restarts so a resumed run
# reproduces the reference chunks it stored in replay.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
LOG_FILE="${LOG_DIR}/rlt_stage2_server.log"
mkdir -p "${LOG_DIR}"

CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} $*"
echo "${CMD}" > "${LOG_FILE}"
${CMD} 2>&1 | tee -a "${LOG_FILE}"
