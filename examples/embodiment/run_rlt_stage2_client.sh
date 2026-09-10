#! /bin/bash
# Launch the robot-side RLT Stage 2 client.
#
# Run this on the machine wired to the arms, after the server is up (the
# client retries the connection, so starting early is safe). No GPU and no
# model are needed here.
#
# The Cobot control stack must already be running; see
# examples/embodiment/run_cobot_control.sh.
#
# Usage: bash examples/embodiment/run_rlt_stage2_client.sh [config_name] [hydra overrides...]
# Example against a remote server:
#   bash examples/embodiment/run_rlt_stage2_client.sh cobot_rlt_stage2_ws_client \
#       client.host=10.0.0.7 transport.is_dummy=false

set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/rlt_stage2_client.py"
# RLinf is used from the source tree, not pip-installed, and running the script
# by path puts only its own directory on sys.path.
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

CONFIG_NAME="${1:-cobot_rlt_stage2_ws_client}"
shift || true

export ROBOT_PLATFORM="${ROBOT_PLATFORM:-cobot}"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
LOG_FILE="${LOG_DIR}/rlt_stage2_client.log"
mkdir -p "${LOG_DIR}"

CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} logging.log_path=${LOG_DIR} $*"
echo "${CMD}" > "${LOG_FILE}"
${CMD} 2>&1 | tee -a "${LOG_FILE}"
