#! /bin/bash
# Bring up the Cobot Magic control stack, then hand off to the RLT client.
#
# This script exists because the RLT client deliberately knows nothing about
# how Cobot is driven: it only speaks the transport data contract. Everything
# hardware-specific — ROS master, arm drivers, cameras, teleoperation — is
# started here and must be healthy before any chunk is executed.
#
# Ray is NOT used by the RLT Stage 2 path. Do not run ray start.
#
# Usage: bash examples/embodiment/run_cobot_control.sh [config_name] [hydra overrides...]

set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

CONFIG_NAME="${1:-cobot_rlt_stage2_ws_client}"
shift || true

# ---------------------------------------------------------------------------
# Site-specific setup. Fill these in for your cell before a real run.
# ---------------------------------------------------------------------------

# 1. Python environment holding the Cobot drivers.
#    source <your_venv_path>/bin/activate

# 2. ROS workspace for the arm and camera drivers.
#    source /opt/ros/noetic/setup.bash
#    source <your_catkin_ws>/devel/setup.bash

# 3. Arm and camera bring-up. These must be running and publishing before the
#    client connects. A missing camera now raises in observe(); it is no longer
#    replaced with a black frame.
#    roslaunch cobot_magic bringup.launch &
#    sleep 10

export ROBOT_PLATFORM="${ROBOT_PLATFORM:-cobot}"

# ---------------------------------------------------------------------------
# Safety gate: refuse to arm the robot against the mock adapter.
# ---------------------------------------------------------------------------
if [[ "$*" != *"transport.is_dummy=false"* ]]; then
  echo "WARNING: transport.is_dummy is not set to false."
  echo "         The client will run MockRewindAdapter: no arm will move and"
  echo "         every camera frame will be black. Pass transport.is_dummy=false"
  echo "         together with a controller factory for a real run."
  echo
fi

exec bash "${EMBODIED_PATH}/run_rlt_stage2_client.sh" "${CONFIG_NAME}" "$@"
