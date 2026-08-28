#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${DREAMZERO_PATH}:$PYTHONPATH

export PYTHONPATH=/vast/users/meng.cao/zhangjian/openpi-repos/RL-Token-nova/openpi-main/src:/vast/users/meng.cao/zhangjian/openpi-repos/RL-Token-nova/openpi-main/packages/openpi-client/src:${PYTHONPATH}
export OPENPI_DATA_HOME=/vast/users/xiaodan/zhangjian/RealRL/openpi-repos/RL-Token-nova/.cache/openpi
export JAX_PLATFORMS=cpu

# Ray AMD GPU manager refuses ROCR_VISIBLE_DEVICES without HIP_VISIBLE_DEVICES.
# Slurm typically injects ROCR; PyTorch ROCm only honors HIP. Copy then drop ROCR.
if [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
  if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
  else
    export HIP_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,3,4,5,6,7}"
  fi
fi
unset ROCR_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

# wandb: SFTRunner logs from the Ray driver (not FSDP ranks). Needs outbound
# access to api.wandb.ai. If the node cannot reach wandb, use:
#   WANDB_MODE=offline bash examples/sft/run_vla_sft.sh franka_pi05_rlinf
# then `wandb sync <log_dir>/wandb` later.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-82c717f44cce17db5f832cbe357a6a52e20677c1}"

# MIOpen: 8 FSDP ranks racing the default /tmp find-db SQLite file causes
# miopenStatusInternalError in SigLIP conv2d. Keep a writable per-user cache
# on local /tmp (not NFS /vast) and skip exhaustive kernel search.
MIOPEN_CACHE_ROOT="${MIOPEN_CACHE_ROOT:-/tmp/${USER}/miopen_rlinf_sft}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$MIOPEN_CACHE_ROOT/user_db}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-$MIOPEN_CACHE_ROOT/kernels}"
export MIOPEN_DEBUG_DISABLE_SQL_WAL="${MIOPEN_DEBUG_DISABLE_SQL_WAL:-1}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
# LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
LOG_DIR="/vast/users/xiaodan/zhangjian/RealRL/rlinf/logs/sft/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}