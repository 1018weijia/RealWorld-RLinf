#!/usr/bin/env bash
# XRobot USB Stage 2: preflight, LeRobot convert, Cal-QL, then the WS server.
# Site paths stay in .private-xrobot-stage2/paths.env. This launcher does not
# override residual_scale or chunk length; those come from the USB YAML (0.4 / 50).
#
# Required in paths.env:
#   XROBOT_USB_STAGE1_CHECKPOINT=/path/to/global_step_*
#   XROBOT_USB_NORM_STATS=/path/to/usb_plug/norm_stats.json
#   XROBOT_USB_DATASET=/path/to/XRobot_USB   # LeRobot v3 root with meta/info.json
# Optional:
#   XROBOT_OFFLINE_BUFFER_ROOT=/data/gxy/realworldRL/offline_rl_buffers
#   XROBOT_USB_OFFLINE_BUFFER=$XROBOT_OFFLINE_BUFFER_ROOT/xrobot_usb_plug/offline_buffer.pt
set -euo pipefail
set +x
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
private="$root/.private-xrobot-stage2"
task="${1:-usb_plug}"
mode="${2:-preflight}"
shift "$(( $# >= 2 ? 2 : $# ))"
case "$mode" in preflight|train|eval|eval-stage1|audit|convert|offline) ;; *)
    echo "Expected preflight|train|eval|eval-stage1|audit|convert|offline" >&2
    exit 2
    ;;
esac
if [[ "$task" != "usb_plug" ]]; then
    echo "Unknown XRobot task: $task" >&2
    exit 2
fi
if [[ "$mode" == "eval-stage1" && -n "${STAGE2_RESUME_DIR:-}" ]]; then
    echo "Pure Stage1 evaluation must not restore Stage2: unset STAGE2_RESUME_DIR" >&2
    exit 2
fi
# shellcheck disable=SC1091
source "$private/paths.env"
: "${XROBOT_USB_STAGE1_CHECKPOINT:?Set XROBOT_USB_STAGE1_CHECKPOINT in $private/paths.env}"
: "${XROBOT_USB_NORM_STATS:?Set XROBOT_USB_NORM_STATS in $private/paths.env}"
export XROBOT_USB_STAGE1_CHECKPOINT XROBOT_USB_NORM_STATS
checkpoint="$XROBOT_USB_STAGE1_CHECKPOINT"
stats="$XROBOT_USB_NORM_STATS"
prompt="Bimanual usb pick and insert"
port=8016
buffer_root="${XROBOT_OFFLINE_BUFFER_ROOT:-/data/gxy/realworldRL/offline_rl_buffers}"
task_dir="$buffer_root/xrobot_${task}"
test -f "$checkpoint/actor/model_state_dict/full_weights.pt"
test -f "$stats"
unset WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID
if [[ -n "${RLT_WANDB_ENTITY:-}" ]]; then export WANDB_ENTITY="$RLT_WANDB_ENTITY"; fi
export WANDB_CONFIG_DIR="$private/wandb/$task/config"
export WANDB_CACHE_DIR="$private/wandb/$task/cache"
export WANDB_DIR="$private/wandb/$task/runs"
export WANDB_PROJECT="${WANDB_PROJECT:-xrobot-usb-offline}"
export WANDB_BASE_URL=https://api.wandb.ai
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
umask 077
mkdir -p "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR" "$WANDB_DIR"
log_backends='[tensorboard]'
if [[ -f "$private/wandb_api_key" ]]; then
    export WANDB_API_KEY="$(< "$private/wandb_api_key")"
    export WANDB_MODE=online
    log_backends='[wandb,tensorboard]'
else
    export WANDB_MODE=offline
fi
if [[ "$mode" == "offline" ]]; then
    experiment_name="usb_plug-offline-calql"
    out="${RLT_RUN_DIR:-$task_dir/pretrain_$(date +%Y%m%d_%H%M%S)}"
    if [[ -e "$out" ]]; then
        echo "Refusing to reuse existing run directory: $out" >&2
        exit 2
    fi
    mkdir -p "$out"
elif [[ "$mode" == "convert" || "$mode" == "audit" ]]; then
    experiment_name="usb_plug-offline-convert"
    out="${RLT_RUN_DIR:-$task_dir}"
    mkdir -p "$out"
else
    experiment_name="xrobot-stage2-$task"
    out="${RLT_RUN_DIR:-$root/results/xrobot_stage2_${task}/$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$out"
fi
args=()
if [[ -n "${STAGE2_RESUME_DIR:-}" ]]; then
    test -f "$STAGE2_RESUME_DIR/stage2_state.pt"
    args+=("runner.resume_dir=$STAGE2_RESUME_DIR")
fi
if [[ "$mode" == "preflight" ]]; then args+=(server.preflight_only=True); fi
if [[ "$mode" == "eval" ]]; then
    : "${STAGE2_RESUME_DIR:?Evaluation requires a trained Stage2 checkpoint directory}"
    args+=(server.eval_only=True)
fi
if [[ "$mode" == "eval-stage1" ]]; then
    args+=(server.eval_only=True +server.vla_only=True)
fi
entry=rlt_stage2_server.py
if [[ "$mode" == "audit" || "$mode" == "convert" || "$mode" == "offline" ]]; then
    entry=cobot_offline.py
    dataset="${XROBOT_USB_DATASET:-}"
    offline_mode="$mode"
    if [[ "$mode" == "offline" ]]; then offline_mode=train; fi
    if [[ "$mode" != "offline" ]]; then
        : "${dataset:?Set XROBOT_USB_DATASET to a verified LeRobot v3 root}"
        test -f "$dataset/meta/info.json"
    fi
    buffer="${OFFLINE_BUFFER:-${XROBOT_USB_OFFLINE_BUFFER:-$task_dir/offline_buffer.pt}}"
    args+=("+offline.mode=$offline_mode" "+offline.dataset_root=$dataset" "+offline.buffer=$buffer"
        "+offline.steps=${NUM_TRAIN_STEPS:-40000}" "+offline.max_episodes=${MAX_EPISODES:-0}"
        "+offline.allow_partial=${ALLOW_PARTIAL_DATASET:-false}"
        "+offline.allow_actor_reconfiguration=${RLT_XROBOT_ALLOW_ACTOR_RECONFIGURATION:-false}"
        "+offline.validation_every=${VALIDATION_EVERY:-500}" "+offline.save_every=${SAVE_EVERY:-5000}")
fi
if [[ "$mode" == "train" || "$mode" == "eval" ]]; then
    args+=("+algorithm.offline_sample_ratio=${RLT_XROBOT_OFFLINE_SAMPLE_RATIO:-0.1}")
fi
cd "$root"
exec "$root/.venv/bin/python" "examples/embodiment/$entry" \
    --config-path "$root/examples/embodiment/config" --config-name xrobot_usb_plug_rlt_stage2_ws_server \
    "server.host=${RLT_SERVER_BIND:-0.0.0.0}" "server.port=${RLT_SERVER_PORT:-$port}" \
    "server.task_prompt=$prompt" "rlt_feature_model.openpi_data.default_prompt=$prompt" \
    "rlt_feature_model.model_path=$checkpoint" "rlt_feature_model.openpi_data.norm_stats_path=$stats" \
    "server.save_dir=$out/checkpoints" "runner.logger.log_path=$out" \
    "runner.logger.project_name=$WANDB_PROJECT" "runner.logger.experiment_name=$experiment_name" \
    "runner.logger.logger_backends=$log_backends" \
    "${args[@]}" "$@"
