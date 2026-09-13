#!/usr/bin/env bash
# Deploy this under RLinf/examples/embodiment. Site paths and credentials stay private.
set -euo pipefail
set +x
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
private="$root/.private-cobot-stage2"
task="${1:-assemble_parts}"
mode="${2:-preflight}"
shift "$(( $# >= 2 ? 2 : $# ))"
case "$mode" in preflight|train|eval|audit|convert|offline) ;; *) echo "Expected preflight|train|eval|audit|convert|offline" >&2; exit 2 ;; esac
source "$private/paths.env"
case "$task" in
    assemble_parts) checkpoint="$COBOT_ASSEMBLE_CHECKPOINT"; stats="$COBOT_ASSEMBLE_STATS"; prompt="assemble parts"; port=8000 ;;
    cube_into_drawer) checkpoint="$COBOT_MIXED_CHECKPOINT"; stats="$COBOT_MIXED_STATS"; prompt="put cube in drawer"; port=8001 ;;
    cook_vegetable) checkpoint="$COBOT_MIXED_CHECKPOINT"; stats="$COBOT_MIXED_STATS"; prompt="cook vegetable"; port=8002 ;;
    pack_and_pour_fruit) checkpoint="$COBOT_MIXED_CHECKPOINT"; stats="$COBOT_MIXED_STATS"; prompt="pack fruit into a container and pour it out"; port=8003 ;;
    *) echo "Unknown Cobot task: $task" >&2; exit 2 ;;
esac
test -f "$checkpoint/actor/model_state_dict/full_weights.pt"
test -f "$stats"
export WANDB_API_KEY="$(< "$private/wandb_api_key")"
unset WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID
if [[ -n "${RLT_WANDB_ENTITY:-}" ]]; then export WANDB_ENTITY="$RLT_WANDB_ENTITY"; fi
export WANDB_CONFIG_DIR="$private/wandb/$task/config"
export WANDB_CACHE_DIR="$private/wandb/$task/cache"
export WANDB_DIR="$private/wandb/$task/runs"
export WANDB_PROJECT="cobot-stage2-$task"
export WANDB_MODE=online WANDB_BASE_URL=https://api.wandb.ai
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
umask 077
mkdir -p "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR" "$WANDB_DIR"
out="${RLT_RUN_DIR:-$root/results/cobot_stage2_${task}/$(date +%Y%m%d_%H%M%S)}"
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
entry=rlt_stage2_server.py
if [[ "$mode" == "audit" || "$mode" == "convert" || "$mode" == "offline" ]]; then
    entry=cobot_offline.py
    dataset="${COBOT_DATASET_ROOT:-}"
    if [[ -z "$dataset" ]]; then
        case "$task" in
            assemble_parts) dataset="${COBOT_ASSEMBLE_DATASET:-}" ;;
            cube_into_drawer) dataset="${COBOT_CUBE_DATASET:-}" ;;
            cook_vegetable) dataset="${COBOT_COOK_DATASET:-}" ;;
            pack_and_pour_fruit) dataset="${COBOT_PACK_DATASET:-}" ;;
        esac
    fi
    offline_mode="$mode"
    if [[ "$mode" == "offline" ]]; then offline_mode=train; fi
    if [[ "$mode" != "offline" ]]; then
        : "${dataset:?Set COBOT_DATASET_ROOT to a verified LeRobot v3 root}"
        test -f "$dataset/meta/info.json"
    fi
    buffer="${OFFLINE_BUFFER:-$root/results/cobot_offline_${task}/offline_buffer.pt}"
    args+=("+offline.mode=$offline_mode" "+offline.dataset_root=$dataset" "+offline.buffer=$buffer"
        "+offline.steps=${NUM_TRAIN_STEPS:-40000}" "+offline.max_episodes=${MAX_EPISODES:-0}"
        "+offline.allow_partial=${ALLOW_PARTIAL_DATASET:-false}"
        "+offline.allow_actor_reconfiguration=${RLT_COBOT_ALLOW_ACTOR_RECONFIGURATION:-false}"
        "+offline.validation_every=${VALIDATION_EVERY:-500}" "+offline.save_every=${SAVE_EVERY:-5000}")
fi
# Training and serving must instantiate the same actor. Conversion keeps the
# original feature contract; fresh training can explicitly reuse that cache.
if [[ "$mode" == "offline" || "$mode" == "train" || "$mode" == "eval" || "$mode" == "preflight" ]]; then
    args+=("actor.model.actor_noise_sigma=${RLT_COBOT_ACTOR_NOISE_SIGMA:-0.1}"
        "actor.model.residual_scale=${RLT_COBOT_RESIDUAL_SCALE:-0.3}")
fi
if [[ "$mode" == "train" || "$mode" == "eval" ]]; then
    warmup_steps="${RLT_COBOT_WARMUP_STEPS:-200}"
    utd_ratio="${RLT_COBOT_UTD_RATIO:-4}"
    demo_batch_ratio="${RLT_COBOT_DEMO_BATCH_RATIO:-0.45}"
    offline_sample_ratio="${RLT_COBOT_OFFLINE_SAMPLE_RATIO:-0.1}"
    expo_base_candidates="${RLT_COBOT_EXPO_BASE_CANDIDATES:-4}"
    expo_edited_candidates="${RLT_COBOT_EXPO_EDITED_CANDIDATES:-4}"
    args+=("server.warmup_steps=$warmup_steps" "server.utd_ratio=$utd_ratio"
        "algorithm.demo_batch_ratio=$demo_batch_ratio" "+algorithm.offline_sample_ratio=$offline_sample_ratio"
        "algorithm.expo.base_candidates=$expo_base_candidates" "algorithm.expo.edited_candidates=$expo_edited_candidates")
fi
cd "$root"
exec "$root/.venv/bin/python" "examples/embodiment/$entry" \
    --config-path "$root/examples/embodiment/config" --config-name cobot_rlt_stage2_ws_server \
    "server.host=${RLT_SERVER_BIND:-0.0.0.0}" "server.port=${RLT_SERVER_PORT:-$port}" \
    "server.task_prompt=$prompt" "rlt_feature_model.openpi_data.default_prompt=$prompt" \
    "rlt_feature_model.model_path=$checkpoint" "rlt_feature_model.openpi_data.norm_stats_path=$stats" \
    rlt_feature_model.num_action_chunks=50 actor.model.ref_num_action_chunks=50 actor.model.num_action_chunks=30 \
    "server.save_dir=$out/checkpoints" "runner.logger.log_path=$out" \
    "runner.logger.project_name=$WANDB_PROJECT" "runner.logger.experiment_name=cobot-stage2-$task" \
    'runner.logger.logger_backends=[wandb,tensorboard]' \
    "${args[@]}" "$@"
