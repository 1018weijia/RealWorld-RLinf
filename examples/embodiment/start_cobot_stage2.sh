#!/usr/bin/env bash
# Deploy this under RLinf/examples/embodiment. Site paths and credentials stay private.
set -euo pipefail
set +x
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
private="$root/.private-cobot-stage2"
task="${1:-assemble_parts}"
mode="${2:-preflight}"
shift "$(( $# >= 2 ? 2 : $# ))"
case "$mode" in preflight|train|eval) ;; *) echo "Expected preflight|train|eval" >&2; exit 2 ;; esac
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
cd "$root"
exec "$root/.venv/bin/python" examples/embodiment/rlt_stage2_server.py \
    --config-path "$root/examples/embodiment/config" --config-name cobot_rlt_stage2_ws_server \
    "server.host=${RLT_SERVER_BIND:-0.0.0.0}" "server.port=${RLT_SERVER_PORT:-$port}" \
    "server.task_prompt=$prompt" "rlt_feature_model.openpi_data.default_prompt=$prompt" \
    "rlt_feature_model.model_path=$checkpoint" "rlt_feature_model.openpi_data.norm_stats_path=$stats" \
    rlt_feature_model.num_action_chunks=50 actor.model.ref_num_action_chunks=50 actor.model.num_action_chunks=30 \
    "server.save_dir=$out/checkpoints" "runner.logger.log_path=$out" \
    "runner.logger.project_name=$WANDB_PROJECT" "runner.logger.experiment_name=cobot-stage2-$task" \
    'runner.logger.logger_backends=[wandb,tensorboard]' \
    "${args[@]}" "$@"
