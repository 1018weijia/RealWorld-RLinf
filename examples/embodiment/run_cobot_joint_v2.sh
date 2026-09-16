#!/usr/bin/env bash
# One versioned entrypoint for a fresh conversion, offline pretrain and online handoff.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task="${1:-assemble_parts}"
mode="${2:-preflight}"
shift "$(( $# >= 2 ? 2 : $# ))"
case "$task" in assemble_parts|cube_into_drawer|cook_vegetable|pack_and_pour_fruit) ;; *) echo "Unknown task: $task" >&2; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
case "$CUDA_VISIBLE_DEVICES" in 6|7) ;; *) echo "This Cobot experiment uses one of GPU 6 or 7 only" >&2; exit 2 ;; esac
export RLT_COBOT_ACTOR_NOISE_SIGMA="${RLT_COBOT_ACTOR_NOISE_SIGMA:-0.025}"
# This legacy scalar is inert for motion v2; physical budgets live in the YAML.
export RLT_COBOT_RESIDUAL_SCALE=0.3
export RLT_COBOT_ALLOW_ACTOR_RECONFIGURATION=false
result="$root/results/cobot_joint_v2_$task"
export OFFLINE_BUFFER="${OFFLINE_BUFFER:-$result/offline_buffer.pt}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-40000}"
export MAX_EPISODES="${MAX_EPISODES:-0}"
case "$mode" in
    prepare)
        if [[ -n "${STAGE2_RESUME_DIR:-}" ]]; then echo "prepare must start fresh; unset STAGE2_RESUME_DIR" >&2; exit 2; fi
        export RLT_RUN_DIR="${RLT_RUN_DIR:-$result/offline_train}"
        RLT_RUN_DIR="${RLT_RUN_DIR}_conversion" bash "$root/examples/embodiment/start_cobot_stage2.sh" "$task" convert "$@"
        exec bash "$root/examples/embodiment/run_cobot_joint_v2.sh" "$task" offline "$@"
        ;;
    offline)
        export RLT_RUN_DIR="${RLT_RUN_DIR:-$result/offline_train}"
        if [[ -d "$RLT_RUN_DIR" && -z "${STAGE2_RESUME_DIR:-}" ]]; then
            echo "Run exists: $RLT_RUN_DIR; set STAGE2_RESUME_DIR to a completed checkpoint or choose a new RLT_RUN_DIR" >&2
            exit 2
        fi
        ;;
    train|eval)
        export STAGE2_RESUME_DIR="${STAGE2_RESUME_DIR:-$result/offline_train/checkpoints/offline_step_40000}"
        test -f "$STAGE2_RESUME_DIR/stage2_state.pt"
        test -f "$STAGE2_RESUME_DIR/offline_buffer.pt"
        export RLT_RUN_DIR="${RLT_RUN_DIR:-$result/${mode}_$(date +%Y%m%d_%H%M%S)}"
        export RLT_SERVER_PORT="${RLT_SERVER_PORT:-8010}"
        ;;
    audit|convert|preflight|eval-stage1) ;;
    *) echo "Expected prepare|audit|convert|offline|train|eval|eval-stage1|preflight" >&2; exit 2 ;;
esac
exec bash "$root/examples/embodiment/start_cobot_stage2.sh" "$task" "$mode" "$@"
