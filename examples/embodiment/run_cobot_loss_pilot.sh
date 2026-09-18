#!/usr/bin/env bash
# Small controlled trials using existing joint-motion-v2 cached features.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task="${1:-assemble_parts}"
variant="${2:-D_family}"
shift "$(( $# >= 2 ? 2 : $# ))"
case "$task" in assemble_parts|cube_into_drawer|cook_vegetable|pack_and_pour_fruit) ;; *) echo "Unknown task: $task" >&2; exit 2 ;; esac
q_weight=0.01
mc_steps=0
calibration=policy_only
case "$variant" in
    A_bc) q_weight=0 ;;
    B_calql) ;;
    C_mc) mc_steps=1000 ;;
    D_family) calibration=reachable_family ;;
    *) echo "Expected A_bc|B_calql|C_mc|D_family" >&2; exit 2 ;;
esac
: "${OFFLINE_BUFFER:?Set OFFLINE_BUFFER to the verified joint-motion-v2 buffer}"
test -f "$OFFLINE_BUFFER"
export RLT_RUN_DIR="${RLT_RUN_DIR:-$root/results/cobot_loss_v3_$task/$variant}"
if [[ -d "$RLT_RUN_DIR" && -z "${STAGE2_RESUME_DIR:-}" ]]; then
    echo "Run exists: $RLT_RUN_DIR; choose a new output or an explicit resume checkpoint" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-5000}" SAVE_EVERY="${SAVE_EVERY:-1000}"
export RLT_COBOT_ACTOR_NOISE_SIGMA=0.025 RLT_COBOT_RESIDUAL_SCALE=0.3
export RLT_COBOT_ALLOW_ACTOR_RECONFIGURATION=false
mkdir -p "$(dirname "$RLT_RUN_DIR")"
bash "$root/examples/embodiment/start_cobot_stage2.sh" "$task" offline \
    ++offline.objective_version=joint-loss-v3 ++offline.calql_alpha=0.1 \
    "++offline.calql_calibration=$calibration" \
    ++offline.bc_weight=1.0 "++offline.q_weight=$q_weight" \
    ++offline.q_warmup_steps=2000 ++offline.q_ramp_steps=2000 \
    "++offline.mc_warmup_steps=$mc_steps" ++offline.q_aggregation=min \
    ++offline.gradient_every=500 ++offline.validation_seed=918 \
    "runner.logger.experiment_name=cobot-loss-v3-$task-$variant" "$@" 2>&1 | tee -a "${RLT_RUN_DIR}.log"
