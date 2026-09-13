#!/usr/bin/env bash
# Fresh Cal-QL training with the compatible Franka actor scalars, never resume.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task="${1:?Expected a Cobot task name}"
shift
case "$task" in
    assemble_parts|cube_into_drawer|cook_vegetable|pack_and_pour_fruit) ;;
    *) echo "Unknown Cobot task: $task" >&2; exit 2 ;;
esac
if [[ -n "${STAGE2_RESUME_DIR:-}" ]]; then
    echo "Fresh training requires STAGE2_RESUME_DIR to be unset; old checkpoints are preserved." >&2
    exit 2
fi
export RLT_COBOT_ACTOR_NOISE_SIGMA=0.1
export RLT_COBOT_RESIDUAL_SCALE=0.3
export RLT_COBOT_ALLOW_ACTOR_RECONFIGURATION=true
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-40000}"
export OFFLINE_BUFFER="${OFFLINE_BUFFER:-$root/results/cobot_offline_${task}/offline_buffer.pt}"
export RLT_RUN_DIR="${RLT_RUN_DIR:-$root/results/cobot_offline_${task}/pretrain_noise01_residual03_$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
test -f "$OFFLINE_BUFFER"
if [[ -e "$RLT_RUN_DIR" ]]; then
    echo "Refusing to reuse existing run directory: $RLT_RUN_DIR" >&2
    exit 2
fi
mkdir -p "$RLT_RUN_DIR"
bash "$root/examples/embodiment/start_cobot_stage2.sh" "$task" offline "$@" \
    2>&1 | tee "$RLT_RUN_DIR/train.log"
