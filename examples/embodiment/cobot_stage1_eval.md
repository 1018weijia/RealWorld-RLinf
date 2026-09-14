# Cobot Stage1-Only Evaluation

Use `start_cobot_stage2.sh <task> eval-stage1` to evaluate the frozen Stage1
checkpoint selected by the project's private task paths. This is not a fresh
Stage2 actor or a resumed offline run: no Stage2 model, optimizer, or replay
buffer is constructed or restored.

```bash
# Run from the RLinf repository root. Confirm that the port is unused first.
unset STAGE2_RESUME_DIR
export CUDA_VISIBLE_DEVICES=6
export RLT_SERVER_PORT=8010
export RLT_RUN_DIR="$PWD/results/cobot_stage1_assemble_parts/eval_$(date +%Y%m%d_%H%M%S)"
bash examples/embodiment/start_cobot_stage2.sh assemble_parts eval-stage1
```

The action path is the same unedited Stage1 reference used during online warmup,
but it remains active indefinitely. Unlike warmup, it never collects replay,
trains, or switches to Stage2. It samples one Stage1 reference, executes the first
30 of 50 actions, and preserves the joint14 left/right order and normalization.
Stage1's own sampling is unchanged; no Stage2 exploration noise or EXPO ranking
is used. Resuming Stage2 or enabling training with `server.vla_only=true` is
rejected before model loading.

The WebSocket handshake reports `eval_only=true`, `vla_only=true`,
`action_selection_mode=vla`, and `edit_scale=0`. Status reports zero online and
offline buffers and update counts.

On the robot computer, use the updated rlt-openpi client:

```bash
export RLT_SERVER_PORT=8010
export NUM_EPISODES=30
export DISPLAY_DATA=true
unset RLT_REQUIRED_STAGE2_UPDATES RLT_REQUIRED_OFFLINE_UPDATES RLT_DATASET_ROOT
bash exp/rlinf_client_cobot.sh assemble_parts eval-stage1
```

The client validates the explicit pure-Stage1 contract before connecting to
robot resources. Local executed-step recordings remain enabled and are labelled
`rlinf_stage1`; inference-wait preview frames are not added to the dataset.
Use `check --stage1-only` or `dry-run --stage1-only` before physical execution.
If another client is using the old service, do not interrupt its active chunk:
stop it safely first, or launch on another free port and match the client port.
