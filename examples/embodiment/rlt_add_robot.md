# Adding a Robot to the RLT Stage 2 Server

The Stage 2 server is not written per robot. The learner, the replay buffer,
the checkpointing and the WebSocket protocol are the same for every
embodiment, and there is no branch anywhere in `rlinf/serving/rlt/` on robot
identity. Joint angles versus end-effector poses are decoded by the frozen
Stage 1 OpenPI transform pipeline, not by the server.

What does vary is the *description* of a robot: how wide an action is, how many
actions the arm executes per chunk, which cameras arrive and which Stage 1
pipeline decodes the result. That description lives in one file.

## What defines a robot, and what does not

`config/embodiment/<name>.yaml` holds the robot contract. It deliberately does
**not** hold the task prompt, the norm statistics or the Stage 1 checkpoint:
those change when you teach the same arm a new skill, and folding them in here
would force a new embodiment per skill. Task-level settings live in the server
config that selects the embodiment.

## Prerequisites

A new robot needs a Stage 1 pipeline before Stage 2 can serve it:

1. An OpenPI data config registered in
   `rlinf/models/embodiment/openpi/dataconfig/__init__.py` — this is what
   actually knows the robot: camera repack, delta/absolute action masks, the
   Aloha decode and un-normalization. `pi05_cobot_magic`, `pi05_xrobot` and
   `pi05_dobot` are already registered.
2. A Stage 1 SFT checkpoint trained with `openpi.use_rlt=True`. A plain SFT
   checkpoint has no `rlt_module.*` weights and preflight rejects it.
3. The `norm_stats.json` produced for that Stage 1 run.

## Step 1: declare the embodiment

Copy `config/embodiment/x2robot.yaml` and set the nine values. The two that are
easy to get wrong:

- `chunk_length` is how many actions the arm executes per chunk, and one chunk
  is one replay transition, so this sets the unit of training data. It may be
  smaller than `ref_chunk_length`, the horizon Stage 1 proposes: Cobot executes
  the first 16 of 20 because the tail of a horizon is its least accurate part.
- `camera_keys` is ordered. The main view comes first, and the remaining views
  reach the Aloha transform as left then right, so swapping two wrist entries
  mirrors the robot without raising anything.

`action_schema` is free-form but should be versioned, e.g. `x2robot-ee14-v1`.
It exists because two embodiments can share every tensor width and still be
incompatible — 14 joint angles and a 14-wide dual-arm pose look identical on
the wire.

## Step 2: write the server config

```yaml
defaults:
  - cobot_rlt_stage2_ws_server
  - override embodiment: my_robot
  - _self_

server:
  task_prompt: "my task"
  save_dir: "../results/my_robot_rlt_stage2/checkpoints"

runner:
  logger:
    experiment_name: "my_robot_rlt_stage2"

rlt_feature_model:
  model_path: /path/to/stage1/global_step_30000
  openpi_data:
    repo_id: "my_robot/my_task"
    norm_stats_path: /path/to/norm_stats.json
    default_prompt: "my task"
```

`task_prompt` and `default_prompt` must be identical; preflight rejects a
disagreement, because the two feed different code paths and a mismatch
conditions the frozen VLA on a task it never learned. `norm_stats_path` is
mandatory even though the OpenPI config names an `asset_id`: without it OpenPI
falls back to that default, which is another task's quantiles, and every action
is de-normalized into wrong robot units.

`cobot_rlt_stage2_ws_server` is the de-facto base config; it is named after the
first robot to use it, but every embodiment number in it interpolates from the
group you just overrode, so nothing robot-specific leaks through. Its *task*
values do leak, which is why the block above overrides all seven of them:
`task_prompt`, `save_dir`, `experiment_name`, `model_path`, `repo_id`,
`norm_stats_path` and `default_prompt`. Anything else it defines — the
learner, the optimizers, the replay and server settings — is meant to be
shared, and preflight warns about deviations from the reference deployment.

## Step 3: verify before arming the robot

```bash
python examples/embodiment/rlt_stage2_server.py \
  --config-name my_robot_rlt_stage2_ws_server \
  server.preflight_only=true
```

This runs every startup check without loading Stage 1, so it takes seconds.
The first line confirms the contract the client will be handed:

```
Preflight: embodiment my_robot (my-robot-v1), 14-D actions, 50-step chunks of 50 proposed
```

The embodiment check runs first, and fails if any config section disagrees with
the profile. That matters because a mismatch is otherwise silent: a Stage 2
head built for 14 actions on top of a pipeline configured for 16 runs fine and
sends wrong numbers to a real arm.

## What the client has to match

Clients are not part of this repository's server code and adapt themselves; the
handshake is the entire contract. The server reports `action_dim`,
`chunk_length`, `proprio_dim`, `camera_keys`, `task_prompt`, `robot_type` and
`action_schema`, and a client should refuse to arm on a mismatch. Chunks always
leave the server in robot units, already un-normalized and Aloha-decoded — a
client must not de-normalize again.

## Optional: offline pretraining

Without offline pretraining the Stage 2 head starts randomly initialized and
the server collects `warmup_steps` transitions on the Stage 1 reference policy
before it begins training. To pretrain instead, supply a reader implementing
`OfflineEpisodeSource` in `rlinf/serving/rlt/cobot_offline_data.py`: `prompt`,
`episodes`, `cameras`, `table()` and `frames()`. `CobotLeRobotV3` is the
reference implementation. Everything downstream of it — conversion, the buffer
and the Cal-QL trainer — is robot-agnostic and needs no changes.

## What you do not touch

Not the learner, the trainer, the replay buffer, the protocol, the policy
router or the inference path. If adding a robot seems to require editing any of
those, the embodiment contract is probably wrong rather than incomplete.
