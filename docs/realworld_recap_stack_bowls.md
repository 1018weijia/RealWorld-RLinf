# Real-World RECAP Stack Bowls Dataset and Training Notes

Last updated: 2026-09-03

This note records the real-world Franka stack-bowls rollout dataset prepared from
SFT-checkpoint rollout logs, and the concrete RLinf RECAP steps for training a
task-specific CFG policy weight.

Feishu target document:

```text
https://mcnklerpx59m.feishu.cn/wiki/Gs60w9DDMibnCIkCUZtcVOIrnhc
```

## 1. Source Machines and Paths

Jump host:

```bash
ssh amax@100.95.122.116
```

Robot / data host, reached from the jump host:

```bash
ssh pnp@192.168.10.110
```

Raw rollout label CSV:

```text
/home/pnp/桌面/franka_deploy_0128_ee/franka_deploy/logs/recap_episode_labels.csv
```

Raw rollout session root:

```text
/home/pnp/桌面/franka_deploy_0128_ee/franka_deploy/logs/session_async_*
```

Prepared lightweight LeRobot / RECAP rollout dataset:

```text
/home/pnp/桌面/franka_oral_data/stack_bowls_recap_rollout_rc
```

The prepared dataset uses symlinks for videos to avoid filling the robot host
disk. It is directly usable on `pnp@192.168.10.110` as long as the original
`franka_deploy` log directory remains in place. To copy it to another training
machine, use `rsync -L` or regenerate with copied/re-encoded videos.

## 2. Label CSV Summary

All rows are for the same stack-bowls task:

```text
Stack the three bowls in size order: the purple bowl first, then the beige bowl.
```

Task id:

```text
stack_bowls_in_size_order_rc
```

CSV columns:

```text
session, session_dir, task, task_id, prompt, trial, is_success,
checkpoint_profile, checkpoint_path, chunks, video_dir, notes, labeled_at
```

Label distribution:

| Item | Count |
| --- | ---: |
| Episodes | 34 |
| Successful episodes | 22 |
| Failed episodes | 12 |

Failure notes include no grasping movement, beige bowl grasp error, wrong order,
and failure to stack the purple bowl into the blue bowl.

## 3. Prepared Dataset Format

Prepared dataset path:

```text
/home/pnp/桌面/franka_oral_data/stack_bowls_recap_rollout_rc
```

Generated structure:

```text
stack_bowls_recap_rollout_rc/
  data/chunk-000/episode_000000.parquet ... episode_000033.parquet
  videos/chunk-000/image/episode_*.mp4
  videos/chunk-000/global_image/episode_*.mp4
  videos/chunk-000/right_image/episode_*.mp4
  videos/chunk-000/wrist_image/episode_*.mp4
  meta/info.json
  meta/tasks.jsonl
  meta/tasks.json
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/stats.json
  meta/recap_episode_labels.csv
  meta/returns_fail300.parquet
```

Dataset summary after validation:

| Item | Value |
| --- | ---: |
| Episodes | 34 |
| Frames | 196,579 |
| Video features | 4 |
| Successful frames | 81,863 |
| Failed frames | 114,716 |

Per-frame parquet columns:

```text
global_image, right_image, wrist_image, image, state, actions, task, task_index,
episode_index, frame_index, timestamp, index, is_success, source_session
```

Camera mapping:

| Raw log camera | LeRobot feature |
| --- | --- |
| `videos/cam_high.mp4` | `global_image`, also aliased as `image` |
| `videos/cam_side.mp4` | `right_image` |
| `videos/cam_wrist.mp4` | `wrist_image` |

The `image` alias is included because the current RECAP value dataset path for
`robot_type=franka` expects an `image` feature. The original 3-camera feature
names are also preserved for visualization and future multi-view training.

## 4. Return Sidecar

The RECAP Step 1 return sidecar has already been generated:

```text
/home/pnp/桌面/franka_oral_data/stack_bowls_recap_rollout_rc/meta/returns_fail300.parquet
```

Return rule:

```text
reward_t = -1 for non-terminal frames
terminal reward = 0 for successful episodes
terminal reward = -300 for failed episodes
gamma = 1.0
G_t = reward_t + gamma * G_{t+1}
```

Observed return/reward stats:

| Metric | Return | Reward |
| --- | ---: | ---: |
| count | 196,579 | 196,579 |
| mean | -5,872.25 | -1.018 |
| std | 5,036.17 | 2.336 |
| min | -20,713 | -300 |
| max | 0 | 0 |

Use `fail300` as the RECAP `data.tag` / `advantage.returns_tag`.

## 5. Rebuild Command

If the dataset needs to be regenerated from the label CSV, use the conversion
tool from this repository:

```bash
cd /path/to/RealWorld-RLinf
python toolkits/lerobot/build_recap_rollout_from_inference_logs.py \
  --labels-csv /home/pnp/桌面/franka_deploy_0128_ee/franka_deploy/logs/recap_episode_labels.csv \
  --output /home/pnp/桌面/franka_oral_data/stack_bowls_recap_rollout_rc \
  --task-id stack_bowls_in_size_order_rc \
  --alignment video_frames \
  --video-mode symlink \
  --returns-tag fail300 \
  --gamma 1.0 \
  --failure-reward -300.0 \
  --overwrite
```

Recommended defaults:

| Option | Recommended | Reason |
| --- | --- | --- |
| `--alignment` | `video_frames` | Keeps rows aligned to original 30 FPS video frames and avoids huge video re-encoding. |
| `--video-mode` | `symlink` | Uses little disk on the robot host. |
| `--returns-tag` | `fail300` | Generates `meta/returns_fail300.parquet` for RECAP Step 2/3. |

Portable dataset options:

```bash
# Copy original videos into the dataset.
--alignment video_frames --video-mode copy

# Expand every low-level action step and re-encode videos to match action frames.
--alignment action_frames --video-mode reencode
```

The portable options need much more disk space. On 2026-09-03 the robot host root
partition had about 37 GB free, so the lightweight symlink dataset was used.

## 6. RECAP Training Plan

Yes, this recap rollout dataset can be used alone to train a stack-bowls
task-specific weight. In all four stages, keep `train_data_paths` as a list with
only this dataset.

Dataset config entry:

```yaml
data:
  train_data_paths:
    - dataset_path: "/home/pnp/桌面/franka_oral_data/stack_bowls_recap_rollout_rc"
      type: "rollout"
      weight: 1.0
      robot_type: "franka"
      model_type: "pi05"
```

If training is run inside Docker, mount the dataset and replace the path with
the in-container path, for example:

```text
/workspace/dataset/stack_bowls_recap_rollout_rc
```

### Step 1: Compute Returns

This step has already been done for the prepared dataset. To recompute:

```bash
cd /workspace/RLinf
export REPO_PATH=/workspace/RLinf

bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
  recap_compute_returns \
  data.train_data_paths='[{dataset_path: "/workspace/dataset/stack_bowls_recap_rollout_rc", type: "rollout"}]' \
  data.gamma=1.0 \
  data.failure_reward=-300.0 \
  data.tag=fail300
```

Output:

```text
meta/returns_fail300.parquet
```

### Step 2: Value Model SFT

Train a value model on the prepared rollout dataset:

```bash
cd /workspace/RLinf
export REPO_PATH=/workspace/RLinf

bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh \
  recap_value_model_sft \
  data.tag=fail300 \
  data.train_data_paths='[{dataset_path: "/workspace/dataset/stack_bowls_recap_rollout_rc", type: "rollout", weight: 1.0, robot_type: "franka", model_type: "pi05"}]' \
  data.eval_data_paths='[{dataset_path: "/workspace/dataset/stack_bowls_recap_rollout_rc", max_samples: 10000, robot_type: "franka", model_type: "pi05"}]' \
  data.robot_type=franka \
  data.model_type=pi05 \
  data.action_dim=7 \
  actor.model.action_dim=7 \
  actor.model.siglip_path=/path/to/siglip2-so400m-patch14-224 \
  actor.model.gemma3_path=/path/to/gemma-3-270m \
  actor.model.tokenizer_path=/path/to/gemma-3-270m \
  runner.logger.experiment_name=stack_bowls_recap_value_sft
```

Record the value checkpoint path:

```text
logs/value_sft/<config>-<timestamp>/value_sft/checkpoints/global_step_<N>/actor/model_state_dict
```

### Step 3: Compute Advantages

Use the value checkpoint from Step 2:

```bash
cd /workspace/RLinf
export REPO_PATH=/workspace/RLinf

bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh \
  recap_compute_advantages \
  advantage.value_checkpoint=/path/to/value_sft/checkpoints/global_step_<N>/actor/model_state_dict \
  advantage.returns_tag=fail300 \
  advantage.tag=fail300_N10_stack_bowls_q30 \
  advantage.positive_quantile=0.3 \
  advantage.model.siglip_path=/path/to/siglip2-so400m-patch14-224 \
  advantage.model.gemma3_path=/path/to/gemma-3-270m \
  advantage.model.tokenizer_path=/path/to/gemma-3-270m \
  data.model_type=pi05 \
  data.train_data_paths='[{dataset_path: "/workspace/dataset/stack_bowls_recap_rollout_rc", robot_type: "franka", type: "rollout", weight: 1.0}]' \
  data.advantage_lookahead_step=10 \
  data.gamma=1.0
```

Output:

```text
meta/advantages_fail300_N10_stack_bowls_q30.parquet
```

Verify:

```bash
python - <<'PY'
import pandas as pd
p = "/workspace/dataset/stack_bowls_recap_rollout_rc/meta/advantages_fail300_N10_stack_bowls_q30.parquet"
df = pd.read_parquet(p)
print(len(df), df.columns.tolist())
print(df["advantage"].value_counts(dropna=False))
print(df["advantage_continuous"].describe())
PY
```

### Step 4: CFG Policy Training

Train the task-specific policy weight:

```bash
cd /workspace/RLinf
export REPO_PATH=/workspace/RLinf

bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
  cfg_rl_openpi \
  data.advantage_tag=fail300_N10_stack_bowls_q30 \
  data.train_data_paths='[{dataset_path: "/workspace/dataset/stack_bowls_recap_rollout_rc", type: "rollout", weight: 1.0}]' \
  actor.model.model_path=/path/to/pi05_base_pytorch \
  actor.model.openpi.config_name=pi05_franka_state \
  actor.model.openpi.positive_only_conditional=true \
  actor.model.openpi.unconditional_prob=0.1 \
  actor.model.openpi.guidance_type=positive \
  actor.optim.lr=1.0e-5 \
  actor.optim.total_training_steps=30000 \
  runner.logger.experiment_name=stack_bowls_recap_cfg
```

The older test config `tests/e2e_tests/offline_rl/recap_cfg_openpi_libero.yaml`
is useful as a smoke-test reference, but the command above uses the normal
offline-RL CFG training entrypoint and overrides the OpenPI config to Franka.

Choose `actor.model.openpi.config_name` according to the checkpoint/interface:

| Interface | Config name |
| --- | --- |
| Single-arm Franka state policy | `pi05_franka_state` |
| Single-arm Franka PnP policy | `pi05_franka_pnp` |
| Dual-Franka TCP rot6d | `pi05_dualfranka_tcp_rot6d` |

For the prepared dataset, `state` and `actions` are 7D EE pose + gripper, so
`pi05_franka_state` is the first config to try unless the SFT checkpoint was
trained with another Franka config.

## 7. Important Caveats

- The robot host `pnp@192.168.10.110` did not have `lerobot` or `rlinf`
  importable in the default Python environment on 2026-09-03, so training should
  run in an RLinf container or on a prepared GPU training machine.
- The lightweight dataset depends on symlinked videos under
  `/home/pnp/桌面/franka_deploy_0128_ee/franka_deploy/logs/session_async_*`.
- If the dataset is copied to a different machine, preserve symlink targets with
  `rsync -L` or regenerate with `--video-mode copy`.
- Keep robot credentials, model weights, camera serials, and private data out of
  git.
- The value-SFT stage currently supports `robot_type=franka` / `franka_co_train`
  transforms. The prepared dataset includes the `image` alias for compatibility.
