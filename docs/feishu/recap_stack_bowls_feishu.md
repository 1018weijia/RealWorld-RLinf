# 叠碗 RECAP 数据整理与单任务训练步骤

更新时间：2026-09-03

飞书目标文档：

```text
https://mcnklerpx59m.feishu.cn/wiki/Gs60w9DDMibnCIkCUZtcVOIrnhc
```

## 结论

可以把这批针对叠碗任务的 SFT 权重 rollout 数据单独整理成一个 RECAP rollout 数据集，并用它单独训练一个叠碗 task-specific CFG 权重。

已经整理好的数据集：

```text
/home/pnp/桌面/franka_oral_data/stack_bowls_recap_rollout_rc
```

已生成 RECAP return sidecar：

```text
/home/pnp/桌面/franka_oral_data/stack_bowls_recap_rollout_rc/meta/returns_fail300.parquet
```

## 远端路径

跳板机：

```bash
ssh amax@100.95.122.116
```

数据机：

```bash
ssh pnp@192.168.10.110
```

原始标签 CSV：

```text
/home/pnp/桌面/franka_deploy_0128_ee/franka_deploy/logs/recap_episode_labels.csv
```

原始 rollout session：

```text
/home/pnp/桌面/franka_deploy_0128_ee/franka_deploy/logs/session_async_*
```

## 标签统计

任务：

```text
Stack the three bowls in size order: the purple bowl first, then the beige bowl.
```

task id：

```text
stack_bowls_in_size_order_rc
```

统计：

| 项 | 数量 |
| --- | ---: |
| episode | 34 |
| success | 22 |
| fail | 12 |

## 已整理数据格式

LeRobot / RECAP 数据集：

```text
/home/pnp/桌面/franka_oral_data/stack_bowls_recap_rollout_rc
```

结构：

```text
data/chunk-000/episode_000000.parquet ... episode_000033.parquet
videos/chunk-000/image/episode_*.mp4
videos/chunk-000/global_image/episode_*.mp4
videos/chunk-000/right_image/episode_*.mp4
videos/chunk-000/wrist_image/episode_*.mp4
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/episodes_stats.jsonl
meta/stats.json
meta/recap_episode_labels.csv
meta/returns_fail300.parquet
```

验证结果：

| 项 | 数量 |
| --- | ---: |
| episode | 34 |
| frame | 196,579 |
| success frame | 81,863 |
| fail frame | 114,716 |

parquet 字段：

```text
global_image, right_image, wrist_image, image, state, actions, task, task_index,
episode_index, frame_index, timestamp, index, is_success, source_session
```

相机映射：

| 原始视频 | LeRobot 字段 |
| --- | --- |
| `cam_high.mp4` | `global_image`，同时别名为 `image` |
| `cam_side.mp4` | `right_image` |
| `cam_wrist.mp4` | `wrist_image` |

当前数据集是轻量版：视频使用软链接指向原始 logs，避免写满机器人主机磁盘。如果要搬到训练机，用 `rsync -L` 展开软链接，或重新生成 copy/reencode 版本。

重建命令：

```bash
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

## Return 设置

tag：

```text
fail300
```

规则：

```text
普通帧 reward = -1
成功 episode 末帧 reward = 0
失败 episode 末帧 reward = -300
gamma = 1.0
```

## RECAP 训练四步

### Step 1: Compute Returns

已完成。重新计算时：

```bash
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
  recap_compute_returns \
  data.train_data_paths='[{dataset_path: "/workspace/dataset/stack_bowls_recap_rollout_rc", type: "rollout"}]' \
  data.gamma=1.0 \
  data.failure_reward=-300.0 \
  data.tag=fail300
```

### Step 2: Value Model SFT

```bash
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

记录输出 value checkpoint：

```text
logs/value_sft/<config>-<timestamp>/value_sft/checkpoints/global_step_<N>/actor/model_state_dict
```

### Step 3: Compute Advantages

```bash
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

输出：

```text
meta/advantages_fail300_N10_stack_bowls_q30.parquet
```

### Step 4: CFG Policy Training

```bash
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

建议优先试 `pi05_franka_state`，因为当前数据是 7D EE pose + gripper 的 `state/actions`。如果 SFT checkpoint 实际使用 `pi05_franka_pnp` 或其他私有配置，需要保持 config name 与原始 SFT 权重一致。

## 注意事项

- 机器人机默认 Python 环境没有 `lerobot` / `rlinf`，训练应放到 RLinf 容器或 GPU 训练机。
- 轻量数据集依赖原始 session 视频软链接；跨机器拷贝要用 `rsync -L`。
- 不要把机器人密码、模型权重、相机 serial、私有数据提交到 git。
