# Franka RECAP Rollout 保存与 is_success 标注

本文档记录当前 amax + pnp C/S 部署下，如何在测 SFT baseline SR 的同时保存 rollout，并给每个 episode 打 `is_success` 标签，供后续 RECAP 使用。

官方 RECAP 文档说明：RECAP 使用 LeRobot 格式数据集，数据分为 SFT 成功轨迹和 rollout 轨迹；rollout 数据包含成功和失败。四步流程是 `compute_returns -> value model SFT -> compute_advantages -> CFG training`。官方文档链接：

```text
https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/recap.html
```

## 1. 当前 pnp 端实际保存位置

pnp 机器：

```text
pnp@192.168.10.110
```

部署目录：

```text
/home/pnp/桌面/franka_deploy_0128_ee/franka_deploy
```

当前异步 client 启动脚本：

```bash
cd /home/pnp/桌面/franka_deploy_0128_ee/franka_deploy
bash start_control_async.sh
bash start_inference_async.sh
```

`start_inference_async.sh` 会运行：

```bash
python3 franka_inference_node_async.py ...
```

该节点每启动一次，会创建一个 session 目录：

```text
/home/pnp/桌面/franka_deploy_0128_ee/franka_deploy/logs/session_async_YYYYMMDD_HHMMSS/
```

实测目录形态如下：

```text
logs/session_async_20260830_162433/
  actions.json
  frame_states.json
  images/
    chunk_000_cam_high.jpg
    chunk_000_cam_wrist.jpg
    chunk_000_cam_side.jpg
    ...
  videos/
    cam_high.mp4
    cam_wrist.mp4
    cam_side.mp4
```

其中：

| 文件 | 内容 |
| --- | --- |
| `actions.json` | task、session、mode、每次模型输出的 action chunk、发送给模型的 state、推理耗时 |
| `frame_states.json` | 视频帧对应的 EE state |
| `images/` | 每个 inference chunk 对应的三视角观测 JPEG |
| `videos/` | 三视角完整视频 |

注意：这套 pnp 原始 session 不是标准 LeRobot 数据集；它是后续转换 LeRobot 的原始 rollout 包。RECAP 的 `compute_returns.py` 最终需要 LeRobot parquet 中有 `episode_index`、`frame_index`、`task` 或 `task_index`、以及 rollout 必需的 `is_success`。

## 2. 2026-08-30 bowls rollout 当前统计

用户已手动删除部分失败和空 session 后，重新统计 pnp：

```text
root: /home/pnp/桌面/franka_deploy_0128_ee/franka_deploy/logs
date prefix: session_async_20260830_*
sessions_total_today: 37
valid_rollout_sessions: 37
empty_or_no_action_sessions: 0
labeled_valid: 0
success_labeled: 0
failure_labeled: 0
unlabeled_valid: 37
total_chunks_valid: 20390
total_actions_valid: 652480
total_images_valid: 61173
total_size_gb_today: 7.828
```

当前 37 个有效 session 的 `actions.json` 已统一为：

```text
task: Stack the three bowls in size order: the purple bowl first, then the beige bowl.
task_id: stack_bowls_in_size_order_rc
```

统一前有 27 个 session 仍记录旧 task id `stack_bowls_in_size_order_rc`。已在 pnp 上为这些 `actions.json` 自动备份：

```text
actions.json.bak_before_prompt_unify
```

截至该统计点，所有有效 session 尚未打 `is_success`，因此当前还不能计算 SR。需要先选定正式纳入统计的 30 个 episode，并逐个运行第 4 节的标注命令。

## 3. 测 SFT baseline 时必须保存的数据

每个 task 跑 30 次。最清楚的做法是每次 trial 都重启一次 `start_inference_async.sh`，这样一个 session 目录就是一个 episode。

每次 trial 结束后记录：

| 字段 | 示例 |
| --- | --- |
| `task_id` | `stack_bowls_in_size_order_rc` |
| `prompt` | `Stack the three bowls in size order: the purple bowl first, then the beige bowl.` |
| `checkpoint_profile` | `front2` 或 `d2` |
| `trial` | `1` 到 `30` |
| `is_success` | 成功为 `1`，失败为 `0` |
| `notes` | 失败原因、人工暂停、安全停止、物体初始摆放异常等 |
| `session_dir` | `logs/session_async_YYYYMMDD_HHMMSS` |

RECAP 后续最关键的是 `is_success`。如果只记录 SR 表格，不保存 session 数据，后续无法从真实 rollout 中计算 return / advantage。

## 4. 标注单个 episode

先把本仓库脚本拷到 pnp：

```bash
scp scripts/franka/label_pnp_session.py pnp@192.168.10.110:/home/pnp/桌面/franka_deploy_0128_ee/franka_deploy/
```

在 pnp 上查看最新 session：

```bash
cd /home/pnp/桌面/franka_deploy_0128_ee/franka_deploy
ls -dt logs/session_async_* | head
```

给成功 episode 打标签：

```bash
python3 label_pnp_session.py logs/session_async_20260830_162433 \
  --success 1 \
  --trial 1 \
  --task-id stack_bowls_in_size_order_rc \
  --prompt "Stack the three bowls in size order: the purple bowl first, then the beige bowl." \
  --checkpoint-profile front2 \
  --notes "success"
```

给失败 episode 打标签：

```bash
python3 label_pnp_session.py logs/session_async_20260830_162433 \
  --success 0 \
  --trial 2 \
  --task-id stack_bowls_in_size_order_rc \
  --prompt "Stack the three bowls in size order: the purple bowl first, then the beige bowl." \
  --checkpoint-profile front2 \
  --notes "failed: bowl slipped before stacking"
```

脚本会写三个位置：

```text
logs/session_async_*/actions.json
logs/session_async_*/episode_label.json
logs/recap_episode_labels.csv
```

检查：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("logs/session_async_20260830_162433/actions.json")
d = json.loads(p.read_text())
print(d["is_success"])
print(d["recap_label"])
PY

tail -n 5 logs/recap_episode_labels.csv
```

## 5. 五个任务的标注命令模板

### task01 bowls

```bash
python3 label_pnp_session.py logs/session_async_YYYYMMDD_HHMMSS \
  --success 1 \
  --trial 1 \
  --task-id stack_bowls_in_size_order_rc \
  --prompt "Stack the three bowls in size order: the purple bowl first, then the beige bowl." \
  --checkpoint-profile front2 \
  --notes ""
```

### task02 ring

```bash
python3 label_pnp_session.py logs/session_async_YYYYMMDD_HHMMSS \
  --success 1 \
  --trial 1 \
  --task-id place_ring_on_rod_rc_0810 \
  --prompt "Place the ring on the rod." \
  --checkpoint-profile front2 \
  --notes ""
```

### task03 fruits

```bash
python3 label_pnp_session.py logs/session_async_YYYYMMDD_HHMMSS \
  --success 1 \
  --trial 1 \
  --task-id place_fruits_on_plate_rc \
  --prompt "Place all the fruits on the plate." \
  --checkpoint-profile d2 \
  --notes ""
```

### task04 charger

```bash
python3 label_pnp_session.py logs/session_async_YYYYMMDD_HHMMSS \
  --success 1 \
  --trial 1 \
  --task-id plug_charger_into_socket_rc \
  --prompt "Plug the charger into the socket." \
  --checkpoint-profile d2 \
  --notes ""
```

### task05 peg

```bash
python3 label_pnp_session.py logs/session_async_YYYYMMDD_HHMMSS \
  --success 1 \
  --trial 1 \
  --task-id insert_peg_into_hole_rc \
  --prompt "Insert the peg into the corresponding hole." \
  --checkpoint-profile d2 \
  --notes ""
```

失败时只需要把 `--success 1` 改成 `--success 0`，并在 `--notes` 写失败原因。

## 6. 后续转换为 RECAP 数据

当前 pnp session 是原始 rollout 包。进入 RECAP 前，需要把它转换成 LeRobot 数据集，至少保证 parquet/meta 中有：

```text
episode_index
frame_index
task_index
task
is_success
observation images
observation state
actions
```

RECAP 的 `compute_returns.py` 对 rollout 数据集的处理逻辑是：

```text
type=sft      -> 默认整条 episode 成功
type=rollout  -> 必须读取 is_success 区分成功/失败
```

因此转换时应把 `episode_label.json` 或 `logs/recap_episode_labels.csv` 中的 `is_success` 写入每个 episode 的最后一帧，最稳妥是写入该 episode 的每一帧。

后续 RECAP 配置应使用类似：

```yaml
data:
  train_data_paths:
    - dataset_path: /path/to/franka_sft_success_dataset
      type: sft
      weight: 1.0
      robot_type: franka
      model_type: pi05
    - dataset_path: /path/to/franka_rollout_lerobot_dataset
      type: rollout
      weight: 1.0
      robot_type: franka
      model_type: pi05
  tag: franka_fail300_v1
```

然后按官方四步：

```bash
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh recap_compute_returns
bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh recap_value_model_sft
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh recap_compute_advantages
bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh cfg_rl_openpi
```

产出新的 CFG/RECAP policy checkpoint 后，再回到 amax + pnp 部署流程，每个 task 重新跑 30 次 SR。

### 6.1 2026-09-03 叠碗 recap 数据集整理结果

叠碗任务的已标注 rollout 已单独整理成 LeRobot / RECAP rollout 数据集：

```text
/home/pnp/桌面/franka_oral_data/stack_bowls_recap_rollout_rc
```

对应原始标签 CSV：

```text
/home/pnp/桌面/franka_deploy_0128_ee/franka_deploy/logs/recap_episode_labels.csv
```

统计：

| 项 | 数量 |
| --- | ---: |
| episodes | 34 |
| successful episodes | 22 |
| failed episodes | 12 |
| frames | 196,579 |
| success frames | 81,863 |
| fail frames | 114,716 |

已生成 RECAP return sidecar：

```text
meta/returns_fail300.parquet
```

这批数据可以单独用于叠碗 task-specific RECAP 训练。完整数据路径、重建命令、`fail300`
tag、Value SFT / Compute Advantages / CFG Training 命令见：

```text
docs/realworld_recap_stack_bowls.md
```

飞书粘贴版见：

```text
docs/feishu/recap_stack_bowls_feishu.md
```

## 7. 切换到 ring 等后续任务

切换任务时必须同时改两处：

1. amax 服务端：`CKPT_PROFILE` 和 `TASK_PROMPT`
2. pnp 客户端：`config.sh` 中的 `export TASK`

原因是服务端 `TASK_PROMPT` 只是 `--default-prompt` fallback；当前 pnp client 每次请求都会发送 `prompt/task` 字段，服务端会优先使用 client payload。也就是说，如果 pnp 端仍是旧 prompt，就会覆盖 amax 的默认 prompt。

推荐在 amax 直接生成命令：

```bash
cd /data/yangky/test/pi05-Recap-Franka
bash scripts/franka/print_5task_launch_commands.sh place_ring_on_rod_rc_0810
```

ring 任务对应：

```text
task_id: place_ring_on_rod_rc_0810
prompt: Place the ring on the rod.
CKPT_PROFILE: front2
checkpoint: sft_franka_shuo_pi05/global_step_15000
norm_stats: franka_shuo_bowls_ring
```

amax：

```bash
tmux kill-session -t pi05_front2 2>/dev/null || true
tmux kill-session -t pi05_place_ring 2>/dev/null || true
tmux new-session -d -s pi05_place_ring \
  'cd /data/yangky/test/pi05-Recap-Franka && export CKPT_PROFILE=front2 TASK_PROMPT="Place the ring on the rod." && bash scripts/franka/run_shuo_pi05_service_amax.sh'
tmux capture-pane -t pi05_place_ring -p | tail -n 80
```

pnp：

```bash
cd /home/pnp/桌面/franka_deploy_0128_ee/franka_deploy
perl -0pi -e 's/export TASK="[^"]+"/export TASK="Place the ring on the rod."/' config.sh
bash start_control_async.sh
bash start_inference_async.sh
```

ring 的每个 trial 结束后标注：

```bash
python3 label_pnp_session.py logs/session_async_YYYYMMDD_HHMMSS \
  --success 1 \
  --trial 1 \
  --task-id place_ring_on_rod_rc_0810 \
  --prompt "Place the ring on the rod." \
  --checkpoint-profile front2 \
  --notes "success"
```

fruits / charger / peg 的流程相同，但 `CKPT_PROFILE=d2`，并使用对应 prompt。完整五任务启动命令见 `docs/franka_5task_launch.md`。

每个新任务建议按下面顺序执行：

1. 在 amax 运行 `bash scripts/franka/print_5task_launch_commands.sh <task_id>`，复制输出的 amax 命令启动 service。
2. 在 pnp 按输出命令修改 `config.sh` 里的 `export TASK`。
3. pnp 分别启动 `bash start_control_async.sh` 和 `bash start_inference_async.sh`。
4. 完成一个 trial 后，在 pnp 记录 `latest_session="$(ls -dt logs/session_async_* | head -1)"`。
5. 用 `label_pnp_session.py "${latest_session}" ...` 标注成功或失败。
6. 每个 task 保留 30 个正式 trial，额外 warmup/debug session 可以保留但不要纳入 SR。
7. 30 个正式 trial 标注完成后，从 `logs/recap_episode_labels.csv` 汇总 `success_count / 30`。

## 8. 当前注意事项

1. pnp 当前 `config.sh` 必须使用训练原文 prompt，不能使用 task id。当前 bowls 应为 `export TASK="Stack the three bowls in size order: the purple bowl first, then the beige bowl."`。
2. 每次 trial 最好独立启动一次 inference node，保证一个 `session_async_*` 对应一个 episode。
3. 失败 episode 不要删除；RECAP 需要失败数据来学习 value / advantage。
4. 如果发生安全急停、Reflex、相机异常、网络断连，也保留 session，并在 `--notes` 中标清楚。
5. 当前脚本只是给原始 session 打标；标准 LeRobot 转换脚本还需要根据最终训练数据格式单独补齐。
