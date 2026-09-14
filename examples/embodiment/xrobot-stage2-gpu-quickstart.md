# XRobot Stage 2 —— GPU 侧速查

面向在 GPU 机器上启动和看守 Stage 2 server 的人。机器人侧看
[xrobot-stage2-robot-quickstart.md](xrobot-stage2-robot-quickstart.md)。

Stage 2 不使用 Ray，不需要 `ray start`。server 既是推理进程也是训练进程，
没有权重同步这一步。

## 1. 环境

```bash
cd /data/gxy/realworldRL/RLinf
source .venv/bin/activate            # 必须，否则会抓到 conda 的 python 并报 ModuleNotFoundError: hydra

export XROBOT_RLT_STAGE1_CHECKPOINT="/data/gxy/realworldRL/RLinf/logs/20260906-06:01:46-xrobot_rlt_stage1_sft_openpi_pi05_put_ring_on_the_rod_franka_legacy_action_expert_base/xrobot_put_ring_franka_legacy_actionexpert_base_fp32master_bf16compute_30k/checkpoints/global_step_30000"
export XROBOT_NORM_STATS="/data/gxy/realworldRL/checkpoints/assets/xrobot/put_ring_on_the_rod/norm_stats.json"
```

## 2. 先跑 preflight（约 7 秒，不加载模型、不开端口）

```bash
bash examples/embodiment/run_rlt_stage2_server.sh xrobot_ee_rlt_stage2_ws_server \
  server.preflight_only=True
```

四行必须全对，尤其 norm stats 指向的是**本任务**：

```text
Preflight: Stage 1 RLT prefix_seq_len=968 matches .../global_step_30000/actor/model_state_dict/full_weights.pt
Preflight: norm stats loaded from .../assets/xrobot/put_ring_on_the_rod/norm_stats.json
Preflight: task prompt = 'put ring on the rod'
Preflight: camera layout = ['image', 'wrist_image', 'side_image']
```

## 3. 启动 server（常驻）

```bash
bash examples/embodiment/run_rlt_stage2_server.sh xrobot_ee_rlt_stage2_ws_server \
  server.host=0.0.0.0 server.port=8000
```

等到这行出现再让机器人侧连：

```text
RLT Stage 2 server ready on 0.0.0.0:8000 (warmup 250 rows, utd 5)
```

`server.task_prompt` 必须和机器人侧 bridge 的 `--task` 完全一致，默认两边都是
`put ring on the rod`。

## 4. 换 checkpoint 时三项要一起改

```bash
bash examples/embodiment/run_rlt_stage2_server.sh xrobot_ee_rlt_stage2_ws_server \
  rlt_feature_model.model_path=<.../checkpoints/global_step_XXXXX> \
  rlt_feature_model.openpi_data.norm_stats_path=<.../norm_stats.json> \
  rlt_feature_model.openpi_data.default_prompt="<stage1 prompt>" \
  server.task_prompt="<stage1 prompt>" \
  server.host=0.0.0.0 server.port=8000
```

`model_path` 指向 `global_step_*` 本身，不要指到 `actor/model_state_dict`。
列出可用的 checkpoint：

```bash
find logs -maxdepth 5 -type d -name 'global_step_*' -path '*xrobot*' | sort -V
```

## 5. XRobot 与 Cobot 的关键差异

| 项目 | Cobot | XRobot |
|---|---:|---:|
| 动作语义 | 关节角 | 末端位姿 xyz+rpy+gripper |
| Stage 2 chunk | 16 步 | **50 步** |
| reference horizon | 20 | **50**（与 Stage 1 训练一致） |
| action_dim / proprio_dim | 14 | 14 |

`warmup_steps: 250` 数的是 **replay 行数，即 chunk 数**，不是环境步数。XRobot 一个
chunk 是 50 步约 1.7 秒动作，250 个 chunk 大约是十几到二十分钟的采集。这期间照常
写 replay 但不产生梯度，`mode` 会一直是 `warmup`。嫌久可加
`server.warmup_steps=80`。

## 6. 运行中看什么

```text
每条请求一行：类型、耗时、replay 行数、pending 数、chunk_id
env/episode_reward, env/episode_chunks, env/episode_train_chunks
env/episode_interventions, env/episode_success
rlt/updates_this_episode
train/critic_loss, train/actor_loss
```

- `mode` 从 `warmup` 变 `actor` 说明 replay 过了 250 行，residual 开始生效。
- 训练只在 `episode_end` 发生，步数 = 本 episode 里 `mode=="actor"` 的 chunk 数 × 5。
  训练期间 server 不响应新请求，机械臂本来就是停走式的，属正常。
- `pending` 长期不为 0 说明有 chunk 既没提交也没丢弃，是客户端循环的 bug；
  正常情况下它只在两次请求之间短暂为 1。

## 7. Checkpoint 与恢复

每 10 个 episode 落一次盘：

```bash
ls ../results/xrobot_ee_rlt_stage2_ws/checkpoints/
# episode_10_step_<N>/ 内应有 stage2_state.pt、replay_buffer/、demo_buffer/
```

恢复：

```bash
bash examples/embodiment/run_rlt_stage2_server.sh xrobot_ee_rlt_stage2_ws_server \
  server.host=0.0.0.0 server.port=8000 \
  runner.resume_dir=../results/xrobot_ee_rlt_stage2_ws/checkpoints/episode_10_step_<N>
```

日志里应出现 `Resuming Stage 2 state from ...` 且 replay 行数接上了。

## 8. 首次运行必须确认的一件事

residual 输出层是零初始化的，所以 Stage 2 刚启动时的动作应与 Stage 1 reference
**逐元素相同**。第一次真机运行务必核对这一点；不一致说明链路上有地方变换错了，
先停下来查，不要继续跑。
