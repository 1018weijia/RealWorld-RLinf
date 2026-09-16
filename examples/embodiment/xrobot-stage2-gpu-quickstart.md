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

## 9. USB 插拔：离线 Cal-QL

套环走上面的 `xrobot_ee_rlt_stage2_ws_server`。插 USB 用独立 overlay 和启动器，
不要改 residual 或 chunk（YAML 已是 `0.4` / `50`）。

在 `.private-xrobot-stage2/paths.env` 写（目录已 gitignore）：

```bash
XROBOT_USB_STAGE1_CHECKPOINT=/data/gxy/realworldRL/RLinf/logs/20260911-012744-xrobot_rlt_stage1_sft_openpi_pi05_usb_plug_franka_legacy_action_expert_base-gpu6/xrobot_usb_plug_franka_legacy_actionexpert_base_fp32master_bf16compute_30k/checkpoints/global_step_20000
XROBOT_USB_NORM_STATS=/data/gxy/realworldRL/checkpoints/assets/xrobot/usb_plug/norm_stats.json
XROBOT_USB_DATASET=/data/gxy/realworldRL/datasets/XRobot_USB_v30/XRobot_USB
XROBOT_OFFLINE_BUFFER_ROOT=/data/gxy/realworldRL/offline_rl_buffers
XROBOT_USB_OFFLINE_BUFFER=/data/gxy/realworldRL/offline_rl_buffers/xrobot_usb_plug/offline_buffer.pt
```

`model_path` 指向最新 USB **RLT** `global_step_*` 目录本身，且 `full_weights.pt` 必须含 `rlt_module.*`。不要用 `20260829` 那份无 RLT 的 SFT。prompt 必须是 `Bimanual usb pick and insert`。

What this does: 1. 核对合同 2. 抽查数据 3. 冻 Stage 1 写成 buffer 4. Cal-QL 4 万步 5. resume 到在线（不再采 250 行 warmup）。

```bash
bash examples/embodiment/start_xrobot_stage2.sh usb_plug preflight

MAX_EPISODES=2 bash examples/embodiment/start_xrobot_stage2.sh usb_plug audit
CUDA_VISIBLE_DEVICES=? MAX_EPISODES=2 \
  bash examples/embodiment/start_xrobot_stage2.sh usb_plug convert

CUDA_VISIBLE_DEVICES=4 bash examples/embodiment/start_xrobot_stage2.sh usb_plug convert

CUDA_VISIBLE_DEVICES=4 bash examples/embodiment/start_xrobot_stage2.sh usb_plug offline

STAGE2_RESUME_DIR=/data/gxy/realworldRL/offline_rl_buffers/xrobot_usb_plug/pretrain_20260915_101150/checkpoints/offline_step_40000 \
  bash examples/embodiment/start_xrobot_stage2.sh usb_plug train
```

转换输出默认 `/data/gxy/realworldRL/offline_rl_buffers/xrobot_usb_plug/offline_buffer.pt`。Cal-QL 日志在同目录 `pretrain_*`。WandB project 是 `xrobot-usb-offline`（需要 `.private-xrobot-stage2/wandb_api_key`）。数据集没有 `episode_success` 时全部当成功。round-trip 失败或 `demo_reachable_fraction` 过低先停，不要只看 BC loss。不要用 GPU 6（Stage 1 还在跑）。

## 10. USB 在线：常驻 server → 等客户端

套环仍用第 3 节（端口 **8000**、`put ring on the rod`）。USB 用 `start_xrobot_stage2.sh`、端口 **8016**、prompt `Bimanual usb pick and insert`。不要用 Cobot 启动器（会把 chunk 打成 30、residual 打成 0.3）。

Cal-QL 已经写进 `offline_total_updates`，resume 后**不会**再采 250 行 warmup。日志里的 `warmup 250 rows` 只是配置打印。第一回合就是 residual actor。第 8 节「必须与 Stage 1 逐元素相同」这次**对不上是正常的**。

在线训练只在机器人发来 `episode_end` 之后才走。server 起来后可以空等几小时，没有客户端就不会迭代。

### 10.1 云机资产与合同

跨机拷贝（rsync / ModelScope）之后，`offline_step_40000/offline_buffer.pt` 里仍可能写着源机的 `model_path`、`norm_stats_path`、`weights_mtime_ns`。旧代码会报：

```text
ValueError: Offline buffer differs from current task/Stage1/norm stats/Stage2 configuration
```

新代码只核 task、actor、gamma、norm sha256、weights size，忽略路径和 mtime。若远端还是旧代码，先把合同改成本机路径（size 和 sha256 对不上就停，不要硬改）：

```bash
cd /mnt/data/lfwj/realworldRL/RLinf
source .venv/bin/activate
python3 - <<'PY'
from pathlib import Path
import hashlib, shutil, torch

resume = Path("/mnt/data/lfwj/realworldRL/offline_rl_buffers/xrobot_usb_plug/offline_step_40000")
model_path = Path("/mnt/data/lfwj/realworldRL/checkpoints/usb_stage1/global_step_20000")
stats = Path("/mnt/data/lfwj/realworldRL/assets/xrobot/usb_plug/norm_stats.json")
weights = model_path / "actor/model_state_dict/full_weights.pt"
buf = resume / "offline_buffer.pt"
digest = hashlib.sha256(stats.read_bytes()).hexdigest()
payload = torch.load(buf, map_location="cpu", weights_only=False)
c = payload["contract"]
assert c["weights_size"] == weights.stat().st_size
assert c["norm_sha256"] == digest
c["feature_model"]["model_path"] = str(model_path)
c["feature_model"]["openpi_data"]["norm_stats_path"] = str(stats)
c["weights_mtime_ns"] = weights.stat().st_mtime_ns
bak = buf.with_suffix(".pt.bak")
if not bak.exists():
    shutil.copy2(buf, bak)
tmp = buf.with_suffix(".pt.tmp")
torch.save(payload, tmp)
tmp.replace(buf)
print("rewrote", buf)
PY
```

What this does: 1. 核对 Stage 1 文件大小和 norm hash 2. 把三处本机字段改成云机路径 3. 留 `.bak` 再覆盖。

`paths.env` 指向同一套文件：

```bash
XROBOT_USB_STAGE1_CHECKPOINT=/mnt/data/lfwj/realworldRL/checkpoints/usb_stage1/global_step_20000
XROBOT_USB_NORM_STATS=/mnt/data/lfwj/realworldRL/assets/xrobot/usb_plug/norm_stats.json
XROBOT_OFFLINE_BUFFER_ROOT=/mnt/data/lfwj/realworldRL/offline_rl_buffers
XROBOT_USB_OFFLINE_BUFFER=/mnt/data/lfwj/realworldRL/offline_rl_buffers/xrobot_usb_plug/offline_step_40000/offline_buffer.pt
```

### 10.2 常驻 train

放进 tmux，先 `nvidia-smi` 选空闲卡：

```bash
cd /mnt/data/lfwj/realworldRL/RLinf
source .venv/bin/activate

STAGE2_RESUME_DIR=/mnt/data/lfwj/realworldRL/offline_rl_buffers/xrobot_usb_plug/offline_step_40000 \
  bash examples/embodiment/start_xrobot_stage2.sh usb_plug train
```

本机 Cal-QL 目录若还在 `pretrain_*/checkpoints/offline_step_40000`，把 `STAGE2_RESUME_DIR` 指过去即可。等到：

```text
Resuming Stage 2 state from .../offline_step_40000
RLT Stage 2 server ready on 0.0.0.0:8016 (warmup 250 rows, utd 5)
RLinf websocket policy server listening on 0.0.0.0:8016
```

空等时不要杀进程、不要再启一份（端口会撞）、不要让 SSH 把进程带走。

云机 SSH 口和 WebSocket 口不是同一个。SSH 若是 `34133`，机器人往往打不到 `8016`。在机器人或跳板上先建隧道：

```bash
ssh -p 34133 -N -L 8016:127.0.0.1:8016 root@<GPU_HOST>
```

然后 bridge 连 `ws://127.0.0.1:8016`。客户端逐步操作见
[xrobot-stage2-robot-quickstart.md](xrobot-stage2-robot-quickstart.md) 的「USB 客户端」。
