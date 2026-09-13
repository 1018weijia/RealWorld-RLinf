# 零件装配：新参数离线模型接续在线训练

## 已验证的模型

2026-09-13 已检查最终 checkpoint，并使用实际在线启动配置在 CPU 恢复 actor/critic、
target、优化器和离线 buffer。没有启动真机或训练服务。

| 项目 | 实际值 |
|---|---|
| 任务 / prompt | `assemble_parts` / `assemble parts` |
| 离线训练目录 | `results/cobot_offline_assemble_parts/pretrain_noise01_residual03_restart_20260913_203655` |
| 最终 checkpoint | 上述目录的 `checkpoints/offline_step_40000` |
| 离线更新数 | 40000 |
| buffer / partial | 6728 条 transition / false |
| actor noise / residual scale | 0.1 / 0.3 |
| 双臂动作维度 | 14，左臂六关节+夹爪、右臂六关节+夹爪 |
| 参考 / 实际执行长度 | 50 / 30，30 Hz，同步 WebSocket |

不要选择 `pretrain_franka_params`，那是保留测试用的旧 `0.2/0.2` 版本。
完成 40000 步且权重有限不等于已证明真机收敛，在线启动仍须操作员验证。

## 1. 在 origin 启动服务

服务器命令在 origin 的 **RLinf 项目根目录**执行，不是在本地 rlt-openpi。
建议另开 zellij tab 用于在线服务，不占用仍在训练的离线 tab。
本站优先使用 GPU 6、7，资源不足时再考虑 0、1、2。以下使用 GPU 6，
启动检查时约剩余 170 GiB，但与其他任务共享；每次启动前重新检查。
8000 在检查时被 XRobot 服务占用，以下使用空闲的 8010，不停止其他服务。

```bash
nvidia-smi
ss -ltnp 'sport = :8010'
```

8010 若已有监听，先确认归属，不要直接终止进程；另选空闲端口并同步修改本地端口。

以下整个代码块在同一个服务器终端运行。路径拆成数条短命令，避免复制时在引号内插入换行。

```bash
export CUDA_VISIBLE_DEVICES=6
export RLT_SERVER_PORT=8010
export RLT_COBOT_ACTOR_NOISE_SIGMA=0.1
export RLT_COBOT_RESIDUAL_SCALE=0.3
export RLT_COBOT_OFFLINE_SAMPLE_RATIO=0.1
export RLT_COBOT_DEMO_BATCH_RATIO=0.45
export RLT_COBOT_UTD_RATIO=4
export RLT_COBOT_WARMUP_STEPS=200
export RLT_COBOT_EXPO_BASE_CANDIDATES=4
export RLT_COBOT_EXPO_EDITED_CANDIDATES=4
export STAGE2_RESUME_DIR="$PWD/results/cobot_offline_assemble_parts"
export STAGE2_RESUME_DIR="$STAGE2_RESUME_DIR/pretrain_noise01_residual03_restart_20260913_203655"
export STAGE2_RESUME_DIR="$STAGE2_RESUME_DIR/checkpoints/offline_step_40000"
export RLT_RUN_DIR="$PWD/results/cobot_stage2_assemble_parts"
export RLT_RUN_DIR="$RLT_RUN_DIR/online_noise01_residual03_$(date +%Y%m%d_%H%M%S)"
test -f "$STAGE2_RESUME_DIR/offline_state.pt" &&
test -f "$STAGE2_RESUME_DIR/stage2_state.pt" &&
test -f "$STAGE2_RESUME_DIR/offline_buffer.pt" &&
bash examples/embodiment/start_cobot_stage2.sh assemble_parts train
```

等待 Stage1 加载、Stage2 恢复和 WebSocket 服务监听。在线服务需要加载冻结 Stage1，
其显存与启动耗时高于只用缓存特征的离线训练。
当前 preflight 的历史参考值仍可能提示 residual_scale、warmup、UTD 偏离；
应核对当前 checkpoint 和实际配置，而不是为了消除警告改回旧参数。
WandB key 由项目私有文件自动加载，无需 `wandb login`，不修改共享用户的全局登录。
`RLT_WANDB_ENTITY` 可选，不设置时使用该账号默认 entity。

这次恢复无需重新传输或转换 LeRobot 数据：checkpoint 已携带离线 buffer。
在线继续训练同一 actor/critic 和优化器，默认约 10% batch 来自离线训练分区，
其余沿用在线/HIL 采样；在线不继续 Cal-QL 保守项和离线 BC。
因为已恢复离线更新，跳过纯 VLA warmup，但训练仍需要采集新的在线 transition。

## 2. 准备本地 ROS2 runtime

下方命令在 **本地 rlt-openpi 项目根目录**执行。
确认两个 follower、两个 leader 和三路相机都可用，且没有另一个客户端向双臂发送动作。
既有 runtime 已处于 policy 模式且工作正常时不必重启。

仅在需要启动或重启 runtime 时，在独立本地终端执行：

```bash
export EVORL_ROOT="${EVORL_ROOT:-$PWD/../Evo-RL}"
test -f "$EVORL_ROOT/scripts/cobot_magic_restart_runtime.sh" &&
bash "$EVORL_ROOT/scripts/cobot_magic_restart_runtime.sh" --mode policy
```

注意：这是实际的硬件 runtime 重启，会停止已有相机/机械臂/CAN runtime，可能需要 sudo。
必须先停下其他机器人操作、确保现场安全；不要在另一客户端运动期间执行。
本地脚本使用既有 `evo-rl-ros2-jazzy` Conda 环境，不需要安装服务器的 RLinf。

## 3. 本地握手及无动作检查

下面在另一个本地终端、rlt-openpi 根目录执行。host 从 SSH config 的 `origin` 解析，
端口明确指定为 8010，以免连到 8000 上的其他机器人。

```bash
unset RLT_SERVER_HOST
export RLT_SERVER_PORT=8010
export RLT_REQUIRED_OFFLINE_UPDATES=40000
export EXECUTE_ACTION_STEPS=30
export RLT_COBOT_CONTROL_HZ=30
export NUM_EPISODES=200
export RLT_DATASET_ROOT="$PWD/data/rlinf_assemble_parts_$(date +%Y%m%d_%H%M%S)"
bash exp/rlinf_client_cobot.sh assemble_parts check &&
bash exp/rlinf_client_cobot.sh assemble_parts dry-run
```

`check` 只握手读取 status；应核对任务是 `assemble parts`，且：

```text
offline_total_updates=40000
offline_buffer_size=6728
offline_partial_data=false
warmup_done=true
```

首次接在线时 `total_updates=0` 是正常的，在线和离线更新分别统计。
`dry-run` 读取真实观测、请求一次推理后 discard，不复位、不发送动作、不写 replay 或数据集。
结果应为 `30x14` 的 robot-space action。它不等于完整运动安全验证。
如果握手、相机、配置合同或推理报错，不要继续执行 train。

## 4. 开始真机在线训练

操作员确认安全并准备好任务物品后，**在刚才通过检查的同一个本地终端**执行：

```bash
bash exp/rlinf_client_cobot.sh assemble_parts train
```

此命令会控制双臂，并按提示执行复位/场景准备；不需要再次启动离线训练脚本。
`RLT_REQUIRED_OFFLINE_UPDATES=40000` 会在连接机器人前检查离线更新门槛。
每次推理 50 步，Stage2 下发并学习前 30 步；完整执行、commit 确认后才请求下一 chunk。

客户端终端保持焦点：`s`/空格成功、`f` 失败、`i` 在 chunk 边界进入双臂 HIL。
`b` 进入倒车暂停，`r` 物理回退，`q` 只回退 credit，`b` 恢复。
物理回退与 credit 回退不能在同一次暂停中混用。遇到硬件危险，使用现场急停流程。

本地数据写入 `RLT_DATASET_ROOT`，包括三路视频、state/action 和动作来源。
只记录实际执行帧，不记录 chunk 间推理等待、暂停或复位；服务器另行保存训练 replay。
相邻 replay transition 复用相同边界观测，不代表录制视频的相邻帧必须逐像素相同。

## 5. 保存与后续恢复

2026-09-13 已在 `cobot-calql-v2` 会话原装配 tab 启动在线服务，tab 名为
`assemble-server-gpu6`，端口 8010。服务器执行 `zellij attach cobot-calql-v2` 查看。
其输出目录是 `results/cobot_stage2_assemble_parts/online_noise01_residual03_20260913_213156`，
包含 `server.log`。再次使用前检查进程和日志，记录不表示服务永久在线。

离线命令完成后，zellij 可能保留已退出的命令窗口；此时按 Enter 是重跑原命令，
不是打开 shell。`Run directory already exists` 是输出目录防覆盖保护，并非训练卡住。
切换用途时应在该 tab 中打开普通 shell 再运行在线命令，不能反复按 Enter 重跑离线任务。

服务端默认每 10 个 episode 保存一次到：

```text
<RLT_RUN_DIR>/checkpoints/episode_<episode数>_step_<在线更新数>/
```

以 `Saved RLT Stage 2 state` 的实际路径为准。离线重训使用的 `SAVE_EVERY=1000`
不控制在线保存间隔；需要更频繁在线保存时，在服务器启动命令末尾追加
`server.save_interval_episodes=1`。

暂停查看 zellij 时按 `Ctrl+O` 再按 `D`，不要退出整个会话。
结束训练前先正常结束当前 episode，确认 checkpoint 已保存，再停止客户端/服务。
异常断网后不要自动重发不确定的 transition；未完成 episode 应检查处理后再连接。

再次在线训练，应把 `STAGE2_RESUME_DIR` 改为实际保存的 **在线 checkpoint**，
保持 `0.1/0.3`，并设置新的 `RLT_RUN_DIR`；不要再次恢复离线 40000 checkpoint，
否则会丢失已完成的在线学习进度。本地复用同一数据集目录时，已有 episode
会计入 `NUM_EPISODES`，该值是该目录的累计目标。

旧 `0.2/0.2` checkpoint、旧数据和其他任务离线结果均保留，不能混用参数。
正式评测另用冻结的 eval 服务，每任务 30 局，并设置验证过的 Stage2 更新门槛。
