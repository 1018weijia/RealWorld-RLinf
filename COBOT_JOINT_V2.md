# Cobot joint motion v2

本分支基于 `93b3eff1`，服务端运行目录是
`/data/gxy/realworldRL/RLinf-cobot-joint-v2`。使用 GPU 6、7，50 步参考/30 步执行/30 Hz。
旧 Stage2 权重和 buffer 已退役；保留 Stage1 和原始 LeRobot 数据。

配置：`examples/embodiment/config/cobot_joint_motion_v2.yaml`。
每臂关节残差预算为 `[0.04,0.04,0.04,0.03,0.03,0.02]` rad；6 个残差时间节点，
统一执行速度/加速度约束，并在 chunk 末端保持。夹爪保留 Stage1 目标。
noise 默认 0.025，离线和在线相同；客户端默认不覆盖。

## 重新转换与离线训练

在 zellij 的 shell 中运行，先检查 GPU 占用，不停止其他人的进程：

```bash
cd /data/gxy/realworldRL/RLinf-cobot-joint-v2
unset STAGE2_RESUME_DIR RLT_RUN_DIR OFFLINE_BUFFER RLT_COBOT_ACTOR_NOISE_SIGMA
unset RLT_WANDB_MODE ALLOW_PARTIAL_DATASET
export CUDA_VISIBLE_DEVICES=7 MAX_EPISODES=0 NUM_TRAIN_STEPS=40000
bash examples/embodiment/run_cobot_joint_v2.sh assemble_parts prepare
```

原始数据已在服务器，路径由 `.private-cobot-stage2/paths.env` 提供，不需要重新上传。
先完整转换，再自动训练；默认 buffer 位于 `results/cobot_joint_v2_assemble_parts/offline_buffer.pt`，
checkpoint 位于该任务目录的 `offline_train/checkpoints/offline_step_40000`。
WandB 仅使用项目 key，不执行共享用户的 `wandb login`。

其他任务名为 `cube_into_drawer`、`cook_vegetable`、`pack_and_pour_fruit`。
每张卡串行准备任务；不要重复向同一输出目录启动两个进程。
转换可断点继续；离线训练中断后显式设置 `STAGE2_RESUME_DIR` 为已有完整 checkpoint。

## 启动在线 server

新离线训练完成后，在 zellij 中运行：

```bash
cd /data/gxy/realworldRL/RLinf-cobot-joint-v2
unset STAGE2_RESUME_DIR RLT_RUN_DIR OFFLINE_BUFFER
CUDA_VISIBLE_DEVICES=6 RLT_SERVER_PORT=8010 bash examples/embodiment/run_cobot_joint_v2.sh assemble_parts train
```

默认恢复新离线 40000 checkpoint，不存在则退出。
继续已有在线实验时，把 `STAGE2_RESUME_DIR` 改成实际在线 checkpoint；这会恢复其 replay。
`reset-server` 不清空 replay，重启也不等于丢弃数据。

## 本地 client

在本地 rlt-openpi 中启动自带 policy runtime，再运行：

```bash
cd /home/guoxiaoyu/rlt-openpi
unset RLT_SERVER_HOST RLT_DATASET_ROOT
export RLT_SERVER_PORT=8010 RLT_REQUIRED_OFFLINE_UPDATES=40000
export EXECUTE_ACTION_STEPS=30 NUM_EPISODES=200 DISPLAY_DATA=true
bash exp/rlinf_client_cobot.sh assemble_parts check &&
bash exp/rlinf_client_cobot.sh assemble_parts dry-run
```

检查 schema 为 `cobot-joint14-motion-v2`，offline_updates=40000、partial=false。
确认现场准备好后执行 `bash exp/rlinf_client_cobot.sh assemble_parts train`。
不依赖 Evo-RL runtime，录制保存在本地，只记录执行帧。

设计、实验限制及完整人工操作见 rlt-openpi 的
`docs/cobot_joint_motion_v2.md`、`docs/cobot_calql_offline_to_online.md`、
`docs/cobot_assemble_parts_online.md`。CPU/GPU短测通过不等于已证实真机任务成功率。
