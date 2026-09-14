# Cobot：LeRobot v3 → Cal-QL → WebSocket 在线训练

以下服务器命令均在 origin 的 RLinf 根目录执行，本地命令在 rlt-openpi 根目录执行。
入口适配当前 RLinf 原生双臂模型，不调用 Franka 的旧训练器。

## 站点配置

服务器 `.private-cobot-stage2/paths.env` 已保存经过核实的装配 Stage1 30k、
归一化统计量和 `COBOT_ASSEMBLE_DATASET`。后者指向包含 `meta/info.json` 的
`lerobot` 根目录，不是它的父目录。服务器副本与本地
`Evo-RL/data/cobot_magic_assemble_parts_v1/lerobot` 的内容哈希一致。
数据共 200 条轨迹、205019 帧，194 次成功、6 次失败，任务为 `assemble parts`。

其它任务沿用四任务启动入口；用 `COBOT_DATASET_ROOT` 指定该任务经核实的 v3 数据根目录。
任务元数据必须与启动任务一致。私有路径、API key、数据和 checkpoint 不进入 Git。
WandB 仍仅从本项目私有 key 文件设置进程环境，不执行共享账号的 `wandb login`。

## 1. 检查全部数据

```bash
bash examples/embodiment/start_cobot_stage2.sh assemble_parts audit
```

检查每个 episode 的索引、字段、时间戳、结果标签和三路视频首尾帧。
完整转换还会检查所有使用到的 chunk 边界帧。`MAX_EPISODES=2` 只用于调试。

## 2. 转换全部离线数据

GPU 编号先用 `nvidia-smi` 确认空闲；下面的 3 是验证时使用的卡，不保证始终空闲。

```bash
CUDA_VISIBLE_DEVICES=3 \
  bash examples/embodiment/convert_lerobot_to_rl_buffer_cobot.sh assemble_parts
```

默认输出为 `$PWD/results/cobot_offline_assemble_parts/offline_buffer.pt`。
可设置 `OFFLINE_BUFFER` 更改输出。转换只运行冻结的 Stage1 模型，不连接机器人，
不训练 actor/critic。每条轨迹完成后落盘 shard；中断后重跑同一命令会继续，
源数据或模型配置改变时拒绝复用旧 shard，应改用新输出目录。

转换合同：

- 56D state 按名字提取左臂六关节＋夹爪、右臂六关节＋夹爪的位置，共 14D。
  从 28D action 提取同名关节目标，不混入末端位姿，也不把速度/力矩当位置。
- 三路图像使用与在线相同的 repacker、Stage1 编码和动作归一化接口。
  保存 50×14 的参考动作及候选；实际训练 action 是 30×14。
- 每 30 步组成一个 transition，边界观测只编码一次，复用于相邻 transition。
  尾端对齐最后一个已记录的观测；丢弃开头不足 30 步的余数，以及缺少后继观测的最后一个动作。
  不补造 hold 动作、视频帧或推理等待间隙。
- 本数据集的 `success` 和 `failure` 都视为显式 episode 终止，关闭 bootstrap。
  成功末步奖励 1，失败末步奖励 0，其余为 0；gamma=0.99 按机器人步数折扣。
  未知标签直接报错。若未来数据用 failure 表示可继续的时间截断，需要另行提供截断语义。
- 保持已有线上 actor/critic 架构、残差幅度、EXPO 和预处理。不会照搬 Franka 的
  7D/8 步或其它 critic 架构。MC return 只作为 Cal-QL 校准下界。

按当前数据长度可得到 6728 个 transition。按完整 episode 和成功/失败类别划分约 10%
验证集，防止相邻 chunk 泄漏。离线 buffer 独立保存，不受在线 replay 的窗口、PER、
HIL 或倒车修正影响。

## 3. Cal-QL 离线预训练

```bash
CUDA_VISIBLE_DEVICES=3 \
RLT_RUN_DIR="$PWD/results/cobot_offline_assemble_parts/pretrain_run1" \
NUM_TRAIN_STEPS=40000 \
  bash examples/embodiment/stage2_cobot_offline_lerobot.sh assemble_parts
```

该阶段只加载缓存特征与 Stage2 小模型，不再加载 Stage1。默认 batch=256，
actor lr=3e-5、critic lr=3e-4，critic 每步更新、actor 每两步更新。
critic 使用当前在线 TD 目标加 Cal-QL 保守项，自适应乘子目标 gap=0.05，
前 1000 步逐渐开启；proposal 为 VLA 候选、actor 动作及参考附近随机扰动。
只对 policy proposal 用 MC return 校准，其余 proposal 保持 CQL 惩罚。
actor 用成功示范的 BC（权重 1→0.1）和延迟开启的 Q 目标（权重最大 0.1）。
失败数据仍参与 critic 训练。示范 BC 投影到当前 actor 的可达残差范围；
日志提供可达比例，不能用低 BC loss 代替真实任务效果。

每 500 步记录验证集指标，每 5000 步及结束时保存完整目录。上述命令的最终结果：

```text
results/cobot_offline_assemble_parts/pretrain_run1/checkpoints/offline_step_40000/
  stage2_state.pt       # 原生模型、target、优化器、在线 schedule/rewind
  replay_buffer/
  demo_buffer/
  offline_buffer.pt    # 不可变离线数据、数据来源、划分、配置合同
  offline_state.pt     # 离线更新数、Cal-QL 乘子/优化器、随机状态
```

整个目录完成后才发布，不覆盖同名 checkpoint。训练中断后，设置
`STAGE2_RESUME_DIR` 为实际存在的 `offline_step_N` 目录，重跑同一个离线命令；
`NUM_TRAIN_STEPS` 是累计目标，不是额外训练次数。更改累计目标会改变 BC 衰减进度。
日志写入本项目 WandB。关注验证 TD 误差、Q 与 MC return、动作误差和残差可达比例，
选择稳定 checkpoint；完成 40000 次更新本身不代表已收敛。

小样本验证必须显式设置 `MAX_EPISODES`、独立 `OFFLINE_BUFFER`，并在训练时设置
`ALLOW_PARTIAL_DATASET=true`。这些产物会标记 partial，本地正式训练门槛和评测拒绝它们。

## 4. 带着离线权重启动在线服务

完成第 3 步后，服务器执行：

```bash
export STAGE2_RESUME_DIR="$PWD/results/cobot_offline_assemble_parts/pretrain_run1/checkpoints/offline_step_40000"
test -f "$STAGE2_RESUME_DIR/offline_state.pt"
CUDA_VISIBLE_DEVICES=3 \
  bash examples/embodiment/start_cobot_assemble_parts.sh train
```

恢复同一 actor/critic、target 和优化器。离线更新数独立记录，在线环境计数从 0 开始。
有已训练离线 checkpoint 时不再强制执行 250 个纯 VLA warmup chunk。
在线阶段恢复原有 TD3/EXPO 目标，Cal-QL 和离线 BC 不继续开启。
默认约 10% 的 batch 来自离线训练分区，其余来自原在线/HIL 采样；
可追加 `+algorithm.offline_sample_ratio=0.1` 调整，取值为 [0,1)。
后续在线 checkpoint 会继续携带离线 buffer；不需要重新转换或重新预训练。

本地先启动既有 Evo-RL ROS2 policy runtime，再执行：

```bash
bash exp/rlinf_client_cobot.sh assemble_parts check
bash exp/rlinf_client_cobot.sh assemble_parts dry-run
RLT_REQUIRED_OFFLINE_UPDATES=40000 \
  bash exp/rlinf_client_cobot.sh assemble_parts train
```

`check` 的 status 应显示 `offline_total_updates=40000`、`offline_buffer_size=6728`、
`offline_partial_data=false`、`warmup_done=true`；首次在线启动时 `total_updates=0` 正常。
本地训练会在创建机器人连接前核对要求的离线更新数，防止漏设服务器恢复目录。
`dry-run` 不发动作，正式 `train` 会控制双臂，需要操作员就绪。
之前现场右腕相机缺失的问题仍需通过真实 dry-run 检查，离线测试不能证明硬件就绪。

客户端继续用 WebSocket；本地执行数据的保存、HIL 和倒车功能沿用现有实现，
只记录实际动作执行帧，不记录 chunk 之间的推理等待。评测仍为每任务默认 30 局，
要求冻结服务端和显式的 `RLT_REQUIRED_STAGE2_UPDATES`（离线＋在线更新之和）。

## 代码与验证

服务端新增 `rlinf/serving/rlt/cobot_offline_data.py`、`offline_trainer.py` 和
`examples/embodiment/cobot_offline.py`；在线入口使用兼容的 trainer 子类。
rlt-openpi 仓库的 `deployment/rlinf/` 保存部署副本；已有服务端入口的接线补丁位于
`deployment/rlinf/cobot_offline_server.patch`，安装后运行 RLinf 的 Ruff 格式化。

服务器运行 `PYTHONPATH=. .venv/bin/python -m pytest tests/unit_tests/test_cobot_offline.py`。
本地运行 `pytest tests/test_rlinf_offline_milestone.py tests/test_rlinf_cobot_client.py`。
CPU 测试覆盖 v3 共享视频、双臂索引、失败奖励、MC 折扣、边界复用、Cal-QL 梯度、
真实 Stage2 权重更新、保存恢复、在线混合更新与不兼容配置拒绝。

部署验证已使用真实装配 Stage1 30k 转换前两条轨迹（137 个 transition），完成
20 次 Cal-QL 更新并恢复在线服务。这是小样本工程验证，未执行真机动作，
不属于全量预训练或正式效果评测。全量数据转换与 40000 步训练仍需按上面的命令启动。
