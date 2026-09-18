# Cobot 离线损失修正与小范围验证

本实验保留 joint motion v2 的 actor、动作预算、Stage1、数据和 noise=0.025，
仅更改训练目标。现有 joint v2 buffer 可复用；旧 Stage2 checkpoint 不作为新目标的起点。

## 目标与指标

显式设置 `offline.objective_version=joint-loss-v3` 才启用新逻辑，旧实验保持原行为。

- 固定 Cal-QL alpha=0.1，停止 alpha 优化器更新；保留 Huber delta=0.5。
- 成功示范 BC 按十二个关节的物理残差预算归一化，权重保持 1，夹爪不计入。
- actor 前 2,000 步只做 BC；其后 2,000 步将 Q 权重升到 0.01，Q 使用 twin minimum。
- 每 500 步记录两项加权梯度范数；固定 seed=918 的 256 个验证 transition，
  不重复抽样且不推进训练随机数状态。验证包含 TD、BC、Q/MC 偏差和 knot 饱和比例。
- 在内存中缓存与模型权重无关的 reference 运动投影，避免每步重复进行 30 步扫描；
  不修改源 buffer 或保存的真实数据。可用 `offline.cache_joint_references=false` 关闭。
- checkpoint 保存完整损失配方，离线恢复时拒绝配方变化；训练步数可以增加。
- 在线恢复新 checkpoint 后保留离线示范 BC；须显式设 `algorithm.q_weight=0.01`
  和正的 `algorithm.offline_sample_ratio`，避免自动回到无 BC、大 Q 权重的旧目标。

真实 replay 动作和对应的观测、奖励始终不改写。可达示范投影只用于 actor 的 BC。
MC 回归只作用于真实示范动作，不为所有候选动作强制设定示范回报下限。

## 对照实验

| 组别 | actor | critic |
|---|---|---|
| A_bc | 全程仅 BC，不使用 Q 梯度 | 与 B 相同，供诊断，不宣称是可用 RL critic |
| B_calql | BC + 延迟引入的小 Q 权重 | TD + 固定小系数 Cal-QL |
| C_mc | 与 B 相同 | 前 1,000 步拟合行为 MC 回报，之后采用 B 的目标 |
| D_family | 与 B 相同 | 保守项同时校准 edited、投影 base、局部 knot 候选 |

每组先运行 5,000 步，模型随机种子、buffer 和验证集保持一致。
保存间隔默认 1,000 步，WandB 名称区分各组，另保存 `validation.jsonl` 和同名 `.log`。

D 是 A/B/C 暴露剩余 Q 漂移后的独立对照：当前 local random 也由相同残差 actor
参数化产生，并非全动作空间的均匀样本，所以把这组可执行策略候选在保守项中同等校准。
该变化只停止对已经低于行为回报的候选继续施加向下惩罚，不裁剪 Q 网络输出、
不把候选的 TD 标签设成行为回报，也不改变真实奖励。使用 `D_family` 选择此对照。

在部署了本目录覆盖文件的 RLinf 根目录运行；`OFFLINE_BUFFER` 使用已经确认存在的
joint v2 buffer，`CUDA_VISIBLE_DEVICES` 使用当前探测后选出的卡：

```bash
bash examples/embodiment/run_cobot_loss_pilot.sh assemble_parts A_bc
bash examples/embodiment/run_cobot_loss_pilot.sh assemble_parts B_calql
bash examples/embodiment/run_cobot_loss_pilot.sh assemble_parts C_mc
```

用不同 zellij tab 启动上述三个命令。入口接受 `RLT_RUN_DIR`、`NUM_TRAIN_STEPS`、
`SAVE_EVERY` 和额外 Hydra 参数，不覆盖已有目录，恢复必须显式提供 `STAGE2_RESUME_DIR`。

## 扩展条件

首先比较 B/C 与 A 的固定验证 BC，再比较旧训练在相同验证样本上的指标。
小试通过应同时满足：无非有限值、TD 与估值漂移明显改善、BC 保持有效、
没有明显新增的残差饱和，且物理轨迹约束仍满足。最后几次验证趋势也需要检查。
若 Q 梯度持续压过 BC，先减小 Q 权重，继续小试，不自动扩展四任务。

这些指标只能支持继续离线实验；不代表已验证真机成功率。通过后才把选定配方
扩展到其他任务及更长训练，在后续在线阶段开始前另做 dry-run 和小规模真机验证。

运行记录与实际服务器路径保存在实验产物中，不写入通用配置。

## 5,000 步对照结果与选定配置

零件装配的四组对照均已完成，并用同一组 256 个验证样本在 CPU 重载复测：

| 组别 | 验证 TD loss | 示范动作平均 Q | Q 与行为 MC 的平均绝对差 | knot 饱和率 |
|---|---:|---:|---:|---:|
| A_bc | 0.24160 | -1.03004 | 1.11187 | 0 |
| B_calql | 0.26843 | -0.98842 | 1.06843 | 0 |
| C_mc | 0.20246 | -0.73822 | 0.81838 | 0 |
| D_family | 0.00292 | 0.08372 | 0.04076 | 0 |

该验证集平均 MC 回报为 0.07345。A/B/C 的 BC 稳定，但 Q 继续向负值漂移；
MC 预训练只延缓了问题。D 的后半程 TD loss 约为 0.0022–0.0034，Q 未继续漂移，
因此选定 `D_family` 扩展四任务至累计 40,000 步，入口默认也改为 D。

D 的固定训练 batch 中，加权 Q/BC 梯度范数比约为 0.0052，actor 仍主要由 BC 驱动。
这是保守的离线预训练配置，不是已经证明 RL 提升真机成功率。保持 Q 权重 0.01，
后续在线数据提供真实策略动作的反馈后，再评估是否需要提高 Q 更新力度。

扩展时设置 `NUM_TRAIN_STEPS=40000 SAVE_EVERY=5000`，给每个任务独立的
`OFFLINE_BUFFER` 和 `RLT_RUN_DIR`，再运行：

```bash
bash examples/embodiment/run_cobot_loss_pilot.sh assemble_parts D_family
```

零件装配可显式设置 `STAGE2_RESUME_DIR` 接续 D 的 5,000 步 checkpoint；
另外三个任务从新初始化的 Stage2 开始，使用各自已经转换好的 joint v2 buffer。
所有旧权重、旧 replay 和对照组 checkpoint 保留。
