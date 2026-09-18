# Cobot RLT Stage 2 对齐目标、现状与问题清单

> 状态：WS 真机路径的**目标算法**已与 `rlt-openpi@f80a804` 对齐；机型差走 `EmbodimentProfile`，不写进 learner。  
> 审查日期：2026-09-10  
> Cobot 基线：`RLinf/rlt@a520205a78d12b3d6de33ff8dbcc14c03f93da1b`  
> Franka 参考（审查时）：`rlt-openpi/remote-franka@2f6aa86cdb698a160b2b457758013ed4340a8192`  
> Franka 参考（对照 / 实施）：`rlt-openpi/remote-franka@f80a804`  
> RLinf 工作分支：`rlt-server-embodiment`（A–E 尚未单独提交）
>
> **修复更新：2026-09-11**，对应提交 `6694f646`、`cea7a95b`、`a367c566`、`76854a49`，以及其后未单独提交的 P1-11 ROS import 清理和 actor preference loss。
> **对照更新：2026-09-14**，对照远端 `exp/config.yaml`、`exp/stage2_server_shuo_rlinf.sh`、`exp/stage2_client_shuo_sync.sh`。客户端不要同步；服务端能迁的按 0.5 节 A–E 做完。
> **盘点：2026-09-15**，当前结论、已落地改动、未决问题和残留冗余见 **第 0.6 节**。第 0.2–0.5 节保留审查当时的条目，其中 0.5 的 1–8 条已按落地结果改状态。
> **USB 在线：2026-09-16**，云机 resume Cal-QL 后常驻 `8016`；跨机合同与空等见 **第 0.7 节**。客户端逐步操作见 `examples/embodiment/xrobot-stage2-robot-quickstart.md` 的「USB 客户端」。
> **对齐与联调：2026-09-18**，对照 `rlt-openpi@origin/remote-franka`（`26378e2`）补打分键 `p`/`o`/`x` 与 `submit`，并修复接管导致的断连与接管回执丢失，见 **第 0.8 节**。**客户端怎么拉代码、怎么操作、每个键什么含义，见第 0.9 节**（机器人侧的人只看那一节即可）。

## 0. 修复状态（2026-09-11 更新）

### 0.1 架构前提已经变了

本提案原本假设保留 Ray。实际采用的方案是：**Stage 2 真机路径不再使用 Ray**，改为两个进程 —— GPU 侧一个 WebSocket server（冻结 Stage 1 + Stage 2 actor/critic + replay + optimizer 全在一个进程内），机器人侧一个不加载模型的轻量 client 直接调用 adapter。

这带来两个后果，读本清单时必须先明确：

1. 本提案中相当一部分问题出在 Ray 链路上（`env_worker.py` 的 trajectory builder、`common/wrappers/` 的 wrapper stack、route 的 `record_transition`、Ray retry dedup）。新路径完全不经过这些代码，所以这些条目标记为**架构变更后不再适用**，而不是"已修复"。
2. **旧的 Ray 路径代码没有改动。** `examples/embodiment/config/cobot_rlt_stage2_td3_mlp.yaml` 及其 env worker/wrapper 链路上的问题原样存在。下面所有"已修复"都只对新的 WebSocket 路径成立。

新增的代码位置：

```text
rlinf/serving/websocket_server.py          WebSocket 传输
rlinf/serving/rlt/protocol.py              八种请求类型与握手 metadata
rlinf/serving/rlt/inference.py             观测 repack、EXPO 选择、动作空间换算
rlinf/serving/rlt/policy.py                请求路由、transition_id pending map
rlinf/serving/rlt/preflight.py             启动检查
rlinf/serving/rlt/trainer.py               Worker-free trainer
rlinf/algorithms/rlt/learner.py            从 Ray worker 抽出的共享算法代码
rlinf/envs/realworld/rlt_client/           transport contract / 循环 / cobot 适配
examples/embodiment/rlt_stage2_{server,client}.py
examples/embodiment/config/cobot_rlt_stage2_ws_{server,client}.yaml
examples/embodiment/run_rlt_stage2_{server,client}.sh, run_cobot_control.sh
```

### 0.2 逐条状态

| 条目 | 状态 | 说明 |
|---|---|---|
| P0-1 `rlt_prefix_seq_len` 968 | 已修复 | 配置改 968，preflight 直接从 checkpoint 读实际值 |
| P0-2 keyboard switch wrapper | 不再适用 | 新架构没有 Stage 2 切换键，warmup 由 server 按 replay 行数判定 |
| P0-3 `record_transition` mask | 不再适用 | 改为 `transition_id` pending map，一次 `act` 最多写一行 |
| P0-4 trajectory `N+1/N` | 不再适用 | server 只为已执行 chunk 写行，没有 final proposal 行 |
| P0-5 相机 OpenPI contract | 已修复 | `RLTObservationRepacker` 显式产出三路语义字段 + preflight 校验 |
| P0-6 normalized 硬裁剪 | 已修复 | clip 可配置，默认 ±1.4；ManiSkill 固定回 ±1.0 |
| P1-1 recovery chunk 丢失 | 不再适用 | rewind 是独立请求，在 commit 之后处理，下一条 chunk 正常提交 |
| P1-2 `update_last_actions` 污染 | 不再适用 | 每行由 `transition_id` 绑定，不依赖"最后一条" |
| P1-3 UTD 补算 1250 次 | 已修复 | 只统计 `mode=="actor"` 的 chunk × `utd_ratio` |
| P1-4 随机 critic 控制真机 | 已修复 | warmup 期间强制执行 Stage 1 reference，EXPO 关闭 |
| P1-5 task prompt 不匹配 | 已修复 | 见 0.3，这是本轮风险最高的一条 |
| P1-6 未 fail-fast 要求 RLT | 已修复 | `require_rlt_checkpoint: True` + 全量 `rlt_module.*` 校验 |
| P1-7 静默降级为 Mock | 已修复 | `is_dummy=false` 且无有效 factory 时直接报错 |
| P1-8 adapter contract 不完整 | **部分修复** | identity/observation/result/event 已定型并在握手校验；混合动作换算已移到 server 侧；`on_action_chunk_begin/end` hook 仍未提供 |
| P1-9 controller state resume | **部分修复** | `session_id` 按进程启动时间生成，重启后 identity 不会与旧 session 冲突；trainer 状态可存可恢复；但机器人侧 controller 的 session/chunk/history 仍未纳入 checkpoint |
| P1-10 bootstrap mask 被覆盖 | 不再适用 | 无 wrapper，bootstrap 由 `ChunkExecutionResult` 直接决定 |
| P1-11 ROS import-time 清理 | 已修复 | 父包 `__init__` 不再调用 `realworld_setup()`；回归测试锁住这条 |
| P1-12 partial chunk 丢弃 | 已修复 | 截断 chunk 照常提交，reward 零补齐，尾部按最后一个已执行动作填充 |
| P2 actor lr `3e-4` | 已修复 | 改为 `3e-5` |
| P2 MLP hidden layers | 已修复 | 改为 3 |
| P2 LayerNorm | **部分修复** | critic 隐层 LN 已加（`critic_use_layer_norm`）；actor input LN 与 critic input LN 均未加，`make_mlp` 只在隐层后插 LN |
| P2 episode horizon | 已修复 | `max_episode_chunks: 150` |
| P2 Stage 2 proprio | **未修复** | 见 0.4 |
| P2 action clip | 已修复 | ±1.4 |
| P2 replay buffer 无界增长 | 未修复 | 未触及 |
| P2 `auto_save=false` index 失配 | 未修复 | 未触及 |
| P2 event-only 增加 `_elapsed_steps` | 不再适用 | 无 env，horizon 按 chunk 在 server 侧计数 |
| P2 缺可执行脚本 | 已修复 | server/client/控制栈三个脚本，并补了 `preflight_only` |
| P2 文档旧提交与失实声明 | 已修复 | 重写并改名为 `cobot-stage2-rollout-online-training-guide.md` |
| P2 `b` 键语义 | 已修复 | 明确为物理回退；无 Stage 2 切换键 |

### 0.3 审查时未发现、修复过程中新暴露的问题

这些不在原清单里，其中前两条与 P1-5 同源，都是"静默用错任务"：

**norm stats 会静默指向另一个任务（P0 级）。** `openpi_data.norm_stats_path` 不配置时，OpenPI 把 `model_path` 当 assets 目录（那里只有 `full_weights.pt`），然后回退到 `pi05_cobot_magic` 内置的默认 `asset_id`，即 `cobot_magic/cube_into_drawer`。实测确认了这条回退路径。后果是反归一化用错任务的分位数，直接把错误关节量下发。已加 `check_norm_stats_configured`，未配置时拒绝启动。

**task prompt 同源问题。** `pi05_cobot_magic` 内置默认 prompt 就是 `put cube in drawer`，而 assemble 那次 Stage 1 实际用的是 `assemble parts`。现在 preflight 校验 `server.task_prompt` 与 `openpi_data.default_prompt` 一致，client 的 `transport.task` 也必须一致。

**`model_path` 多写一层会掉进 safetensors 分支。** 加载器只认 `<model_path>/model_state_dict/full_weights.pt` 和 `<model_path>/actor/model_state_dict/full_weights.pt`。旧文档写的 `.../global_step_15000/actor/model_state_dict` 两者都不匹配，会在启动看起来正常之后才失败。已加 `resolve_stage1_weights`，并提示往上退一级。

**EXPO 选中的 base 没有写回 `ref_chunk`（高影响）。** Ray 路径在 `rlt_td3_mlp_policy.py` 里会把选中的 base 写回 `ref_chunk`，而 server 路径最初存的是 `candidates[0]`。BC target 因此可能对应一条没被选中的候选。已修正。

**msgpack 解出的数组是只读的。** `msgpack_numpy` 返回的是接收缓冲区上的只读视图，`np.asarray` 会保留这一属性，`torch.as_tensor` 则会 alias 该缓冲区。只在真 socket 上跑第二个 episode 时才暴露（`assignment destination is read-only`）。已在 5 处解码边界改为 `np.array` 拷贝。这类 bug 所有 loopback 测试都测不出来，这也是后来把真 socket 用例加进常规测试的原因。

**resume 后 checkpoint 互相覆盖。** 目录名原来只有 `episode_<N>`，恢复后 episode 编号从头开始会覆盖旧目录。改为 `episode_<N>_step_<update_step>`。

**`reset()` 会复用 identity。** 原来不递增 `episode_id`，同一 session 内两个 episode 的 identity 会重合。已递增。

**启动脚本从未成功运行过。** `run_rlt_stage2_server.sh` 和 `run_rlt_stage2_client.sh` 没设 `PYTHONPATH`；RLinf 是源码树直接用、非 pip 安装，按路径执行脚本时只有脚本自身目录进 `sys.path`，两者都死在 `No module named 'rlinf'`。已补齐。

**共享 learner 的 actor 漏了 rewind Bradley-Terry 项。** 抽 `RLTLearnerCore` 时只把 `_add_critic_preference_loss` 带过来了，`forward_actor` 停在 `-Q + BC`。WebSocket 配置里 `rewind_preference.actor_weight: 0.1` 因此是死的：回退 pair 会改 Q，但不直接推 actor 靠近 recovery、远离坏分支。对照是 `fsdp_rlt_td3_policy_worker.py` 自己的 `forward_actor`，不是还要用 Ray 跑真机。已在共享 `forward_actor` 补上 `_add_actor_preference_loss`，WebSocket 与仿真 Ray AC 路径一起修好。

**reference horizon 与 Stage 1 不一致（已记录，未改动）。** `pi05_cobot_magic` 默认 `action_horizon` 为 50，两次 Stage 1 SFT 都没覆盖它，所以是按 50 步 action block 训练的；Stage 2 推理覆盖成 20。checkpoint 里没有任何按 horizon 定长的张量（`968` 只与 3 相机 × 256 patch + `max_token_len` 200 的图文 prefix 有关），所以两种取值都能加载，但 action expert 训练时始终看到 50 个 action token、推理只给 20 个，属于 attention 序列长度上的 train/inference 偏移。该 16/20 设置是从既有 Ray 配置继承的（其数值源自 `pi05_franka_state`，那边 Stage 1 horizon 本来就是 20），不是新引入，因此保持原值并写入操作文档作为 dry run 核对项。

### 0.4 仍未修复、上真机前需要决策的一项

**P2 Stage 2 proprio 仍是 raw state。** `inference.py` 的 `encode()` 里 `proprio = torch.as_tensor(env_obs["states"])`，而 `env_obs["states"]` 是 client 上传的原始关节量；经 OpenPI 输入变换归一化后的 `openpi_obs.state` 只被留作反归一化用（`_openpi_state`），没有喂给 Stage 2 actor/critic。Franka 参考实现用的是 normalized state。后果是 actor/critic 的输入尺度与参考实现不同，而 critic 又没有 input LayerNorm 来吸收这个尺度差（见上面 LayerNorm 那条部分修复）。这两条建议一起处理。它不阻塞启动和数据流，但影响训练效果，建议低速 HIL 前定下来。

### 0.5 2026-09-14 对照 `rlt-openpi@f80a804`

本地 `/data/gxy/realworldRL/rlt-openpi` 已从审查时的 `2f6aa86` fast-forward 到 `f80a804`（70 个提交）。对照的三份文件：

| 文件 | 角色 |
|---|---|
| `exp/config.yaml` | 客户端 + 服务端 + 离线共用的一份旋钮。算法相关全部在 `stage2.server.franka`。 |
| `exp/stage2_server_shuo_rlinf.sh` | 单进程 WebSocket server。Stage 1 走 RLinf checkpoint（`rlinf_adapter.py`），不再走自家 OpenPI serve。 |
| `exp/stage2_client_shuo_sync.sh` | Franka + Gello + ROS Humble 的同步客户端。`QUEUE_LOW=0` 的 stop-and-go，与 XRobot V2 同一类节奏。 |

两边现在是**同一类架构**：GPU 上一个进程持有冻结 Stage 1 + Stage 2 + replay，机器人侧一个不加载模型的客户端。`stage2_server_shuo_rlinf.sh` 甚至直接加载 RLinf 的 Stage 1 权重。通信协议仍然各自实现，不能把他们的 `.sh` 接到我们的端口上跑。

#### 不要同步的（客户端 / 站点 / 实验分叉）

这些是 Franka 机架和他们仓库的部署细节，搬过来只会污染 embodiment 无关的服务端：

- `min_z` / `fault_mode` / `gello_takeover_mode=stop_and_go` / 相机 serial / ROS topic / `QUEUE_LOW`。XRobot 的 V2 supervisor 已经覆盖接管、回滚、急停；Cobot 走 adapter。
- `RL_TOKEN_MODE=linear|mlp|resnet34` 以及 reconstruction probe。我们只服务冻结的 Stage 1 `rl_token`。
- `MAX_ENTROPY_ACTOR=1`（TanhNormal + 在线调 α）。我们的 WS 路径显式关掉熵（`entropy_tuning.initial_alpha: 0`，learner 对 `forward_alpha` 直接 `NotImplementedError`）。他们脚本默认打开，但 `config.yaml` 没有暴露这个键；迁过来等于换算法，不是对齐。
- 他们脚本里的 wandb key、`/root/lianglin/RLinf`、`/mnt/data/...` 路径。
- 把 `exp/config.yaml` 整份搬进 Hydra。机型合同已经在 `config/embodiment/<name>.yaml`，任务旋钮在 server YAML。再加一份平行的 `config.yaml` 会变成第三份真相。

#### 已经对齐、或我们这边更多的

| 项 | `rlt-openpi` 现在 | RLinf WS 现在 |
|---|---|---|
| 单进程 WS + 冻结 Stage 1 | 是（还反过来加载 RLinf ckpt） | 是 |
| EXPO 执行端 4 base + 4 edit | `rl_algo_act: expo` | `action_selection_mode: expo`，`expo_num_*: 4` |
| actor 纯 `-Q`，BC=0 | `EXPO_PURE_Q_ACTOR=1` | `q_weight: 1` / `bc_weight: 0` |
| actor lr / critic lr / 3 层 MLP / clip ±1.4 | 3e-5 / 3e-4 / 3 / ±1.4 | 同 |
| Huber δ=0.5、γ=0.99、τ=0.005 | 是 | 是 |
| warmup + episode 边界 UTD | 是 | `warmup_steps` × `server.utd_ratio` |
| rewind preference（critic hinge + actor BT） | 是 | 是（`rank_slope: 0.2`，近重复 pair 自动松弛） |
| 离线 Cal-QL → 在线混采样 | 有，在线 Cal-QL 权重默认 0 | `RLTOfflineTrainer` + `offline_sample_ratio` |
| PER | 默认关 | 默认开（`prioritized: True`） |
| 机型合同 / preflight / 握手 `robot_type`+`action_schema` | 无（写死 Franka） | 有。这是我们多出来的，留下。 |

#### 缺的、而且能迁到服务端的

按「改完 Q 学的还是不是同一个对象」排序。都落在 `rlinf/algorithms/rlt/`、`rlinf/models/embodiment/mlp_policy/rlt_td3_mlp_policy.py`、`rlinf/serving/rlt/{trainer,policy}.py`，**不需要改客户端协议**（滑窗那条除外）。

**1. act / TD-backup 拆开，以及 `expo_decoupled`（已实施）。**  
他们把「机器人执行什么」和「critic 备份什么」拆成两个键：

```text
rl_algo_act:        expo | td3
rl_algo_td_backup:  expo | expo_decoupled | td3
```

`config.yaml` 当前部署是 `act=expo, backup=td3`（注释里写的诊断组合）。最终算法注释写的是 `act=expo, backup=expo_decoupled`：同一份候选做 Expected-Max，但用一个 target 子集选、不相交的另一个子集评（Double-DQN），去掉 max 的系统性上偏。  
已迁：`rl_algo_act` / `rl_algo_td_backup`，部署 `expo` + `expo_decoupled`；ensemble 宽度不够时 preflight 拒绝。旧 replay 在 `expo`/`expo_decoupled` 之间可以共用。

**2. critic 从 Twin-Q 换成 REDQ ensemble（已实施）。**  
他们默认 10 个 Q、每次抽 2 个取 min；actor 目标对全集做 **mean**。  
已迁：`EnsembleQCritic` 10/2，`actor_q_aggregation: mean`。构造默认仍是 2 头以兼容旧测试。改了 checkpoint 布局，旧 Twin-Q 权重不能直接 resume。

**3. delayed Polyak：只在 actor 步更新 target（已实施）。**  
`1241fde`：actor target 和 critic target 只在 actor 更新那一步做 Polyak。  
已迁：`target_update_on_actor_step: True`，`update_once` 只在 actor 步 Polyak。

**4. 残差幅度和夹爪独立参数化（已实施；XRobot/Cobot 都有夹爪维）。**

| 键 | 他们 | 我们现在 |
|---|---|---|
| 手臂 `edit_scale` / `residual_scale` | 0.4 | 0.4 |
| `gripper_edit_scale` | 2.0 | 2.0 |
| `actor_gripper_absolute_output` | 开：`scale * tanh(u)` | 开 |
| 夹爪 clip | ±1.0 | ±1.0（手臂仍 ±1.4） |
| clip 反传 | `inward` | `inward` |

夹爪在归一化空间贴着 ±0.98，加性残差几乎只能顶出 [-1,1]。这是他们 `c93e4fa` / `d8e23fe` 的理由，对 14-D 带夹爪机型成立，不是 Franka 特例。构造默认仍是 scale=0.2 + hard clip，旧单测不用改。

**5. preference / intervention rank 已换语义（已实施）。**  
他们把常数 margin 改成 `slope * ||e' - e||`（每维 RMS），近重复的 pair 自动松弛。另外多了：

- `intervention_rank`：`Q(a_human) > Q(a_neg)`，人的动作是 Q 的序约束，不再当 actor 的监督目标（`INTERVENTION_BC_BETA` 在 EXPO 下是 0）。
- `CRITIC_RANK_LOCAL_STEPS=0.2,0.4,0.6,0.8`：在 `a_neg → a_human` 的弦上取探针，约束 actor 实际站立处的方向导数。他们上一轮只做端点 pair 时，Q gap 满足了 3 倍、actor 却不动。

`rewind_preference.rewind_*_reward` 三个键全仓零引用，他们的 `rewind_*_exit_reward` 是客户端写进 transition 的，不是服务端 loss 键。  
已迁：`losses.py` 用 `rank_slope` 取代只靠常数 margin；加了 intervention rank + local probes。pair 仍由 rewind 独立请求构造。

**6. TD target 下截断（已实施）。**  
他们 `TD_TARGET_CLIP_MIN=-1.0`，上界空着。理由：reward 落在 chunk 末步，bootstrap 把 `r / (1-γ^C)` 放大到十几倍。  
已迁：`td_target_clip_min: -1.0`。

**7. 滑窗 transition（不做；能迁但要动协议和客户端）。**  
`041f2af`：纯 policy 段内按 `step_window_stride` 切中间观测，client 先把图 resize 到 224 再发。我们一个 chunk 一行，没有 mid-chunk 观测字段。  
迁：handshake 下发 stride；`transition` 允许多个 `next_observation`；`policy.py` 在 `episode_end` 切窗。XRobot / Cobot 客户端都要跟着改。**不要单独先做服务端。**

**8. eval episode 写回 replay（已实施）。**  
`63d110a`：周期性 eval 的轨迹默认写回。  
已迁：`eval_interval_episodes: 5`，`store_eval_episodes: True`。整场 `eval_only` 仍不写、不训。

#### 我们多出来的、不要删去对齐

- `EmbodimentProfile` + `config/embodiment/` + preflight（norm stats / prompt / prefix_seq_len / 几何一致性）。他们没有等价物，`config.yaml` 把任务路径写死。
- PER 默认开。他们默认关。两边都是合法选择，改之前先看 replay 规模，不要为了对齐而关。
- XRobot EE14 已跑通的闭环（握手、pending map、`episode_end` 更新、checkpoint `episode_N_step_M`）。
- 文档在 `examples/embodiment/`，不再散落根目录。

#### 当时的迁移顺序（A–E 已做完，F 仍等客户端）

1–3 改的是「critic 在备份哪一个策略」，必须一起做。4 是 actor 几何，和旧 Stage 2 权重不兼容。5–6 是 loss 形状。7 等客户端。8 是开关。

```text
A. trainer：Polyak 只在 actor 步     （3，行为变、权重兼容）
B. TwinQ → ensemble 10/2 + act/backup 拆开 + expo_decoupled
   （1+2，改 ckpt 布局；config.yaml 现网是 backup=td3，最终目标 backup=expo_decoupled）
C. 夹爪绝对输出 + inward clip + residual_scale 0.4
   （4，改 actor 几何；旧 Stage 2 头作废）
D. rank slope + intervention rank + local steps + TD clip
   （5+6）
E. eval 写回                                        （8）
F. 滑窗 —— 等 XRobot/Cobot 客户端能发 mid-chunk 观测   （7）
```

A–D 都不碰 WebSocket 帧类型，XRobot 现网客户端可以继续连。B/C 之后必须新开 Stage 2 头，不能 `resume_dir` 到 `episode_10_step_0` 那种 Twin-Q + scale=0.2 的目录。

实施状态（2026-09-14）：

| 步 | 状态 | 落点 |
|---|---|---|
| A delayed Polyak | 已实施 | `trainer.update_once` + `algorithm.target_update_on_actor_step` |
| B ensemble + act/backup + `expo_decoupled` | 已实施 | `EnsembleQCritic`、`compute_rlt_critic_loss`、`rl_algo_*`；部署 `backup=expo_decoupled`、`critic_num_qs=10` |
| C 夹爪绝对输出 + inward + scale 0.4 | 已实施 | `DirectGaussianActor`；YAML 部署值，构造默认仍兼容旧测试 |
| D rank slope / intervention / local / TD clip | 已实施 | `losses.py` + `learner.forward_critic`；`rewind_preference.rank_slope=0.2`，`intervention_rank` + `td_target_clip_min=-1.0` |
| E eval 写回 | 已实施 | `eval_interval_episodes=5`，`store_eval_episodes=True`；`eval_only` 整场仍不写 |
| F 滑窗 | 不做 | 等客户端能发 mid-chunk 观测 |

### 0.6 2026-09-15 盘点：对齐、机型、已落地、未决、冗余

这一节是现在读这份清单时的入口。0.2–0.5 是审查当时的条目；这里只写**当前还成立的结论**。

#### 和 `rlt-openpi` 对齐了吗

**目标算法对齐，不是把他们仓库整份搬过来。**

对照的是 `rlt-openpi@f80a804` 注释里的最终配方（`act=expo, backup=expo_decoupled`），不是他们 `exp/config.yaml` 里现网诊断组合（`backup=td3`）。我们的 WS YAML 配的是最终配方，备份模式比他们检入的 deploy 配置更靠前一档。

已落地、两边学到的 Q 是同一类对象：

| 项 | 落点 |
|---|---|
| delayed Polyak（只在 actor 步更新 target） | `trainer.update_once`，`algorithm.target_update_on_actor_step` |
| REDQ ensemble 10/2，actor Q 取 mean | `EnsembleQCritic`，`actor_q_aggregation: mean` |
| 执行 EXPO 4+4，备份 `expo_decoupled` | `action_selection_mode` + `rl_algo_td_backup` |
| 夹爪绝对 tanh、inward clip、手臂 `residual_scale=0.4` | `DirectGaussianActor`；夹爪 ±1.0，手臂 ±1.4 |
| rewind `rank_slope=0.2`；intervention rank + 弦上探针 | `losses.py` + `learner.forward_critic` |
| TD target `clip_min=-1.0` | `compute_rlt_critic_loss` |
| 训练中每隔 5 个 episode 一局 eval，默认写回 replay | `store_eval_episodes` / `eval_interval_episodes` |
| 纯 −Q actor、Huber、γ/τ、warmup × UTD | 与对照脚本同值 |

故意不迁、也不该迁：

- Franka 客户端栈（Gello / `min_z` / ROS / 相机 serial）。
- `MAX_ENTROPY_ACTOR`（TanhNormal + 在线调 α）。我们固定方差、α=0。
- `RL_TOKEN_MODE=linear|mlp|resnet34`。我们只服务冻结 Stage 1 `rl_token`。
- 把 `exp/config.yaml` 整份搬进 Hydra。机型在 `config/embodiment/`，任务在 server YAML。
- 滑窗 mid-chunk 观测（F）：要改协议和两边客户端，不能只做服务端。

两边合法分叉、不要为对齐而改：PER 我们默认开、他们默认关；在线 Cal-QL 权重他们默认 0，我们的 Cal-QL 在离线入口，WS YAML 没挂上。

B/C 改了 critic 头数和 actor 几何。旧 Twin-Q + `residual_scale=0.2` 的 `resume_dir`（例如 `episode_10_step_0`）不能用，必须新开 Stage 2 头。

#### 不同类型的机器怎么封装

**是。** learner / trainer / protocol / replay 不认机型。差的是一份合同和一份 OpenPI 流水线名。

```text
config/embodiment/<robot>.yaml     宽度、chunk、相机、openpi_config_name、
                                   robot_type、action_schema
server YAML 任务层                 prompt、Stage 1 ckpt、norm_stats、保存目录
EmbodimentProfile.check_config     各节数字必须和合同一致，否则拒绝启动
握手 metadata                      robot_type + action_schema，客户端拒接错机
关节 vs 末端                       只在冻结 Stage 1 的 OpenPI transform 里
夹爪维                             14-D 双臂按 (6,13)，其它布局取最后一维
```

现网两份：`cobot_magic`（关节 14、执行 16 / 提案 20）和 `x2robot`（EE14、50/50）。`xrobot_ee_rlt_stage2_ws_server.yaml` 继承整份算法，只改任务路径。换机 = 加一份 embodiment YAML + 任务 overlay，不改 Python。

他们那边没有等价物，`config.yaml` 把 Franka 路径写死。这是我们多出来的，留下。

#### 已经落地的改动（按时间）

**2026-09-11 通信与调度（已提交）**

Ray 真机路径弃用，改为单进程 WS。八种请求、`transition_id` pending map、warmup 强制 Stage 1、`episode_end` 按 `train_chunks × utd_ratio` 更新、rewind 独立请求、preflight（prefix_seq_len / prompt / norm stats / `model_path`）、normalized↔robot 在 server 侧换算、actor preference BT、文档收到 `examples/embodiment/`。

**2026-09-14/15 算法 A–E（工作区未提交）**

见 0.5 实施表。构造默认值仍兼容旧单测（2 头、`residual_scale=0.2`、hard clip）；部署值只写在 `cobot_rlt_stage2_ws_server.yaml`，XRobot 继承。

#### 还没解决的问题

按「上真机前要不要拍板」排序。

| 优先级 | 项 | 说明 |
|---|---|---|
| 要决策 | **P2 Stage 2 proprio** | `encode()` 喂 raw 关节；Franka 用 OpenPI 归一化 state。critic 也没有 input LN 吸收尺度。不阻塞启动，影响学到的 Q。见 0.4。 |
| 要决策 | **P2 actor input LN** | critic 隐层 LN 已开；actor MLP 无 input LN。建议和 proprio 一起改。 |
| 要决策 | **P1-8 adapter hook** | identity / obs / result 已定型；`on_action_chunk_begin/end` 仍没有。 |
| 要决策 | **P1-9 controller resume** | trainer 可存可恢复；机器人侧 session/chunk/history 不在 checkpoint 里。重启靠新 `session_id` 避免撞号，不能从断点续控。 |
| 操作风险 | **replay 无界** | 没有全局 cap，只有采样窗 `sample_window_size: 200`。长跑吃内存。 |
| 操作风险 | **`auto_save: False`** | 崩溃会丢掉尚未写入 `episode_N_step_M` 的 replay。 |
| 操作风险 | **旧 Stage 2 头作废** | 不能 resume Twin-Q + scale=0.2；也不能直接吃 rlt-openpi 的 `offline_rl_step*.pt`。 |
| 等客户端 | **F 滑窗** | 协议没有 mid-chunk 观测字段。 |
| 不迁 | **Ray `cobot_rlt_stage2_td3_mlp.yaml`** | 仍走 `RLTTD3LossMixin`，读 `algorithm.expo.enable` / `actor_agg_q`，不走 `RLTLearnerCore` 的 backup。真机不要用这条。 |
| 分叉留下 | **PER 默认开** | 不要为了对齐关掉。 |
| 未接线 | **在线混离线 / Cal-QL** | USB `start_xrobot_stage2.sh usb_plug train` 已挂 `offline_sample_ratio=0.1`；Cobot WS YAML 仍没挂。 |
| 未做 | **两边固定 batch 的 loss 数值对照** | 第 12 节完成条件。要对着 `rlt-openpi` 取数，还没做。 |

`rewind_preference.rewind_*_reward` 三个键从来不是服务端 loss：客户端 rewind 请求自己带 terminal/prefix reward。不是未实现，是死键。

#### 还在的冗余（看起来像旋钮、WS 不读）

这些不必为了对齐删 Ray YAML，但 WS 文件里容易误导。标了「可从 WS YAML 拿掉」。

| 键 / 符号 | 谁读 | 处理 |
|---|---|---|
| `algorithm.expo.{enable,base_candidates,edited_candidates}` | 只有旧 Ray `RLTTD3LossMixin` | WS 可删；活旋钮是 `actor.model.action_selection_mode` / `expo_num_*` |
| `algorithm.agg_q` / `actor_agg_q` | 只有旧 Ray actor | WS 可删；活旋钮是 `actor_q_aggregation` |
| `algorithm.adv_type` / `loss_type` / `loss_agg_func` | Ray registry | WS 可删 |
| `rewind_preference.rewind_*_reward` 三个 | **全仓零引用** | WS 和 Ray YAML 都可删 |
| `algorithm.actor_update_action_noise` / `critic_subsample_size` / `backup_entropy` | WS 不读 | WS 可删 |
| `algorithm.rl_algo_act` | 只在 preflight 和 `action_selection_mode` 对账 | 留下当文档 + 启动校验 |
| `losses.expo_backup_num_edit_samples` | learner 传进去，loss 里 `del` 掉 | 实现或删掉这根线 |
| `server.robot_type` / `action_schema` / `camera_keys` | 已删 | 握手只读 `embodiment` |
| `RLTOfflineTrainer` vs `RLTStage2Trainer` | 入口用前者 | 不是重复，留下 |
| `rlinf/serving/rlt/cobot_offline_data.py` | 离线 Cobot LeRobot | 不在 WS 热路径；以后再改名 |

本轮**没有**再删这些死键，避免和 Ray 配置的 diff 搅在一起。真机路径以 WS YAML 的活旋钮为准。

### 0.7 2026-09-16 USB 云机常驻与客户端

USB 不走套环启动器，也不走 Cobot launcher（后者会把 chunk 打成 30、residual 打成 0.3）。

已完成的路径：

1. 本机非重叠 50 步 convert + Cal-QL 4 万步，buffer 在 `offline_rl_buffers/xrobot_usb_plug/`，WandB project `xrobot-usb-offline`。
2. 资产拷到云机 `/mnt/data/lfwj/realworldRL`（Stage 1 `full_weights.pt`、USB `norm_stats.json`、`offline_step_40000`）。不要拷 LeRobot 全集。
3. 跨机合同曾拒收：buffer 里写死了源机 `model_path` / `norm_stats_path` / `weights_mtime_ns`。处理：改这三处本机字段（先核 `weights_size` 和 `norm_sha256`），或拉忽略路径/mtime 的新代码。
4. 云机 `STAGE2_RESUME_DIR=.../offline_step_40000` + `usb_plug train`，听 `0.0.0.0:8016`。`warmup 250 rows` 只是打印；`offline_total_updates>0` 时第一回合就是 residual。
5. 在线训练要等机器人 `episode_end`。server 空等几小时没有问题。SSH `34133` 和 WS `8016` 不是同一口，打不通就在机器人侧 `-L 8016:127.0.0.1:8016`。

客户端（机器人，GPU 已 ready 之后）：

1. 需要时先打隧道，然后 `RLT_UPSTREAM_URI=ws://<GPU或127.0.0.1>:8016`，`RLT_TASK_PROMPT="Bimanual usb pick and insert"`。
2. 先 probe（不设 `RLT_EE_ALLOW_MOTION`），DesktopClient 模型地址 `127.0.0.1:33057`，确认没有 `ProtocolError`。
3. 再开 V2 适配器，再 `--stop` 后带 `RLT_EE_ALLOW_MOTION=true` 重拉 bridge，最后开一个没用过的 V2 session。
4. V2：`s` 接管，`f` 成功。每轮必须收尾，否则 server 不训练。

操作文档：`examples/embodiment/xrobot-stage2-gpu-quickstart.md` 第 10 节；`examples/embodiment/xrobot-stage2-robot-quickstart.md` 「USB 客户端」。

### 0.8 2026-09-18 对齐审计：`rlt-openpi@origin/remote-franka`

本地 `/data/gxy/realworldRL/rlt-openpi` 已 fetch 到 `origin/remote-franka`（`26378e2`，比 0.5 节当时的 `f80a804` 新 21 个提交，新增 `exp/stage1-2_shuo_ddp_rlinf.sh`、`exp/stage2-2_offline_shuo_rlinf.sh`）。对照 `exp/stage2_server_shuo_rlinf.sh`、`exp/stage2_client_shuo_sync.sh`、`exp/start_franka_control_rlt_sync.sh`、`exp/config.yaml`。

#### 已一致

`rl_algo_act=expo`、`rl_algo_td_backup=expo_decoupled`、`expo_num_base_samples=4`、`expo_num_edit_samples=4`、`edit_scale/residual_scale=0.4`（对齐 `exp/config.yaml:85`；脚本里的 `EDIT_SCALE=0.3` 是旧默认）、`gripper_edit_scale=2.0`、`actor_lr=3e-5`、`ACTOR_ACTION_CLIP_GRADIENT_MODE=inward`。

#### 差异与处理

| 项 | 他们 | 我们 | 处理 |
|---|---|---|---|
| `actor_noise_sigma` | 0.1（`stage2_server_shuo_rlinf.sh:250`） | YAML 0.2 | **不改 YAML**：改了会让 `offline_step_40000` 的在线 resume 合同不匹配（`actor_noise_sigma` 在 `contract().actor_model` 里，而 `allow_actor_reconfiguration` 只在 `offline_mode` 下可用）。改为客户端 per-act 覆盖 `RLT_EXPLORATION_NOISE_SIGMA=0.1`，已写进机器人速查。 |
| `TARGET_NOISE_SIGMA` / `TARGET_NOISE_CLIP` | 0.007 / 0.014 | 无 | **未实现**。我们只有 `intervention_critic_action_noise_*` 和 `rewind_critic_action_noise_*`（各 0.002/0.005），主 TD backup 没有 target 平滑噪声。这两个键在 `algorithm.` 下、不进合同，可随时加。留作单独决策。 |
| 失败奖励 | `f` = failure，reward **0** | `server.failure_reward: -1.0` | 有意分叉，保持 −1.0：我们要显式惩罚失败分支。 |
| `MAX_ENTROPY_ACTOR` | 1（TanhNormal + 在线调 α） | 固定方差、α=0 | 有意不迁移，见 0.6 节。 |
| 打分键 `p` / `o` / `x` | 有（0.5 / +0.1 累加 / −0.5，**不结束回合**） | 原先完全没有 | **本轮补上**，见下。 |

#### 打分键与 submit（本轮实现）

`exp/stage2_client_shuo_sync.sh` 的键位是 `s`=成功 1、`f`=失败 0、`p`=进展 0.5、`o`=小进展 +0.1（可累加）、`x`=退步 −0.5、`y`=接管暂停时提交当前 chunk 并继续。

RLT 协议**不需要改**：`transition` 本来就带 `rewards` 数组，打分只是把 `rewards[-1]` 写成非零而保持 `done=False`、`bootstrap_mask=1`。实现落在客户端侧：`QueuedDecision.score`（累加）、`RLTSession.commit(score=...)`、`operator_cli` 的 `progress` / `small_progress` / `regress`，以及 adapter 的 `session_progress` / `session_small_progress` / `session_regress` 事件。终止判定优先于累计分。

`y` 对应新增的 `submit` 命令：`DecisionInbox.pop_commit_now` + `RLTBridgeCore.flush_queued_submit`，在 policy 暂停、没有新观测时把当前 chunk 连同接管回执提交，**不**结束 episode。故意做成显式命令而不是“见到 intervention 就冲”，否则 `r` → `r` 的倒车流程会在 `rewind_exit` 判定到达前就把 chunk 提交掉。

#### 接管断连（本轮修复）

现象：接管后服务端与客户端断开，接管期间的动作到不了服务端。根因是 upstream 连接和 `RLTSession` 都是**按 DesktopClient 连接**创建的，`_serve_robot` 的 `finally` 无条件 `core.abort()` + `upstream.close()`。V2 在接管时关掉模型连接，于是整局被 `episode_end(aborted=True)` 判死，排队中的 intervention 回执一起丢掉，服务端还会打印 `dropping N pending transitions`。

修法：新增 `BridgeRuntime` 持有进程级 upstream 和当前 episode；DesktopClient 断开只解绑 downstream，不 abort、不关 upstream，重连后继续同一局；只有 episode 自身已结束时才开新 session。协议异常仍然 abort 那一局，但保留 upstream。

### 0.9 客户端操作（机器人侧，2026-09-18 起）

本节面向在 X2Robot 机器上操作的人，照做即可，不需要读前面的算法部分。完整版在
`examples/embodiment/xrobot-stage2-robot-quickstart.md`。

#### 0.9.1 先拉代码

新的打分键和接管修复都在机器人侧的 `toolkits/inference/xrobot_rlt_ee/`，**不拉代码不会生效**。

```bash
cd /home/xr/lfwj/RealWorld-RLinf
git pull
git log -1 --oneline        # 应为 4c8be6a9 或更新
```

如果这台机器解析不了 github（`ssh: Could not resolve hostname github.com`），让 GPU 侧的人用
`git bundle` 传一份过来，不要去改机器人的 DNS：

```bash
# 在有代码的机器上
git bundle create /tmp/rlt.bundle <机器人当前HEAD>..rlt-server-embodiment
scp /tmp/rlt.bundle <robot>:/tmp/rlt.bundle
# 在机器人上
cd /home/xr/lfwj/RealWorld-RLinf
git fetch /tmp/rlt.bundle rlt-server-embodiment && git merge --ff-only FETCH_HEAD
```

#### 0.9.2 启动顺序（三步，顺序不能换）

GPU 侧先要有 `RLT Stage 2 server ready on 0.0.0.0:8016`。打不通 8016 就先开隧道
（云机 SSH 口是 34133，和 WS 口不是一个），之后上游地址写 `ws://127.0.0.1:8016`：

```bash
ssh -p 34133 -N -L 8016:127.0.0.1:8016 root@<GPU_HOST>
```

```bash
cd /home/xr/lfwj/RealWorld-RLinf

# ① probe：机械臂不动，只验握手和形状，日志里不能有 ProtocolError
RLT_UPSTREAM_URI=ws://127.0.0.1:8016 \
  RLT_TASK_PROMPT="Bimanual usb pick and insert" \
  RLT_EXPLORATION_NOISE_SIGMA=0.1 \
  bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh

# ② V2 事件适配器，必须在开 V2 会话之前
bash toolkits/inference/run_xrobot_rlt_ee_v2_adapter.sh

# ③ 放开真实动作，再开一个没用过的 session id
bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh --stop
RLT_EE_ALLOW_MOTION=true \
  RLT_UPSTREAM_URI=ws://127.0.0.1:8016 \
  RLT_TASK_PROMPT="Bimanual usb pick and insert" \
  RLT_EXPLORATION_NOISE_SIGMA=0.1 \
  bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh

cd /home/xr/lfwj
env -u HISTORY_DRY_RUN MODEL_ADDRESS=127.0.0.1:33057 \
  ./collect/run_interactive_session_v2.sh --session-id v2_YYYYMMDD_NN
```

三个固定值：prompt 必须一字不差 `Bimanual usb pick and insert`；DesktopClient 模型地址
`127.0.0.1:33057`；`RLT_EXPLORATION_NOISE_SIGMA=0.1`（对齐 rlt-openpi 的 `ACTOR_NOISE_SIGMA`，
**不要**去改服务端 YAML 的 0.2，改了离线合同就对不上、`offline_step_40000` 无法 resume）。

#### 0.9.3 按键含义

**注意 V2 和 Cobot 那套是反的：V2 里 `s` 是接管、`f` 是成功。**

| 键 | 含义 | 回合是否结束 | RLT 侧动作 |
|---|---|---|---|
| `r` 第一次 | 暂停 policy，开始物理倒车 | 否 | 当前 chunk 按实测 `/end_pose` 重采样成 `[50,14]`，标 intervention 排队 |
| `r` 第二次 | 停止倒放，收敛到 `PolicyPaused` | 否 | 校验回退健康后发 `rewind_exit`，终止奖励 −1.0，带实测终态 |
| `s` | 进入人工接管（必须在倒车停止后） | 否 | 无。人工遥操那一段不进在线 replay |
| `h` | 交还 policy，开新 policy phase | 否 | 无。policy 恢复后自动接上 |
| `f` | **成功**结束 | 是 | 发 `success`，最后一步写 **+1** |
| `d` | **失败**结束 | 是 | 发 `failure`，服务端在最后一步写 **−1** |
| `p` | 打分：进展 | **否** | 发 `progress`，当前 chunk **+0.5** |
| `o` | 打分：小进展，可连按 | **否** | 发 `small_progress`，当前 chunk 每次 **+0.1**，累加 |
| `x` | 打分：退步 | **否** | 发 `regress`，当前 chunk **−0.5** |
| `y` | 接管暂停时提交当前 chunk 并继续 | **否** | 发 `submit`，把接管回执立刻交给服务端 |
| `q` | 人工 abort，返回码 2 属正常 | 是 | 发 `abort`，丢弃未完成 transition，**不给任何判定** |

几条容易踩的：

- `p` / `o` / `x` / `y` 需要 V2 按 0.9.4 的契约发事件才生效；在那之前用 `operator_cli` 手工发，效果完全一样。
- 打分落在**当前这个 chunk** 上；两个 chunk 之间按的键会留在队列里，落到下一个提交的 chunk。
- `s` / `f` 的终止奖励**优先于**累计打分，不是相加。
- 倒车流程请等第二次 `r` 之后再 `y`，否则 `rewind_exit` 判定会落到下一个 chunk 上。
- 每轮必须用 `f` / `d` / `q` 收尾，否则服务端不训练（训练只在 `episode_end` 发生）。
- `q` 中止仍会按已提交的 actor chunk 跑 UTD。想少更新就少收尾。

#### 0.9.4 V2 要发的事件契约

发到 `/take_over_data`，adapter 会转成对应的 RLT operator 命令。`event_id` 必须唯一（按它去重）：

```json
{"event_id": "...", "event": "session_failed", "detail": {"failure_end": <与 success_end 同结构的终态 sample>}}
{"event_id": "...", "event": "session_progress"}
{"event_id": "...", "event": "session_small_progress"}
{"event_id": "...", "event": "session_regress", "detail": {"reward": -0.25}}
{"event_id": "...", "event": "session_submit"}
```

打分和 `submit` 的 `detail` 可省略；`detail.reward` 用来覆盖默认分值。打分事件**不要求**当前有
pending chunk，两个 chunk 之间按的键会留在队列里。

#### 0.9.5 手工命令（V2 没接好之前全部可用）

```bash
docker exec desktop-robot_client-1 bash -lc \
  'source /opt/xr/py_env/bin/activate && PYTHONPATH=/tmp python -m x2robot_rlt_ee.operator_cli <命令>'
```

可用 `<命令>`：`success`、`failure`、`progress`、`small_progress`、`regress`、`submit`、
`abort`、`rewind_exit`、`rewind_credit`、`sigma`、`status`。
`rewind_*` 用 `--chunks N`（只改 replay 不动机械臂）；打分用 `--reward` 覆盖分值；
`sigma --sigma 0.05` 在线改探索噪声（`--sigma default` 回到服务端的 0.2，`--sigma 0` 关噪声）。

`status` 是联调时最有用的一条，会回 `pending`、`queued_score`、`exploration_noise_sigma`。

#### 0.9.6 三个验收点

1. `operator_cli status` 里 `exploration_noise_sigma` 是 `0.1`。
2. 按一次 `progress`，`status` 的 `queued_score` 变成 0.5；等这个 chunk 提交后，服务端
   `env/episode_reward` 增加而回合**没有**结束。
3. 接管一次：bridge 日志出现 `RLT episode kept open`，服务端**不再**出现接管后的
   `episode_end` + `dropping N pending transitions`。看到了说明 bridge 还是旧版本，回 0.9.1。

在线 replay 未满 `warmup_steps`（250 行）之前，服务端只存不训，日志是
`filling online replay (N/250 rows)`、`warmup_done=false`，这是正常的。

## 1. 目标

本提案的目标是在保留 RLinf/Ray 分布式架构的前提下，使 Cobot 的 RLT Stage 2 在核心语义上与 `rlt-openpi` 的 `remote-franka` 实现一致。

> 已调整：实际方案改为 WebSocket 单进程，不再保留 Ray。见第 0.1 节。

允许 Cobot 与 Franka 使用不同的真机控制协议、通信 SDK、关节布局和安全实现，但以下部分必须保持等价：

1. Stage 1 checkpoint 加载和 reference action 生成。
2. Stage 2 residual actor、EXPO action selection 与 zero-init 行为。
3. rollout、动作执行结果回传、trajectory 构建和 replay ingestion。
4. 人工接管动作对 policy proposal 的替换和标记。
5. rewind/event-only、坏分支 credit patch 和 recovery preference 构造。
6. 回退后的第一条有效、已提交动作，无论来自人工还是 policy，都能作为 positive preference。
7. warmup、UTD、episode-boundary update 和模型权重同步。
8. critic、actor、preference 和 intervention/rewind augmentation 的 loss 语义。
9. checkpoint/resume 后 learner、replay 和真机 controller identity 保持一致。

## 2. 允许不同的部分

Cobot 控制侧只需要满足清晰的 adapter contract，不要求复制 Franka 的控制代码：

- 真实机器人 SDK 和控制消息格式。
- 低层位置、速度、力矩或其他控制模式。
- 人工接管设备及按键映射。
- 物理回退的执行方法。
- 本地 watchdog、限速、碰撞检测和急停实现。

这些差异不得改变上传到 RLinf 的 observation、normalized action、transition identity、intervention、rewind 和 terminal metadata 的语义。

## 3. 目标执行流程

目标链路应为：

```text
Cobot observation
    -> OpenPI 输入映射和 Stage 1 normalization
    -> Stage 1 reference chunks / RLT feature
    -> Stage 2 residual actor 生成 edited candidates
    -> selection critic 在 base + edited candidates 中选择
    -> warmup/switch gate 决定执行 Stage 1 reference 或 Stage 2 action
    -> Cobot adapter 将 normalized mixed action 转成机器人控制指令
    -> adapter 返回实际执行的 normalized action 和 metadata
    -> env worker 只为真实执行且允许记录的完整 chunk 构造 transition
    -> Ray 上传 trajectory/event
    -> learner 去重、写 replay、处理 rewind credit/preference
    -> episode boundary 按新增 transition 的 UTD 预算在线更新
    -> actor 和 selection critic 权重同步到 rollout worker
```

回退流程应为：

```text
坏分支最后一个已提交 chunk
    -> 真机侧在 chunk boundary 产生 rewind event
    -> event-only payload，不生成普通 replay row
    -> learner patch 坏分支 reward/bootstrap
    -> 物理回退或 credit-only 回退
    -> 第一条有效、已提交 recovery chunk
    -> recovery action 作为 positive，bad fork action 作为 negative
    -> preference loss + anchor next-action override
```

## 4. Stage 1 权重基线

> **更新（2026-09-11）：** 两次 Stage 1 SFT 都以 30k 为目标，本节记录的 15k/10k 已不是最新。assemble **已跑完，最终权重为 `global_step_30000`**；mixed 当前到 `global_step_20000` 且仍在训练。三个 assemble checkpoint（15k/25k/30k）的 RLT prefix 长度都是 968。WS 配置现在默认指向 assemble 30k。
>
> 另注意：本节给出的是权重**文件**路径（`.../actor/model_state_dict/full_weights.pt`），而配置项 `rlt_feature_model.model_path` 要填的是 `global_step_*` **目录**本身 —— 加载器会在其下按 `actor/model_state_dict/full_weights.pt` 查找。写到 `model_state_dict` 层会两种布局都不匹配，见第 0.3 节。查当前可用 step：
>
> ```bash
> find /data/gxy/realworldRL/RLinf/logs -maxdepth 5 -type d -name 'global_step_*' \
>   -path '*cobot_*legacy_action_expert_base*' | sort -V
> ```

### 4.1 Assemble 15k

训练会话：

```text
RLT_stage1_Cobot_assemble_franka_legacy_actionexpert_gpu5
```

权重：

```text
/data/gxy/realworldRL/RLinf/logs/20260906-06:01:46-cobot_rlt_stage1_sft_openpi_pi05_assemble_parts_franka_legacy_action_expert_base/cobot_assemble_franka_legacy_actionexpert_base_fp32master_bf16compute_30k/checkpoints/global_step_15000/actor/model_state_dict/full_weights.pt
```

Norm stats：

```text
/data/gxy/realworldRL/checkpoints/assets/cobot_magic/assemble_parts/norm_stats.json
```

该 checkpoint 的关键 RLT 配置为：

```yaml
rlt_prefix_seq_len: 968
rlt_architecture: legacy
rlt_image_only: false
rlt_use_mask: true
```

### 4.2 Mixed cook/cube/pack 10k

训练会话：

```text
RLT_stage1_Cobot_mixed_cook_cube_pack_gpu7
```

权重：

```text
/data/gxy/realworldRL/RLinf/logs/20260907-09:26:30-cobot_rlt_stage1_sft_openpi_pi05_mixed_cook_cube_pack_franka_legacy_action_expert_base/cobot_mixed_cook_cube_pack_franka_legacy_actionexpert_base_fp32master_bf16compute_30k/checkpoints/global_step_10000/actor/model_state_dict/full_weights.pt
```

Norm stats：

```text
/data/gxy/realworldRL/checkpoints/assets/cobot_magic/mixed_cook_cube_pack/norm_stats.json
```

Mixed 实验当前只有 10k，不能作为 assemble 15k 的等价替代，也不能与 assemble 的 norm stats 混用。

> **更新：** mixed 现已到 `global_step_20000`（仍在训练）。跨任务混用 norm stats 的风险已由 preflight 拦住一半 —— `norm_stats_path` 未配置时会静默回退到 `cube_into_drawer` 的分位数，现在这种情况直接拒绝启动。但**指定了一个存在却不匹配的 norm stats 仍然拦不住**，切换 checkpoint 时 `model_path`、`norm_stats_path`、prompt 三项必须一起改。
>
> mixed 是三任务模型，prompt 必须是它训练过的三条之一：`cook vegetable`、`put cube in drawer`、`pack fruit into a container and pour it out`。

## 5. 动作空间说明

OpenPI quantile normalization 使用：

```text
normalized = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
```

因此：

- 模型中心空间是 `[-1, 1]`，不是 `[0, 1]`。
- `q01` 和 `q99` 是 1%/99% 分位点，不是绝对硬边界；模型输出可能超出 `[-1, 1]`。
- Cobot 的两个 gripper 物理量近似在 `[0, 1]`，但经过 quantile normalization 后仍映射到约 `[-1, 1]`。
- 另外 12 个 arm action 是有符号 joint delta，物理空间本来就包含负值。
- normalized action、反归一化后的 robot-space action 和最终硬件控制命令必须作为三个不同空间处理。

Cobot 的目标转换 contract 是：

```text
normalized 14-D mixed action
    -> 使用与 Stage 1 相同的 norm stats 反归一化
    -> 12-D arm delta 基于 chunk-start state 转 absolute target
    -> 2-D gripper 保持 absolute
    -> 按约定频率和时间戳执行
```

人工接管产生的绝对机器人动作必须经过逆变换，转换回同一 normalized delta/absolute mixed representation 后才能写入 replay。

## 6. 当前实现结论

当前实现不能直接进行 Cobot 真机 Stage 2。现有 16 个 Stage 2 alignment unit tests 能通过，但它们没有覆盖真实 checkpoint 加载、Cobot 相机 transform、键盘 switch wrapper、完整 RealWorld trajectory 构建和首批 replay ingestion。

当前 loss 数学主体已经大体对齐 Franka；主要风险集中在 loss 之前的数据路由、动作空间、trajectory 对齐、recovery preference 和训练调度。即使 loss 公式正确，输入数据错误仍会导致训练语义错误。

> **更新（2026-09-11）：** 本节列出的测试盲区已大部分补上 —— 真实 checkpoint 加载、三相机 transform、完整 trajectory 构建与首批 replay ingestion 现在都有覆盖（含真 socket 的整 episode 用例），相关套件现为 64 passed，详见第 14.2 节。键盘 switch wrapper 那项因新架构取消切换键而不再需要。
>
> 本段"风险集中在 loss 之前的数据路由和动作空间"的判断在修复过程中得到了印证：本轮修掉的最高影响问题正是这一类 —— norm stats 静默指向别的任务、prompt 继承了错误默认值、以及 EXPO 选中的 base 没有写回 `ref_chunk` 导致 BC target 对错候选。这三项都不会报错，只会让训练在语义上悄悄跑偏。
>
> 阻塞启动的 P0/P1 已处置。XRobot 路径已跑通一轮闭环。仍建议低速 HIL：真机 adapter 和键盘事件要站点侧实现，第 0.4 节 proprio 尺度尚未对齐，且第 0.5 节列出的算法缺口（尤其 act/backup 拆开、ensemble critic、夹爪绝对输出）会让两边学到的 Q 不是同一个对象。

## 7. P0：阻止启动或破坏基本训练链路的问题

### P0-1 Stage 1 15k checkpoint 与 Stage 2 RLT shape 不兼容

> **状态：已修复。** WS 配置 `rlt_prefix_seq_len: 968`；`preflight.read_rlt_prefix_seq_len()` 用 `mmap` 从 checkpoint 读实际长度（约 0.05 秒），不匹配时报错并打印应填的数字。

Stage 1 使用 `rlt_prefix_seq_len: 968`，当前 Stage 2 配置为 `1024`：

- `logs/...assemble.../tensorboard/config.yaml:30`
- `examples/embodiment/config/cobot_rlt_stage2_td3_mlp.yaml:277`
- `rlinf/models/embodiment/modules/rlt_token_transformer.py:134-136`

`prefix_pos_enc` 参数 shape 依赖该长度。`load_state_dict(..., strict=False)` 不会忽略同名 tensor 的 shape mismatch，所以文档中的 15k 启动命令会在加载阶段失败。

期望修复：Stage 2 配置与 checkpoint 的完整 RLT architecture 参数一致，并在启动前做显式 checkpoint compatibility preflight。

### P0-2 Cobot factory 没有安装 Stage 2 keyboard switch wrapper

> **状态：架构变更后不再适用。** 新架构没有 Stage 2 切换键 —— server 按 replay 行数自行决定 warmup/actor，客户端再加手动开关只会让两边不同步。`b` 现在表示物理回退。旧 Ray 路径的该问题未改动。

`rlinf/envs/realworld/cobot/tasks.py:19-37` 只安装 `RLTInterventionMetadataWrapper`。标准 wrapper stack 中才会根据 `keyboard_reward_wrapper: rlt_policy_switch` 安装 `KeyboardRLTPolicySwitchWrapper`：

- `rlinf/envs/realworld/common/wrappers/apply.py:83-100`

当前结果是 `b` 键无效、`rlt_switch_flags` 缺失，route 会一直使用默认分支。现有操作文档对 `b` 的描述与代码不一致。

期望修复：Cobot factory 复用统一 wrapper builder，或最小化地显式安装 keyboard switch wrapper，并增加真实 factory wrapper-stack 测试。

### P0-3 RealWorld replay 忽略 route 生成的 `record_transition`

> **状态：架构变更后不再适用。** 改为 server 侧 `transition_id` pending map：`act` 签发 id 并挂起观测，`transition` 消费它并恰好写一行，`discard` 消费但不写。取过未执行的 chunk 不可能漏进 replay，重复提交也不可能写两次。旧 Ray 路径未改动。

route 将 actor switch 写入：

- `rlinf/algorithms/rlt/route.py:135-143`

但 env worker/build branch 使用的是 adapter/env info：

- `rlinf/workers/env/env_worker.py:1145-1206`
- `rlinf/algorithms/rlt/transition.py:197-247`

Cobot 对所有非 safety action 默认 `record_transition=True`：

- `rlinf/envs/realworld/cobot/cobot_env.py:223-225`

RealWorld ingestion 又直接添加整条 trajectory：

- `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:649-711`

因此即使没有切换到 Stage 2，Stage 1 reference chunk 也可能写入 replay、计入 warmup并被错误标成 policy action。

期望修复：定义一个唯一的 effective record mask，将 route switch、env safety 和完整 chunk commit 三者合并，并贯穿 builder、trajectory 与 learner ingestion。

### P0-4 RealWorld trajectory 首尾字段长度不一致

> **状态：架构变更后不再适用。** server 只为客户端汇报的已执行 chunk 写行，不存在"末尾多一条未执行 proposal"。截断 chunk 的 reward 由 `_pad_rewards` 零补齐到 chunk 长度。旧 Ray 路径未改动。

当前循环首轮保存 proposal 但没有对应 reward，末尾又保存一条尚未执行的 final proposal，常见结果为：

```text
actions = N + 1
rewards/curr_obs/next_obs/identity/record_transition = N
```

相关代码：

- `rlinf/workers/env/env_worker.py:1096-1211`
- `rlinf/workers/env/env_worker.py:1270-1369`
- `rlinf/data/schema/embodied_trajectory_builder.py:100-145`
- `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:397-434`
- `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:753-780`

learner 按 action 长度 reshape/索引较短的 identity 和 record 字段时可能在第一批 RealWorld 数据上崩溃。

期望修复：只提交已经执行、具有完整 `(s, a, r, s', done, identity, record)` 的行；final bootstrap proposal 不得作为 action row 进入 replay trajectory。

### P0-5 Cobot 相机字段不满足 OpenPI Aloha contract

> **状态：已修复。** `RLTObservationRepacker` 按 `CameraLayout(main, wrist)` 显式产出 `states` / `main_images` / `wrist_images`（后者堆成 Aloha transform 需要的 `[2,H,W,3]`）。缺任何一路直接报错，不补黑帧。`check_camera_layout` 在启动时校验相机数量与 `num_images_in_input` 一致，并有 transform 冒烟测试。

`RealWorldEnv` 只输出：

```text
main_images
extra_view_images
```

但 OpenPI repack 只有在输入存在 `wrist_images` 时才创建 `observation/wrist_image`，而 Aloha input transform 会强制读取该字段：

- `rlinf/envs/realworld/realworld_env.py:245-277`
- `rlinf/models/embodiment/openpi_rlinf/eval_action_model.py:142-174`
- `rlinf/models/embodiment/openpi/policies/aloha_policy.py:196-207`

`extra_image_keys: [wrist_image, side_image]` 只决定 stack 顺序，不能建立 OpenPI 所需的命名 contract。

期望修复：在 RealWorld observation schema 中保留每个相机的语义字段，明确生成 `wrist_images` 以及 side/extra view 对应字段，并增加完整 input-transform smoke test。

### P0-6 normalized action 硬裁剪破坏 zero-init 等价性

> **状态：已修复。** `action_clip_min/max` 配置化，默认 ±1.4 并同步到 TD target smoothing；`cobot_env.py` 里第二处 `[-1,1]` 断言已移除（它会让放宽后的 clip 在 Ray cobot 路径上崩）；ManiSkill 配置显式固定回 ±1.0，因为它的动作空间确实是单位盒。robot-space 限幅只在 adapter 内做。

当前 residual actor：

```text
action = clamp(ref + edit_scale * tanh(residual), -1, 1)
```

位置：

- `rlinf/models/embodiment/mlp_policy/rlt_td3_mlp_policy.py:114-115`
- `rlinf/envs/realworld/cobot/cobot_env.py:212-219`
- `rlinf/envs/realworld/realworld_env.py:317-331`
- `rlinf/envs/realworld/cobot/control.py:288-297`

Franka actor 的默认 normalized clip 为 `[-1.4, 1.4]`：

- `rlt-openpi/exp/stage2_server_shuo.sh:535-537`

当 Stage 1 reference 超过 `[-1,1]` 时，即便 residual head 为零，Cobot Stage 2 输出也会变成 `clamp(ref)`，不再严格等于 Stage 1。

期望修复：把模型空间 clip 配置化并与参考实现一致；robot-space safety clipping 只在 adapter 内进行，不能复用 normalized 模型边界。

## 8. P1：会造成错误数据、错误偏好或不安全运行的问题

### P1-1 回退后的第一条 recovery chunk 丢失

> **状态：架构变更后不再适用。** rewind 走独立请求，不占 transition、不产生额外 replay 行；client 在当前 chunk **提交之后**才处理它，回退完成后的下一条 chunk 按普通 `act`/`transition` 正常提交，因此能成为 preference positive。无论来自人工接管还是 policy 重试都成立。

event-only 后，`env_worker.py:1182-1206` 将 `rlt_pending_obs` 清空。第一条 recovery action 随后会在真机执行，但无法构成 transition，因此不能成为 preference positive。

期望修复：event-only 必须保留回退后的 observation 作为新分支起点，并确保下一条已提交 chunk 构造完整 transition。

### P1-2 人工 recovery 可能污染上一条 bad action

> **状态：架构变更后不再适用。** 不再使用 `update_last_actions()` 这种位置关系。每行由 `transition_id` 绑定到签发它的那次 `act`，executed/intervention 动作只能更新自己那一行。旧 Ray 路径未改动。

`EmbodiedTrajectoryBuilder.update_last_actions()` 会覆盖 builder 中最后一个 action：

- `rlinf/data/schema/embodied_trajectory_builder.py:163-203`

如果第一条 recovery 是人工接管动作，它可能被写到回退前的旧 transition 上，而 observation、reward 和 identity 仍属于旧 transition。

期望修复：executed/intervention action 必须通过 transition identity 更新当前已执行 chunk，不能依赖“builder 最后一条”这种位置关系。

### P1-3 warmup/UTD 更新预算计算错误

> **状态：已修复。** 更新只在 `episode_end` 发生，预算为「本 episode 中 `pending.mode == "actor"` 的 chunk 数 × `utd_ratio`」。warmup 和 eval 的 chunk 照常采集但不换取梯度步，因此不会出现补算 1250 次的情况。已有单元测试覆盖该预算。

当前配置：

```yaml
warmup_min_size: 250
warmup_post_collect_updates: 0
utd_ratio: 5
episode_boundary_only: true
```

但 `_rlt_updates_to_run()` 使用全部历史 transition 计算目标更新数：

- `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:1097-1169`

首次达到 warmup 且遇到 episode boundary 时可能补做约 `250 * 5 = 1250` 次更新。Franka 语义是前 250 条只采集，之后只按新增 online transition 计算 UTD。

期望修复：UTD budget 从 warmup-ready 水位后的新增 transition 开始计算；若要求 warmup 后预训练，应使用独立、显式的配置项。

### P1-4 warmup 未完成时可能使用随机 critic 做 EXPO

> **状态：已修复。** 正是按本条的期望拆成了两个状态：`replay_buffer.total_samples < warmup_steps` 期间 `mode="warmup"`，下发 Stage 1 reference 并关闭 EXPO，但照常写 replay；越过水位后才 `mode="actor"` 启用 residual 与 EXPO 选择。每次 `act` 的响应都带 `mode`，客户端会打进日志。

Simulator route 有 readiness gate，RealWorld route 没有。操作员切换后，即使 replay 未达到 250 条，也可能立即由随机 critic 在候选动作中选择。

期望修复：将“开始记录 Stage 2 数据”和“允许 actor/EXPO 控制真机”拆成两个状态。warmup 期间可以记录，但继续执行 Stage 1 reference；达到 readiness 条件并完成必要更新后才能启用 Stage 2 action selection。

### P1-5 Stage 1 task prompt 与 Stage 2 默认 prompt 不匹配

> **状态：已修复。** 本轮风险最高的一条，且根因比原文描述更广 —— 不只是配置抄错，而是 `pi05_cobot_magic` 的内置默认 prompt 本身就是 `put cube in drawer`，继承默认值就会出错。现在 WS 配置三处统一为 `assemble parts`（`server.task_prompt`、`openpi_data.default_prompt`、client 的 `transport.task`），`check_task_prompt` 同时拒绝占位符和两处不一致。task 随 `RobotObservation` 一起上传，不再依赖静态缓存。mixed checkpoint 需要 episode 级 prompt 时，仍须从它训练过的三条里选，一次运行只跑一条。

Stage 1 assemble prompt 是 `assemble parts`，Stage 2 默认是：

```text
pack fruit into a container and pour it out
```

位置：`examples/embodiment/config/env/cobot.yaml:35`。

此外，`CobotObservation.task` 虽然存在，但 `CobotEnv._to_raw_obs()` 忽略它，`RealWorldEnv` 只缓存静态 `task_description`：

- `rlinf/envs/realworld/cobot/control.py:17-24`
- `rlinf/envs/realworld/cobot/cobot_env.py:132-151`
- `rlinf/envs/realworld/realworld_env.py:108-116`

期望修复：assemble 命令显式设置正确 prompt；mixed checkpoint 支持 episode-level task prompt，并保证 task 与 observation 一起传输。

### P1-6 Stage 2 feature model 没有 fail-fast 要求 RLT checkpoint

> **状态：已修复。** WS 配置显式 `require_rlt_checkpoint: True`，`check_stage1_checkpoint()` 在启动时校验 `rlt_module.*` 存在且 prefix 长度匹配。已在真实 15k/25k/30k 三个 checkpoint 上实测通过，并验证过它会拒绝声称 1024 的配置。

loader 只在 `openpi.task == "rl"` 时默认 `expect_rlt=True`：

- `rlinf/models/embodiment/openpi_rlinf/__init__.py:165-182`

当前 Stage 2 feature model 配置却是 `task: eval`：

- `examples/embodiment/config/cobot_rlt_stage2_td3_mlp.yaml:258-260`

误传普通 SFT checkpoint 时只会告警，RLT module 可能保持随机初始化。

期望修复：Stage 2 配置显式设置 `require_rlt_checkpoint: true`，并验证所有 `rlt_module.*` tensor 都存在且 shape 匹配。

### P1-7 真机漏配 controller 时静默降级为 Mock

> **状态：已修复。** `build_cobot_transport()` 在 `is_dummy=false` 且没有有效 `controller_factory` 时直接抛错，不再返回 Mock。`run_cobot_control.sh` 里也加了 is_dummy 安全提示。

`rlinf/envs/realworld/cobot/cobot_env.py:100-110` 在 `controller_factory` 为空时返回 Mock，即使已经设置 `is_dummy=false`。

期望修复：`is_dummy=false` 且没有有效 factory 时立即报错；启动日志明确输出 adapter class、action schema 和安全模式。

### P1-8 adapter contract 不完整

> **状态：部分修复。** 已定型于 `rlinf/envs/realworld/rlt_client/transport.py`：`ChunkIdentity(episode_id, session_id, env_id, chunk_id)`、`RobotObservation`、`ChunkExecutionResult`、`OperatorEvent` 和 `RLTRobotTransport` Protocol，且 `action_dim`/chunk 长度/proprio 宽度/相机 key 会在握手时校验，不符即拒绝启动。
>
> 混合动作换算不再是 adapter 的责任：`to_robot_space()` 走 OpenPI 完整 `output_transform`（含 `Unnormalize` 与 dataset 的 absolute/Aloha 解码），adapter 直接收到可下发的机器人量；人工动作的逆变换由 `to_normalized_space()` 用仿射逆完成，并在管线非仿射时抛错而不是静默写坏 replay。
>
> 仍缺：`on_action_chunk_begin/end` hook 没有纳入 contract。

当前 Protocol 没有强制规定：

- 12 arm delta + 2 gripper absolute 的 mixed action mask。
- norm stats 标识及 normalized/robot-space round trip。
- chunk-start state、控制频率、时间戳和 partial execution。
- `on_action_chunk_begin/end` hook。
- 人工动作逆变换。
- 每个有效 action 必须返回的 `episode_id/session_id/env_id/chunk_id`。

缺少 identity 时，Ray retry dedup、rewind patch 和 preference pairing 都无法可靠工作。

期望修复：把上述字段纳入正式 adapter contract，并在 env 边界 fail-fast 校验。

### P1-9 controller 状态没有进入 checkpoint/resume 链路

> **状态：部分修复。** identity 冲突这一半已经解决：`session_id` 由 server 进程启动时的 wall-clock 生成，所以即使 resume 后 `episode_id` 从 0 重新计数，也不会与旧 session 的 identity 重合。trainer 侧 `save/load` 覆盖模型、target、两个 optimizer、rewind/schedule 状态和 replay/demo buffer，checkpoint 目录名带上 `_step_<update_step>` 避免 resume 后互相覆盖。
>
> 仍缺：机器人侧 controller 的 session/chunk/history 没有纳入 checkpoint，crash recovery 行为也没有明文规定。

`CobotEnv.controller_state_dict/load_controller_state_dict` 定义于：

- `rlinf/envs/realworld/cobot/cobot_env.py:237-255`

但 runner 的 save/load 只处理 actor/replay/rewind learner state，没有调用 controller 状态接口。恢复后真机 session/chunk/history 可能与 learner 的 dedup 和 rewind state 不一致。

期望修复：在一致的 chunk boundary 保存 controller identity/history，恢复时先校验 robot state，再恢复或开启新 session，并规定 crash recovery 行为。

### P1-10 metadata wrapper 覆盖 adapter 的 bootstrap 语义

> **状态：架构变更后不再适用。** 新路径不经过 wrapper。`ChunkExecutionResult` 直接携带 `terminated`/`truncated`，`bootstrap_mask` 由其推导，adapter 的显式语义不会被改写。旧 Ray 路径的该 wrapper 未改动。

`RLTInterventionMetadataWrapper` 无条件重写：

```python
info["rlt_bootstrap_mask"] = 0.0 if safety_terminal or operator_abort else 1.0
```

位置：`rlinf/envs/realworld/common/wrappers/rlt_intervention_metadata.py:55-65`。

adapter 对自定义不可 bootstrap terminal 返回的 `0` 会被覆盖为 `1`，除非同时设置 wrapper 已知的 safety/operator 字段。

期望修复：尊重 adapter 显式值，仅在字段缺失时推导默认值，并验证 terminal reason 与 bootstrap mask 的组合。

### P1-11 ROS 初始化和异常清理有真机风险

> **状态：已修复。** 按方案一删掉了 `rlinf/envs/realworld/__init__.py` 里的 `RealWorldEnv.realworld_setup()` 调用。`realworld_setup()` 仍可显式调用，但不再作为 import 副作用。`ROSController` 发现已有 roscore 会直接复用，本来就不需要先清场。回归测试 `test_rlt_client_import_does_not_kill_roscore` 会丢掉已缓存模块、按 client 入口重新 import，并断言没有进程被 kill。
>
> 异常路径这一半此前已实现：`RLTRobotLoop` 用 `try/finally` 保证 `transport.stop()`/`close()`，执行期间抛出的异常会先 stop 再向上传播，挂起的 `transition_id` 一定被 `discard`。

导入 realworld package 会清理节点上的 `roscore/rosmaster/rosout`：

- `rlinf/envs/realworld/realworld_env.py:84-106`
- `rlinf/envs/realworld/__init__.py:36`

这可能终止 Cobot 已有 ROS 控制栈。worker 异常退出路径也没有保证调用 adapter `stop()`；键盘 listener 没有可靠的显式关闭流程。

期望修复：禁止 import-time 全局进程清理；只管理当前进程创建且有明确 owner/PID 的资源，并用 `try/finally` 保证 stop/close。

### P1-12 partial chunk 的已执行动作被整块丢弃

> **状态：已修复。** 截断 chunk 照常作为一行提交：reward 数组短于 chunk 长度时由 server 零补齐，`steps_executed` 一并上报。尾部未执行的位置不会被当成"已执行"——client 用最后一个真正执行的动作填充，因为机械臂停下后物理上就停在那里。

发生急停、timeout 或连接中断时，当前逻辑会丢弃整个 partial chunk。已经真实执行的前几步因此没有可追溯记录，影响安全复盘和训练数据一致性。

期望修复：训练 replay 可以排除 partial chunk，但必须单独保存 executed prefix、终止原因、时间戳和 identity；不得把未执行 proposal 当成 executed action。

## 9. P2：与 Franka 的配置和工程差异

这些差异不一定单独导致崩溃，但会让两套 Stage 2 的行为和稳定性明显不同。

下表的"WS 现值"一列为修复后的 `cobot_rlt_stage2_ws_server.yaml` 实际值，已由 server 启动时打印的超参清单核对。

| 项目 | 原 Cobot 值 | Franka 参考值 | WS 现值 | 状态 |
|---|---|---|---|---|
| actor learning rate | `3e-4` | `3e-5` | `3e-5` | 已修复 |
| MLP hidden layers | 默认 2 | 3 | 3 | 已修复 |
| LayerNorm | actor/critic 均无 | actor input LN；critic input + hidden LN | critic 隐层 LN | **部分修复** |
| episode horizon | 480 low-level steps，约 30 chunks | 150 chunks | 150 chunks | 已修复 |
| Stage 2 proprio | raw Cobot state | OpenPI normalized state | raw state | **未修复** |
| action clip | `[-1,1]` | `[-1.4,1.4]` | `[-1.4,1.4]` | 已修复 |

LayerNorm 只做到 critic 隐层：`make_mlp(use_layer_norm=True)` 在每个隐层 `Linear` 后插 `nn.LayerNorm`，但不会在首个 `Linear` 之前对输入做 LN，因此 actor input LN 和 critic input LN 都还没有。它与上面 proprio 那条相关 —— 输入尺度不同、又没有 input LN 吸收，建议一起处理。

其他工程问题：

- **未修复：** `TrajectoryReplayBuffer` 的 trajectory object、dedup set、processed event set 和 rewind row index 长期无界增长。长时间真机 session 仍需关注。
- **未修复：** `auto_save=false` 时 checkpoint 只保存最近窗口的 flat cache 文件，却可能保留全量 index/metadata；恢复后旧 index 可能指向不存在的数据文件。
- **不再适用：** event-only rewind 增加 `_elapsed_steps`。新路径没有 env，horizon 按 chunk 在 server 侧计数，rewind 请求不占 chunk 预算。
- **已修复：** 补齐了三个可执行脚本 —— `run_rlt_stage2_server.sh`（训练/推理侧）、`run_rlt_stage2_client.sh`（真机 client）、`run_cobot_control.sh`（先拉起控制栈再转交 client），职责与 Franka 三个脚本对应。另加 `server.preflight_only` 用于几秒内单独校验 checkpoint/prompt/norm-stats 组合。注意这三个脚本最初都跑不起来（缺 `PYTHONPATH`），见第 0.3 节。
- **已修复：** 操作说明已重写，并改名为 `cobot-stage2-rollout-online-training-guide.md`（原名写死 15k，Stage 1 继续训练后即失效）。删除了 switch 键等未成立的能力声明，补了命令速查和带命令的 11 步首次真机顺序。
- **已修复：** 文档中的 checkpoint 路径已改为在目标机器上实测存在的完整路径。
- **已修复：** 按键语义已明确 —— `b` 物理回退、`q` credit-only 回退、`s`/`f` 成功失败、`escape` 中止；明文说明没有 Stage 2 切换键，避免操作员沿用旧文档或 Franka 习惯。

## 10. Loss 对齐结论

已确认大体一致的部分：

- chunk discounted return。
- twin-Q 和 min target Q。
- Huber critic loss。
- EXPO expected-max target，包含 4 个 base + 4 个 deterministic edited candidates。
- intervention critic augmentation。
- rewind critic augmentation。
- preference hinge rank loss。
- fixed-std preference actor loss。
- pure `-Q` actor objective。
- actor delay 和 target update 主链路。

不要误报以下两项：

- 当前 EXPO target 不是“只有单动作”；代码确实构造了 base 与 edited candidates。
- 当前 actor 并非硬编码只用 Q1；配置路径会使用 min twin-Q。

仍需在数据链路修复后用同一批固定 synthetic transitions 对两仓库做数值对照测试。只有公式相似不足以证明端到端 loss 等价，因为当前 action、reward、bootstrap、identity 和 preference pair 可能在进入 loss 前已经错位。

## 11. 最小修改计划

> **执行情况（2026-09-11）：** 阶段 A 全部完成。阶段 B、C 的条目大多因架构改为 WebSocket 而不再适用（相关 Ray 代码不在新链路上），其目的由 `transition_id` pending map 和"只为已执行 chunk 写行"达成；共享 learner 的 actor Bradley-Terry preference 也已补上。阶段 D 完成 1、2、3（部分）、5，未完成 4（proprio）。阶段 E 完成 1（部分）、4、5、6，未完成 2、3。逐条见第 0.2 节。

### 阶段 A：先恢复可启动性

1. 将 Stage 2 RLT architecture 与 assemble 15k checkpoint 完全对齐，至少修正 `rlt_prefix_seq_len=968`。
2. 设置 `require_rlt_checkpoint: true`，增加 checkpoint shape/preflight 测试。
3. 修复相机语义映射，跑通 Cobot observation 到 OpenPI input 的 CPU smoke test。
4. 真机模式缺少 `controller_factory` 时 fail-fast。
5. 将 model-space clip 配置化，并与 Franka 的 `[-1.4,1.4]` 语义对齐。

### 阶段 B：修复 rollout/replay 正确性

1. 为 Cobot 安装有效的 switch wrapper。
2. 合并 route 与 env 的 effective record mask。
3. 重构 builder，使每个 replay row 只对应一条已经执行的完整 transition。
4. 修复 bootstrap/final proposal 导致的 `N+1/N` 长度问题。
5. 用 identity 绑定 executed/intervention action，禁止通过“最后一条 action”跨分支覆盖。
6. 增加 RealWorld 首批 trajectory ingestion 端到端测试。

### 阶段 C：修复 rewind/recovery/preference

1. event-only 后保存回退完成时的新 observation。
2. 保证第一条人工或 policy recovery chunk 都进入 replay。
3. 验证该 chunk 被标为 `recovery_root` 并生成唯一 preference pair。
4. 验证坏分支 reward/bootstrap patch、anchor override 和 PER priority 同步更新。
5. 对 physical rewind 和 credit-only rewind 分别增加测试。

### 阶段 D：对齐训练调度与模型结构

1. warmup 前只采集并执行 Stage 1 reference，不使用随机 critic 控制真机。
2. UTD 只按 warmup 后新增 transition 计算。
3. 对齐 actor LR、MLP 深度和 LayerNorm，除非实验明确记录偏差原因。
4. Stage 2 proprio 使用与 Franka 等价的 normalized state，或提供可验证的尺度适配。
5. 将 episode horizon 按 chunk 计数并与参考实现对齐。

### 阶段 E：完善真机 contract 和运维交付

1. 定义 Cobot mixed-action conversion、identity、chunk hooks、partial execution 和人工动作逆变换 contract。
2. 将 controller state 接入 checkpoint/resume，或明确规定 resume 必须创建全新 session 并清理不兼容 pending state。
3. 修复 bootstrap metadata 合并策略。
4. 移除 import-time ROS 全局清理，补齐异常 stop/close。
5. 增加与 Franka 三个脚本职责对应的 Cobot 启动脚本，并提供统一 preflight。
6. 最后再更新真机操作说明，删除尚未满足的能力声明。

## 12. 完成条件

只有同时满足以下条件，才可认为本提案完成并允许进入低速真机 HIL：

### Checkpoint 和模型

- Assemble 15k checkpoint 能按实际路径加载，无 missing/mismatched RLT tensor。
- Stage 2 feature/reference 输出可复现 Stage 1。
- residual head zero-init 时，允许范围内的 Stage 2 deterministic action 与 Stage 1 reference 数值一致。
- 超出 quantile 分位区间的 reference 不被错误截到 `[-1,1]`。

### Observation 和 action

- 三路相机按明确语义进入 OpenPI，完整 transform 不出现 `KeyError`。
- 14-D action round trip 测试覆盖 12-D delta 和 2-D absolute gripper。
- policy 与人工 action 都能转换为同一 normalized representation。
- robot-space safety limit 只在 adapter 层生效。

### Rollout 和 replay

- 未切换时执行 Stage 1 且不写 Stage 2 replay。
- 切换并满足 readiness 后才执行 Stage 2/EXPO。
- 每个 replay row 的 action、reward、obs、next_obs、done、record 和 identity 长度完全一致。
- final bootstrap proposal 不进入 replay。
- Ray retry 不会重复提交相同 identity。

### 接管和回退

- 人工接管 action 正确替换对应 proposal，不污染前一条 transition。
- event-only rewind 不产生普通 replay row，也不占 committed chunk ID。
- 回退后的第一条有效人工动作能成为 positive preference。
- 回退后的第一条有效 policy 动作也能成为 positive preference。
- bad fork action 为 negative，reward/bootstrap/anchor override 均 patch 到正确 identity。

### 训练调度和 loss

- 前 250 条有效 transition 只 warmup，不使用随机 critic 接管真机。
- warmup 后每新增 `N` 条 transition 产生 `N * 5` 更新预算，不补算 warmup 数据的 1250 次更新。
- episode boundary 行为与 Franka 一致。
- 固定 synthetic batch 下，两套实现的 critic target、critic loss、actor loss 和 preference loss 在容差内一致。

### 真机安全和恢复

- `is_dummy=false` 且 adapter 缺失时启动失败。
- 断开 Ray、controller exception、keyboard exception 和进程退出都能触发本地 stop/close。
- RLinf 不会杀掉不属于当前进程的 ROS master/control stack。
- checkpoint/resume 后 identity 不冲突，pending rewind/preference 不跨 session 错配。

### 文档和脚本

- Cobot 训练侧、Ray worker 侧和真机控制侧均有可执行脚本。
- 脚本启动前检查 checkpoint、norm stats、task prompt、camera schema、adapter 和 Ray placement。
- 操作说明明确区分 Stage 2 switch、人工 takeover、rewind 和 episode end 信号。
- 文档中的分支、提交、checkpoint 文件路径和命令均在目标机器验证通过。

## 13. 建议测试矩阵

| 测试 | Mock/CPU | GPU | Cobot HIL |
|---|---:|---:|---:|
| 15k checkpoint shape/load | 是 | 是 | 否 |
| 三相机 OpenPI transform | 是 | 是 | 否 |
| normalized action round trip | 是 | 否 | 是 |
| switch/record mask | 是 | 否 | 是 |
| RealWorld trajectory 首尾对齐 | 是 | 否 | 否 |
| Ray retry dedup | 是 | 否 | 否 |
| human intervention replacement | 是 | 否 | 是 |
| physical rewind recovery positive | 是 | 否 | 是 |
| policy rewind recovery positive | 是 | 是 | 是 |
| credit-only rewind patch | 是 | 否 | 否 |
| warmup/readiness/UTD | 是 | 是 | 否 |
| Franka/Cobot loss 数值对照 | 是 | 是 | 否 |
| disconnect/exception stop | 否 | 否 | 是 |
| checkpoint/resume identity | 是 | 是 | 是 |

## 14. 当前验证记录

### 14.1 审查时（2026-09-10）

```bash
cd /data/gxy/realworldRL/RLinf
source .venv/bin/activate
PYTHONPATH=. pytest -q tests/unit_tests/test_rlt_stage2_alignment.py
```

结果：

```text
16 passed
```

该结果只说明现有单元测试通过，不覆盖本提案列出的真实端到端阻塞项。

### 14.2 修复后（2026-09-11）

```bash
.venv/bin/python -m pytest -q \
  tests/unit_tests/test_rlt_stage2_websocket.py \
  tests/unit_tests/test_rlt_stage2_alignment.py \
  tests/unit_tests/test_rlt_proposal_smoke.py \
  tests/unit_tests/test_rlt_client_import_does_not_kill_roscore.py
```

结果：

```text
64 passed
```

其中 `test_rlt_stage2_websocket.py`（35 个）为本轮新增，覆盖协议 round-trip、pending/discard/重复提交、warmup gate 与 UTD 预算、rewind exit/credit 的 preference pair、normalized↔robot 往返、三相机 transform、以及四项 preflight 检查。另有 import-client 不杀 roscore 的回归，以及共享 `forward_actor` 会加上 actor preference 的回归。

关键的一个用例是 `test_full_episode_over_a_real_websocket`：起真 server、连真 client、跑完整 episode。**所有 loopback 测试都绕过了 msgpack 编解码，因此测不出只读数组那一类 bug**（第 0.3 节），这是把真 socket 用例纳入常规测试的原因。

此外在真实 checkpoint 上做过的验证：

```text
run_preflight 在 assemble global_step_30000 上全绿
该 checkpoint 729 个张量全部可加载且有限，prefix_seq_len = 968
15k / 25k / 30k 三个 checkpoint 的 prefix 长度均为 968
声称 1024 的配置会被拒绝，并打印应填的数字
norm_stats_path 未配置 / prompt 与 Stage 1 不一致 / model_path 多写一层，三种误配均被拦下
server.preflight_only 约 6 秒返回，退出码 0
CPU 上用真实 server 配置构建 RLTStage2Trainer：写入 6 条 transition、跑 3 次更新、save/load 往返通过
ManiSkill 配置的 action clip 仍为 (-1.0, 1.0)
```

### 14.3 2026-09-11 结论

P0 六条已全部处置（三条修复、三条因架构变更不再适用），P1 十二条中八条修复、三条不再适用、两条部分修复。

通信和在线更新数据流（握手、八种请求、pending map、warmup/actor、`episode_end` 同进程更新、rewind pair + critic/actor preference）已对齐。仍建议先低速 HIL：真机 adapter / 操作员事件要站点实现；第 0.4 节 proprio 仍是 raw state。

第 12 节当时尚未满足的是：Stage 2 proprio 与参考实现等价（P2）；controller state 的 resume 语义（P1-9 后半）；以及两套实现在固定 synthetic batch 上的 loss 数值对照。`RLinf 不会杀掉不属于当前进程的 ROS 控制栈` 已满足。

### 14.4 2026-09-15 算法落地后

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/unit_tests/test_rlt_algo_migration.py \
  tests/unit_tests/test_rlt_stage2_websocket.py \
  tests/unit_tests/test_rlt_stage2_alignment.py \
  tests/unit_tests/test_rlt_proposal_smoke.py \
  tests/unit_tests/test_cobot_offline.py \
  tests/unit_tests/test_cobot_stage2_launcher.py \
  tests/unit_tests/test_rlt_embodiment.py \
  tests/unit_tests/test_rlt_stage1_eval.py \
  tests/unit_tests/test_verify_rlt_stage2_training.py
```

结果：`133 passed`。新增 `test_rlt_algo_migration.py` 锁 delayed Polyak、disjoint ensemble、`expo_decoupled` 乐观间隙、inward clip、夹爪绝对输出、rank slope、intervention local probes、TD clip。websocket 用例锁 eval 写回 / 不写回，以及 `expo_decoupled` 在 2 头上被 preflight 拒绝。

Hydra compose `cobot_rlt_stage2_ws_server` 与 `xrobot_ee_rlt_stage2_ws_server` 后，`check_stage2_algorithm` 通过：`backup=expo_decoupled`，`critic_num_qs=10`，`residual_scale=0.4`，`eval_interval_episodes=5`，`store_eval_episodes=True`。

**结论：** WS 路径的目标算法与 `rlt-openpi@f80a804` 对齐；机型走 `EmbodimentProfile`。未决项见 0.6。A–E 工作区改动尚未提交。不能 resume 旧 Twin-Q + `residual_scale=0.2` 目录。
