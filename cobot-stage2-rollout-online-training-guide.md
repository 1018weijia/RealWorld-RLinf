# Cobot Stage 2 Rollout、接管与在线训练操作说明

适用代码：`RLinf` 的 `rlt` 分支，WebSocket 单进程架构。

本文面向两类人员：

- GPU/训练侧：启动 Stage 2 server，加载最新 Stage 1 权重，跑在线训练。
- Cobot 真机侧：实现 transport，启动本地控制程序和 client，执行 rollout、接管和回退。

当前推荐的 Stage 1 权重是 assemble **30k**（该次 SFT 的最终权重），见第 2 节。

## 0. 命令速查

按顺序执行。前四条不需要机械臂，第 1、2 条连 GPU 都不需要。

```bash
cd /data/gxy/realworldRL/RLinf

# 1. 协议/循环冒烟：真 WebSocket + 假模型，跑完一整个 episode
.venv/bin/python -m pytest tests/unit_tests/test_rlt_stage2_websocket.py -q

# 2. 列出可用的 Stage 1 checkpoint，每组最后一行是该 run 最新的 step
find /data/gxy/realworldRL/RLinf/logs -maxdepth 5 -type d -name 'global_step_*' \
  -path '*cobot_*legacy_action_expert_base*' | sort -V

# 3. 只跑启动检查，不加载 Stage 1、不开端口（约 6 秒）
bash examples/embodiment/run_rlt_stage2_server.sh cobot_rlt_stage2_ws_server \
  server.preflight_only=True

# 4. 启动 server（加载 16 GB Stage 1，需要 GPU，之后常驻）
bash examples/embodiment/run_rlt_stage2_server.sh cobot_rlt_stage2_ws_server \
  server.host=0.0.0.0 server.port=8000

# 5. 空跑 client：机械臂不动、相机全黑，只验证握手和链路
bash examples/embodiment/run_rlt_stage2_client.sh cobot_rlt_stage2_ws_client \
  client.host=<GPU_HEAD_IP> client.port=8000 client.num_episodes=1

# 6. 真机 client（在接机械臂的那台机器上跑）
bash examples/embodiment/run_cobot_control.sh cobot_rlt_stage2_ws_client \
  client.host=<GPU_HEAD_IP> client.port=8000 \
  transport.is_dummy=false \
  transport.controller_factory=your_package.controller:create_adapter \
  transport.task="assemble parts"
```

第 5 条用的是 `run_rlt_stage2_client.sh`（不拉起控制栈），第 6 条用的是 `run_cobot_control.sh`（先拉起 ROS/驱动/相机再转交 client）。两个脚本吃同一份 config。

各步的含义、需要确认什么、以及换 checkpoint 时要一起改哪几项，见下面对应章节。

## 1. 架构与边界

Stage 2 不使用 Ray。整套系统只有两个进程：

```text
GPU 节点                                    Cobot 节点
┌────────────────────────────┐             ┌─────────────────────────┐
│ rlt_stage2_server.py       │  WebSocket  │ rlt_stage2_client.py    │
│  冻结 Stage 1 VLA          │ ◄─────────► │  RLTRobotLoop           │
│  Stage 2 actor/critic      │             │  CobotTransport         │
│  replay + optimizer        │             │  你的 control adapter   │
└────────────────────────────┘             └─────────────────────────┘
```

职责划分：

1. **server 拥有全部模型和数据**：Stage 1 归一化、OpenPI input transform、reference chunk、EXPO 选择、反归一化到机器人空间、replay 写入、梯度更新，全部在 server 内完成。它既是推理进程也是训练进程，因此没有权重同步这一步。
2. **client 不加载任何模型，也不需要 GPU**：它上传原始相机图 + state + prompt，执行 server 返回的机器人空间 chunk，并汇报实际发生了什么。
3. **真机侧独占安全职责**：限幅、watchdog、急停、rewind 回放必须在 adapter 内本地实现，且在 learner 断连时仍然可用。
4. 默认配置 `transport.is_dummy: true`，跑的是 `MockRewindAdapter`：机械臂不动，所有相机帧全黑。真机运行必须显式设 `is_dummy=false` 并提供 `controller_factory`，否则 client 直接拒绝启动。

## 2. Stage 1 权重和 norm stats

Stage 1 checkpoint 放在 `rlt_feature_model.model_path`。`actor.model` 是 Stage 2 的 MLP head，不要往里填 Stage 1 路径。

两个 Stage 1 任务都以 30k 步为目标。assemble 已经跑完（`global_step_30000` 是最终权重），**mixed 还在跑**，所以用 mixed 时先查一下现在最新的是哪一步，不要照抄本文的数字：

```bash
find /data/gxy/realworldRL/RLinf/logs -maxdepth 5 -type d -name 'global_step_*' \
  -path '*cobot_*legacy_action_expert_base*' | sort -V
```

`sort -V` 会按数字后缀正确排序并按 run 分组，每组最后一行就是该 run 当前最新的 step。截至本文修订，assemble 到 **30000**（已完成），mixed 到 **20000**。

挑到新 step 之后确认它写完了再用 —— checkpoint 有 16 GB，正在写入的文件字节数会偏小。preflight 会读实际张量，读得通就是完整的。

### 2.1 Assemble / legacy action expert（推荐，30k 最终权重）

```text
model_path:
/data/gxy/realworldRL/RLinf/logs/20260906-06:01:46-cobot_rlt_stage1_sft_openpi_pi05_assemble_parts_franka_legacy_action_expert_base/cobot_assemble_franka_legacy_actionexpert_base_fp32master_bf16compute_30k/checkpoints/global_step_30000

norm_stats_path:
/data/gxy/realworldRL/checkpoints/assets/cobot_magic/assemble_parts/norm_stats.json

task_prompt: "assemble parts"
```

### 2.2 Mixed cook/cube/pack（备选，20k）

```text
model_path:
/data/gxy/realworldRL/RLinf/logs/20260907-09:26:30-cobot_rlt_stage1_sft_openpi_pi05_mixed_cook_cube_pack_franka_legacy_action_expert_base/cobot_mixed_cook_cube_pack_franka_legacy_actionexpert_base_fp32master_bf16compute_30k/checkpoints/global_step_20000

norm_stats_path:
/data/gxy/realworldRL/checkpoints/assets/cobot_magic/mixed_cook_cube_pack/norm_stats.json

task_prompt: "cook vegetable" / "put cube in drawer" /
             "pack fruit into a container and pour it out"
```

mixed 是三任务模型，prompt 必须是训练时用过的三条之一，一次运行只跑一条。

### 2.3 三个容易写错的点

**`model_path` 指向 `global_step_*` 本身，不要指到 `actor/model_state_dict`。** 加载器只认 `<model_path>/model_state_dict/full_weights.pt` 和 `<model_path>/actor/model_state_dict/full_weights.pt` 两种布局；多写一层会两个都不匹配，然后掉进 safetensors 分支，在启动看起来一切正常之后才报错。preflight 现在会当场拦下并告诉你该往上退一级。

**`norm_stats_path` 必须显式配置，它不是可选项。** 不填的话 OpenPI 会拿 `model_path` 当 assets 目录（那里只有 `full_weights.pt`），并回退到 `pi05_cobot_magic` 内置的默认 `asset_id`，也就是 `cobot_magic/cube_into_drawer` —— **另一个任务的分位数**。反归一化会静默地用错统计量，直接把错误的关节量下发到机械臂。preflight 现在拒绝在未配置时启动。

**`task_prompt` 必须是 Stage 1 训练时用的那条。** 同样地，`pi05_cobot_magic` 的内置默认 prompt 是 `"put cube in drawer"`，而 assemble 那次 Stage 1 训练用的是 `"assemble parts"`。冻结的 VLA 是 prompt 条件化的，写错不会报错，只会产出自信但错误的动作。preflight 会校验 `server.task_prompt` 与 `openpi_data.default_prompt` 一致，两者必须同时改。

不要跨任务复用 norm stats。Stage 1 模型、OpenPI `config_name`、图像数量、`action_dim` 和 state schema 必须一致。

### 2.4 关于 `rlt_prefix_seq_len = 968`

目前 assemble 和 mixed 的所有 checkpoint 都是 **968**，不是 OpenPI 默认的 1024。这个数字是图文 prefix 的 token 数：

```text
3 相机 × 256 patch + max_token_len 200 = 968
```

所以它由 `num_images_in_input` 和 `max_token_len` 决定，跟 action horizon 无关。改相机数量或 `max_token_len` 就得跟着改。preflight 会从 checkpoint 里读出 `rlt_module.encoder.prefix_pos_enc` 的实际长度，不匹配时报错并打印应该填的数字。

## 3. 真机侧需要提供什么

client 通过 `CobotTransport` 调用你的 adapter。它是既有的 `CobotControlAdapter` 协议，逐步执行：

```python
class MyCobotAdapter:
    action_dim = 14

    def reset(self) -> CobotObservation: ...
    def observe(self) -> CobotObservation: ...
    def execute(self, action) -> CobotStepResult: ...
    def stop(self, reason: str) -> None: ...
    def close(self) -> None: ...

    # 物理回退需要这两个；只做 credit-only 回退可以省略
    def poll_rewind_event(self) -> CobotRewindEvent | OperatorEvent | None: ...
    def rewind_chunks(self, count: int) -> CobotObservation: ...
```

与旧版本的三个区别：

**动作是机器人空间的，不是 `[-1, 1]` 归一化的。** server 已经跑完 OpenPI 的反归一化，`execute()` 收到的就是可以直接下发的关节量。相应地 `validate_action` 不再强制 `[-1, 1]`：OpenPI 的分位数归一化把 q01/q99 而不是极值映射到 ±1，合法动作本来就可以越界。

**`observe()` 必须提供三路相机。** key 为 `image`（cam_high）、`wrist_image`（左腕）、`side_image`（右腕），顺序不能换 —— server 按这个顺序堆成 OpenPI Aloha transform 需要的 `[2, H, W, 3]` 腕部张量，交换两项等于交换两条手臂的视角。缺任何一路，client 直接报错而不是补黑帧。

**`poll_rewind_event()` 可以返回 `OperatorEvent`。** `CobotRewindEvent` 只能表达回退，而键盘还要产出成功、失败、中止。返回 `OperatorEvent(kind="success")` 等即可，transport 原样透传。

`execute()` 的返回里，`info["executed_action"]` 必须是**真正发给机器人的动作**。当它与 server 下发的值不同（人工接管、本地限幅），client 会把整条 chunk 回传给 server，由 server 归一化后写入 replay。人工接管时另外置 `info["human_intervention"] = True`。

## 4. 启动

不需要 `ray start`。两个进程独立启动，先 server 后 client（client 会重试连接，早启动也没问题）。

### 4.1 GPU 节点：启动 server

`cobot_rlt_stage2_ws_server.yaml` 里已经填好了 2.1 的 assemble 30k 路径、norm stats 和 prompt，所以默认情况直接：

```bash
cd /data/gxy/realworldRL/RLinf
bash examples/embodiment/run_rlt_stage2_server.sh cobot_rlt_stage2_ws_server \
  server.host=0.0.0.0 server.port=8000
```

换成别的 checkpoint 时，三项要一起改，不能只改路径：

```bash
bash examples/embodiment/run_rlt_stage2_server.sh cobot_rlt_stage2_ws_server \
  rlt_feature_model.model_path=<.../checkpoints/global_step_XXXXX> \
  rlt_feature_model.openpi_data.norm_stats_path=<.../norm_stats.json> \
  rlt_feature_model.openpi_data.default_prompt="<stage1 prompt>" \
  server.task_prompt="<stage1 prompt>" \
  server.host=0.0.0.0 server.port=8000
```

启动时 preflight 会依次检查：`model_path` 下能否按加载器的两种布局找到 `full_weights.pt`、checkpoint 里 `rlt_module.encoder.prefix_pos_enc` 的实际长度与 `rlt_prefix_seq_len` 是否一致、`rlt_module.*` 是否存在（`require_rlt_checkpoint: true` 时缺失即报错）、`norm_stats_path` 是否配置且可读、task prompt 是否是占位符或与 Stage 1 prompt 不一致、相机数量是否与 `num_images_in_input` 匹配。任何一项失败都在机械臂上电之前终止。全绿时的输出形如：

```text
Preflight: Stage 1 RLT prefix_seq_len=968 matches .../global_step_30000/actor/model_state_dict/full_weights.pt
Preflight: norm stats loaded from .../cobot_magic/assemble_parts/norm_stats.json
Preflight: task prompt = 'assemble parts'
Preflight: camera layout = ['image', 'wrist_image', 'side_image']
```

随后 server 打印全部关键超参，并对每个偏离 remote-franka 参考部署的值发一条警告。警告不阻塞启动 —— 有意的 sweep 应该跑得起来 —— 但意外的偏差（比如 actor 学习率差一个数量级）在几小时的真机运行里几乎不可能靠肉眼发现。

### 4.2 Cobot 节点：启动控制栈和 client

```bash
cd /data/gxy/realworldRL/RLinf
bash examples/embodiment/run_cobot_control.sh cobot_rlt_stage2_ws_client \
  client.host=<GPU_HEAD_IP> client.port=8000 \
  transport.is_dummy=false \
  transport.controller_factory=your_package.controller:create_adapter \
  transport.task="assemble parts"
```

`transport.task` 必须与 server 的 `task_prompt` 一致。

`run_cobot_control.sh` 里预留了站点相关的 ROS/驱动/相机 bring-up 段落，填好之后它会把 adapter 起好再转交给 `run_rlt_stage2_client.sh`。只想跑通链路不动机械臂时，去掉 `transport.is_dummy=false` 即可。

client 连上后先校验握手：`action_dim`、chunk 长度、proprio 宽度、相机 key 有任何一项与 transport 不符就拒绝启动。这把本来会在半个 episode 之后表现为奇怪动作的形状错误，变成了启动失败。

### 4.3 Stage 2 关键默认值

| 项目 | 值 |
|---|---:|
| action_dim | 14 |
| Stage 2 chunk | 16 steps |
| Stage 1 reference horizon | 20 steps |
| residual scale | 0.2 |
| residual clip | ±1.4 |
| EXPO | 4 个 base + 4 个 edited candidate |
| actor MLP | 3 隐层，critic 隐层带 LayerNorm |
| actor lr / critic lr | 3e-5 / 3e-4 |
| gamma | 0.99 |
| critic | twin-Q Huber，delta=0.5 |
| replay warmup | 250 行 |
| UTD | 每条 policy chunk 5 次更新 |
| target tau | 0.005 |
| 每 episode chunk 上限 | 150 |

Stage 2 actor 输出为：

```text
action = clip(ref_chunk + 0.2 * tanh(delta), -1.4, 1.4)
```

clip 边界是 ±1.4 而不是 ±1.0：分位数归一化下合法动作可以超出单位盒，卡在 1.0 会连同它加在上面的 Stage 1 reference 一起削掉，直接删掉一部分可达动作。

residual 输出层零初始化，所以刚启动时 Stage 2 行为应与 Stage 1 reference 一致。首次真机运行必须先验证这一点。

### 4.4 已知注意点：reference horizon 20 vs Stage 1 的 50

`pi05_cobot_magic` 的默认 `action_horizon` 是 50，两次 Stage 1 SFT 都没有覆盖它，所以它们是按 **50 步** action block 训练的。而 Stage 2 把推理时的 horizon 覆盖成 **20**（`rlt_feature_model.num_action_chunks`），执行前 16 步。

这不是权重形状问题 —— checkpoint 里没有任何按 horizon 定长的张量，20 和 50 都能正常加载，`968` 也只跟图文 prefix 有关。但 action expert 在训练时始终看到 50 个 action token，推理时只给 20 个，属于 attention 序列长度上的 train/inference 偏移，可能让 reference chunk 的质量比 Stage 1 评估时差一些。

这个 16/20 设置是从既有的 Ray 版 `cobot_rlt_stage2_td3_mlp.yaml` 继承的（那边的数值最初来自 `pi05_franka_state`，它的 Stage 1 horizon 本来就是 20），WebSocket 配置只是保持一致，并非新引入。第 11 节的 dry run 会暴露它：如果 zero-init 的 Stage 2 输出与直接跑 Stage 1（horizon 50）的动作明显不同，就把 `rlt_feature_model.num_action_chunks` 和 `actor.model.ref_num_action_chunks` 一起改成 50 再比一次。

## 5. Rollout：warmup 由 server 自动切换

**没有 Stage 2 切换键。** 旧版本的 `b` 键切换已经删除：server 自己按 replay 行数决定用哪个策略，客户端再加一个手动开关只会让两边不同步。

```text
replay < 250 行 → mode="warmup"，下发 Stage 1 reference chunk，不跑 EXPO
replay ≥ 250 行 → mode="actor"，跑 EXPO 选择，residual 生效
```

每次 `act` 的响应里都带 `mode` 字段，client 会打进日志，所以当前处于哪个阶段一眼可见。warmup 期间照常写 replay —— 收集数据正是它的目的 —— 只是不产生梯度步。

每个 chunk 的时序是固定的四步：机械臂静止时采观测 → 向 server 要 chunk → 执行 → 汇报结果。server 在 `act` 时签发一个 `transition_id` 并把观测挂起，`transition` 消费它并恰好写一行 replay，`discard` 消费它但什么都不写。取过但没执行的 chunk 因此不可能漏进 replay，重复提交的 chunk 也不可能写两次。

## 6. 人工接管

接管在本地 teleop 里实现。被人控制的每个 low-level step 返回：

```python
info["human_intervention"] = True
info["executed_action"] = actual_human_action   # 机器人空间
```

client 检测到实际动作与下发动作不一致时，把整条 chunk 连同 `intervention=True` 一起回传。server 会把它归一化后写入 replay，并标记为 human action source，进入 intervention sampling partition（训练时约 50% online + 50% intervention，intervention 不足时回退到全量 replay）。

真机侧不需要另外上报一份"接管动作日志"。

## 7. 回退 / rewind

两种回退都由本地触发，在 chunk 边界排队一个事件：

| 按键 | 事件 | 机械臂 | 效果 |
|---|---|---|---|
| `b` | `rewind_exit` | 物理倒放 N 个 chunk | 坏分支末尾写终止惩罚、切断 bootstrap，并设置 fork anchor 生成 preference pair |
| `q` | `rewind_credit` | 不动 | 只 patch replay 的 credit |

事件走独立的请求，不占用 transition，也不会产生额外的 replay 行。运行时顺序：

1. 操作员在本地触发 rewind，`poll_rewind_event()` 返回一次事件。
2. client 在当前 chunk **提交之后**处理它 —— 回退 patch 的是已经存在的行，顺序反了就无行可 patch。
3. `rewind_exit` 时 client 先调 `rewind_chunks(N)` 让机械臂退回去，再发请求；adapter 没实现物理回退时自动降级为 credit-only 并告警。
4. server 把坏分支最后一行写入负终止奖励并切断 bootstrap。
5. 回退后第一条有效、已提交的 chunk 成为 recovery positive，可以来自人工接管，也可以来自 policy 自己重试成功。
6. server 以 bad fork action 为 negative 构造 preference pair，并 patch anchor 的 next-action override。

回退请求的 `chunks_rewound` 超过 adapter 保留的历史长度时会被截断并告警。

## 8. 操作员按键

```text
s        标记本 episode 成功，终止奖励记在当前 chunk 上，episode 结束
f        标记失败，episode 结束
b        物理回退：把机械臂退出坏分支
q        credit-only 回退：只改 replay，机械臂不动
escape   停止机械臂并结束 episode，不给判定
```

成功/失败的判定必须在观察完当前 chunk 之后给出，所以 client 在**提交这条 transition 之前**把终止奖励折进它的最后一步 reward，同时置 `terminated=True` 切断 bootstrap —— 任务已经解决，没有未来回报可估。一旦这行写进 replay，再改就只能靠 credit 回退了。

## 9. 安全故障和中止

安全故障由 adapter 本地处理并返回：

```python
info = {
    "rlt_safety_fault": True,
    "executed_action": actual_safe_action,
}
```

`stop_on_safety_fault: true`（默认）时 client 立即中断剩余 low-level step，把已执行的部分作为截断 chunk 提交（reward 数组短于 chunk 长度，server 按零补齐），然后结束 episode。

截断 chunk 里没跑到的尾部不会被当成"已执行"上报：client 用最后一个真正执行的动作填充剩余位置，因为机械臂停下后物理上就停在那里。

异常路径有两道保证：任何在执行期间抛出的异常都会先 `transport.stop()` 再向上传播，而挂起的 `transition_id` 一定会被 `discard` 掉。断开 WebSocket 不影响真机侧急停和 stop。

## 10. 在线训练何时发生

真机侧不发送显式"update once"信号。训练发生在 `episode_end`：

```text
本 episode 中 mode=="actor" 的 chunk 数 × utd_ratio = 本次更新步数
```

只数 policy 驱动的 chunk。warmup 和 eval 的 chunk 照常收集数据但不换取梯度步，否则 warmup 阶段会把实际 UTD 抬到配置值以上。训练在请求锁内同步执行，这期间 server 不响应新请求 —— 机械臂本来就是停走式的，`ping_timeout` 默认给到 600 秒就是为了让 keepalive 熬过训练突发。

checkpoint 按 `server.save_interval_episodes` 落在 `server.save_dir/episode_<N>_step_<update_step>/`，包含模型、target、两个 optimizer、rewind/schedule 状态和 replay/demo buffer。目录名带上 update step 是为了让恢复后再存不会覆盖掉恢复前同编号的 episode。恢复用 `runner.resume_dir` 指向该目录。

## 11. 推荐的首次真机顺序

**第 1 步：不碰 GPU、不碰机械臂，先验协议和循环。**

```bash
.venv/bin/python -m pytest tests/unit_tests/test_rlt_stage2_websocket.py -q
```

其中 `test_full_episode_over_a_real_websocket` 会起一个真 WebSocket server、连一个真 client，用假模型跑完一整个 episode，覆盖 warmup、UTD 预算、成功判定和 rewind。这一步失败就不用往下走了。

**第 2 步：确认 checkpoint 组合是对的。**

```bash
bash examples/embodiment/run_rlt_stage2_server.sh cobot_rlt_stage2_ws_server \
  server.preflight_only=True
```

约 6 秒返回，不加载 Stage 1 也不开端口。要确认的是：prefix_seq_len 匹配、**norm stats 那行指向的是本任务**（不是 `cube_into_drawer`）、prompt 是 Stage 1 训练用的那条、相机是三路，以及超参警告里没有意料之外的项。

**第 3 步：启动 server，让它常驻。**

```bash
bash examples/embodiment/run_rlt_stage2_server.sh cobot_rlt_stage2_ws_server \
  server.host=0.0.0.0 server.port=8000
```

等到出现 `RLT Stage 2 server ready on 0.0.0.0:8000 (warmup 250 rows, utd 5)` 再继续。

**第 4 步：空跑 client，验证握手。**

```bash
bash examples/embodiment/run_rlt_stage2_client.sh cobot_rlt_stage2_ws_client \
  client.host=<GPU_HEAD_IP> client.port=8000 client.num_episodes=1
```

默认 `is_dummy: true`，机械臂不动、相机全黑。要看到握手通过、每个 chunk 打出 `mode=warmup`、`episode_end` 正常返回。这一步验证的是网络和形状约定，不验证策略质量（输入是黑图，动作没有意义）。

**第 5 步：校验你自己的 adapter。** observation 三路相机、state 宽度、动作范围、watchdog、stop、fault、rewind。这部分没有通用命令，取决于你的控制栈。

**第 6 步：启动本地控制栈，先禁止真实动作或用最低速度/最小限幅。**

```bash
bash examples/embodiment/run_cobot_control.sh cobot_rlt_stage2_ws_client \
  client.host=<GPU_HEAD_IP> client.port=8000 \
  transport.is_dummy=false \
  transport.controller_factory=your_package.controller:create_adapter \
  transport.task="assemble parts"
```

**第 7 步：Dry run。** 确认 Stage 2 zero-init 输出与 Stage 1 reference 一致（residual 输出层零初始化，两者应当逐元素相同）；同时按 4.4 比一次 horizon 20 与 50 的差异。

**第 8 步：单条低速 chunk。** 确认每步返回的 `executed_action` shape 和量纲正确。

**第 9 步：分别实测**正常 rollout、人工接管、物理回退后人工恢复、物理回退后 policy 恢复。

**第 10 步：跑过 warmup。** 累计 replay 超过 250 行后，确认 `act` 响应里的 `mode` 从 `warmup` 变成 `actor`，且 `episode_end` 开始出现 critic/actor update 日志。

**第 11 步：验证恢复。** 确认 checkpoint 落盘：

```bash
ls ../results/cobot_rlt_stage2_ws/checkpoints/
# episode_10_step_<N>/  里应有 stage2_state.pt、replay_buffer/、demo_buffer/
```

然后停掉 server，指向该目录重启一次，确认日志里出现 `Resuming Stage 2 state from ...` 且 replay 行数接上了：

```bash
bash examples/embodiment/run_rlt_stage2_server.sh cobot_rlt_stage2_ws_server \
  server.host=0.0.0.0 server.port=8000 \
  runner.resume_dir=../results/cobot_rlt_stage2_ws/checkpoints/episode_10_step_<N>
```

## 12. 运行中应观察的日志/指标

server 侧：

```text
每条请求一行：类型、耗时、replay 行数、pending 数、chunk_id
env/episode_reward, env/episode_chunks, env/episode_train_chunks
env/episode_interventions, env/episode_success
rlt/updates_this_episode
train/critic_loss, train/actor_loss
```

client 侧：

```text
每个 chunk 一行：chunk_id、mode、执行步数、reward、耗时
episode 结束一行：chunks、reward、success、server 跑的更新步数
```

`pending` 长期不为 0 说明有 chunk 既没提交也没丢弃，是 client 循环的 bug；正常情况下它在两次请求之间才短暂为 1。
