# XRobot Stage 2 —— 机器人侧速查

面向在 X2Robot 机器上操作的人。GPU 侧看
[xrobot-stage2-gpu-quickstart.md](xrobot-stage2-gpu-quickstart.md)。

RLT bridge 不发布任何 ROS 命令、不碰控制器。它只把 DesktopClient 的观测转成 RLT
协议，再把 server 返回的 chunk 转回 DesktopClient。物理动作、接管、回退全部由现有
的 V2 流程负责。

## 0. 先记住这个

**V2 的按键和 Cobot 那套是反的。**

```text
Cobot RLT:  s = 成功    f = 失败
V2:         s = 接管    f = 成功
```

套环连 `ws://<GPU_IP>:8000`、任务 `put ring on the rod`。USB 按下面这一节做。

## USB 客户端（当前任务）

GPU 侧 `0.0.0.0:8016` 出现 `server ready` 就可以连。中间隔几小时没有问题：没有
`episode_end` 就不会训练。USB 已 resume Cal-QL，第一回合就是 residual，不要用套环
第 8 节「必须与 Stage 1 逐元素相同」当验收。

Prompt 必须一字不差：`Bimanual usb pick and insert`。DesktopClient 模型地址填
`127.0.0.1:33057`。

**连不上 8016 时先打隧道**（云机 SSH 口往往不是 8016）：

```bash
ssh -p 34133 -N -L 8016:127.0.0.1:8016 root@<GPU_HOST>
```

之后把下面的 `<GPU_IP>` 换成 `127.0.0.1`。能直连 8016 就写真实 GPU IP。

在机器人仓库根目录 `/home/xr/lfwj/RealWorld-RLinf` 按顺序做：

**① Probe（机械臂不动）**

```bash
cd /home/xr/lfwj/RealWorld-RLinf
RLT_UPSTREAM_URI=ws://<GPU_IP>:8016 \
  RLT_TASK_PROMPT="Bimanual usb pick and insert" \
  RLT_EXPLORATION_NOISE_SIGMA=0.1 \
  bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh
```

`0.1` 对齐 rlt-openpi `stage2_server_shuo_rlinf.sh` 的 `ACTOR_NOISE_SIGMA`。
服务端 YAML 保持 `0.2` 不动，否则离线合同对不上、`offline_step_40000` 无法 resume。

X2Robot 模型地址填 `127.0.0.1:33057`，跑一轮。只验证握手和形状。日志里不能有
`ProtocolError`：

```bash
bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh --status
```

**② V2 事件适配器（必须在 V2 会话之前）**

```bash
bash toolkits/inference/run_xrobot_rlt_ee_v2_adapter.sh
```

**③ 放动作，再开一个没用过的 V2 session**

```bash
bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh --stop
RLT_EE_ALLOW_MOTION=true \
  RLT_UPSTREAM_URI=ws://<GPU_IP>:8016 \
  RLT_TASK_PROMPT="Bimanual usb pick and insert" \
  RLT_EXPLORATION_NOISE_SIGMA=0.1 \
  bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh

cd /home/xr/lfwj
env -u HISTORY_DRY_RUN MODEL_ADDRESS=127.0.0.1:33057 \
  ./collect/run_interactive_session_v2.sh --session-id v2_YYYYMMDD_NN
```

V2：`s` 接管，`f` 成功，`d` 失败（V2 需发 `session_failed`）。每轮必须收尾。
前一两回合看夹爪和动作还像不像插 USB。

`q` 中止仍会按已提交的 actor chunk 做 UTD。想少更新就少收尾。

接管时 V2 断开模型连接**不再**判死这一局：bridge 保留到 GPU 的连接和当前
chunk，重连后接着走。日志会打 `RLT episode kept open`。

### 打分键 p / o / x（回合不结束）

对齐 rlt-openpi `stage2_client_shuo_sync.sh`：`p` = +0.5、`o` = +0.1（可连按累加）、
`x` = −0.5。分数落在**当前这个 chunk** 上，回合继续；`s` / `f` 的终止奖励优先于累计分。

```bash
docker exec desktop-robot_client-1 bash -lc \
  'source /opt/xr/py_env/bin/activate && PYTHONPATH=/tmp python -m x2robot_rlt_ee.operator_cli progress'
docker exec desktop-robot_client-1 bash -lc \
  'source /opt/xr/py_env/bin/activate && PYTHONPATH=/tmp python -m x2robot_rlt_ee.operator_cli small_progress'
docker exec desktop-robot_client-1 bash -lc \
  'source /opt/xr/py_env/bin/activate && PYTHONPATH=/tmp python -m x2robot_rlt_ee.operator_cli regress --reward -0.25'
```

`operator_cli status` 里的 `queued_score` 是还没落盘的累计分。

### submit：接管期间把动作交给服务端

policy 暂停时 DesktopClient 不发观测，chunk 会一直挂着。`submit`（对应 rlt-openpi 的
`y`）把当前 chunk 连同接管回执立刻提交，回合不结束：

```bash
docker exec desktop-robot_client-1 bash -lc \
  'source /opt/xr/py_env/bin/activate && PYTHONPATH=/tmp python -m x2robot_rlt_ee.operator_cli submit'
```

倒车流程请等第二次 `r` 之后再 `submit`，否则 rewind 判定会落到下一个 chunk 上。

### 在线改探索噪声（下一 chunk 生效，不用重启 server）

不要改 `actor.model.actor_noise_sigma` 再 resume，合同会对不上。用 per-act 覆盖：

```bash
docker exec desktop-robot_client-1 bash -lc \
  'source /opt/xr/py_env/bin/activate && PYTHONPATH=/tmp python -m x2robot_rlt_ee.operator_cli status'
docker exec desktop-robot_client-1 bash -lc \
  'source /opt/xr/py_env/bin/activate && PYTHONPATH=/tmp python -m x2robot_rlt_ee.operator_cli sigma --sigma 0.05'
docker exec desktop-robot_client-1 bash -lc \
  'source /opt/xr/py_env/bin/activate && PYTHONPATH=/tmp python -m x2robot_rlt_ee.operator_cli sigma --sigma default'
```

`--sigma 0` 是确定性执行（仍是 residual，不是纯 Stage 1）。`--sigma default` 回到服务端
YAML 的 0.2。`mode=eval` 的自动评测局本来就关噪声。

抖动对照：第一局收尾前 = Cal-QL+当前 sigma；`f`/`failure` 之后变抖 = 在线 −Q。eval 也抖 = residual 大了；只有 actor 抖 = 噪声 + EXPO。

## 1. 套环启动顺序

GPU 侧套环 server 先听 `8000`。下面三步在机器人上执行。

**① 先 probe，动作不会下发到机械臂**

```bash
cd /home/xr/lfwj/RealWorld-RLinf
RLT_UPSTREAM_URI=ws://<GPU_IP>:8000 bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh
```

把 X2Robot 客户端的模型地址指向 `127.0.0.1:33057`，跑一次。probe 模式下 bridge
取到 chunk 就丢弃并结束 episode，只验证握手和形状。看日志确认没有 `ProtocolError`：

```bash
bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh --status
```

**② 启动 V2 事件适配器 —— 必须在 V2 会话之前**

```bash
bash toolkits/inference/run_xrobot_rlt_ee_v2_adapter.sh
```

适配器只采纳「当前有 pending chunk、且事件时间戳晚于该 chunk 开始时刻」的事件。
启动晚了会漏掉瞬时事件。

**③ 允许真实动作，然后开 V2 会话**

```bash
bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh --stop
RLT_EE_ALLOW_MOTION=true RLT_UPSTREAM_URI=ws://<GPU_IP>:8000 \
  bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh

cd /home/xr/lfwj
env -u HISTORY_DRY_RUN MODEL_ADDRESS=127.0.0.1:33057 \
  ./collect/run_interactive_session_v2.sh --session-id v2_YYYYMMDD_NN
```

session ID 必须是从未用过的。

查看与停止（两个脚本都支持）：

```bash
bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh --status
bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh --stop
bash toolkits/inference/run_xrobot_rlt_ee_v2_adapter.sh --status
bash toolkits/inference/run_xrobot_rlt_ee_v2_adapter.sh --stop
```

## 2. 键盘：你按 V2 的键，RLT 自动跟随

| 键 | V2 做什么 | RLT 自动做什么 |
|---|---|---|
| `r` 第一次 | 暂停 policy，冻结历史，开始可取消的物理倒车 | 把当前 chunk 用实测 `/end_pose` 重采样成 `[50,14]` 回执，标 intervention 提交 |
| `r` 第二次 | 停止倒放，投影到正向路径最近的成对样本，收敛后进 `PolicyPaused` | 校验回退健康后发 `rewind_exit`，chunk 数 = `ceil(回退时长 / 1.67秒)`，终止奖励 -1.0，带实测终态 |
| `s` | 进入人工接管（必须在倒车停止后） | 无。人工段不进在线 replay |
| `h` | 交还 policy，开新 policy phase | 无。policy 恢复后 RLT 循环自动接上 |
| `f` | **成功**结束，生成 Traj_B（只在 `Policy` 状态有效） | 发 `success` + 实测终态 |
| `d` | **失败**结束（V2 需发 `session_failed`） | 发 `failure`，server 在最后一步写 `-1` |
| `p` | 打分：进展（回合继续） | 发 `progress`，当前 chunk +0.5 |
| `o` | 打分：小进展（可连按，回合继续） | 发 `small_progress`，当前 chunk 每次 +0.1，累加 |
| `x` | 打分：退步（回合继续） | 发 `regress`，当前 chunk −0.5 |
| `y` | 接管暂停时提交当前 chunk 并继续 | 发 `submit`，把接管回执立刻交给 server |
| `q` | 人工 abort，返回码 `2`，属正常不是故障 | 发 `abort`，丢弃未完成的 transition，**不给任何判定** |

`p` / `o` / `x` / `y` 需要 V2 按第 3 节的契约发事件；未接入前用 `operator_cli` 等效。

一次典型的接管：`r`（倒车）→ `r`（停在满意位置）→ `s`（接管）→ 人工做完 →
`h`（交还）→ 任务最终成功后 `f`。

按 `s` 之后 V2 会做物理切换（停 debug task → controller mode `8` 主从臂对齐 →
启用 databridge → mode `13` 末端遥操作），并在 5 秒内验证
`/master_{left,right}_arm/end_pose` 和 `/{left,right}_arm_cartesian_controller/pose_cmd`
四路各至少 3 条新鲜数据，才宣布接管成功。任一路缺失会关掉 databridge 并保持
policy 暂停 —— 这是防止"只有夹爪能动"被误判成接管成功。

## 3. 给 V2 的事件契约

`f` 仍是成功。失败和打分需要 V2 往 `/take_over_data` 发下面这些事件，adapter 会转成
对应的 RLT operator 命令。**只改 RLinf 不会让新键生效**，V2 collect 脚本要接：

```json
{"event_id": "...", "event": "session_failed", "detail": {"failure_end": <与 success_end 同结构的终态 sample>}}
{"event_id": "...", "event": "session_progress"}
{"event_id": "...", "event": "session_small_progress"}
{"event_id": "...", "event": "session_regress", "detail": {"reward": -0.25}}
{"event_id": "...", "event": "session_submit"}
```

`event_id` 必须唯一（adapter 按它去重）。打分和 `submit` 的 `detail` 可以省略；
`detail.reward` 用来覆盖默认分值。打分事件**不要求**当前有 pending chunk：两个 chunk
之间按的键会留在队列里，落到下一个提交的 chunk 上。

没接好之前全部可以手工发，效果一样：

```bash
docker exec desktop-robot_client-1 bash -lc \
  'source /opt/xr/py_env/bin/activate && PYTHONPATH=/tmp python -m x2robot_rlt_ee.operator_cli failure'
```

可用的 `command`：`success`、`failure`、`progress`、`small_progress`、`regress`、
`submit`、`abort`、`rewind_exit`、`rewind_credit`、`sigma`、`status`
（`rewind_*` 用 `--chunks N`，只改 replay 不动机械臂；打分用 `--reward` 覆盖分值）。

`status` 用来查当前有没有 pending chunk，联调时很有用：

```bash
docker exec desktop-robot_client-1 bash -lc \
  'source /opt/xr/py_env/bin/activate && PYTHONPATH=/tmp python -m x2robot_rlt_ee.operator_cli status'
```

## 4. 约束

- 倒车过程中其他普通键会被忽略。
- 每次 intervention 只能在**当前 policy phase** 的历史内倒车，单轮累计最多 30 秒，
  不能穿过上一段人类纠正轨迹。
- 在 `PolicyPaused` 里再按 `r`，只增加同一 intervention 的 `rollback_pass_id`，
  不会新开一条 Traj_A。
- 只有在 `Policy` 状态下按 `f` 才会成功结束。
- 回退不健康时适配器发的是 `abort` 而不是 `rewind_exit`，日志里会写明原因。

## 5. 两件要知道的事

**人工遥操那一段不进在线 replay。** 接管期间 policy 是暂停的，DesktopClient 不向
bridge 要 chunk，所以你手动操作的那一段进的是 V2 的 bag 和 Traj_A/B（离线管线）。
进在线 replay 的是**被打断的那个 policy chunk**：它按实测 `/end_pose` 重采样后标
intervention 提交。以前这条要等 policy 恢复才发得出去，现在按 `y` / `submit` 就能
立刻交给 server。在线学到的是：策略自己跑的 chunk + 坏分支上的 -1 + 你打的分。

**episode 一定要结束。** 训练只在 `episode_end` 发生。bridge 目前不读
`max_episode_chunks`，不会自己封顶，所以每轮都得靠 `f` / `q` 或手工命令收尾。
有人值守没问题；要跑长时间或无人值守之前，需要先给 bridge 补上 chunk 预算。

## 6. 端口

```text
127.0.0.1:33057   bridge <- DesktopClient（模型地址填这个）
127.0.0.1:33058   bridge 的 operator 控制口（operator_cli / V2 适配器用）
<GPU_IP>:8000     RLT Stage 2 server（套环）
<GPU_IP>:8016     RLT Stage 2 server（USB）
```
