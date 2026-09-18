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
  bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh
```

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
  bash toolkits/inference/run_xrobot_rlt_ee_bridge.sh

cd /home/xr/lfwj
env -u HISTORY_DRY_RUN MODEL_ADDRESS=127.0.0.1:33057 \
  ./collect/run_interactive_session_v2.sh --session-id v2_YYYYMMDD_NN
```

V2：`s` 接管，`f` 成功，`d` 失败（V2 需发 `session_failed`）。每轮必须收尾。
前一两回合看夹爪和动作还像不像插 USB。

`q` 中止仍会按已提交的 actor chunk 做 UTD。想少更新就少收尾。

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

`--sigma 0` 是确定性执行（仍是 residual，不是纯 Stage 1）。`mode=eval` 的自动评测局本来就关噪声。启动时也可设 `RLT_EXPLORATION_NOISE_SIGMA=0.05` 再拉 bridge。

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
| `q` | 人工 abort，返回码 `2`，属正常不是故障 | 发 `abort`，丢弃未完成的 transition，**不给任何判定** |

一次典型的接管：`r`（倒车）→ `r`（停在满意位置）→ `s`（接管）→ 人工做完 →
`h`（交还）→ 任务最终成功后 `f`。

按 `s` 之后 V2 会做物理切换（停 debug task → controller mode `8` 主从臂对齐 →
启用 databridge → mode `13` 末端遥操作），并在 5 秒内验证
`/master_{left,right}_arm/end_pose` 和 `/{left,right}_arm_cartesian_controller/pose_cmd`
四路各至少 3 条新鲜数据，才宣布接管成功。任一路缺失会关掉 databridge 并保持
policy 暂停 —— 这是防止"只有夹爪能动"被误判成接管成功。

## 3. 失败：V2 发 `session_failed`，或手工 `failure`

`f` 仍是成功。失败请用 **`d`**（或你们指定的失败键），让 V2 往 `/take_over_data` 发：

```json
{"event_id": "...", "event": "session_failed", "detail": {"failure_end": <与 success_end 同结构的终态 sample>}}
```

adapter 会转成 RLT `failure`。**V2 collect 脚本要接这个事件**；只改 RLinf 不会让 `d` 生效。

没接好之前仍可手工：

```bash
docker exec desktop-robot_client-1 bash -lc \
  'source /opt/xr/py_env/bin/activate && PYTHONPATH=/tmp python -m x2robot_rlt_ee.operator_cli failure'
```

同一条命令可用的 `command`：`failure`、`success`、`abort`、`rewind_credit`、`sigma`、`status`
（`--chunks N`，只改 replay 不动机械臂）、`status`。

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

**人工段不进在线 replay。** 接管期间 policy 是暂停的，DesktopClient 不向 bridge 要
chunk，所以你手动操作的那一段进的是 V2 的 bag 和 Traj_A/B（离线管线），不进 RLT
的在线 replay。在线学到的是：策略自己跑的 chunk + 坏分支上的 -1。

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
