# Cobot Stage 2 Rollout、接管与在线训练操作说明

适用代码：`RLinf` 的 `rlt` 分支，提交 `18bd8c85`。

本文面向两类人员：

- GPU/训练侧：启动 Ray、加载 Stage 1 15k 权重、运行 Stage 2 learner。
- Cobot 真机侧：实现控制 adapter，启动本地控制程序，执行 rollout、接管和回退。

## 1. 先确认几个边界

1. Stage 1 权重是冻结的 feature/reference 模型；Stage 2 是新建的 residual TD3/EXPO MLP。
2. Stage 2 的 rollout、replay、loss、在线更新和权重同步由 RLinf/Ray 完成。
3. Cobot 的真实控制协议、归一化转换、限幅、watchdog、急停和 rewind 由真机侧 adapter 完成。
4. 当前配置默认 `is_dummy: true`。真实运行前必须指定 `controller_factory`，并完成低速 HIL 验证。
5. Cobot 不需要向 GPU 训练进程发送单独的“开始训练”网络命令。动作执行结果和事件元数据会随 Ray trajectory 自动上传。

## 2. Stage 1 15k 权重和 norm stats

Stage 1 checkpoint 必须放在 `rollout.rlt_feature_model.model_path`，不要放到 `rollout.model.model_path` 或 `actor.model.model_path`。后两个字段属于 Stage 2 MLP。

本机已确认的 Stage 1 权重如下。

### 2.1 Assemble / legacy action expert（推荐，15k）

对应 tmux 会话：`RLT_stage1_Cobot_assemble_franka_legacy_actionexpert_gpu5`。

```text
/data/gxy/realworldRL/RLinf/logs/20260906-06:01:46-cobot_rlt_stage1_sft_openpi_pi05_assemble_parts_franka_legacy_action_expert_base/cobot_assemble_franka_legacy_actionexpert_base_fp32master_bf16compute_30k/checkpoints/global_step_15000/actor/model_state_dict
```

对应 norm stats：

```text
/data/gxy/realworldRL/checkpoints/assets/cobot_magic/assemble_parts/norm_stats.json
```

### 2.2 Mixed cook/cube/pack（备选，当前只有 10k）

对应 tmux 会话：`RLT_stage1_Cobot_mixed_cook_cube_pack_gpu7`。

该实验目前只保存到 `global_step_10000`，没有 `global_step_15000`，因此它不是本次 15k 运行的等价替代。

```text
/data/gxy/realworldRL/RLinf/logs/20260907-09:26:30-cobot_rlt_stage1_sft_openpi_pi05_mixed_cook_cube_pack_franka_legacy_action_expert_base/cobot_mixed_cook_cube_pack_franka_legacy_actionexpert_base_fp32master_bf16compute_30k/checkpoints/global_step_10000/actor/model_state_dict
```

对应 mixed norm stats：

```text
/data/gxy/realworldRL/checkpoints/assets/cobot_magic/mixed_cook_cube_pack/norm_stats.json
```

Stage 1 权重目录通常都包含 `actor/model_state_dict`，但启动前仍建议检查：

```text
/data/gxy/realworldRL/logs/<stage1-run>/<experiment>/checkpoints/global_step_15000/actor/model_state_dict
```

真机侧/训练侧先确认实际目录：

```bash
find /data/gxy/realworldRL/RLinf/logs \
  -path '*cobot_rlt_stage1*' \
  -type d \( -name 'global_step_15000' -o -name 'global_step_10000' \) \
  -print
```

不要跨任务复用 norm stats。Stage 1 模型、OpenPI `config_name`、图像数量、action_dim 和 state schema 必须一致。

## 3. Cobot adapter 必须提供的接口

训练进程通过 `controller_factory` 加载真机 adapter。最少需要：

```python
class MyCobotAdapter:
    action_dim = 14

    def reset(self) -> CobotObservation: ...
    def observe(self) -> CobotObservation: ...
    def execute(self, action) -> CobotStepResult: ...
    def stop(self, reason: str) -> None: ...
    def close(self) -> None: ...

    # 需要支持物理/credit-only rewind
    def poll_rewind_event(self) -> CobotRewindEvent | None: ...
    def rewind_chunks(self, count: int) -> CobotObservation: ...
```

`execute()` 接收 RLinf normalized action，adapter 自己完成 normalized action 到真实机器人控制协议的转换。返回信息必须包含实际发给机器人并执行的 normalized action：

```python
return CobotStepResult(
    observation=obs_after_step,
    reward=reward,
    terminated=terminated,
    truncated=truncated,
    info={
        "executed_action": actual_normalized_action,
        "intervene_flag": human_is_controlling,
    },
)
```

要求：`executed_action` shape 与 proposal 相同、无 NaN/Inf、范围为 `[-1, 1]`。如果有人接管，`intervene_flag=True`；可同时提供兼容字段 `intervene_action`。

## 4. 两节点 Ray 启动

### 4.1 GPU 节点

在启动 Ray 前设置 rank：

```bash
cd /data/gxy/realworldRL/RLinf
export RLINF_NODE_RANK=0
ray start --head --port=6379 --node-ip-address=<GPU_HEAD_IP>
```

### 4.2 Cobot 节点

```bash
cd /data/gxy/realworldRL/RLinf
export RLINF_NODE_RANK=1
ray start --address=<GPU_HEAD_IP>:6379
```

必须在 `ray start` 之前设置 `RLINF_NODE_RANK`。两节点都加入后，在 GPU 节点检查：

```bash
ray status
```

应看到一个 GPU 节点和一个 Cobot 节点。Cobot 节点不运行训练入口，只承载 env/controller worker。

## 5. 从 Stage 1 15k 启动 Stage 2

在 GPU 节点执行。下面给出已确认的两个命令，二选一。`controller_factory` 必须替换成真机侧实际 Python 模块路径。

### 5.1 使用 assemble/legacy 的 15k 权重（推荐）

```bash
cd /data/gxy/realworldRL/RLinf
export EMBODIED_PATH=/data/gxy/realworldRL/RLinf

.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-name cobot_rlt_stage2_td3_mlp \
  rollout.rlt_feature_model.model_path=/data/gxy/realworldRL/RLinf/logs/20260906-06:01:46-cobot_rlt_stage1_sft_openpi_pi05_assemble_parts_franka_legacy_action_expert_base/cobot_assemble_franka_legacy_actionexpert_base_fp32master_bf16compute_30k/checkpoints/global_step_15000/actor/model_state_dict \
  +rollout.rlt_feature_model.openpi_data.norm_stats_path=/data/gxy/realworldRL/checkpoints/assets/cobot_magic/assemble_parts/norm_stats.json \
  env.train.override_cfg.is_dummy=false \
  env.train.override_cfg.controller_factory=your_package.controller:create_adapter
```

### 5.2 使用 mixed 三任务的 10k 权重（备选）

只有在要运行 mixed cook/cube/pack 任务时使用：

```bash
cd /data/gxy/realworldRL/RLinf
export EMBODIED_PATH=/data/gxy/realworldRL/RLinf

.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-name cobot_rlt_stage2_td3_mlp \
  rollout.rlt_feature_model.model_path=/data/gxy/realworldRL/RLinf/logs/20260907-09:26:30-cobot_rlt_stage1_sft_openpi_pi05_mixed_cook_cube_pack_franka_legacy_action_expert_base/cobot_mixed_cook_cube_pack_franka_legacy_actionexpert_base_fp32master_bf16compute_30k/checkpoints/global_step_10000/actor/model_state_dict \
  +rollout.rlt_feature_model.openpi_data.norm_stats_path=/data/gxy/realworldRL/checkpoints/assets/cobot_magic/mixed_cook_cube_pack/norm_stats.json \
  env.train.override_cfg.is_dummy=false \
  env.train.override_cfg.controller_factory=your_package.controller:create_adapter
```

不要把 mixed 10k 权重和 `assemble_parts/norm_stats.json` 混用，也不要把 assemble 15k 权重和 mixed norm stats 混用。

第一次真机验证建议使用有限步数和独立输出目录。把上面命令的最后三行替换为：

```bash
  env.train.override_cfg.controller_factory=your_package.controller:create_adapter \
  runner.max_steps=10000 \
  runner.logger.log_path=/data/gxy/realworldRL/results/cobot_stage2_15k
```

这三行需要接在 `.venv/bin/python ... --config-name ...` 之后，形成一条完整命令。确认 HIL 稳定后可去掉最后两项 `runner.max_steps` 和 `runner.logger.log_path`，按需要长时间运行。`your_package.controller:create_adapter` 必须替换成真机侧实际的 Python 模块和工厂函数。

Stage 2 的关键默认值：

| 项目 | 值 |
|---|---:|
| action_dim | 14 |
| Stage 2 chunk | 16 steps |
| Stage 1 reference horizon | 20 steps |
| residual scale | 0.2 |
| EXPO | 4 个 base + 4 个 edited candidate |
| gamma | 0.99 |
| critic | twin-Q Huber，delta=0.5 |
| replay warmup | 250 条有效 transition |
| UTD | 每条有效 transition 5 次 critic update |
| actor delay | 每 2 次 critic update 更新一次 |
| target tau | 0.005 |

Stage 2 actor 输出为：

```text
action = clip(ref_chunk + 0.2 * tanh(delta), -1, 1)
```

residual 输出层零初始化，所以刚启动时 Stage 2 行为应与 Stage 1 reference 一致。首次真机运行必须先验证这一点。

## 6. Rollout 阶段和 `b` 切换

环境 reset 后，`KeyboardRLTPolicySwitchWrapper` 的状态为 false：

```text
按 b 前：Stage 1 ref_chunk 执行，record_transition=False，不写 Stage 2 replay
按 b 后：Stage 2 actor/EXPO 执行，record_transition=True，写入 replay
```

`b` 是“进入 Stage 2 critical phase”的信号，不是人工接管信号。它只在当前 episode 内生效，reset 后恢复 false。

建议在即将开始需要 Stage 2 学习的任务阶段前按一次 `b`。按键后从下一个 action chunk 开始切换；重复按 `b` 不会重复切换。

## 7. 人工接管流程

人工接管必须在 Cobot 本地 teleop/controller 中实现。每个被人控制的 low-level step：

```python
info["intervene_flag"] = True
info["executed_action"] = actual_human_action_normalized
```

没有接管时：

```python
info["intervene_flag"] = False
info["executed_action"] = actual_policy_action_normalized
```

RLinf 会：

1. 以 `executed_action` 覆盖原始 proposal。
2. 保留 chunk 内逐步 intervention mask。
3. 只把完整且 `record_transition=True` 的 chunk 写入 replay。
4. 将含人工接管的 chunk 放入 intervention sampling partition。
5. 训练时通常使用约 50% online replay + 50% intervention replay；intervention 不足时回退到全量 replay。

因此真机侧不需要再向 GPU 发送一份“接管动作日志”；只要 adapter 返回上述字段即可。

## 8. 回退/rewind 流程

回退由真机侧本地控制程序触发，并在 chunk 边界排队一个事件。物理回退示例：

```python
CobotRewindEvent(
    mode="exit",
    chunks_rewound=N,
    terminal_reward=-1.0,
    confidence=1.0,
    episode_id=episode_id,
    session_id=session_id,
    env_id=0,
    chunk_id=last_committed_chunk_id,
    rewind_exit_mode="physical",
)
```

credit-only 回退示例：

```python
CobotRewindEvent(
    mode="credit",
    chunks_rewound=N,
    terminal_reward=-1.0,
    prefix_reward=0.1,
    confidence=1.0,
    episode_id=episode_id,
    session_id=session_id,
    env_id=0,
    chunk_id=last_committed_chunk_id,
    rewind_exit_mode="credit_only",
)
```

运行时顺序：

1. 操作员在本地触发 rewind。
2. adapter 的 `poll_rewind_event()` 返回一次事件。
3. RLinf 在下一个 chunk 边界消费事件。
4. `mode="exit"` 时调用 `rewind_chunks(N)`；`mode="credit"` 时不移动机器人，只做 credit patch。
5. 当前待执行 policy proposal 被事件抢占，执行 0 个 low-level step。
6. 事件通过 Ray 发送为 event-only payload，`record_transition=False`，不会产生 replay row。
7. learner 将坏分支最后一行写入负终止奖励并切断 bootstrap。
8. 回退后的第一条有效、已提交 chunk成为 recovery positive。这个 positive 可以来自人工接管，也可以来自 policy 自己重试成功。
9. learner 将 bad fork action 作为 negative，构造 preference pair，并 patch anchor 的 next-action override。

真机侧不要把 rewind 命令发送给 GPU learner，也不要把 rewind event 当成普通 action transition 上报。

## 9. 安全故障和中止

安全故障应由 adapter 本地处理，并返回：

```python
info = {
    "rlt_safety_fault": True,
    "record_transition": False,
    "executed_action": actual_safe_action,
}
```

操作员主动中止：

```python
info = {
    "rlt_operator_abort": True,
    "record_transition": False,
}
```

安全故障、急停、watchdog 或中间断开时，adapter 必须立即停止剩余 low-level action。RLinf 会丢弃 partial chunk、停止 bootstrap，并保留本地 stop reason。断开 Ray 不能影响真机侧急停和 stop。

## 10. 在线训练何时发生

真机侧不发送显式“update once”信号。训练条件由 learner 自动判断：

```text
完整 chunk 且 record_transition=True
    -> replay 达到 250 条有效 transition
    -> 收到有效的已记录 episode boundary
    -> learner 执行累积 UTD 更新
    -> actor/critic/target 更新
    -> actor 权重和 selection critic 自动同步到 rollout worker
```

为了触发一次正常的训练 drain，建议在一个已提交的完整 chunk 上结束 episode，例如 adapter 返回：

```python
CobotStepResult(
    observation=obs,
    reward=terminal_reward,
    terminated=True,
    truncated=False,
    info={
        "executed_action": actual_action,
        "intervene_flag": False,
        "record_transition": True,
    },
)
```

注意：中间 safety fault/timeout 造成的 `record_transition=False` partial chunk 不会单独触发 episode-boundary training。需要训练时，应在后续有效 chunk 上完成并记录 episode 结束。

## 11. 推荐的真机首次运行顺序

1. 检查 adapter 的 observation、action、normalized round-trip、watchdog、stop、fault 和 rewind contract。
2. 启动 Cobot 本地控制程序，但先禁止真实动作或使用最低速度/最小限幅。
3. 启动两节点 Ray，确认 GPU 节点和 Cobot 节点都在线。
4. 在 GPU 节点用 Stage 1 15k checkpoint 启动 Stage 2。
5. 做 dry run：确认 Stage 2 zero-init 输出与 Stage 1 reference 一致。
6. 运行单个低速 action chunk，确认每一步返回的 `executed_action` shape/range 正确。
7. 在任务关键阶段按一次 `b`，确认后续 chunk 的 `rlt_switch_flags=True` 且开始记录 replay。
8. 分别实测：正常 rollout、人工接管、physical rewind 后人工恢复、physical rewind 后 policy 恢复。
9. 在完整已提交 chunk 边界结束若干 episode，确认 replay 达到 250 后出现 critic/actor update 日志。
10. 确认新权重同步后，后续 rollout 使用更新后的 Stage 2 actor；任何异常先由本地 stop/急停处理。

## 12. 运行中应观察的日志/指标

训练侧重点观察：

```text
rlt/global_min_replay_size
rlt/updates_to_run
rlt/critic_updates_run
rlt/actor_updates_run
rlt/update_step
replay/reward_positive_rate
preference_critic_active
```

真机侧重点记录：

```text
episode_id, session_id, env_id, chunk_id
proposal_action, executed_action
intervene_flag
rlt_switch_flags
rlt_rewind_event / rewind_exit_mode
rlt_safety_fault / rlt_operator_abort
record_transition
```

`chunk_id` 只给已提交的完整 transition 使用；event-only、partial、discarded chunk 不应占用 committed chunk ID。
