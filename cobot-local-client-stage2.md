# Cobot RLinf WebSocket Stage2

本地使用 `exp/rlinf_client_cobot.sh`。服务器使用 RLinf 的
`examples/embodiment/start_cobot_stage2.sh`，不再把本仓库的
`exp/stage2_server_cobot.sh` 和 RLinf checkpoint 混用。

## 四任务启动

| task | prompt | 默认端口 | Stage1 / norm stats |
|---|---|---|---|
| assemble_parts | assemble parts | 8000 | assemble 30k / assemble_parts |
| cube_into_drawer | put cube in drawer | 8001 | mixed 30k / mixed_cook_cube_pack |
| cook_vegetable | cook vegetable | 8002 | mixed 30k / mixed_cook_cube_pack |
| pack_and_pour_fruit | pack fruit into a container and pour it out | 8003 | mixed 30k / mixed_cook_cube_pack |

在 origin 的 RLinf 根目录执行以下命令；替换 task 即可选择其它任务：

```bash
bash examples/embodiment/start_cobot_stage2.sh assemble_parts preflight
CUDA_VISIBLE_DEVICES=3 bash examples/embodiment/start_cobot_stage2.sh assemble_parts train
```

也有独立入口 `start_cobot_assemble_parts.sh`、`start_cobot_cube_into_drawer.sh`、
`start_cobot_cook_vegetable.sh`、`start_cobot_pack_and_pour_fruit.sh`，参数为
`preflight|train|eval` 及后续 Hydra overrides。不指定模式时只做 preflight。
GPU 编号按当前空闲情况选择，不同任务同时运行须显式分配 GPU。

模型/统计量的实际路径从项目私有 `.private-cobot-stage2/paths.env` 加载，
包含 `COBOT_ASSEMBLE_CHECKPOINT`、`COBOT_ASSEMBLE_STATS`、
`COBOT_MIXED_CHECKPOINT`、`COBOT_MIXED_STATS`；路径必须存在，否则立即退出。
Stage1 模型同时包含 RLToken，不需要虚构独立的 `rl_token_step*.pt` 文件。

本地先按站点步骤启动 Evo-RL ROS2 policy runtime，再在本仓库执行：

```bash
bash exp/rlinf_client_cobot.sh assemble_parts check
bash exp/rlinf_client_cobot.sh assemble_parts dry-run
bash exp/rlinf_client_cobot.sh assemble_parts train
```

`check` 仅检查握手；`dry-run` 读取真实相机并请求一次推理后 discard，不重置机械臂、
不发动作、不写 replay、不创建数据集。`train` 会控制真机，需操作员在场，
确认两个 follower、两个 leader 和三路相机均由 policy runtime 提供。
脚本从 SSH config 的 `origin` 条目解析地址，也可设置 `RLT_SERVER_HOST` 和
`RLT_SERVER_PORT`（两端端口要一致）。网络失败会在本地停止，不能自动重发不确定的 transition。

客户端不需要安装 RLinf；使用本项目与已有 Evo-RL 的 ROS2/LeRobot 依赖。
默认 Conda 环境为 `evo-rl-ros2-jazzy`，`EVORL_ROOT` 和 `RLT_CONDA_ENV` 可覆盖。
动作是左臂 6 关节+夹爪、右臂 6 关节+夹爪的物理单位。
不交换手臂，不再次反归一化，不对关节量施加错误的 `[-1,1]` 裁剪。
本地默认拒绝偏离当前实测关节超过 0.15 rad 的目标（不含夹爪）；
`RLT_COBOT_MAX_JOINT_DELTA` 可按站点验证结果设置。

## 操作与数据

每次 VLA 预测 50 步；Stage2 actor/critic 控制并学习前 30 步，WebSocket 下发 30 步。
客户端必须完整回传这 30 步的实际命令；不能把没执行的后 20 步放进 replay。
30 Hz、无预取，一次完整 commit 确认后才请求下一 chunk。

`s`/空格成功、`f` 失败、`p` 进展；`i` 在 chunk 边界进入双臂 HIL。
HIL 也通过同步 act/transition 收集，chunk 间会等待服务器，不承诺连续遥操作无等待。
中途结束/HIL 释放后，剩余步数真实下发当前位姿保持命令，并记录这些实际动作；
不以数组 padding 冒充执行。硬件异常立即停止并 discard 未完整完成的 chunk。

`b` 在完整 commit 后暂停，`r` 回退一段真实双臂历史，`q` 只改 credit，
再次 `b` 恢复，`i` 接管。物理和 credit 模式不可在一次暂停中混用。
回退命令走 RLinf 原生 `rewind_*_correction`，不使用旧客户端专用的
`intervention_transition`。默认奖励分别为 -0.14 / -0.1，credit prefix 为 0.1。

默认保存到本地 `data/rlinf_<task>_<启动时间>`，可用 `RLT_DATASET_ROOT` 覆盖。
复用同一目录时已保存 episode 计入 `NUM_EPISODES`。记录三路视频、56D state、
28D action，以及 policy/HIL/rewind 来源；命令成功下发后才追加帧。
推理等待、暂停、复位和 leader 同步期间的保持不写入执行数据集。

边界对齐的合同是 `transition.next_observation == 下一次 act.observation`：
复制图像缓冲并复用同一个 endpoint，不能各自重新采样。录制行是动作前的观测，
因此 chunk 最后一行与下个 chunk 第一行并不应逐像素相等：中间实际执行了一步，
且机械臂可能继续收敛到最后的目标。时间压缩视频不代表真实墙钟连续。
相机采样同步、实际到位和机械安全仍需真机验证；单元测试不能证明真实训练收敛。

## 凭据、恢复和评测

项目 key 存在 `.private-cobot-stage2/wandb_api_key`（600，目录 700，Git 忽略）。
脚本仅给子进程设置 `WANDB_API_KEY` 与项目独立的目录，不执行 `wandb login`，
不修改 `~/.netrc`。可选 `RLT_WANDB_ENTITY` 指定团队；不设则使用账号默认 entity。
同一个 Linux 用户的进程仍可访问该用户私有文件，项目隔离不是账号间的权限隔离。

服务端输出默认到 `results/cobot_stage2_<task>/<启动时间>/checkpoints`。
恢复使用 `STAGE2_RESUME_DIR`，该目录应有 `stage2_state.pt` 和 replay 子目录。
必须是同任务、相同 30/50 维度配置，不能拿 Stage1 目录当 Stage2 resume。

```bash
# 已导出实际 STAGE2_RESUME_DIR 后，在 origin 启动冻结评测服务
bash examples/embodiment/start_cobot_stage2.sh assemble_parts eval
# 本地需导出经实验计划确认的 RLT_REQUIRED_STAGE2_UPDATES，再执行 30 局
bash exp/rlinf_client_cobot.sh assemble_parts eval
```

评测客户端核对冻结标志和实际 update 数，关闭 HIL/rewind、replay 上传及训练。
训练默认 200 局，评测默认 30 局，均可用 `NUM_EPISODES` 覆盖。

这些入口目前运行在线 RL；不会自动把 LeRobot 演示填入 buffer，也未实现 Cal-QL
离线预训练。那需要另外完成 RLinf 的数据转换与训练算法接口，不能用旧项目
生成的 buffer/actor 直接替换这里的 checkpoint。

## 验证

本地：`pytest tests/test_rlinf_cobot_client.py`。
装有 RLinf 的 origin：把本仓库 `src` 加入 `PYTHONPATH` 后运行
`tests/test_rlinf_cobot_integration.py`，它使用真实 RLinf WebSocket handler、
真实协议和假模型/假机器人，验证动作反向归一化、replay 和训练触发。
`deployment/rlinf/` 保存可审查的部署脚本及服务端启动回归测试。
