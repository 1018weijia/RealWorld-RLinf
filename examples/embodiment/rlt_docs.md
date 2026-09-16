# RLT Stage 2 文档索引

Stage 2 的文档都放在本目录（`examples/embodiment/`），与它们描述的配置和启动脚本同级。
新增 RLT 文档请一并放在这里，不要放到仓库根目录。

## 按场景查

| 场景 | 文档 |
| --- | --- |
| 接入一台新机器人 | [rlt_add_robot.md](rlt_add_robot.md)（英文） |
| XRobot 上手：GPU 侧 | [xrobot-stage2-gpu-quickstart.md](xrobot-stage2-gpu-quickstart.md) |
| XRobot USB 离线 Cal-QL | [xrobot-stage2-gpu-quickstart.md](xrobot-stage2-gpu-quickstart.md) 第 9 节 |
| XRobot USB 在线（GPU 常驻 / 合同 / 空等） | [xrobot-stage2-gpu-quickstart.md](xrobot-stage2-gpu-quickstart.md) 第 10 节 |
| XRobot USB 客户端（probe → 真机） | [xrobot-stage2-robot-quickstart.md](xrobot-stage2-robot-quickstart.md) 「USB 客户端」 |
| XRobot 上手：机器人侧（套环） | [xrobot-stage2-robot-quickstart.md](xrobot-stage2-robot-quickstart.md) |
| Cobot 客户端部署 | [cobot-local-client-stage2.md](cobot-local-client-stage2.md) |
| 现场操作：rollout、接管、回滚 | [cobot-stage2-rollout-online-training-guide.md](cobot-stage2-rollout-online-training-guide.md) |
| 离线预训练后接在线 | [cobot_calql_offline_to_online.md](cobot_calql_offline_to_online.md) |
| 换参数后接续在线训练 | [cobot_assemble_parts_online.md](cobot_assemble_parts_online.md) |
| 只评测 Stage 1，不训练 | [cobot_stage1_eval.md](cobot_stage1_eval.md) |

## 关于文档名里的机器人

以 `cobot` 开头的几份是按当时唯一跑通的机型写的，但除了里面的具体路径和任务名，
流程对所有机型一致 —— 服务端不按机器人分支，机型差异只在
`config/embodiment/<name>.yaml` 里。要理解哪些是机型相关的，看
[rlt_add_robot.md](rlt_add_robot.md)。
