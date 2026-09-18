# Cobot 需要迁移的权重与路径

2026-09-18 已在 origin 核实。只保留 Stage1 与 Stage2 离线模型，
放弃本次在线训练的权重和数据。此文档仅列迁移清单，不包含迁移脚本。

## 1. Stage1：两份模型文件，必须保留

零件装配，30000 步，约 15.3 GiB：

```text
/data/gxy/realworldRL/RLinf/logs/20260906-06:01:46-cobot_rlt_stage1_sft_openpi_pi05_assemble_parts_franka_legacy_action_expert_base/cobot_assemble_franka_legacy_actionexpert_base_fp32master_bf16compute_30k/checkpoints/global_step_30000/actor/model_state_dict/full_weights.pt
```

炒蔬菜、方块入抽屉、水果装倒共用的 mixed 模型，30000 步，约 15.3 GiB：

```text
/data/gxy/realworldRL/RLinf/logs/20260907-09:26:30-cobot_rlt_stage1_sft_openpi_pi05_mixed_cook_cube_pack_franka_legacy_action_expert_base/cobot_mixed_cook_cube_pack_franka_legacy_actionexpert_base_fp32master_bf16compute_30k/checkpoints/global_step_30000/actor/model_state_dict/full_weights.pt
```

这两份是包含 VLA 和 RLT 模块的完整权重。Stage2 推理和在线训练仍然需要它们。
同级的 `actor/dcp_checkpoint/` 是 Stage1 分布式训练状态，本次不要求迁移；
因此不承诺恢复 Stage1 原优化器状态。

## 2. Stage2：四个最终离线 checkpoint，必须迁移整个目录

均为新 `cobot-joint-motion-v2`、noise=0.025、离线训练 40000 步；
已读取确认 `offline_total_updates=40000`、`partial_conversion=false`。

| 任务 | 目录体积 | 离线 transition 数 |
| --- | --- | --- |
| 零件装配 | 348 MiB | 6728 |
| 炒蔬菜 | 408 MiB | 8072 |
| 方块入抽屉 | 461 MiB | 9261 |
| 水果装倒 | 230 MiB | 4075 |

四个完整目录：

```text
/data/gxy/realworldRL/RLinf-cobot-joint-v2/results/cobot_joint_v2_assemble_parts/offline_train/checkpoints/offline_step_40000
/data/gxy/realworldRL/RLinf-cobot-joint-v2/results/cobot_joint_v2_cook_vegetable/offline_train/checkpoints/offline_step_40000
/data/gxy/realworldRL/RLinf-cobot-joint-v2/results/cobot_joint_v2_cube_into_drawer/offline_train/checkpoints/offline_step_40000
/data/gxy/realworldRL/RLinf-cobot-joint-v2/results/cobot_joint_v2_pack_and_pour_fruit/offline_train/checkpoints/offline_step_40000
```

每个目录需要完整保留：

- `stage2_state.pt`：actor、critic、target 与优化器等；
- `offline_state.pt`：离线更新计数、Cal-QL 与随机数状态；
- `offline_buffer.pt`：恢复模型所需的离线特征数据；
- `replay_buffer/`、`demo_buffer/` 等已有目录和 metadata。

不能只拿 `stage2_state.pt`。这里的 buffer 属于离线 checkpoint 的恢复依赖，
不是放弃的在线试验数据。

如需保留不同离线步数用于比较，保存上述各任务的整个 `offline_train/` 目录。
其中包含 5000、10000、15000、20000、25000、30000、35000、40000 步 checkpoint。
四任务这些目录合计约 11.7 GiB，含训练配置与日志。
尤其可保留 20000 步用于和 40000 步比较；训练步数不等同于实际任务效果。

## 3. 必要的统计量和配置

两份归一化统计必须与 Stage1 一起保留：

```text
/data/gxy/realworldRL/checkpoints/assets/cobot_magic/assemble_parts/norm_stats.json
/data/gxy/realworldRL/checkpoints/assets/cobot_magic/mixed_cook_cube_pack/norm_stats.json
```

各任务实际离线训练配置：

```text
/data/gxy/realworldRL/RLinf-cobot-joint-v2/results/cobot_joint_v2_assemble_parts/offline_train/effective_config.yaml
/data/gxy/realworldRL/RLinf-cobot-joint-v2/results/cobot_joint_v2_cook_vegetable/offline_train/effective_config.yaml
/data/gxy/realworldRL/RLinf-cobot-joint-v2/results/cobot_joint_v2_cube_into_drawer/offline_train/effective_config.yaml
/data/gxy/realworldRL/RLinf-cobot-joint-v2/results/cobot_joint_v2_pack_and_pour_fruit/offline_train/effective_config.yaml
```

路径配置的实际文件：

```text
/data/gxy/realworldRL/RLinf/.private-cobot-stage2/paths.env
```

该目录中的 `wandb_api_key` 如需继续使用，单独私下保管，不提交 Git。

tokenizer 的已缓存文件可一并保存，避免新服务器重新下载：

```text
/home/xiaoyu/.cache/openpi/big_vision/paligemma_tokenizer.model
```

## 4. 代码仓库与版本

| 用途 | 仓库 | 分支 | 实现基准提交 |
| --- | --- | --- | --- |
| 本地客户端及独立 Cobot runtime | https://github.com/jianzhang96/rlt-openpi | `remote-cobot` | `95882f9` |
| 服务器 joint v2 训练与推理 | https://github.com/1018weijia/RealWorld-RLinf | `cobot-joint-motion-v2` | `65bb1f1d` |

基准提交后的提交主要补充本文档。服务器代码位于：

```text
/data/gxy/realworldRL/RLinf-cobot-joint-v2
```

它是 Git worktree，不能只复制 `.git` 文件；应从对应分支恢复仓库。
其 `.venv` 和 `.private-cobot-stage2` 也指向原 RLinf 目录，迁移链接本身不能恢复环境。

已生成的辅助清单与代码快照位于：

```text
/data/gxy/realworldRL/RLinf-cobot-joint-v2/results/cobot_backup_manifest_20260918
```

其中 `inventory.json` 记录文件路径、大小、纳秒时间戳、SHA256 与模型配置；
`weights.sha256` 对应两份 Stage1 和四任务全部离线 checkpoint。
`rlinf-all-refs.bundle`、`rlinf-cobot-code.tar.gz` 为生成清单时的服务器代码快照。
`server-requirements.txt`、`environment.json` 记录环境；
`installed-policy-packages.tar.gz` 保存实际安装的 OpenPI、OpenPI client 和
Transformers 补丁代码。新机器只安装同名 pip 包不一定与原环境一致。

## 5. 容量和恢复注意事项

- 最小权重集：两份 Stage1 + 四个最终离线 checkpoint，约 **32 GiB**。
- 保留全部离线 checkpoint：两份 Stage1 + 四个完整 `offline_train/`，约 **43 GiB**。
- 当前恢复合同校验 Stage1 的绝对路径、文件大小和纳秒 mtime，以及 norm_stats 的内容。
  新服务器应保持原路径或建立兼容路径，并保留时间戳；仅改 `paths.env` 会触发
  `Offline buffer differs`，不能通过关闭校验来绕过。
- 不迁移本次 `train_*` 在线实验、在线 replay、原始数据集、转换 shards 和 20 步测试模型。
- 上述清单和 Git push 不等于权重已经下载到其他机器；服务器回收前需实际转出所选文件。

