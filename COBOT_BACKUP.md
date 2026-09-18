# Cobot：服务器回收前备份与恢复

适用版本：客户端 `remote-cobot`，服务器 `cobot-joint-motion-v2`。
只保留 Stage1 和 Stage2 **离线**训练结果。本次在线训练的权重、replay 和原始采集数据不在备份范围内。
Git push 只保存代码；权重必须下载到不会随服务器回收的磁盘。

## 已核实的保存范围

| 内容 | 来源 | 备份内位置 |
| --- | --- | --- |
| 装配 Stage1，30000 步 | 私有配置的 `COBOT_ASSEMBLE_CHECKPOINT` | `stage1/assemble_parts/global_step_30000/actor/model_state_dict/full_weights.pt` |
| 其余三任务共用 Stage1，30000 步 | `COBOT_MIXED_CHECKPOINT` | `stage1/mixed_cook_cube_pack/global_step_30000/actor/model_state_dict/full_weights.pt` |
| 两份归一化统计 | `COBOT_ASSEMBLE_STATS`、`COBOT_MIXED_STATS` | `norm_stats/` |
| 四任务离线训练，5000～40000 步共 8 个 checkpoint/任务 | `results/cobot_joint_v2_<task>/offline_train/` | 保持同样的相对目录 |
| tokenizer、安装环境版本和 OpenPI/Transformers 实际代码 | 已核实的服务器安装环境 | `cache/openpi/`、`metadata/` |
| 项目 WandB key、原路径配置 | 项目私有文件 | `private/`，不进 Git |
| 离线训练日志、启动脚本 | `results/joint_v2_parallel/`、`results/joint_v2_queues/` | 保持同样的相对目录 |

四任务为 `assemble_parts`、`cook_vegetable`、`cube_into_drawer`、
`pack_and_pour_fruit`。最终 checkpoint 都已读取确认：
`offline_total_updates=40000`、`partial_conversion=false`；
对应离线 transition 数为 6728、8072、9261、4075。

两份 Stage1 模型文件各约 15.3 GiB；四任务离线训练目录合计约 11.7 GiB，
总计约 **43 GiB**，建议目标盘至少空出 50 GiB。
保留全部中间离线模型，方便比较 20000 和 40000 步的效果；跑满步数不代表策略已经收敛。

Stage1 的 `full_weights.pt` 是完整 wrapper 权重，含 VLA 和 RLT 模块；
不包含原 Stage1 的 DCP 优化器分片，所以支持推理、后续 Stage2 和重新微调，
但不能完整恢复 Stage1 原优化器训练状态。

Stage2 必须保留**整个** `offline_step_N` 目录，包括：

- `stage2_state.pt`：actor、critic、target 和优化器等状态；
- `offline_state.pt`：离线更新数、Cal-QL 状态、随机数状态；
- `offline_buffer.pt`：该离线模型使用的特征与 transition；
- `replay_buffer/`、`demo_buffer/` 等目录及 metadata。

这些是离线 checkpoint 自带的恢复依赖，不是本次在线实验数据。
只取 `stage2_state.pt` 会导致当前启动入口或恢复校验失败。
不复制 `train_*`、`eval_*`、原始数据集、转换 shards、20 步验证模型及整套虚拟环境。

## 代码已经独立于原 worktree 保存

- 客户端：`git@github.com:jianzhang96/rlt-openpi.git`，分支 `remote-cobot`。
- 服务端：`git@github.com:1018weijia/RealWorld-RLinf.git`，分支 `cobot-joint-motion-v2`。
- `metadata/rlinf-all-refs.bundle` 保存服务器仓库所有本地 refs；
  `metadata/rlinf-cobot-code.tar.gz` 保存 Cobot 分支的受跟踪文件快照。
- 本地另有 `results/cobot_server_backup/rlt-openpi-all-refs.bundle`。

服务器 Cobot 目录的 `.git` 是指向主仓库的文件，
`.venv`、`.private-cobot-stage2` 也是链接。不要只复制这些指针。
恢复代码用 Git clone 或 bundle；实际私有文件和安装包已单独列入。

## 在本地下载

在本地 rlt-openpi 项目根目录运行。本次核实的 SSH alias、服务器目录和 tokenizer 路径
已写入被 Git 忽略的 `results/cobot_server_backup/source.env`。
这个文件仅含路径；API key 不会输出到终端。

```bash
bash scripts/backup_cobot_server.sh plan
bash scripts/backup_cobot_server.sh pull
bash scripts/backup_cobot_server.sh verify
```

- `plan` 只下载很小的路径配置并检查远端文件清单，不传输权重。
- `pull` 使用 rsync 断点继续，默认下载至 `results/cobot_server_backup/archive/`。
  中断后重跑同一命令，不用删除已有文件。脚本不删除服务器文件。
- `verify` 只读比较服务器与本地校验和、文件列表和纳秒时间戳；有差异返回非零。
- 改目的地可先 `export RLT_BACKUP_DEST="/已挂载的备份盘/cobot/archive"`，
  后续三步必须使用相同目的地。SSH 断开后需重跑；长时间下载请在**本地** zellij 中执行。

换电脑执行时，将已核实的 `source.env` 一起带走，或配置：
`RLT_BACKUP_REMOTE`、`RLT_BACKUP_SERVER_ROOT`、
`RLT_BACKUP_SERVER_SNAPSHOT`、`RLT_BACKUP_TOKENIZER`。
来源绝对路径、文件大小、SHA256 和时间戳保存在私有 `metadata/inventory.json`，不写入 Git。

服务器回收后仍可进行离线完整性验证：

```bash
backup_root="${RLT_BACKUP_DEST:-$PWD/results/cobot_server_backup/archive}"
cd "$backup_root"
sha256sum -c metadata/weights.sha256
git bundle verify metadata/rlinf-all-refs.bundle
```

`git bundle verify` 需要在 Git 仓库中执行；如果备份放在仓库外，改为：
`git -C /已有Git仓库 bundle verify /备份目录/metadata/rlinf-all-refs.bundle`。
SHA256 必须全部为 OK。还要将客户端的 bundle 和 `source.env` 保存在独立磁盘；
仅服务器内生成清单或 Git push 不等于完成权重异地备份。

## 恢复到新服务器

先将 archive 上传至新服务器，在新目录克隆代码：

```bash
git clone --branch cobot-joint-motion-v2 git@github.com:1018weijia/RealWorld-RLinf.git RLinf-cobot-joint-v2
git clone --branch remote-cobot git@github.com:jianzhang96/rlt-openpi.git rlt-openpi
```

无法访问 GitHub 时，服务端可用：
`git clone -b cobot-joint-motion-v2 /备份目录/metadata/rlinf-all-refs.bundle RLinf-cobot-joint-v2`。

**当前 checkpoint contract 比较 Stage1 绝对路径及权重文件的纳秒 mtime。**
最直接的恢复方式是根据 `private/paths.original.env` 在新服务器重建原来的
两份 Stage1 路径和两份 norm_stats 路径，并用 rsync 保留文件时间。
只修改 `paths.env` 指向新路径会触发 offline buffer mismatch，不能靠关闭检查恢复。

在新服务器设置实际备份与代码目录后：

```bash
export COBOT_ARCHIVE="/实际备份目录/archive"
export COBOT_SERVER="/实际代码目录/RLinf-cobot-joint-v2"
source "$COBOT_ARCHIVE/private/paths.original.env"

mkdir -p "$COBOT_ASSEMBLE_CHECKPOINT/actor/model_state_dict" "$COBOT_MIXED_CHECKPOINT/actor/model_state_dict"
mkdir -p "$(dirname "$COBOT_ASSEMBLE_STATS")" "$(dirname "$COBOT_MIXED_STATS")"
rsync -a --modify-window=-1 "$COBOT_ARCHIVE/stage1/assemble_parts/global_step_30000/" "$COBOT_ASSEMBLE_CHECKPOINT/"
rsync -a --modify-window=-1 "$COBOT_ARCHIVE/stage1/mixed_cook_cube_pack/global_step_30000/" "$COBOT_MIXED_CHECKPOINT/"
rsync -a "$COBOT_ARCHIVE/norm_stats/assemble_parts/norm_stats.json" "$COBOT_ASSEMBLE_STATS"
rsync -a "$COBOT_ARCHIVE/norm_stats/mixed_cook_cube_pack/norm_stats.json" "$COBOT_MIXED_STATS"

mkdir -p "$COBOT_SERVER/.private-cobot-stage2" "$COBOT_SERVER/results"
install -m 600 "$COBOT_ARCHIVE/private/paths.original.env" "$COBOT_SERVER/.private-cobot-stage2/paths.env"
install -m 600 "$COBOT_ARCHIVE/private/server/wandb_api_key" "$COBOT_SERVER/.private-cobot-stage2/wandb_api_key"
chmod 700 "$COBOT_SERVER/.private-cobot-stage2"
rsync -a "$COBOT_ARCHIVE/results/" "$COBOT_SERVER/results/"
export OPENPI_DATA_HOME="$COBOT_ARCHIVE/cache/openpi"
```

以上应在新的、没有同名实验数据的目标目录执行。原路径若无写权限，
先为这些明确的目录配置挂载或路径兼容链接；不要改 buffer 内的合同来绕过校验。
外置盘如果不能保留纳秒 mtime，按照 `inventory.json` 中对应文件的 `mtime_ns`
恢复元数据并验证，不能只靠文件大小认定恢复正确。

新服务器需重建 Python 3.11 环境，结合仓库安装说明和
`metadata/server-requirements.txt`、`metadata/environment.json` 恢复依赖。
原环境的 OpenPI 没有 wheel 元信息，且 Transformers 有补丁；
只安装同名 pip 包可能不等价。安装依赖后，把
`metadata/installed-policy-packages.tar.gz` 中的
`openpi/`、`openpi_client/`、`transformers/` 恢复到新环境 site-packages。
不要直接复用指向旧服务器 Python 的 `.venv` 链接。
新机器环境与真机执行仍需验证。

预检查通过后，从离线模型开始一个新的在线实验：

```bash
cd "$COBOT_SERVER"
unset STAGE2_RESUME_DIR RLT_RUN_DIR OFFLINE_BUFFER
export CUDA_VISIBLE_DEVICES=6 RLT_SERVER_PORT=8010
export RLT_COBOT_ACTOR_NOISE_SIGMA=0.025
bash examples/embodiment/run_cobot_joint_v2.sh assemble_parts preflight
bash examples/embodiment/run_cobot_joint_v2.sh assemble_parts train
```

当前启动脚本限制 GPU 6/7；新服务器若无这些卡号，需要按实际拓扑调整 launcher，
不要把物理 GPU 号与进程内 `cuda:0` 混淆。
Stage1=装配 30000 步、Stage2=新 motion v2 离线 40000 步。
本地先运行 `check` 和 `dry-run`；确认 schema、40000 次离线更新、partial=false。
新的在线 replay 应为空，不恢复已经放弃的这次在线试验。

