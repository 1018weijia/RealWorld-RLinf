# Franka 5 Task Launch Commands

当前 C/S 部署：

- amax: pi0.5/OpenPI/RLinf policy service，监听 `192.168.10.114:33050`
- pnp: Franka 控制节点 + 异步推理 client
- 每个 task 测 30 次，SR = success_count / 30

## Prompt 要求

`TASK_PROMPT` / `TASK` 应精确匹配训练数据里的 task description。当前这套
Shuo pi0.5 Franka 权重使用下面五个自然语言 prompt；task id 只作为任务选择器和
实验记录名，不应直接作为模型语言条件。

不要 paraphrase，例如不要把 `Place the ring on the rod.` 改成
`put the ring on the pole`。大小写、标点也建议保持一致。

amax 的 `TASK_PROMPT` 和 pnp 的 `config.sh` 里的 `TASK` 必须完全一致。

服务端 `TASK_PROMPT` 只会作为 `--default-prompt` fallback。当前 pnp client
会在每个 WebSocket 请求里发送 `prompt/task`，服务端会优先使用 client payload。
因此只在 amax 上 `export TASK_PROMPT=...` 不足以切换任务；pnp 的 `export TASK`
如果没改，仍会覆盖服务端默认 prompt。

## Task/Weight 对应表

| Task id | Prompt | CKPT_PROFILE | Checkpoint | Norm stats |
| --- | --- | --- | --- | --- |
| `stack_bowls_in_size_order_rc` | `Stack the three bowls in size order: the purple bowl first, then the beige bowl.` | `front2` | `/home/amax/checkpoints/pi05-Recap-Franka/sft_franka_shuo_pi05/checkpoints/global_step_15000/actor/model_state_dict/full_weights.pt` | `/home/amax/checkpoints/pi05-Recap-Franka/assets/franka_shuo_bowls_ring/norm_stats.json` |
| `place_ring_on_rod_rc_0810` | `Place the ring on the rod.` | `front2` | `/home/amax/checkpoints/pi05-Recap-Franka/sft_franka_shuo_pi05/checkpoints/global_step_15000/actor/model_state_dict/full_weights.pt` | `/home/amax/checkpoints/pi05-Recap-Franka/assets/franka_shuo_bowls_ring/norm_stats.json` |
| `place_fruits_on_plate_rc` | `Place all the fruits on the plate.` | `d2` | `/home/amax/checkpoints/pi05-Recap-Franka/20260828-080659-franka_pi05_rlinf_d2/sft_franka_shuo_pi05/checkpoints/global_step_23000/actor/model_state_dict/full_weights.pt` | `/home/amax/checkpoints/pi05-Recap-Franka/assets/franka_shuo_fruits_charger_peg/norm_stats.json` |
| `plug_charger_into_socket_rc` | `Plug the charger into the socket.` | `d2` | `/home/amax/checkpoints/pi05-Recap-Franka/20260828-080659-franka_pi05_rlinf_d2/sft_franka_shuo_pi05/checkpoints/global_step_23000/actor/model_state_dict/full_weights.pt` | `/home/amax/checkpoints/pi05-Recap-Franka/assets/franka_shuo_fruits_charger_peg/norm_stats.json` |
| `insert_peg_into_hole_rc` | `Insert the peg into the corresponding hole.` | `d2` | `/home/amax/checkpoints/pi05-Recap-Franka/20260828-080659-franka_pi05_rlinf_d2/sft_franka_shuo_pi05/checkpoints/global_step_23000/actor/model_state_dict/full_weights.pt` | `/home/amax/checkpoints/pi05-Recap-Franka/assets/franka_shuo_fruits_charger_peg/norm_stats.json` |

## 通用切换流程

1. pnp 上停掉当前控制节点和推理节点：两个终端分别 `Ctrl+C`。
2. amax 上停掉当前 policy service。
3. amax 按目标 task 启动对应 `CKPT_PROFILE` 和自然语言 `TASK_PROMPT`。
4. pnp 的 `config.sh` 中把 `TASK` 改成完全相同的自然语言 prompt。
5. pnp 启动 `bash start_control_async.sh` 和 `bash start_inference_async.sh`。
6. 每个 trial 结束后记录最新 `logs/session_async_*` 目录，并用 `label_pnp_session.py` 标注 `is_success`。
7. 每个 task 选定 30 个正式 trial 后，计算 `SR = success_count / 30`。

检查 amax service：

```bash
ss -ltnp | grep 33050
tmux capture-pane -t <session> -p | tail -n 80
```

检查 pnp ROS graph：

```bash
source /opt/ros/humble/setup.bash
ros2 node list
ros2 topic info /franka/infer_request -v
ros2 topic info /franka/action_command -v
ros2 topic info /franka/ee_states -v
```

正常应看到：

```text
/franka_control_node_async
/franka_inference_node_async

/franka/infer_request    Publisher count: 1, Subscription count: 1
/franka/action_command   Publisher count: 1, Subscription count: 1
/franka/ee_states        Publisher count: 1, Subscription count: 1
```

## 一键生成命令

在 amax 仓库中可以用脚本打印某个 task 的启动命令：

```bash
cd /data/yangky/test/pi05-Recap-Franka
bash scripts/franka/print_5task_launch_commands.sh stack_bowls_in_size_order_rc
```

每个 trial 结束后，在 pnp 上记录并标注最新 session：

```bash
cd ~/桌面/franka_deploy_0128_ee/franka_deploy
latest_session="$(ls -dt logs/session_async_* | head -1)"
echo "${latest_session}"

python3 label_pnp_session.py "${latest_session}" \
  --success 1 \
  --trial 1 \
  --task-id <task_id> \
  --prompt "<exact_prompt>" \
  --checkpoint-profile <front2_or_d2> \
  --notes "success"
```

失败时把 `--success 1` 改成 `--success 0`，并在 `--notes` 写失败原因。
完整 rollout 保存和 RECAP 标注说明见 `docs/franka_recap_rollout_labeling.md`。

## 1. stack_bowls_in_size_order_rc

amax:

```bash
tmux kill-session -t pi05_front2 2>/dev/null || true
tmux kill-session -t pi05_stack_bowls 2>/dev/null || true
tmux new-session -d -s pi05_stack_bowls \
  'cd /data/yangky/test/pi05-Recap-Franka && export CKPT_PROFILE=front2 TASK_PROMPT="Stack the three bowls in size order: the purple bowl first, then the beige bowl." && bash scripts/franka/run_shuo_pi05_service_amax.sh'
tmux capture-pane -t pi05_stack_bowls -p | tail -n 80
```

pnp:

```bash
cd ~/桌面/franka_deploy_0128_ee/franka_deploy
perl -0pi -e 's/export TASK="[^"]+"/export TASK="Stack the three bowls in size order: the purple bowl first, then the beige bowl."/' config.sh
bash start_control_async.sh
bash start_inference_async.sh
```

## 2. place_ring_on_rod_rc_0810

amax:

```bash
tmux kill-session -t pi05_front2 2>/dev/null || true
tmux kill-session -t pi05_place_ring 2>/dev/null || true
tmux new-session -d -s pi05_place_ring \
  'cd /data/yangky/test/pi05-Recap-Franka && export CKPT_PROFILE=front2 TASK_PROMPT="Place the ring on the rod." && bash scripts/franka/run_shuo_pi05_service_amax.sh'
tmux capture-pane -t pi05_place_ring -p | tail -n 80
```

pnp:

```bash
cd ~/桌面/franka_deploy_0128_ee/franka_deploy
perl -0pi -e 's/export TASK="[^"]+"/export TASK="Place the ring on the rod."/' config.sh
bash start_control_async.sh
bash start_inference_async.sh
```

## 3. place_fruits_on_plate_rc

amax:

```bash
tmux kill-session -t pi05_front2 2>/dev/null || true
tmux kill-session -t pi05_place_fruits 2>/dev/null || true
tmux new-session -d -s pi05_place_fruits \
  'cd /data/yangky/test/pi05-Recap-Franka && export CKPT_PROFILE=d2 TASK_PROMPT="Place all the fruits on the plate." && bash scripts/franka/run_shuo_pi05_service_amax.sh'
tmux capture-pane -t pi05_place_fruits -p | tail -n 80
```

pnp:

```bash
cd ~/桌面/franka_deploy_0128_ee/franka_deploy
perl -0pi -e 's/export TASK="[^"]+"/export TASK="Place all the fruits on the plate."/' config.sh
bash start_control_async.sh
bash start_inference_async.sh
```

## 4. plug_charger_into_socket_rc

amax:

```bash
tmux kill-session -t pi05_front2 2>/dev/null || true
tmux kill-session -t pi05_plug_charger 2>/dev/null || true
tmux new-session -d -s pi05_plug_charger \
  'cd /data/yangky/test/pi05-Recap-Franka && export CKPT_PROFILE=d2 TASK_PROMPT="Plug the charger into the socket." && bash scripts/franka/run_shuo_pi05_service_amax.sh'
tmux capture-pane -t pi05_plug_charger -p | tail -n 80
```

pnp:

```bash
cd ~/桌面/franka_deploy_0128_ee/franka_deploy
perl -0pi -e 's/export TASK="[^"]+"/export TASK="Plug the charger into the socket."/' config.sh
bash start_control_async.sh
bash start_inference_async.sh
```

## 5. insert_peg_into_hole_rc

amax:

```bash
tmux kill-session -t pi05_front2 2>/dev/null || true
tmux kill-session -t pi05_insert_peg 2>/dev/null || true
tmux new-session -d -s pi05_insert_peg \
  'cd /data/yangky/test/pi05-Recap-Franka && export CKPT_PROFILE=d2 TASK_PROMPT="Insert the peg into the corresponding hole." && bash scripts/franka/run_shuo_pi05_service_amax.sh'
tmux capture-pane -t pi05_insert_peg -p | tail -n 80
```

pnp:

```bash
cd ~/桌面/franka_deploy_0128_ee/franka_deploy
perl -0pi -e 's/export TASK="[^"]+"/export TASK="Insert the peg into the corresponding hole."/' config.sh
bash start_control_async.sh
bash start_inference_async.sh
```
