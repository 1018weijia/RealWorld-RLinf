# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ray ``Worker`` host for the RLT Stage 2 learner cores.

All RLT algorithm semantics live in :mod:`rlinf.algorithms.rlt.learner` so the
single-process WebSocket server and this FSDP worker cannot drift apart.  This
module only adds the worker-specific plumbing: ``Worker.timer`` instrumentation,
``torch.distributed`` reductions, RLinf ``Channel`` ingest, and checkpointing
alongside the FSDP SAC parent.
"""

import queue

import torch

from rlinf.algorithms.rlt.learner import (
    RLTLossCore,
    RLTReplayCore,
    RLTRewindCore,
    RLTScheduleCore,
)
from rlinf.algorithms.rlt.preference import RewindPreferenceBuffer
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
    AsyncEmbodiedSACFSDPPolicy,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class RLTWorkerHostHooks:
    """Route the learner-core hooks onto ``Worker`` and ``torch.distributed``."""

    def _rlt_warn(self, message: str) -> None:
        self.log_warning(message)

    def _rlt_reduce_sum(self, values: dict[str, float]) -> dict[str, float]:
        return all_reduce_dict(values, op=torch.distributed.ReduceOp.SUM)

    def _rlt_reduce_min(self, values: dict[str, float]) -> dict[str, float]:
        return all_reduce_dict(values, op=torch.distributed.ReduceOp.MIN)

    @property
    def _rlt_metric_reducer(self):
        return all_reduce_dict


class RLTACLossMixin(RLTWorkerHostHooks, RLTLossCore):
    """RLT actor-critic losses with worker timing instrumentation."""

    def get_rollout_sync_version(self) -> int:
        """Expose learner update count when RLT warmup gates actor rollout."""
        if not self.use_rlt_schedule:
            return int(self.version)
        return int(self.update_step)

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        return RLTLossCore.forward_critic(self, batch)

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        return RLTLossCore.forward_actor(self, batch)

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        return RLTLossCore.forward_alpha(self, batch)


class RLTACReplayMixin(RLTWorkerHostHooks, RLTReplayCore):
    """Rollout-to-replay ingestion for sync and async RLT AC workers."""


class RLTACFSDPPolicy(
    RLTACLossMixin,
    RLTACReplayMixin,
    RLTRewindCore,
    RLTScheduleCore,
    EmbodiedSACFSDPPolicy,
):
    """Synchronous RLT AC worker with transition replay and warmup scheduling."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.init_rlt_schedule_state()
        preference_cfg = cfg.algorithm.get("rewind_preference", {}) or {}
        self.rewind_preference_buffer = RewindPreferenceBuffer(
            int(preference_cfg.get("capacity", 1024))
        )
        self.init_rlt_rewind_state()

    def setup_sac_components(self):
        """Initialize replay components and let RLT schedule own readiness."""
        super().setup_sac_components()
        if self.use_rlt_schedule:
            self.buffer_dataset.min_replay_buffer_size = 1

    def save_checkpoint(self, save_base_path, step):
        super().save_checkpoint(save_base_path, step)
        import os

        state_path = os.path.join(
            save_base_path, f"sac_components/rewind_state_rank_{self._rank}.pt"
        )
        torch.save(
            {
                **self.rlt_rewind_state_dict(),
                "schedule": self.rlt_schedule_state_dict(),
            },
            state_path,
        )

    def load_checkpoint(self, load_base_path):
        super().load_checkpoint(load_base_path)
        import os

        state_path = os.path.join(
            load_base_path, f"sac_components/rewind_state_rank_{self._rank}.pt"
        )
        if not os.path.exists(state_path):
            return
        state = torch.load(state_path, map_location="cpu")
        self.load_rlt_rewind_state_dict(state)
        self.load_rlt_schedule_state_dict(state.get("schedule", {}))

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel):
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        added, completed = self._ingest_rollout_trajectories(recv_list)
        self._update_rollout_ingest_counters(added, completed)

    def run_training(self):
        if not self.use_rlt_schedule:
            mean_metric_dict = super().run_training()
            replay_metrics = getattr(self, "_last_replay_metrics", {})
            if replay_metrics:
                mean_metric_dict = {**mean_metric_dict, **replay_metrics}
            return mean_metric_dict

        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        updates_to_run, schedule_metrics = self._rlt_updates_to_run()
        if updates_to_run <= 0:
            mean_metric_dict = self.process_train_metrics(schedule_metrics)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.empty_cache()
            return mean_metric_dict

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        critic_updates_run = 0
        actor_updates_run = 0
        for _ in range(updates_to_run):
            update_actor = int(self.update_step) % int(self.critic_actor_ratio) == 0
            metrics_data = self.update_one_epoch(train_actor=True)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1
            critic_updates_run += 1
            actor_updates_run += int(update_actor)

        schedule_metrics["rlt/critic_updates_run"] = float(critic_updates_run)
        schedule_metrics["rlt/actor_updates_run"] = float(actor_updates_run)
        self.pending_update_budget = max(
            int(self.pending_update_budget) - critic_updates_run,
            0,
        )
        schedule_metrics["rlt/pending_update_budget"] = float(
            self.pending_update_budget
        )
        append_to_dict(metrics, schedule_metrics)
        mean_metric_dict = self.process_train_metrics(metrics)
        self.transitions_since_train = 0
        self.episodes_since_train = 0

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict


class AsyncRLTACFSDPPolicy(
    RLTACLossMixin,
    RLTACReplayMixin,
    RLTRewindCore,
    RLTScheduleCore,
    AsyncEmbodiedSACFSDPPolicy,
):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.init_rlt_schedule_state()
        preference_cfg = cfg.algorithm.get("rewind_preference", {}) or {}
        self.rewind_preference_buffer = RewindPreferenceBuffer(
            int(preference_cfg.get("capacity", 1024))
        )
        self.init_rlt_rewind_state()

    def _drain_received_trajectories(self, max_trajectories: int | None = None):
        if getattr(self, "_recv_queue", None) is None:
            return
        recv_list = []
        processed = 0
        while True:
            try:
                recv_list.append(self._recv_queue.get_nowait())
                processed += 1
                if max_trajectories is not None and processed >= max_trajectories:
                    break
            except queue.Empty:
                break
        if not recv_list:
            return

        added, completed = self._ingest_rollout_trajectories(recv_list)
        self._update_rollout_ingest_counters(added, completed)

    async def run_training(self):
        mean_metric_dict = await super().run_training()
        replay_metrics = getattr(self, "_last_replay_metrics", {})
        if replay_metrics:
            mean_metric_dict = {**mean_metric_dict, **replay_metrics}
        return mean_metric_dict
