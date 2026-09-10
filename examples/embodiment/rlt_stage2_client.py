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

"""Entry point for the robot-side RLT Stage 2 client.

Runs on the machine wired to the arms. It loads no model and needs no GPU: it
uploads raw observations, executes the robot-space chunks the server returns,
and relays the operator's verdicts.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.realworld.rlt_client.cobot import build_cobot_transport
from rlinf.envs.realworld.rlt_client.loop import RLTRobotLoop
from rlinf.serving.websocket_client import RLinfWebsocketClient

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="config", config_name=None)
def main(cfg: DictConfig) -> None:
    """Connect to the server and run episodes until the budget is spent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Client config:\n%s", OmegaConf.to_yaml(cfg))

    transport = build_cobot_transport(cfg)
    client = RLinfWebsocketClient(
        host=str(cfg.client.host),
        port=int(cfg.client.port),
        connect_timeout=float(cfg.client.get("connect_timeout", 600.0)),
        retry_interval=float(cfg.client.get("retry_interval", 2.0)),
        recv_timeout=cfg.client.get("recv_timeout"),
    )
    logger.info(
        "Connected to %s; server metadata: %s", client.uri, client.server_metadata
    )

    loop = RLTRobotLoop(
        transport=transport,
        client=client,
        max_episode_chunks=int(cfg.client.max_episode_chunks),
        exploration_noise_sigma=cfg.client.get("exploration_noise_sigma"),
        success_reward=float(cfg.client.get("success_reward", 1.0)),
        failure_reward=float(cfg.client.get("failure_reward", 0.0)),
        rewind_terminal_reward=float(cfg.client.get("rewind_terminal_reward", -1.0)),
        rewind_prefix_reward=float(cfg.client.get("rewind_prefix_reward", 0.0)),
        validate_handshake=bool(cfg.client.get("validate_handshake", True)),
    )

    outcomes = loop.run(int(cfg.client.num_episodes))
    successes = sum(1 for outcome in outcomes if outcome.success)
    logger.info(
        "Finished %d episodes, %d successful (%.1f%%)",
        len(outcomes),
        successes,
        100.0 * successes / max(1, len(outcomes)),
    )


if __name__ == "__main__":
    main()
