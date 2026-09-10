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

"""Ray-free serving entrypoints for real-robot RLinf deployments.

The modules here run a policy in a single process behind a WebSocket socket.
Nothing in this package imports :mod:`rlinf.scheduler`, so a robot workstation
can host a learner without a Ray cluster.
"""

from rlinf.serving.websocket_client import RLinfWebsocketClient
from rlinf.serving.websocket_server import (
    RLinfWebsocketPolicyServer,
    WebsocketPolicy,
)

__all__ = [
    "RLinfWebsocketClient",
    "RLinfWebsocketPolicyServer",
    "WebsocketPolicy",
]
