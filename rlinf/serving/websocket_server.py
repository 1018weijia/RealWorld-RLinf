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

"""WebSocket transport for single-process RLinf policy servers.

The wire format is byte-compatible with ``openpi``: msgpack with the
``openpi_client.msgpack_numpy`` numpy extension, one metadata frame pushed on
connect, then request/response frames.  This lets an existing ``openpi_client``
robot client talk to an RLinf learner unchanged.

The handler is implemented here rather than reused from ``openpi`` because RLT
Stage 2 needs two things the packaged server lacks: configurable
``ping_interval``/``ping_timeout``, and inference dispatched through
:func:`asyncio.to_thread` so long GPU work (a training burst at
``episode_end`` can take minutes) does not stall keepalive and drop the robot
connection mid-episode.
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class WebsocketPolicy(Protocol):
    """Minimal policy surface the transport needs.

    ``infer`` is called from a worker thread, so implementations are
    responsible for their own locking.
    """

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle one decoded request frame and return the response payload."""
        ...


class RLinfWebsocketPolicyServer:
    """Serve a policy over WebSocket using the openpi msgpack wire format.

    Args:
        policy: Object exposing ``infer(request) -> response``.
        host: Bind address.
        port: Bind port.
        metadata: Handshake payload pushed as the first frame of every
            connection. Clients use it to validate protocol compatibility
            before sending any request.
        ping_interval: Seconds between keepalive pings, or ``None`` to disable.
        ping_timeout: Seconds to wait for a pong before dropping the
            connection, or ``None`` to wait forever. Real-robot runs usually
            want ``None`` or a large value because a training burst blocks the
            worker thread.
        max_size: Maximum inbound frame size in bytes, or ``None`` for
            unlimited. Camera frames easily exceed the library default.
    """

    def __init__(
        self,
        policy: WebsocketPolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict[str, Any] | None = None,
        ping_interval: float | None = 20.0,
        ping_timeout: float | None = None,
        max_size: int | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._max_size = max_size
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """Block the calling thread serving requests until interrupted."""
        asyncio.run(self.run())

    async def run(self) -> None:
        """Bind the socket and serve until the server is closed."""
        import websockets.asyncio.server as _server

        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=self._max_size,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            process_request=_health_check,
        ) as server:
            logger.info(
                "RLinf websocket policy server listening on %s:%s",
                self._host,
                self._port,
            )
            await server.serve_forever()

    async def _handler(self, websocket) -> None:
        import websockets
        import websockets.frames
        from openpi_client import msgpack_numpy

        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                request = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                # Policy work is synchronous and can run for minutes during a
                # training burst. Run it on a worker thread so ping/pong
                # keepalive keeps flowing on the event loop.
                response = await asyncio.to_thread(self._policy.infer, request)
                infer_time = time.monotonic() - infer_time

                response["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    # Only the previous round trip is known here, because the
                    # current one is not finished until the send completes.
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                # Send the traceback as a text frame so the robot client can
                # surface the server-side failure, then close hard.
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback in previous frame.",
                )
                raise


def _health_check(connection, request):
    """Answer ``GET /healthz`` without upgrading to WebSocket."""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
