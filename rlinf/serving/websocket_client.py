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

"""Blocking WebSocket client for RLinf single-process policy servers."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class RLinfWebsocketClient:
    """Synchronous request/response client for :class:`RLinfWebsocketPolicyServer`.

    The robot control loop is inherently synchronous and single-threaded, so
    this deliberately uses ``websockets.sync`` rather than asyncio.

    Args:
        host: Server hostname, IP, or a full ``ws://``/``wss://`` URI.
        port: Server port, ignored when ``host`` already carries a URI.
        connect_timeout: Seconds to keep retrying the initial connect before
            giving up. ``0`` fails on the first refused connection.
        retry_interval: Seconds between connect attempts.
        recv_timeout: Seconds to wait for one response. ``None`` waits
            forever, which is what real runs want because ``episode_end``
            blocks on a training burst.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        connect_timeout: float = 120.0,
        retry_interval: float = 2.0,
        recv_timeout: float | None = None,
    ) -> None:
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        netloc = self._uri.split("//", 1)[-1]
        if port is not None and ":" not in netloc:
            self._uri = f"{self._uri}:{port}"
        self._connect_timeout = float(connect_timeout)
        self._retry_interval = max(0.05, float(retry_interval))
        self._recv_timeout = recv_timeout

        from openpi_client import msgpack_numpy

        self._packer = msgpack_numpy.Packer()
        self._unpackb = msgpack_numpy.unpackb
        self._connection, self._server_metadata = self._wait_for_server()

    @property
    def uri(self) -> str:
        """The resolved server URI."""
        return self._uri

    @property
    def server_metadata(self) -> dict[str, Any]:
        """Handshake metadata received on connect."""
        return self._server_metadata

    def _wait_for_server(self):
        import websockets.sync.client

        deadline = time.monotonic() + self._connect_timeout
        last_error: Exception | None = None
        while True:
            try:
                connection = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=None,
                )
                metadata = self._unpackb(connection.recv())
                logger.info("Connected to RLinf policy server at %s", self._uri)
                return connection, metadata
            except Exception as error:  # noqa: BLE001 - retried below
                last_error = error
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Could not reach RLinf policy server at {self._uri} "
                        f"within {self._connect_timeout:.0f}s: {last_error}"
                    ) from last_error
                logger.info(
                    "Waiting for RLinf policy server at %s (%s)", self._uri, error
                )
                time.sleep(self._retry_interval)

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return the decoded response.

        Raises:
            RuntimeError: The server replied with a text frame, which the
                transport reserves for server-side tracebacks.
        """
        self._connection.send(self._packer.pack(payload))
        response = self._connection.recv(timeout=self._recv_timeout)
        if isinstance(response, str):
            raise RuntimeError(f"RLinf policy server error:\n{response}")
        return self._unpackb(response)

    def close(self) -> None:
        """Close the underlying connection, ignoring shutdown races."""
        try:
            self._connection.close()
        except Exception:  # noqa: BLE001 - best effort on shutdown
            logger.debug("Ignoring error while closing %s", self._uri, exc_info=True)
