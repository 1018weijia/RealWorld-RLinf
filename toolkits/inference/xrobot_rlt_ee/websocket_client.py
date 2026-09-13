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

"""Synchronous, lightweight WebSocket client for an RLinf RLT server."""

from __future__ import annotations

import logging
import time
from typing import Any

from . import codec

LOG = logging.getLogger("x2robot_rlt.websocket")


class RLTWebSocketClient:
    def __init__(
        self,
        uri: str,
        *,
        connect_timeout: float = 600.0,
        retry_interval: float = 2.0,
        recv_timeout: float | None = 900.0,
    ) -> None:
        self.uri = str(uri)
        self._connect_timeout = float(connect_timeout)
        self._retry_interval = max(0.05, float(retry_interval))
        self._recv_timeout = recv_timeout
        self._connection = None
        self.metadata: dict[str, Any] = {}
        self._connect()

    def _connect(self) -> None:
        import websockets.sync.client

        deadline = time.monotonic() + self._connect_timeout
        last_error: BaseException | None = None
        while True:
            try:
                self._connection = websockets.sync.client.connect(
                    self.uri,
                    compression=None,
                    max_size=None,
                    open_timeout=None,
                )
                metadata = codec.unpackb(self._connection.recv())
                if not isinstance(metadata, dict):
                    raise RuntimeError("RLT metadata is not a mapping")
                self.metadata = metadata
                return
            except BaseException as exc:
                last_error = exc
                self.close()
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"could not connect to RLT server {self.uri}: {last_error}"
                    ) from last_error
                LOG.info("waiting for RLT server %s (%s)", self.uri, exc)
                time.sleep(self._retry_interval)

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError("RLT WebSocket is closed")
        self._connection.send(codec.packb(payload))
        raw = self._connection.recv(timeout=self._recv_timeout)
        if isinstance(raw, str):
            raise RuntimeError(f"RLT server error:\n{raw}")
        response = codec.unpackb(raw)
        if not isinstance(response, dict):
            raise RuntimeError("RLT response is not a mapping")
        return response

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                LOG.debug("ignoring WebSocket close race", exc_info=True)
