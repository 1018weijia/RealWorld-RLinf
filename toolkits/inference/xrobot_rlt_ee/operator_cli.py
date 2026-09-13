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

"""Queue one operator outcome for the X2Robot RLT bridge.

``V2EventAdapter`` already forwards takeover, physical rewind, success and
abort automatically. This CLI covers the verdicts V2 does not emit -- most
importantly ``failure``, which has no V2 event -- and gives a way to inspect
the bridge from a shell.
"""

from __future__ import annotations

import argparse
import json
import socket

COMMANDS = (
    "success",
    "failure",
    "intervention",
    "abort",
    "rewind_exit",
    "rewind_credit",
    "status",
)


def main() -> int:
    """Send one command to the bridge operator port and print the reply."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=33058)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--terminal-reward", type=float, default=0.0)
    parser.add_argument("--prefix-reward", type=float, default=0.1)
    parser.add_argument("--confidence", type=float, default=1.0)
    args = parser.parse_args()
    payload = {
        "command": args.command,
        "chunks": args.chunks,
        "terminal_reward": args.terminal_reward,
        "prefix_reward": args.prefix_reward,
        "confidence": args.confidence,
    }
    with socket.create_connection((args.host, args.port), timeout=5.0) as connection:
        connection.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        reply = connection.makefile("rb").readline(4096)
    decoded = json.loads(reply.decode("utf-8"))
    print(json.dumps(decoded, ensure_ascii=False))
    return 0 if decoded.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
