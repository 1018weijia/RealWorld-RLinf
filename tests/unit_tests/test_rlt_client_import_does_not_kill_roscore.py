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

"""Importing the Stage 2 client must not tear down a running ROS stack.

The client entry is ``from rlinf.envs.realworld.rlt_client.cobot import ...``.
Python runs the parent package ``__init__`` first. That file used to call
``RealWorldEnv.realworld_setup()``, which killed every ``roscore`` /
``rosmaster`` / ``rosout`` on the node. ``run_cobot_control.sh`` starts ROS
and then hands off to the client, so the import would take down the stack
it had just brought up.
"""

from __future__ import annotations

import sys

import psutil
import pytest

_REALWORLD_PREFIX = "rlinf.envs.realworld"


class _RecordingProcess:
    """Wrap a live ``psutil.Process`` so ``kill`` is recorded, not executed."""

    def __init__(self, proc: psutil.Process, killed: list[str]) -> None:
        self._proc = proc
        self._killed = killed

    def name(self) -> str:
        return self._proc.name()

    def kill(self) -> None:
        self._killed.append(self.name())

    def __getattr__(self, item: str):
        return getattr(self._proc, item)


def _drop_realworld_modules() -> None:
    """Force the next import to re-run ``rlinf.envs.realworld.__init__``."""
    for name in list(sys.modules):
        if name == _REALWORLD_PREFIX or name.startswith(_REALWORLD_PREFIX + "."):
            del sys.modules[name]


def test_importing_the_rlt_client_does_not_kill_roscore(
    monkeypatch: pytest.MonkeyPatch,
):
    """The client import path must leave roscore / rosmaster / rosout alone."""
    killed: list[str] = []
    original_iter = psutil.process_iter

    def wrapped_iter(*args, **kwargs):
        for proc in original_iter(*args, **kwargs):
            yield _RecordingProcess(proc, killed)

    monkeypatch.setattr(psutil, "process_iter", wrapped_iter)
    _drop_realworld_modules()

    from rlinf.envs.realworld.rlt_client.cobot import build_cobot_transport

    assert build_cobot_transport is not None
    assert killed == []


def test_realworld_setup_still_kills_when_called_explicitly(
    monkeypatch: pytest.MonkeyPatch,
):
    """The helper remains available; only the import-time call was removed."""
    killed: list[str] = []

    class _FakeProc:
        def __init__(self, name: str) -> None:
            self._name = name

        def name(self) -> str:
            return self._name

        def kill(self) -> None:
            killed.append(self._name)

    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda *args, **kwargs: iter(
            [_FakeProc("roscore"), _FakeProc("rosmaster"), _FakeProc("unrelated")]
        ),
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.realworld_env.FileLock",
        lambda *args, **kwargs: _NullLock(),
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.realworld_env.time.sleep", lambda *_: None
    )

    from rlinf.envs.realworld.realworld_env import RealWorldEnv

    RealWorldEnv.realworld_setup()
    assert killed == ["roscore", "rosmaster"]


class _NullLock:
    """Stand-in for ``FileLock`` so the explicit-call test stays offline."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
