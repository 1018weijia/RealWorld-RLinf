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

"""Backward-compatible entrypoint for Cobot cube_into_drawer.

Prefer::

    bash toolkits/inference/run_infer_server.sh cobot_cube_into_drawer 0 8000
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    """Launch with the cube_into_drawer YAML if ``--config`` was not passed."""
    here = Path(__file__).resolve().parent
    default_cfg = here / "config" / "cobot_cube_into_drawer.yaml"
    argv = sys.argv[1:]
    if "--config" not in argv:
        argv = ["--config", str(default_cfg), *argv]
        sys.argv = [sys.argv[0], *argv]
    # Run the YAML-driven server as __main__.
    runpy.run_path(str(here / "openpi_rlinf_infer_server.py"), run_name="__main__")


if __name__ == "__main__":
    main()
