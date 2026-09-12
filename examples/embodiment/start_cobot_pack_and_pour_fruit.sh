#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/start_cobot_stage2.sh" pack_and_pour_fruit "$@"
