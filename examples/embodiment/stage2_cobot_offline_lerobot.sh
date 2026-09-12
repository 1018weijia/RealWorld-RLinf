#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task="${1:-assemble_parts}"
if (( $# )); then shift; fi
exec bash "$root/start_cobot_stage2.sh" "$task" offline "$@"
