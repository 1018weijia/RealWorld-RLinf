#!/usr/bin/env bash
# Compatibility wrapper. Prefer: bash toolkits/lerobot/run_sft.sh <config> <gpu> [resume]
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_sft.sh" "$@"
