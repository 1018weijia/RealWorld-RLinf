#!/usr/bin/env bash
# Subscribe to V2 takeover events and feed real EE execution receipts to RLT.
# This process only reads ROS events and talks to the local bridge control port.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${CONTAINER:-desktop-robot_client-1}"
OPERATOR_PORT="${RLT_EE_OPERATOR_PORT:-33058}"
RUNTIME_DIR="/tmp/x2robot_rlt_ee"
LOG_PATH="/tmp/x2robot_rlt_ee_v2_adapter.log"

running() {
  docker exec "$CONTAINER" pgrep -f '[r]lt_ee.v2_event_adapter' >/dev/null 2>&1
}

stop_adapter() {
  docker exec "$CONTAINER" pkill -TERM -f '[r]lt_ee.v2_event_adapter' >/dev/null 2>&1 || true
}

case "${1:-}" in
  --status)
    if running; then
      echo "[rlt-ee-v2] adapter running"
      docker exec "$CONTAINER" tail -n 30 "$LOG_PATH" || true
    else
      echo "[rlt-ee-v2] adapter stopped"
    fi
    exit 0
    ;;
  --stop)
    stop_adapter
    echo "[rlt-ee-v2] adapter stopped"
    exit 0
    ;;
  "") ;;
  *) echo "Usage: $0 [--status|--stop]" >&2; exit 2 ;;
esac

# Refresh the same package used by the bridge; neither copy operation starts control.
docker exec "$CONTAINER" rm -rf "$RUNTIME_DIR"
docker cp "$SCRIPT_DIR/xrobot_rlt_ee" "$CONTAINER:$RUNTIME_DIR"
stop_adapter
docker exec -d \
  -e PYTHONPATH=/tmp \
  -e OPERATOR_PORT="$OPERATOR_PORT" \
  "$CONTAINER" bash -lc '
    source /opt/xr/py_env/bin/activate
    set +u
    source /opt/xr/bot/setup.bash
    set -u
    exec python -m x2robot_rlt_ee.v2_event_adapter \
      --bridge-host 127.0.0.1 --bridge-port "$OPERATOR_PORT" \
      --ros-topic /take_over_data --chunk-length 50 --control-hz 30 \
      --rewind-mode rewind_exit \
      > /tmp/x2robot_rlt_ee_v2_adapter.log 2>&1
  '

for _ in {1..50}; do
  running && break
  sleep 0.1
done
running || {
  echo "ERROR: RLT V2 adapter failed to start" >&2
  docker exec "$CONTAINER" tail -n 60 "$LOG_PATH" >&2 || true
  exit 1
}
echo "[rlt-ee-v2] /take_over_data -> RLT operator port ${OPERATOR_PORT}"
echo "[rlt-ee-v2] healthy V2 physical rollback -> RLT rewind_exit"
