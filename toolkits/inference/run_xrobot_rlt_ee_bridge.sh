#!/usr/bin/env bash
# Start only the X2Robot EE14 <-> RLinf RLT protocol bridge.
# This script doesn't start control, publish ROS commands, or home the robot.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${CONTAINER:-desktop-robot_client-1}"
UPSTREAM_URI="${RLT_UPSTREAM_URI:-ws://127.0.0.1:8000}"
LISTEN_PORT="${RLT_EE_BRIDGE_PORT:-33057}"
OPERATOR_PORT="${RLT_EE_OPERATOR_PORT:-33058}"
TASK_PROMPT="${RLT_TASK_PROMPT:-put ring on the rod}"
ALLOW_MOTION="${RLT_EE_ALLOW_MOTION:-false}"
EXPLORATION_SIGMA="${RLT_EXPLORATION_NOISE_SIGMA:-}"
RUNTIME_DIR="/tmp/x2robot_rlt_ee"
LOG_PATH="/tmp/x2robot_rlt_ee_bridge.log"

bridge_running() {
  docker exec "$CONTAINER" pgrep -f '[r]lt_ee.bridge' >/dev/null 2>&1
}

stop_bridge() {
  docker exec "$CONTAINER" pkill -TERM -f '[r]lt_ee.bridge' >/dev/null 2>&1 || true
}

case "${1:-}" in
  --status)
    if bridge_running; then
      echo "[rlt-ee] bridge running"
      docker exec "$CONTAINER" tail -n 30 "$LOG_PATH" || true
    else
      echo "[rlt-ee] bridge stopped"
    fi
    exit 0
    ;;
  --stop)
    stop_bridge
    echo "[rlt-ee] bridge stopped (robot control was not changed)"
    exit 0
    ;;
  "") ;;
  *) echo "Usage: $0 [--status|--stop]" >&2; exit 2 ;;
esac

case "$ALLOW_MOTION" in true|false) ;; *)
  echo "ERROR: RLT_EE_ALLOW_MOTION must be true or false" >&2; exit 2 ;;
esac
[[ "$UPSTREAM_URI" =~ ^wss?:// ]] || {
  echo "ERROR: RLT_UPSTREAM_URI must start with ws:// or wss://" >&2; exit 2;
}

stop_bridge
docker exec "$CONTAINER" rm -rf "$RUNTIME_DIR"
docker cp "$SCRIPT_DIR/xrobot_rlt_ee" "$CONTAINER:$RUNTIME_DIR"

PROBE_ARG="--probe-only"
if [[ "$ALLOW_MOTION" == true ]]; then
  PROBE_ARG="--no-probe-only"
  echo "[rlt-ee] WARNING: motion forwarding explicitly enabled"
else
  echo "[rlt-ee] probe-only: upstream actions cannot reach DesktopClient"
fi

SIGMA_ARG=()
if [[ -n "$EXPLORATION_SIGMA" ]]; then
  SIGMA_ARG=(--exploration-noise-sigma "$EXPLORATION_SIGMA")
fi
sigma_cli="${SIGMA_ARG[*]}"

docker exec -d \
  -e PYTHONPATH=/tmp \
  -e UPSTREAM_URI="$UPSTREAM_URI" \
  -e LISTEN_PORT="$LISTEN_PORT" \
  -e OPERATOR_PORT="$OPERATOR_PORT" \
  -e TASK_PROMPT="$TASK_PROMPT" \
  -e PROBE_ARG="$PROBE_ARG" \
  -e SIGMA_CLI="$sigma_cli" \
  "$CONTAINER" bash -lc '
    source /opt/xr/py_env/bin/activate
    # shellcheck disable=SC2086
    exec python -m x2robot_rlt_ee.bridge \
      --upstream-uri "$UPSTREAM_URI" \
      --listen-host 127.0.0.1 --listen-port "$LISTEN_PORT" \
      --operator-host 127.0.0.1 --operator-port "$OPERATOR_PORT" \
      --task "$TASK_PROMPT" --chunk-length 50 "$PROBE_ARG" \
      $SIGMA_CLI \
      > /tmp/x2robot_rlt_ee_bridge.log 2>&1
  '

for _ in {1..50}; do
  bridge_running && break
  sleep 0.1
done
bridge_running || {
  echo "ERROR: RLT EE bridge failed to start" >&2
  docker exec "$CONTAINER" tail -n 60 "$LOG_PATH" >&2 || true
  exit 1
}

echo "[rlt-ee] bridge: ws://127.0.0.1:${LISTEN_PORT} -> ${UPSTREAM_URI}"
echo "[rlt-ee] operator control: tcp://127.0.0.1:${OPERATOR_PORT}"
