#!/usr/bin/env bash
# Local GPU backend for generating committed demo fixtures. The deployed browser
# is cache-only and never connects to this process.
set -euo pipefail

PORT="${PORT:-8000}"
PY="${PY:-/home/ubuntu/doc-to-lora/.venv/bin/python}"
CAPTURE_FIXTURES="${CAPTURE_FIXTURES:-0}"
CAPTURE_SIZES="${CAPTURE_SIZES:-small medium}"
LOGDIR="${LOGDIR:-logs}"
SERVER_PID=""

mkdir -p "$LOGDIR"

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$PY" -m uvicorn agenthn.webapp.app:app --host 127.0.0.1 --port "$PORT" \
  >"$LOGDIR/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null; then
    echo "fixture backend ready at http://127.0.0.1:$PORT"
    if [ "$CAPTURE_FIXTURES" = "1" ]; then
      CAPTURE_BASE="http://127.0.0.1:$PORT" CAPTURE_SIZES="$CAPTURE_SIZES" \
        "$PY" scripts/capture_fixtures.py
      exit 0
    fi
    wait "$SERVER_PID"
    exit $?
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "fixture backend exited; see $LOGDIR/server.log" >&2
    exit 1
  fi
  sleep 1
done

echo "fixture backend did not become healthy; see $LOGDIR/server.log" >&2
exit 1
