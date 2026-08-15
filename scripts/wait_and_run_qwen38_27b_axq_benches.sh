#!/usr/bin/env bash
# Sleep until the first attempt, then retry hourly until the Qwen 3.8 27B
# AXQ benches can start on $AX_BENCH_HOST.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INITIAL_SLEEP_SECONDS="${AX_BENCH_INITIAL_SLEEP_SECONDS:-86400}"
RETRY_SLEEP_SECONDS="${AX_BENCH_RETRY_SLEEP_SECONDS:-3600}"
RUNNER="$ROOT/scripts/run_qwen38_27b_axq_benches.sh"

echo "[wait] first attempt in ${INITIAL_SLEEP_SECONDS}s"
sleep "$INITIAL_SLEEP_SECONDS"

attempt=1
while true; do
  echo "[wait] attempt ${attempt} at $(date '+%Y-%m-%dT%H:%M:%S%z')"
  set +e
  bash "$RUNNER"
  status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    echo "[wait] benches started and finished"
    exit 0
  fi
  if [ "$status" -ne 75 ]; then
    echo "[wait] runner failed with status $status; retrying anyway" >&2
  fi
  echo "[wait] retry in ${RETRY_SLEEP_SECONDS}s"
  sleep "$RETRY_SLEEP_SECONDS"
  attempt=$((attempt + 1))
done
