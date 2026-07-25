#!/usr/bin/env bash
# Cheapest binary-level validation that TensorRT-LLM and TensorRT Edge-LLM
# session construction fail closed without machine-readable runtime identity.
# Does not require a live worker, GPU model weights, or network.
#
# Usage:
#   AX_BIN=./target/release-server/ax-engine-server \
#     bash scripts/validate_tensorrt_identity_fail_closed.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AX_BIN="${AX_BIN:-}"
if [[ -z "$AX_BIN" ]]; then
  for candidate in \
    ./target/release-server/ax-engine-server \
    ./target/release/ax-engine-server \
    ./target/debug/ax-engine-server
  do
    if [[ -x "$candidate" ]]; then
      AX_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "${AX_BIN}" || ! -x "$AX_BIN" ]]; then
  echo "error: set AX_BIN to a built ax-engine-server binary" >&2
  exit 2
fi

LOG_DIR="${LOG_DIR:-/tmp}"
mkdir -p "$LOG_DIR"

fail_closed() {
  local name="$1"
  shift
  local log="${LOG_DIR}/ax-identity-${name}.log"
  set +e
  "$AX_BIN" "$@" >"$log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    echo "FAIL ${name}: expected non-zero exit, got 0" >&2
    tail -40 "$log" >&2
    return 1
  fi
  if ! grep -Eiq 'machine-readable runtime identity|upstream-version|execution-backend' "$log"; then
    echo "FAIL ${name}: log missing identity error; exit=${rc}" >&2
    tail -40 "$log" >&2
    return 1
  fi
  echo "PASS ${name}: fail-closed exit=${rc}"
}

echo "AX_BIN=${AX_BIN}"
fail_closed tensor-rt-llm-missing-identity \
  --support-tier tensor-rt-llm \
  --tensorrt-llm-server-url http://127.0.0.1:8000 \
  --model-id smoke-model \
  --host 127.0.0.1 \
  --port 0

fail_closed tensor-rt-edge-llm-missing-identity \
  --support-tier tensor-rt-edge-llm \
  --edge-llm-server-url http://127.0.0.1:8090 \
  --model-id smoke-model \
  --host 127.0.0.1 \
  --port 0

echo "IDENTITY_FAIL_CLOSED_OK"
