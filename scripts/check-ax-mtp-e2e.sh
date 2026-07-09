#!/usr/bin/env bash
# Direct AX Engine MTP + n-gram integration smoke.
#
# Starts ax-engine-server with MTP + n-gram, sends a chat completion through
# its OpenAI-compatible /v1 endpoint, and checks the response for the
# corruption patterns reported in issue #36 / #37.  No Docker required.
#
# Required env vars:
#   AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR — path to MTP model snapshot (e.g. Qwen3.6-27B-MTP)
#
# Optional env vars:
#   AX_MTP_E2E_PROMPT       — prompt string (default: "what is agi ?")
#   AX_MTP_E2E_MAX_TOKENS   — max generation tokens (default: 96)
#   AX_MTP_E2E_TIMEOUT_SECS — readiness + request timeout in seconds (default: 120)
#   AX_MTP_E2E_MTP_DEPTH    — MTP max draft depth (default: 3)
#   AX_MTP_E2E_NGRAM        — 1=enable n-gram stacking (default: 1)
#   AX_MTP_E2E_REPORT       — path for JSON report (default: /tmp)
#   AX_MTP_E2E_MODEL_ID     — model-id served by ax-engine (default: ax-mtp-e2e)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

ax_require_env AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR is required"

PROMPT="${AX_MTP_E2E_PROMPT:-what is agi ?}"
MAX_TOKENS="${AX_MTP_E2E_MAX_TOKENS:-96}"
TIMEOUT_SECS="${AX_MTP_E2E_TIMEOUT_SECS:-120}"
MTP_DEPTH="${AX_MTP_E2E_MTP_DEPTH:-3}"
NGRAM="${AX_MTP_E2E_NGRAM:-1}"
# Leave MODEL_ID empty to auto-detect from /v1/models after server starts
MODEL_ID="${AX_MTP_E2E_MODEL_ID:-}"
REPORT="${AX_MTP_E2E_REPORT:-$(ax_tmp_file ax-mtp-e2e .json)}"

AX_PORT="$(ax_allocate_port)"
AX_BASE_URL="http://127.0.0.1:${AX_PORT}"
export AX_BASE_URL
AX_PID=""
AX_LOG="$(ax_tmp_file ax-mtp-e2e-server .log)"

AX_BIN="${ROOT_DIR}/target/release/ax-engine-server"
if [[ ! -x "$AX_BIN" ]]; then
    echo "[ax-mtp-e2e] ax-engine-server binary not found at ${AX_BIN}" >&2
    echo "[ax-mtp-e2e] run: cargo build --release -p ax-engine-server" >&2
    exit 1
fi

cleanup() {
    ax_kill_pid "$AX_PID"
    ax_rm_rf "$AX_LOG"
}
trap 'ax_run_cleanup "$?" cleanup' EXIT

cd "$ROOT_DIR"

NGRAM_FLAG=""
if [[ "$NGRAM" == "0" ]]; then
    NGRAM_FLAG="--mlx-mtp-disable-ngram-stacking"
fi

echo "[ax-mtp-e2e] starting ax-engine-server port=${AX_PORT} depth=${MTP_DEPTH} ngram=${NGRAM}"
AX_MLX_MTP_MAX_DEPTH="$MTP_DEPTH" \
    "$AX_BIN" \
    --mlx \
    --mlx-model-artifacts-dir "$AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR" \
    --port "$AX_PORT" \
    --prefill-chunk 2048 \
    --max-batch-tokens 2048 \
    ${NGRAM_FLAG} \
    >"$AX_LOG" 2>&1 &
AX_PID="$!"

# Auto-detect model-id if not set: wait for server then read first model from /v1/models
if [[ -z "$MODEL_ID" ]]; then
    echo "[ax-mtp-e2e] waiting for server to detect model-id..."
    for _ in $(seq 1 60); do
        DETECTED="$("$PYTHON_BIN" - <<'PY'
import sys, urllib.request, json, os
try:
    url = os.environ["AX_BASE_URL"] + "/v1/models"
    with urllib.request.urlopen(url, timeout=3) as r:
        d = json.loads(r.read())
    ids = [m["id"] for m in d.get("data", []) if isinstance(m.get("id"), str)]
    print(ids[0] if ids else "")
except Exception:
    print("")
PY
        )" 2>/dev/null || true
        if [[ -n "$DETECTED" ]]; then
            MODEL_ID="$DETECTED"
            break
        fi
        sleep 1
    done
    if [[ -z "$MODEL_ID" ]]; then
        echo "[ax-mtp-e2e] failed: could not detect model-id from server" >&2
        exit 1
    fi
    echo "[ax-mtp-e2e] detected model-id: ${MODEL_ID}"
fi

echo "[ax-mtp-e2e] probe url=${AX_BASE_URL} model=${MODEL_ID} prompt=${PROMPT}"
"$PYTHON_BIN" "$SCRIPT_DIR/openwebui_e2e.py" \
    --ax-direct \
    --openwebui-base-url "$AX_BASE_URL" \
    --model-id "$MODEL_ID" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --timeout-secs "$TIMEOUT_SECS" \
    --report "$REPORT"

echo "[ax-mtp-e2e] report: $REPORT"
