#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

if [[ "${AX_OPENWEBUI_E2E:-0}" != "1" ]]; then
    echo "[openwebui-e2e] skipped: set AX_OPENWEBUI_E2E=1 to run"
    exit 0
fi

command -v docker >/dev/null 2>&1 || {
    echo "[openwebui-e2e] failed: AX_OPENWEBUI_E2E=1 requires docker" >&2
    exit 1
}

OPENWEBUI_IMAGE="${AX_OPENWEBUI_IMAGE:-ghcr.io/open-webui/open-webui:main}"
OPENWEBUI_HOST="${AX_OPENWEBUI_HOST:-127.0.0.1}"
OPENWEBUI_PORT="${AX_OPENWEBUI_PORT:-$(ax_allocate_port)}"
OPENWEBUI_URL="http://${OPENWEBUI_HOST}:${OPENWEBUI_PORT}"
OPENWEBUI_CONTAINER="${AX_OPENWEBUI_CONTAINER:-ax-engine-openwebui-e2e-${OPENWEBUI_PORT}}"
OPENWEBUI_DATA_DIR_TEMP=0
if [[ -n "${AX_OPENWEBUI_DATA_DIR:-}" ]]; then
    OPENWEBUI_DATA_DIR="$AX_OPENWEBUI_DATA_DIR"
else
    OPENWEBUI_DATA_DIR="$(ax_tmp_dir ax-openwebui-data)"
    OPENWEBUI_DATA_DIR_TEMP=1
fi
OPENWEBUI_REPORT="${AX_OPENWEBUI_REPORT:-$(ax_tmp_file ax-openwebui-e2e .json)}"

AX_BASE_URL="${AX_OPENWEBUI_AX_BASE_URL:-http://127.0.0.1:8080}"
MODEL_ID="${AX_OPENWEBUI_MODEL_ID:-ax-engine-openwebui-smoke}"
PROMPT="${AX_OPENWEBUI_PROMPT:-what is agi ?}"
MAX_TOKENS="${AX_OPENWEBUI_MAX_TOKENS:-96}"
TIMEOUT_SECS="${AX_OPENWEBUI_TIMEOUT_SECS:-180}"

AX_SHIM_PID=""
AX_SHIM_LOG=""

cleanup() {
    if [[ "${AX_OPENWEBUI_KEEP_CONTAINER:-0}" != "1" ]]; then
        docker rm -f "$OPENWEBUI_CONTAINER" >/dev/null 2>&1 || true
    fi
    ax_kill_pid "$AX_SHIM_PID"
    if [[ "${AX_OPENWEBUI_KEEP_CONTAINER:-0}" != "1" ]] && [[ "$OPENWEBUI_DATA_DIR_TEMP" == "1" ]]; then
        ax_rm_rf "$OPENWEBUI_DATA_DIR"
    fi
    ax_rm_rf "$AX_SHIM_LOG"
}

trap 'ax_run_cleanup "$?" cleanup' EXIT

cd "$ROOT_DIR"

if [[ "${AX_OPENWEBUI_START_AX_SHIM:-0}" == "1" ]]; then
    ax_require_env AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR is required when AX_OPENWEBUI_START_AX_SHIM=1"
    ax_require_env AX_OPENWEBUI_TOKENIZER "AX_OPENWEBUI_TOKENIZER is required when AX_OPENWEBUI_START_AX_SHIM=1"
    AX_SHIM_PORT="${AX_OPENWEBUI_AX_PORT:-$(ax_allocate_port)}"
    AX_BASE_URL="http://127.0.0.1:${AX_SHIM_PORT}"
    AX_SHIM_LOG="$(ax_tmp_file ax-openwebui-ax-shim .log)"
    "$PYTHON_BIN" -m ax_engine.openai_server \
        --model-id "$MODEL_ID" \
        --mlx-model-artifacts-dir "$AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR" \
        --tokenizer "$AX_OPENWEBUI_TOKENIZER" \
        --host 127.0.0.1 \
        --port "$AX_SHIM_PORT" >"$AX_SHIM_LOG" 2>&1 &
    AX_SHIM_PID="$!"
fi

DOCKER_AX_BASE_URL="$("$PYTHON_BIN" "$SCRIPT_DIR/openwebui_e2e.py" \
    --print-docker-openai-base-url "$AX_BASE_URL")"

echo "[openwebui-e2e] starting OpenWebUI image=${OPENWEBUI_IMAGE} url=${OPENWEBUI_URL}"
docker rm -f "$OPENWEBUI_CONTAINER" >/dev/null 2>&1 || true
docker run -d \
    --name "$OPENWEBUI_CONTAINER" \
    -p "${OPENWEBUI_HOST}:${OPENWEBUI_PORT}:8080" \
    -e WEBUI_AUTH=False \
    -e ENABLE_PERSISTENT_CONFIG=False \
    -e OPENAI_API_BASE_URL="$DOCKER_AX_BASE_URL" \
    -e OPENAI_API_BASE_URLS="$DOCKER_AX_BASE_URL" \
    -e OPENAI_API_KEY="${AX_OPENWEBUI_OPENAI_API_KEY:-ax-engine-local}" \
    -e OPENAI_API_KEYS="${AX_OPENWEBUI_OPENAI_API_KEY:-ax-engine-local}" \
    -v "${OPENWEBUI_DATA_DIR}:/app/backend/data" \
    "$OPENWEBUI_IMAGE" >/dev/null

"$PYTHON_BIN" "$SCRIPT_DIR/openwebui_e2e.py" \
    --openwebui-base-url "$OPENWEBUI_URL" \
    --model-id "$MODEL_ID" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --timeout-secs "$TIMEOUT_SECS" \
    --report "$OPENWEBUI_REPORT"

echo "[openwebui-e2e] report: $OPENWEBUI_REPORT"
