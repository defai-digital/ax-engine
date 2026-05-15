#!/usr/bin/env bash
# Launch ax-engine-server with stderr captured + send one inference request.
# Reproduces (or rules out) the RemoteDisconnected failure path with full
# server stderr visible.
set -uo pipefail
[ "$#" -lt 2 ] && { echo "usage: $0 SLUG MODEL_DIR [PORT]"; exit 1; }
SLUG="$1"; MODEL_DIR="$2"; PORT="${3:-8093}"
REPO_ROOT="/Users/akiralam/code/ax-engine-v4"
cd "$REPO_ROOT"

LOG_DIR="/tmp/ax-diagnose"
mkdir -p "$LOG_DIR"
SERVER_STDERR="$LOG_DIR/${SLUG}-server.stderr"
SERVER_STDOUT="$LOG_DIR/${SLUG}-server.stdout"
DIAG_LOG="$LOG_DIR/${SLUG}-diag.log"

: > "$SERVER_STDERR"
: > "$SERVER_STDOUT"
: > "$DIAG_LOG"

pkill -f 'ax-engine-server' >/dev/null 2>&1 || true
sleep 2

echo "[diag] launching ax-engine-server (stderr → $SERVER_STDERR)" | tee -a "$DIAG_LOG"
AX_MLX_PREFILL_FFN_COMPILE=1 AX_MLX_PREFILL_FFN_COMPILE_SWIGLU=1 \
RUST_LOG=info AX_BENCH_LOG=1 \
nohup target/release/ax-engine-server \
    --mlx --mlx-model-artifacts-dir "$MODEL_DIR" \
    --port "$PORT" \
    --prefill-chunk 2048 --max-batch-tokens 2048 \
    > "$SERVER_STDOUT" 2> "$SERVER_STDERR" &
SERVER_PID=$!
echo "[diag] server pid=$SERVER_PID" | tee -a "$DIAG_LOG"
disown

# Wait up to 180s for /health
for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[diag] /health ready after ${i}s" | tee -a "$DIAG_LOG"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[diag] FAIL_TYPE=died_before_health" | tee -a "$DIAG_LOG"
        echo "--- server stderr tail ---" | tee -a "$DIAG_LOG"
        tail -80 "$SERVER_STDERR" | tee -a "$DIAG_LOG"
        exit 2
    fi
    sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[diag] FAIL_TYPE=health_timeout" | tee -a "$DIAG_LOG"
    tail -80 "$SERVER_STDERR" | tee -a "$DIAG_LOG"
    kill "$SERVER_PID" 2>/dev/null
    exit 3
fi

# Send a small chat-completion request — mimics what the bench does
echo "[diag] sending /v1/generate request (MLX preview endpoint)" | tee -a "$DIAG_LOG"
RESPONSE=$(curl -sS -m 120 -w "\nHTTPSTATUS:%{http_code}" \
    "http://127.0.0.1:$PORT/v1/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Write a short paragraph about why fall leaves change color.",
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": false
    }' 2>&1) || {
    EXIT=$?
    echo "[diag] FAIL_TYPE=curl_failed (exit=$EXIT)" | tee -a "$DIAG_LOG"
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[diag] (server is dead)" | tee -a "$DIAG_LOG"
    fi
    echo "--- server stderr tail ---" | tee -a "$DIAG_LOG"
    tail -80 "$SERVER_STDERR" | tee -a "$DIAG_LOG"
    exit 4
}

STATUS_LINE=$(echo "$RESPONSE" | grep "^HTTPSTATUS:")
BODY=$(echo "$RESPONSE" | sed '/^HTTPSTATUS:/d')
echo "[diag] response: $STATUS_LINE" | tee -a "$DIAG_LOG"
echo "[diag] body (first 400 chars):" | tee -a "$DIAG_LOG"
echo "$BODY" | head -c 400 | tee -a "$DIAG_LOG"
echo "" | tee -a "$DIAG_LOG"

if echo "$STATUS_LINE" | grep -q "200"; then
    echo "[diag] OUTCOME=ok (server processed request successfully)" | tee -a "$DIAG_LOG"
    OUTCOME=0
else
    echo "[diag] OUTCOME=non200" | tee -a "$DIAG_LOG"
    echo "--- server stderr tail ---" | tee -a "$DIAG_LOG"
    tail -80 "$SERVER_STDERR" | tee -a "$DIAG_LOG"
    OUTCOME=5
fi

# Try a second request
echo "[diag] sending second request to test stability" | tee -a "$DIAG_LOG"
STATUS2=$(curl -sS -m 120 -o /dev/null -w "%{http_code}" \
    "http://127.0.0.1:$PORT/v1/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Say hi.", "max_tokens": 8, "temperature": 0.0, "stream": false}' \
    2>&1) || STATUS2="curl_fail"
echo "[diag] second request status: $STATUS2" | tee -a "$DIAG_LOG"

kill "$SERVER_PID" 2>/dev/null || true
sleep 2
kill -9 "$SERVER_PID" 2>/dev/null || true
echo "[diag] done" | tee -a "$DIAG_LOG"
exit $OUTCOME
