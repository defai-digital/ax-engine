#!/usr/bin/env bash
# Manually launch ax-engine-server for one model with stderr captured to a
# file, then run the standard bench with --no-build-ax-engine and --skip-server.
# Used to diagnose RemoteDisconnected failures where the bench script's
# stderr=PIPE swallowed the real server panic.
set -uo pipefail
[ "$#" -lt 3 ] && { echo "usage: $0 SLUG MODEL_DIR OUTDIR [extra_server_args...]"; exit 1; }
SLUG="$1"; MODEL_DIR="$2"; OUTDIR="$3"; shift 3
REPO_ROOT="/Users/akiralam/code/ax-engine-v4"
cd "$REPO_ROOT"
mkdir -p "$OUTDIR/logs"

SERVER_LOG="$OUTDIR/logs/${SLUG}-server.stderr"
BENCH_LOG="$OUTDIR/logs/${SLUG}-bench.log"
PORT=8092

pkill -f 'ax-engine-server' >/dev/null 2>&1 || true
sleep 1

echo "[retest] launching ax-engine-server with stderr → $SERVER_LOG" | tee -a "$BENCH_LOG"
# Note: pass-through env (AX_MLX_PREFILL_FFN_COMPILE*, AX_MLX_LOG, RUST_LOG)
AX_MLX_PREFILL_FFN_COMPILE=1 AX_MLX_PREFILL_FFN_COMPILE_SWIGLU=1 \
RUST_LOG=info AX_BENCH_LOG=1 \
nohup target/release/ax-engine-server \
    --mlx --mlx-model-artifacts-dir "$MODEL_DIR" \
    --port "$PORT" \
    --prefill-chunk 2048 --max-batch-tokens 2048 \
    "$@" \
    > "$SERVER_LOG.stdout" 2> "$SERVER_LOG" &
SERVER_PID=$!
echo "[retest] server pid=$SERVER_PID" | tee -a "$BENCH_LOG"
disown

# Wait for /health
echo "[retest] waiting for /health" | tee -a "$BENCH_LOG"
for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[retest] /health ready after ${i}s" | tee -a "$BENCH_LOG"
        break
    fi
    sleep 1
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[retest] FAIL: server died before /health responded" | tee -a "$BENCH_LOG"
        echo "--- server stderr tail ---" | tee -a "$BENCH_LOG"
        tail -50 "$SERVER_LOG" | tee -a "$BENCH_LOG"
        exit 2
    fi
done

# Now run the bench against the running server
echo "[retest] running bench against pre-launched server" | tee -a "$BENCH_LOG"
PYTHONUNBUFFERED=1 python3 scripts/bench_mlx_inference_stack.py \
    --model-dir "$MODEL_DIR" \
    --prompt-tokens 128,512 \
    --generation-tokens 128 \
    --repetitions 5 \
    --cooldown 15 \
    --ax-compare-policies \
    --no-build-ax-engine \
    --axengine-port "$PORT" \
    --axengine-skip-server-launch \
    --reuse-reference-results-from "benchmarks/results/mlx-inference/2026-05-13-full-fresh/${SLUG}.json" \
    --output "$OUTDIR/${SLUG}.json" \
    2>&1 | tee -a "$BENCH_LOG"
BENCH_EXIT=${PIPESTATUS[0]}

echo "[retest] bench exit=$BENCH_EXIT" | tee -a "$BENCH_LOG"
echo "--- server stderr tail (post-bench) ---" | tee -a "$BENCH_LOG"
tail -50 "$SERVER_LOG" | tee -a "$BENCH_LOG"

# Cleanup
if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    sleep 1
    kill -9 "$SERVER_PID" 2>/dev/null || true
fi
exit $BENCH_EXIT
