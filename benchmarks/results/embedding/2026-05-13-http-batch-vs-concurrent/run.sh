#!/usr/bin/env bash
# Measure the new /v1/embeddings batch-input contract against the
# concurrent-single-input path that the microbatcher serves. Both
# paths should produce roughly the same per-sentence throughput; the
# batch path saves the caller from fanning out N HTTP requests.
set -uo pipefail

REPO_ROOT="/Users/akiralam/code/ax-engine-v4"
cd "$REPO_ROOT"
source .venv/bin/activate

OUT="benchmarks/results/embedding/2026-05-13-http-batch-vs-concurrent"
mkdir -p "$OUT"

cleanup() {
    pkill -f 'ax-engine-server' >/dev/null 2>&1 || true
    sleep 2
}
trap cleanup EXIT

run_one() {
    local label="$1"
    local model_dir="$2"
    cleanup
    echo ""
    echo "=== $label ==="
    "$REPO_ROOT/target/release/ax-engine-server" \
        --model-id qwen3_dense --support-tier mlx-preview --mlx \
        --mlx-model-artifacts-dir "$model_dir" \
        --host 127.0.0.1 --port 8083 \
        > "$OUT/server-$label.log" 2>&1 &
    server_pid=$!
    # wait for port
    for _ in $(seq 1 60); do
        if curl -sSf http://127.0.0.1:8083/health >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -sSf http://127.0.0.1:8083/health >/dev/null 2>&1; then
        echo "server failed to start"; kill $server_pid 2>/dev/null; return
    fi

    python - "$label" "$OUT" <<'PYEND'
import json, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
import http.client

label, outdir = sys.argv[1:]
sentences = [list(range(n)) for n in [10,15,13,8,3,8,10,8,10,10]]
total_tokens = sum(len(s) for s in sentences)

def post_single(ids):
    conn = http.client.HTTPConnection("127.0.0.1", 8083, timeout=60)
    body = json.dumps({"input": ids, "pooling": "last", "normalize": True}).encode()
    conn.request("POST", "/v1/embeddings", body=body, headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    assert r.status == 200, r.status
    r.read()
    conn.close()

def post_batch(batch):
    conn = http.client.HTTPConnection("127.0.0.1", 8083, timeout=60)
    body = json.dumps({"input": batch, "pooling": "last", "normalize": True}).encode()
    conn.request("POST", "/v1/embeddings", body=body, headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    assert r.status == 200, r.status
    r.read()
    conn.close()

# Warmup both paths.
post_batch(sentences)
with ThreadPoolExecutor(max_workers=len(sentences)) as ex:
    list(ex.map(post_single, sentences))

# A: concurrent single-input (existing microbatcher path)
times_concurrent = []
with ThreadPoolExecutor(max_workers=len(sentences)) as ex:
    for _ in range(10):
        time.sleep(0.5)  # small cooldown between trials
        t0 = time.perf_counter()
        list(ex.map(post_single, sentences))
        times_concurrent.append((time.perf_counter() - t0) * 1000.0)

# B: single batched POST (new explicit batch contract)
times_batch = []
for _ in range(10):
    time.sleep(0.5)
    t0 = time.perf_counter()
    post_batch(sentences)
    times_batch.append((time.perf_counter() - t0) * 1000.0)

def stats(name, ts):
    m = statistics.median(ts)
    return {
        "name": name,
        "trials_ms": ts,
        "median_ms": m,
        "ms_per_sentence": m / len(sentences),
        "tok_s": total_tokens / (m / 1000.0),
    }

s_concurrent = stats("concurrent_10_single", times_concurrent)
s_batch = stats("single_batched", times_batch)
print(f"  concurrent x10: med={s_concurrent['median_ms']:.2f}ms tok/s={s_concurrent['tok_s']:.0f}")
print(f"  batched POST  : med={s_batch['median_ms']:.2f}ms tok/s={s_batch['tok_s']:.0f}")
print(f"  batch vs concurrent: {(s_concurrent['median_ms']/s_batch['median_ms'] - 1) * 100:+.1f}% (positive = batch is faster)")

import pathlib
pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
pathlib.Path(outdir, f"{label}.json").write_text(json.dumps({
    "label": label,
    "concurrent": s_concurrent,
    "batched": s_batch,
}, indent=2) + "\n")
PYEND
    kill $server_pid 2>/dev/null
    sleep 5
}

run_one qwen3-embedding-0.6b-8bit   .internal/models/qwen3-embedding-0.6b-8bit
run_one qwen3-embedding-4b-4bit     .internal/models/qwen3-embedding-4b-4bit
run_one qwen3-embedding-8b-4bit-dwq .internal/models/qwen3-embedding-8b-4bit-dwq
echo ""
echo "All done."
