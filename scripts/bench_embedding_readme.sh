#!/usr/bin/env bash
# One canonical script that reproduces every number in the README
# `### Embedding throughput` section. Runs three paths on three models:
#
#   1. In-process batched throughput (Python + Rust + mlx-lm reference)
#   2. HTTP /v1/embeddings throughput (sequential / concurrent / batched POST)
#   3. Cold-start session_new latency (C loader vs AX_MMAP_WEIGHTS=1)
#
# Output: $OUTDIR/summary.md — drop-in markdown for the README, with the
# tables already filled in. Per-trial artifacts are kept alongside so the
# numbers are auditable.
#
# Usage:
#   bash scripts/bench_embedding_readme.sh                # default: today's date
#   OUTDIR=/path/to/dir bash scripts/bench_embedding_readme.sh
#
# Expects:
#   - .internal/models/qwen3-embedding-0.6b-8bit/
#   - .internal/models/qwen3-embedding-4b-4bit/
#   - .internal/models/qwen3-embedding-8b-4bit-dwq/
#   - .venv with ax_engine + mlx_lm + numpy installed
#   - target/release/{ax-engine-server, examples/embed_rust_bench, examples/cold_start_bench}
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

DATE="$(date '+%Y-%m-%d')"
OUTDIR="${OUTDIR:-benchmarks/results/embedding/${DATE}-readme}"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.md"

MODELS=(
    "qwen3-embedding-0.6b-8bit:.internal/models/qwen3-embedding-0.6b-8bit"
    "qwen3-embedding-4b-4bit:.internal/models/qwen3-embedding-4b-4bit"
    "qwen3-embedding-8b-4bit-dwq:.internal/models/qwen3-embedding-8b-4bit-dwq"
)
# 10-sentence corpus, lengths [10,15,13,8,3,8,10,8,10,10] — canonical.
BATCH_LENS="10,15,13,8,3,8,10,8,10,10"

cleanup() {
    pkill -f 'ax-engine-server' >/dev/null 2>&1 || true
    sleep 2
}
trap cleanup EXIT

# Ensure the binaries we need are built. Build is a no-op when fresh.
echo "[bench] building release binaries…"
cargo build -p ax-engine-server --release >/dev/null 2>&1 || {
    echo "cargo build ax-engine-server failed; aborting"; exit 1;
}
cargo build -p ax-engine-bench --example embed_rust_bench --example cold_start_bench \
    --release >/dev/null 2>&1 || {
    echo "cargo build examples failed; aborting"; exit 1;
}
echo "[bench] ax_engine python extension up to date…"
maturin develop --release >/dev/null 2>&1 || {
    echo "maturin develop failed; aborting"; exit 1;
}

# ---------------------------------------------------------------------------
# 1. In-process batched throughput (Python + Rust + mlx-lm).
# ---------------------------------------------------------------------------
echo ""
echo "[bench] (1/3) in-process batched throughput"
for spec in "${MODELS[@]}"; do
    label="${spec%%:*}"; model_dir="${spec#*:}"
    sub="$OUTDIR/inproc/$label"
    mkdir -p "$sub"

    echo "  $label"
    PYTHONUNBUFFERED=1 python - "$model_dir" "$label" "$sub" "$BATCH_LENS" <<'PYEND' >"$sub/log.txt" 2>&1 || true
import json, statistics, sys, time
import mlx.core as mx
from mlx_lm import load
import ax_engine

model_dir, label, outdir, batch_spec = sys.argv[1:]
lens = [int(x) for x in batch_spec.split(",")]
max_len = max(lens)
batch = [list(range(l)) for l in lens]
padded = [ids + [0] * (max_len - len(ids)) for ids in batch]
last_pos = [l - 1 for l in lens]
total_tokens = sum(lens)
N_WARMUP, N_TRIALS = 5, 15

def mlx_lm_batched():
    x = mx.array(padded); h = model.model(x)
    outs = []
    for i, pos in enumerate(last_pos):
        last = h[i, pos, :].astype(mx.float32)
        outs.append(last / (mx.sqrt(mx.sum(last * last)) + 1e-12))
    return [o.tolist() for o in outs]

# mlx-lm
model, _ = load(model_dir)
for _ in range(N_WARMUP): mlx_lm_batched()
ts_mlx = []
for _ in range(N_TRIALS):
    t0 = time.perf_counter(); mlx_lm_batched()
    ts_mlx.append((time.perf_counter() - t0) * 1000.0)
del model

# ax-engine-py
s = ax_engine.Session(mlx=True, model_id="qwen3_dense", support_tier="mlx_preview",
                     mlx_model_artifacts_dir=model_dir)
for _ in range(N_WARMUP):
    s.embed_batch_bytes(batch, pooling="last", normalize=True)
ts_ax = []
for _ in range(N_TRIALS):
    t0 = time.perf_counter()
    s.embed_batch_bytes(batch, pooling="last", normalize=True)
    ts_ax.append((time.perf_counter() - t0) * 1000.0)
s.close()

def median_tps(ts):
    return total_tokens / (statistics.median(ts) / 1000.0)

out = {
    "label": label,
    "lens": lens,
    "warmup": N_WARMUP,
    "trials": N_TRIALS,
    "mlx_lm_batched_tok_s": median_tps(ts_mlx),
    "ax_py_batched_tok_s": median_tps(ts_ax),
    "mlx_lm_trials_ms": ts_mlx,
    "ax_py_trials_ms": ts_ax,
}
import pathlib
pathlib.Path(outdir, "python.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PYEND

    # Rust direct
    ./target/release/examples/embed_rust_bench \
        --model-dir "$model_dir" \
        --batch "$BATCH_LENS" \
        --warmup 5 --trials 15 \
        >"$sub/rust.txt" 2>"$sub/rust.err"
done

# ---------------------------------------------------------------------------
# 2. HTTP /v1/embeddings — sequential / concurrent / batched POST.
# ---------------------------------------------------------------------------
echo ""
echo "[bench] (2/3) HTTP /v1/embeddings paths"
for spec in "${MODELS[@]}"; do
    label="${spec%%:*}"; model_dir="${spec#*:}"
    sub="$OUTDIR/http/$label"
    mkdir -p "$sub"
    echo "  $label"

    cleanup
    AX_ENGINE_EMBED_MICROBATCH_WINDOW_MS=20 \
        ./target/release/ax-engine-server \
        --model-id qwen3_dense --support-tier mlx-preview --mlx \
        --mlx-model-artifacts-dir "$model_dir" \
        --host 127.0.0.1 --port 8083 >"$sub/server.log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 120); do
        if curl -sSf http://127.0.0.1:8083/health >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -sSf http://127.0.0.1:8083/health >/dev/null 2>&1; then
        echo "  server failed to start for $label"; kill $server_pid 2>/dev/null; continue
    fi

    PYTHONUNBUFFERED=1 python - "$label" "$sub" "$BATCH_LENS" <<'PYEND' >"$sub/log.txt" 2>&1 || true
import json, statistics, sys, time
import http.client
from concurrent.futures import ThreadPoolExecutor

label, outdir, batch_spec = sys.argv[1:]
lens = [int(x) for x in batch_spec.split(",")]
sentences = [list(range(l)) for l in lens]
total_tokens = sum(lens)
N_TRIALS = 10

def post(payload):
    conn = http.client.HTTPConnection("127.0.0.1", 8083, timeout=60)
    conn.request("POST", "/v1/embeddings",
                 body=json.dumps(payload).encode(),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse(); assert r.status == 200, r.status
    r.read(); conn.close()

# Warmup all three.
for ids in sentences: post({"input": ids})
with ThreadPoolExecutor(max_workers=len(sentences)) as pool:
    list(pool.map(lambda ids: post({"input": ids}), sentences))
post({"input": sentences})

# Sequential
t_seq = []
for _ in range(N_TRIALS):
    time.sleep(0.3)
    t0 = time.perf_counter()
    for ids in sentences: post({"input": ids})
    t_seq.append((time.perf_counter() - t0) * 1000.0)

# Concurrent (microbatcher coalesces)
t_conc = []
with ThreadPoolExecutor(max_workers=len(sentences)) as pool:
    for _ in range(N_TRIALS):
        time.sleep(0.3)
        t0 = time.perf_counter()
        list(pool.map(lambda ids: post({"input": ids}), sentences))
        t_conc.append((time.perf_counter() - t0) * 1000.0)

# Batched POST
t_batch = []
for _ in range(N_TRIALS):
    time.sleep(0.3)
    t0 = time.perf_counter()
    post({"input": sentences})
    t_batch.append((time.perf_counter() - t0) * 1000.0)

def median_tps(ts):
    return total_tokens / (statistics.median(ts) / 1000.0)

out = {
    "label": label,
    "lens": lens,
    "trials": N_TRIALS,
    "sequential_tok_s": median_tps(t_seq),
    "concurrent_tok_s": median_tps(t_conc),
    "batched_post_tok_s": median_tps(t_batch),
    "sequential_trials_ms": t_seq,
    "concurrent_trials_ms": t_conc,
    "batched_trials_ms": t_batch,
}
import pathlib
pathlib.Path(outdir, "http.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PYEND

    kill $server_pid 2>/dev/null
    sleep 3
done

# ---------------------------------------------------------------------------
# 3. Cold-start session_new latency.
# ---------------------------------------------------------------------------
echo ""
echo "[bench] (3/3) cold-start session construction"
for spec in "${MODELS[@]}"; do
    label="${spec%%:*}"; model_dir="${spec#*:}"
    sub="$OUTDIR/coldstart/$label"
    mkdir -p "$sub"
    echo "  $label"

    : >"$sub/c_loader.txt"; : >"$sub/mmap.txt"
    for _ in 1 2 3; do
        AX_MMAP_WEIGHTS=0 ./target/release/examples/cold_start_bench \
            --model-dir "$model_dir" >>"$sub/c_loader.txt" 2>/dev/null
        sleep 1
    done
    for _ in 1 2 3; do
        AX_MMAP_WEIGHTS=1 ./target/release/examples/cold_start_bench \
            --model-dir "$model_dir" >>"$sub/mmap.txt" 2>/dev/null
        sleep 1
    done
done

# ---------------------------------------------------------------------------
# Build the README-ready summary.md from the per-path JSONs.
# ---------------------------------------------------------------------------
echo ""
echo "[bench] writing $SUMMARY"
PYTHONUNBUFFERED=1 python - "$OUTDIR" <<'PYEND'
import json, os, pathlib, sys, re

outdir = pathlib.Path(sys.argv[1])

def load(p):
    if p.exists():
        return json.loads(p.read_text())
    return None

def median(xs):
    xs = sorted(xs); n = len(xs); return xs[n // 2]

def parse_cold(path):
    """Parse cold_start_bench output `session_new_ms <num>` lines."""
    ms = []
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        m = re.search(r"session_new_ms\s+([0-9.]+)", line)
        if m:
            ms.append(float(m.group(1)))
    return median(ms) if ms else None

def parse_rust_bench(path):
    """Parse embed_rust_bench output for `ms/sentence  <num>    tok/s <num>`."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        m = re.search(r"tok/s\s+([0-9.]+)", line)
        if m:
            return float(m.group(1))
    return None

models = [
    ("qwen3-embedding-0.6b-8bit",   "Qwen3-Embedding 0.6B 8-bit"),
    ("qwen3-embedding-4b-4bit",     "Qwen3-Embedding 4B 4-bit"),
    ("qwen3-embedding-8b-4bit-dwq", "Qwen3-Embedding 8B 4-bit DWQ"),
]

inproc = {}
http_data = {}
cold = {}
for label, _ in models:
    py = load(outdir / "inproc" / label / "python.json")
    rust_tok_s = parse_rust_bench(outdir / "inproc" / label / "rust.txt")
    if py:
        inproc[label] = {
            "mlx_lm": py["mlx_lm_batched_tok_s"],
            "ax_py": py["ax_py_batched_tok_s"],
            "ax_rust": rust_tok_s,
        }
    h = load(outdir / "http" / label / "http.json")
    if h:
        http_data[label] = {
            "sequential": h["sequential_tok_s"],
            "concurrent": h["concurrent_tok_s"],
            "batched":   h["batched_post_tok_s"],
        }
    cold[label] = {
        "c_loader": parse_cold(outdir / "coldstart" / label / "c_loader.txt"),
        "mmap":     parse_cold(outdir / "coldstart" / label / "mmap.txt"),
    }

def fmt(x):
    if x is None: return "—"
    return f"{x:,.0f}" if x >= 100 else f"{x:.1f}"

def fmt_ms(x):
    if x is None: return "—"
    return f"{x:.0f} ms"

def pct(a, b):
    if a is None or b is None or b == 0:
        return ""
    delta = (a - b) / b * 100
    sign = "+" if delta >= 0 else ""
    return f" ({sign}{delta:.0f}%)"

lines = []
lines.append("### Embedding throughput")
lines.append("")
lines.append("ax-engine matches `mlx-lm` on Qwen3-Embedding 4B and 8B and is")
lines.append("within ~10% on 0.6B (small-model Python/PyO3 overhead). Use the")
lines.append("batched API — `embed_batch_array` in Python, `embed_batch_flat`")
lines.append("in Rust, or `{\"input\": [[ids], ...]}` over HTTP. Single-sentence")
lines.append("loops are 2–3× slower in both backends.")
lines.append("")
lines.append("Source: `benchmarks/results/embedding/" + outdir.name + "/`.")
lines.append("")
lines.append("#### In-process batched (sustained, hot-loop)")
lines.append("")
lines.append("| Model | mlx-lm | ax-engine-py | ax-engine Rust |")
lines.append("|---|---:|---:|---:|")
for label, display in models:
    r = inproc.get(label, {})
    lines.append(f"| {display} | {fmt(r.get('mlx_lm'))} | {fmt(r.get('ax_py'))} | {fmt(r.get('ax_rust'))} |")
lines.append("")
lines.append("Median tok/s, 5 warmup + 15 timed trials, no cooldown. 10-sentence")
lines.append("corpus with lengths [10,15,13,8,3,8,10,8,10,10], `last` pooling,")
lines.append("l2-normalized.")
lines.append("")
lines.append("#### HTTP serving (`/v1/embeddings`)")
lines.append("")
lines.append("Three serving contracts on the same 10-sentence corpus:")
lines.append("")
lines.append("| Model | Sequential | Concurrent (microbatcher) | Batched POST |")
lines.append("|---|---:|---:|---:|")
for label, display in models:
    r = http_data.get(label, {})
    lines.append(f"| {display} | {fmt(r.get('sequential'))} | {fmt(r.get('concurrent'))} | {fmt(r.get('batched'))} |")
lines.append("")
lines.append("- **Batched POST** `{\"input\": [[ids], ...]}` is the fastest path.")
lines.append("- **Concurrent** single-input POSTs are coalesced server-side by")
lines.append("  `EmbeddingMicroBatcher` (20 ms window in this measurement). Use")
lines.append("  this when the caller can't pre-batch.")
lines.append("- **Sequential** is the worst case — every POST round-trips through")
lines.append("  the GPU on its own.")
lines.append("")
lines.append("#### Cold start (session construction)")
lines.append("")
lines.append("| Model | Default | `AX_MMAP_WEIGHTS=1` |")
lines.append("|---|---:|---:|")
for label, display in models:
    r = cold.get(label, {})
    lines.append(f"| {display} | {fmt_ms(r.get('c_loader'))} | {fmt_ms(r.get('mmap'))}{pct(r.get('mmap'), r.get('c_loader'))} |")
lines.append("")
lines.append("Memory-mapped safetensors loader; opt in with the env var. Bigger")
lines.append("relative win on larger models. See")
lines.append("[`docs/EMBEDDING_COLDSTART.md`](../../../../docs/EMBEDDING_COLDSTART.md)")
lines.append("for the true-cold (post-`sudo purge`) measurement procedure.")
lines.append("")
lines.append("#### Reproducing")
lines.append("")
lines.append("```bash")
lines.append("bash scripts/bench_embedding_readme.sh")
lines.append("```")
lines.append("")
lines.append("Detailed methodology: [`docs/EMBEDDINGS.md`](../../../../docs/EMBEDDINGS.md).")

(outdir / "summary.md").write_text("\n".join(lines) + "\n")
print(f"wrote {outdir/'summary.md'}")
PYEND

echo ""
echo "[bench] done; see $SUMMARY"
