#!/usr/bin/env bash
# Sustained batched embedding throughput — hot-loop (no cooldown).
#
# This corpus / shape matches `scripts/bench_embedding_models.py`'s
# canonical 10 sentences (lengths 10,15,13,8,3,8,10,8,10,10). The script
# warms up 5 reps, then times 15 reps back-to-back. Designed to measure
# what a sustained workload (vector-DB ingest, batch evaluation, async
# worker pool) sees, in contrast to the 10s-cooldown bench profile which
# captures intermittent-call behavior.
#
# Captures three batched paths on the same input:
#   1. mlx-lm:        model.model(padded_batch) + per-sentence pooling + .tolist()
#   2. ax-engine-py:  session.embed_batch_bytes(batch, ...)
#   3. ax-engine-rust: EngineSession::embed_batch(...) via the
#                      examples/embed_rust_bench example binary
set -uo pipefail

REPO_ROOT="/Users/akiralam/code/ax-engine-v4"
cd "$REPO_ROOT"

source .venv/bin/activate

OUTDIR="benchmarks/results/embedding/2026-05-12-sustained-batched"
BATCH_LENS=10,15,13,8,3,8,10,8,10,10

# Make sure the Rust example is built.
cargo build -p ax-engine-bench --example embed_rust_bench --release \
    >"$OUTDIR/cargo_build.log" 2>&1 || {
    echo "cargo build failed; see $OUTDIR/cargo_build.log"; exit 1;
}

run_python() {
    local label="$1" model_dir="$2"
    echo "  [py] $label"
    python - "$model_dir" "$label" "$OUTDIR" "$BATCH_LENS" <<'PYEND'
import json, statistics, sys, time
import mlx.core as mx
import ax_engine
from mlx_lm import load

model_dir, label, outdir, batch_spec = sys.argv[1:]
lens = [int(x) for x in batch_spec.split(",")]
max_len = max(lens)
batch = [list(range(l)) for l in lens]
padded = [ids + [0] * (max_len - len(ids)) for ids in batch]
last_pos = [l - 1 for l in lens]
total_tokens = sum(lens)

# mlx-lm hot-loop
model, _ = load(model_dir)
def mlxlm_step():
    x = mx.array(padded)
    h = model.model(x)
    outs = []
    for i, pos in enumerate(last_pos):
        last = h[i, pos, :].astype(mx.float32)
        outs.append(last / (mx.sqrt(mx.sum(last * last)) + 1e-12))
    return [o.tolist() for o in outs]
for _ in range(5): mlxlm_step()
ts_mlx = []
for _ in range(15):
    t0 = time.perf_counter(); mlxlm_step()
    ts_mlx.append((time.perf_counter() - t0) * 1000.0)
del model

# ax-engine-py hot-loop
s = ax_engine.Session(
    mlx=True, model_id="qwen3_dense", support_tier="mlx_preview",
    mlx_model_artifacts_dir=model_dir,
)
for _ in range(5):
    s.embed_batch_bytes(batch, pooling="last", normalize=True)
ts_ax = []
for _ in range(15):
    t0 = time.perf_counter()
    s.embed_batch_bytes(batch, pooling="last", normalize=True)
    ts_ax.append((time.perf_counter() - t0) * 1000.0)
s.close()

def stats(ts):
    ts_sorted = sorted(ts)
    n = len(ts_sorted)
    return {
        "trials_ms": ts,
        "median_ms": ts_sorted[n // 2],
        "min_ms": ts_sorted[0],
        "max_ms": ts_sorted[-1],
        "mean_ms": sum(ts) / n,
        "median_ms_per_sentence": ts_sorted[n // 2] / 10.0,
        "median_tokens_per_sec": total_tokens / (ts_sorted[n // 2] / 1000.0),
    }

out = {
    "schema_version": "ax.embedding_sustained_batched.v1",
    "model_label": label,
    "model_dir": model_dir,
    "batch_lengths": lens,
    "total_tokens": total_tokens,
    "warmup": 5,
    "trials": 15,
    "cooldown_s": 0,
    "results": {
        "mlx_lm_batched": stats(ts_mlx),
        "ax_engine_py_batched": stats(ts_ax),
    },
}
import pathlib
pathlib.Path(outdir, label).mkdir(parents=True, exist_ok=True)
pathlib.Path(outdir, label, "python_sustained.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
print(f"  wrote {outdir}/{label}/python_sustained.json")
PYEND
}

run_rust() {
    local label="$1" model_dir="$2"
    echo "  [rust] $label"
    local out_file="$OUTDIR/${label}/rust_sustained.txt"
    mkdir -p "$OUTDIR/${label}"
    ./target/release/examples/embed_rust_bench \
        --model-dir "$model_dir" \
        --batch "$BATCH_LENS" \
        --warmup 5 --trials 15 \
        2>"$out_file.err" >"$out_file"
}

mkdir -p "$OUTDIR"
for spec in \
    "qwen3-embedding-0.6b-8bit:.internal/models/qwen3-embedding-0.6b-8bit" \
    "qwen3-embedding-4b-4bit:.internal/models/qwen3-embedding-4b-4bit" \
    "qwen3-embedding-8b-4bit-dwq:.internal/models/qwen3-embedding-8b-4bit-dwq"; do
    label="${spec%%:*}"; model_dir="${spec#*:}"
    run_python "$label" "$model_dir"
    run_rust "$label" "$model_dir"
done
echo "Done."
