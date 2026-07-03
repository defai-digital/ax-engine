#!/usr/bin/env bash
# Full README refresh: inference models (ax direct + n-gram) then embedding models.
#
# Inference: reuses mlx_lm/mlx_swift_lm reference rows from the 2026-05-12
# production build; only re-runs ax_engine policies with the layer-mask fix.
# Updates README rows for each model during the 45-second inter-model cooldown.
#
# Embedding: runs mlx_lm vs ax_engine_py using the maturin-built Python extension.
# Updates the README embedding table after all three models complete.
set -euo pipefail

INFDIR="benchmarks/results/mlx-inference/2026-05-12-ax-refresh-post-layer-mask-fix"
REFDIR="benchmarks/results/mlx-inference/2026-05-12-production-build-readme-refresh"
EMBDIR="benchmarks/results/embedding/2026-05-12-readme-refresh-v2"
SCRIPT="scripts/bench_mlx_inference_stack.py"
UPDATE="scripts/update_readme_from_bench.py"
EMBBENCH="scripts/bench_embedding_models.py"
EMBUPDATE="scripts/update_readme_embedding.py"

# ── Inference benchmarks ─────────────────────────────────────────────────────

run_inference_model() {
    local slug="$1"
    local model_dir="$2"
    echo ""
    echo "=========================================="
    echo "  [inference] $slug"
    echo "=========================================="
    python3 "$SCRIPT" \
        --model-dir "$model_dir" \
        --prompt-tokens 128,512 \
        --generation-tokens 128 \
        --repetitions 5 \
        --cooldown 15 \
        --ax-compare-policies \
        --reuse-reference-results-from "$REFDIR/${slug}.json" \
        --no-build-ax-engine \
        --output "$INFDIR/${slug}.json" \
        2>&1 | tee "$INFDIR/logs/${slug}.log"

    echo ""
    echo "  [updating README during 45s cooldown]"
    python3 "$UPDATE" --slug "$slug" --json "$INFDIR/${slug}.json"
    sleep 45
}

run_inference_model gemma-4-e2b-it-4bit          .internal/models/gemma-4-e2b-it-4bit
run_inference_model gemma-4-e2b-it-5bit          .internal/models/gemma-4-e2b-it-5bit
run_inference_model gemma-4-e2b-it-6bit          .internal/models/gemma-4-e2b-it-6bit
run_inference_model gemma-4-e2b-it-8bit          .internal/models/gemma-4-e2b-it-8bit
run_inference_model gemma-4-e4b-it-4bit          .internal/models/gemma-4-e4b-it-4bit
run_inference_model gemma-4-26b-a4b-it-4bit      .internal/models/gemma-4-26b-a4b-it-4bit
run_inference_model gemma-4-31b-it-4bit          .internal/models/gemma-4-31b-it-4bit
run_inference_model qwen3_5-9b-mlx-4bit          .internal/models/Qwen3.5-9B-MLX-4bit
run_inference_model qwen3_6-35b-a3b-ud-mlx-4bit  .internal/models/Qwen3.6-35B-A3B-UD-MLX-4bit
run_inference_model qwen3_6-35b-a3b-5bit         .internal/models/Qwen3.6-35B-A3B-5bit
run_inference_model qwen3_6-35b-a3b-6bit         .internal/models/Qwen3.6-35B-A3B-6bit
run_inference_model qwen3_6-35b-a3b-8bit         .internal/models/Qwen3.6-35B-A3B-8bit
run_inference_model qwen3-coder-next-4bit        .internal/models/Qwen3-Coder-Next-4bit
run_inference_model glm-4.7-flash-4bit           .internal/models/GLM-4.7-Flash-4bit

echo ""
echo "=== All inference models done. Building Python extension for embedding. ==="
maturin develop --release

# ── Embedding benchmarks ──────────────────────────────────────────────────────

run_embedding_model() {
    local label="$1"
    local model_dir="$2"
    echo ""
    echo "=========================================="
    echo "  [embedding] $label"
    echo "=========================================="
    python3 "$EMBBENCH" \
        --model-dir "$model_dir" \
        --model-label "$label" \
        --trials 5 \
        --cooldown 3 \
        --skip-ax-http \
        --skip-swift \
        --output-dir "$EMBDIR" \
        2>&1 | tee "$EMBDIR/${label}.log"

    echo ""
    echo "  [cooling 45s before next embedding model]"
    sleep 45
}

run_embedding_model qwen3-embedding-0.6b-8bit    .internal/models/qwen3-embedding-0.6b-8bit
run_embedding_model qwen3-embedding-4b-4bit      .internal/models/qwen3-embedding-4b-4bit
run_embedding_model qwen3-embedding-8b-4bit-dwq  .internal/models/qwen3-embedding-8b-4bit-dwq

echo ""
echo "=== All embedding models done. Updating README embedding rows. ==="
python3 "$EMBUPDATE" --results-dir "$EMBDIR"

echo ""
echo "Full README refresh complete — inference + embedding."
