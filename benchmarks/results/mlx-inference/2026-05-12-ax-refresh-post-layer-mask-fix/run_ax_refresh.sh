#!/usr/bin/env bash
# Refresh ax-engine direct + n-gram rows for all README models.
# Reuses mlx_lm and mlx_swift_lm reference rows from the 2026-05-12 production
# build run; only re-runs the ax_engine_mlx and ax_engine_mlx_ngram_accel policies.
# Updates README.md rows for each model during the 45-second inter-model cooldown.
#
# Built with: cargo build -p ax-engine-server --release
# Change captured: build_layer_masks seq==1 fast path (no HashMap alloc on decode)
set -euo pipefail

OUTDIR="benchmarks/results/mlx-inference/2026-05-12-ax-refresh-post-layer-mask-fix"
REFDIR="benchmarks/results/mlx-inference/2026-05-12-production-build-readme-refresh"
SCRIPT="scripts/bench_mlx_inference_stack.py"
UPDATE="scripts/update_readme_from_bench.py"

run_model() {
    local slug="$1"
    local model_dir="$2"
    echo ""
    echo "=========================================="
    echo "  $slug"
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
        --output "$OUTDIR/${slug}.json" \
        2>&1 | tee "$OUTDIR/logs/${slug}.log"

    echo ""
    echo "  [updating README during 45s cooldown]"
    python3 "$UPDATE" --slug "$slug" --json "$OUTDIR/${slug}.json"
    sleep 45
}

run_model gemma-4-e2b-it-4bit          .internal/models/gemma-4-e2b-it-4bit
run_model gemma-4-e2b-it-5bit          .internal/models/gemma-4-e2b-it-5bit
run_model gemma-4-e2b-it-6bit          .internal/models/gemma-4-e2b-it-6bit
run_model gemma-4-e2b-it-8bit          .internal/models/gemma-4-e2b-it-8bit
run_model gemma-4-e4b-it-4bit          .internal/models/gemma-4-e4b-it-4bit
run_model gemma-4-26b-a4b-it-4bit      .internal/models/gemma-4-26b-a4b-it-4bit
run_model gemma-4-31b-it-4bit          .internal/models/gemma-4-31b-it-4bit
run_model qwen3_5-9b-mlx-4bit          .internal/models/Qwen3.5-9B-MLX-4bit
run_model qwen3_6-35b-a3b-ud-mlx-4bit  .internal/models/Qwen3.6-35B-A3B-UD-MLX-4bit
run_model qwen3_6-35b-a3b-5bit         .internal/models/Qwen3.6-35B-A3B-5bit
run_model qwen3_6-35b-a3b-6bit         .internal/models/Qwen3.6-35B-A3B-6bit
run_model qwen3_6-35b-a3b-8bit         .internal/models/Qwen3.6-35B-A3B-8bit
run_model qwen3-coder-next-4bit        .internal/models/Qwen3-Coder-Next-4bit
run_model glm-4.7-flash-4bit           .internal/models/GLM-4.7-Flash-4bit

echo ""
echo "All models done — ax-engine post-layer-mask-fix refresh complete."
