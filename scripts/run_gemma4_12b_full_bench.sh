#!/usr/bin/env bash
# One-shot driver for the dedicated Gemma 4 12B (gemma4_unified) benchmark.
#
# Produces the two artifact trees that back the README "Gemma 4 12B" section:
#   1. Direct decode/prefill/TTFT, AX-native vs llama.cpp Metal (2-way; mlx_lm
#      cannot load gemma4_unified so it is necessarily absent).
#   2. Assistant-MTP and assistant MTP+n-gram decode, same methodology as the
#      published 26B/31B run (1000 tokens, 5 reps, 10s/5s cooldowns).
#
# Run sequentially on purpose: both phases saturate the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

DATE=2026-06-08
PY=python3
LLAMA_BENCH=/opt/homebrew/bin/llama-bench
GGUF=$(ls "$HOME"/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/*/gemma-4-12B-it-Q4_K_M.gguf | head -1)
# AX-ready base snapshot (the one carrying model-manifest.json).
BASE_DIR=$(dirname "$(ls "$HOME"/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-4bit/snapshots/*/model-manifest.json | head -1)")

DIRECT_DIR="benchmarks/results/mlx-inference/${DATE}-gemma-4-12b-it-4bit-direct"
MTP_DIR="benchmarks/results/gemma4-assistant-mtp/${DATE}-gemma4-12b-assistant-mtp"
mkdir -p "$DIRECT_DIR"

echo "===== GGUF: $GGUF"
echo "===== BASE_DIR: $BASE_DIR"

# Record llama.cpp GGUF provenance next to the direct artifact (this 12B GGUF
# is not in the bartowski-only llama_cpp_metal inventory because bartowski has
# not published a Gemma 4 12B GGUF).
cat > "$DIRECT_DIR/llama_cpp_gguf_provenance.json" <<JSON
{
  "schema": "ax.gemma4_12b_llama_cpp_provenance.v1",
  "model": "Gemma 4 12B",
  "mlx_repo_id": "mlx-community/gemma-4-12B-it-4bit",
  "gguf_repo": "ggml-org/gemma-4-12B-it-GGUF",
  "gguf_file": "gemma-4-12B-it-Q4_K_M.gguf",
  "gguf_quant_target": "Q4_K_M",
  "llama_cpp_arch": "gemma4",
  "publisher_deviation_note": "bartowski has not published a Gemma 4 12B GGUF (404 at bartowski/google_gemma-4-12B-it-GGUF as of 2026-06-08). Uses the official ggml-org conversion, the same publisher as the on-disk 26B/31B/E2B/E4B GGUFs. Shape-compatible external baseline only; not prompt-hash parity.",
  "mlx_lm_baseline": "absent_unsupported_architecture",
  "mlx_lm_note": "mlx_lm 0.31.3 has no graph for model_type gemma4_unified (ValueError: Model type gemma4_unified not supported); the MLX-side reference is AX Engine's repo-owned native runtime."
}
JSON

echo "===== Phase 1/2: direct decode (AX native vs llama.cpp Metal) ====="
$PY scripts/bench_mlx_inference_stack.py \
  --model gemma-4-12b-it --model-dir "$BASE_DIR" \
  --prompt-tokens 128,512,2048 \
  --generation-tokens 128 \
  --repetitions 5 \
  --cooldown 15 \
  --ax-direct --skip-mlx-lm --no-build-ax-engine \
  --llama-cpp-bench "$LLAMA_BENCH" --llama-cpp-gguf "$GGUF" \
  --output "$DIRECT_DIR/gemma-4-12b-it-4bit.json"

echo "===== Phase 2/2: assistant-MTP and MTP+n-gram ====="
$PY scripts/bench_gemma4_assistant_mtp.py \
  --models 12b-4bit --modes mtp,mtp-ngram \
  --suites flappy,long_code,python_modules_long \
  --max-tokens 1000 --repetitions 5 --cooldown 10 --inter-case-cooldown 5 \
  --no-build-ax-engine \
  --output-dir "$MTP_DIR"

echo "===== DONE. Direct: $DIRECT_DIR  MTP: $MTP_DIR"
