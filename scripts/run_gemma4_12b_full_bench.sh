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

# Fair-quant AX artifact: the upstream mlx-community 4bit snapshot keeps the FFN
# at 8-bit (overrides every mlp.{gate,up,down}_proj to bits:8), so it weighs
# ~10.4 GiB and reads ~1.5x the FFN bytes of the llama.cpp Q4_K_M GGUF (6.87
# GiB). Decode is memory-bandwidth-bound, so benchmarking the 8-bit-FFN artifact
# against a 4-bit GGUF handicaps AX. We re-quantize the FFN to 4-bit (uniform
# 4-bit group-64, ~4.5 bpw, comparable to Q4_K_M's ~4.8 bpw) for an apples-to-
# apples decode comparison. Build it once if absent.
FFN4_DIR="$HOME/.cache/huggingface/hub/models--ax-local--gemma-4-12B-it-4bit-ffn4/snapshots/v1"
if [ ! -f "$FFN4_DIR/model-manifest.json" ]; then
  echo "===== Building fair-quant 4-bit-FFN artifact (one-time) -> $FFN4_DIR"
  mkdir -p "$FFN4_DIR"
  $PY scripts/requantize_gemma4_12b_ffn_4bit.py "$BASE_DIR" "$FFN4_DIR"
  cargo run -q --release -p ax-engine-core --bin generate-manifest -- --force --validate "$FFN4_DIR"
fi

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
  "mlx_repo_id": "ax-local/gemma-4-12B-it-4bit-ffn4 (FFN re-quantized 8->4 bit from mlx-community/gemma-4-12B-it-4bit)",
  "ax_quant_note": "Upstream mlx-community 4bit keeps FFN at 8-bit (~10.4 GiB); re-quantized to uniform 4-bit group-64 (~6.3 GiB, ~4.5 bpw) for bit-comparable fairness vs Q4_K_M (~6.87 GiB, ~4.8 bpw). See scripts/requantize_gemma4_12b_ffn_4bit.py.",
  "gguf_repo": "ggml-org/gemma-4-12B-it-GGUF",
  "gguf_file": "gemma-4-12B-it-Q4_K_M.gguf",
  "gguf_quant_target": "Q4_K_M",
  "decode_comparison_contract": "llama.cpp decode measured at matched context depth (--llama-cpp-decode-at-depth: llama-bench -n GEN -d PROMPT), not depth-0 tg, to match AX's prompt-context decode.",
  "llama_cpp_arch": "gemma4",
  "publisher_deviation_note": "bartowski has not published a Gemma 4 12B GGUF (404 at bartowski/google_gemma-4-12B-it-GGUF as of 2026-06-08). Uses the official ggml-org conversion, the same publisher as the on-disk 26B/31B/E2B/E4B GGUFs. Shape-compatible external baseline only; not prompt-hash parity.",
  "mlx_lm_baseline": "absent_unsupported_architecture",
  "mlx_lm_note": "mlx_lm 0.31.3 has no graph for model_type gemma4_unified (ValueError: Model type gemma4_unified not supported); the MLX-side reference is AX Engine's repo-owned native runtime."
}
JSON

echo "===== Phase 1/2: direct decode (AX native vs llama.cpp Metal) ====="
# Fairness corrections vs the prior run:
#   1. AX uses the fair-quant 4-bit-FFN artifact ($FFN4_DIR), comparable in bits
#      to the Q4_K_M GGUF (the upstream 8-bit-FFN snapshot handicapped AX).
#   2. --llama-cpp-decode-at-depth measures llama.cpp decode at the SAME context
#      depth AX decodes at (after the prompt prefill). Plain llama-bench `tg`
#      reports decode from an EMPTY context (depth 0), llama.cpp's best case,
#      which flatters it relative to AX's prompt-context decode.
$PY scripts/bench_mlx_inference_stack.py \
  --model gemma-4-12b-it-ffn4 --model-dir "$FFN4_DIR" \
  --prompt-tokens 128,512,2048 \
  --generation-tokens 128 \
  --repetitions 5 \
  --cooldown 15 \
  --ax-direct --skip-mlx-lm --no-build-ax-engine \
  --llama-cpp-bench "$LLAMA_BENCH" --llama-cpp-gguf "$GGUF" \
  --llama-cpp-decode-at-depth \
  --output "$DIRECT_DIR/gemma-4-12b-it-4bit.json"

echo "===== Phase 2/2: assistant-MTP and MTP+n-gram ====="
$PY scripts/bench_gemma4_assistant_mtp.py \
  --models 12b-4bit --modes mtp,mtp-ngram \
  --suites flappy,long_code,python_modules_long \
  --max-tokens 1000 --repetitions 5 --cooldown 10 --inter-case-cooldown 5 \
  --no-build-ax-engine \
  --output-dir "$MTP_DIR"

echo "===== DONE. Direct: $DIRECT_DIR  MTP: $MTP_DIR"
