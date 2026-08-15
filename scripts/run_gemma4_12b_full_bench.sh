#!/usr/bin/env bash
# One-shot driver for the dedicated Gemma 4 12B (gemma4_unified) benchmark.
#
# Produces the two artifact trees that back the retained Gemma 4 12B case study:
#   1. Direct decode/prefill/TTFT on the AX-native MLX route (mlx_lm cannot
#      load gemma4_unified). Historical llama.cpp Metal comparison rows stay
#      in the checked-in artifacts; this driver no longer reruns llama.cpp.
#   2. Assistant-MTP and assistant MTP+n-gram decode, same methodology as the
#      published 26B/31B run (1000 tokens, 5 reps, 10s/5s cooldowns).
#
# Run sequentially on purpose: both phases saturate the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

DATE=2026-06-09
PY=python3
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
FFN4_MTP_DIR="$HOME/.cache/huggingface/hub/models--ax-local--gemma-4-12b-it-4bit-ffn4-assistant-mtp/snapshots/v1"
if [ ! -f "$FFN4_DIR/model-manifest.json" ]; then
  echo "===== Building fair-quant 4-bit-FFN artifact (one-time) -> $FFN4_DIR"
  mkdir -p "$FFN4_DIR"
  $PY scripts/requantize_gemma4_12b_ffn_4bit.py "$BASE_DIR" "$FFN4_DIR"
  cargo run -q --release -p ax-engine-core --bin generate-manifest -- --force --validate "$FFN4_DIR"
fi

DIRECT_DIR="benchmarks/results/inference/mlx-inference/${DATE}-gemma-4-12b-it-4bit-direct"
MTP_DIR="benchmarks/results/gemma4-assistant-mtp/${DATE}-gemma4-12b-ffn4-mtp-speedup"
mkdir -p "$DIRECT_DIR"

if [ ! -f "$FFN4_MTP_DIR/model-manifest.json" ]; then
  echo "===== Building fair-quant 4-bit-FFN assistant-MTP package -> $FFN4_MTP_DIR"
  $PY scripts/prepare_gemma4_assistant_mtp.py \
    --target "$FFN4_DIR" \
    --assistant mlx-community/gemma-4-12B-it-assistant-4bit \
    --target-model-id gemma-4-12b-it \
    --assistant-model-id gemma-4-12b-it-assistant \
    --output "$FFN4_MTP_DIR" \
    --max-depth 2
fi

echo "===== BASE_DIR: $BASE_DIR"

# Record the historical llama.cpp GGUF provenance next to the direct artifact.
cat > "$DIRECT_DIR/llama_cpp_gguf_provenance.json" <<JSON
{
  "schema": "ax.gemma4_12b_llama_cpp_provenance.v1",
  "model": "Gemma 4 12B",
  "mlx_repo_id": "ax-local/gemma-4-12B-it-4bit-ffn4 (FFN re-quantized 8->4 bit from mlx-community/gemma-4-12B-it-4bit)",
  "ax_quant_note": "Upstream mlx-community 4bit keeps FFN at 8-bit (~10.4 GiB); re-quantized to uniform 4-bit group-64 (~6.3 GiB, ~4.5 bpw) for bit-comparable fairness vs Q4_K_M (~6.87 GiB, ~4.8 bpw). See scripts/requantize_gemma4_12b_ffn_4bit.py.",
  "gguf_repo": "unsloth/gemma-4-12b-it-GGUF",
  "gguf_file": "gemma-4-12b-it-Q4_K_M.gguf",
  "gguf_quant_target": "Q4_K_M",
  "decode_comparison_contract": "llama.cpp decode measured at matched context depth (--llama-cpp-decode-at-depth: llama-bench -n GEN -d PROMPT), not depth-0 tg, to match AX's prompt-context decode.",
  "llama_cpp_arch": "gemma4",
  "publisher_note": "Uses the Unsloth Gemma 4 12B standard Q4_K_M GGUF for the shape-compatible external llama.cpp baseline. Not prompt-hash parity.",
  "mlx_lm_baseline": "absent_unsupported_architecture",
  "mlx_lm_note": "mlx_lm 0.31.3 has no graph for model_type gemma4_unified (ValueError: Model type gemma4_unified not supported); the MLX-side reference is AX Engine's repo-owned native runtime."
}
JSON

echo "===== Phase 1/2: direct decode (AX native) ====="
# AX uses the fair-quant 4-bit-FFN artifact ($FFN4_DIR). Historical llama.cpp
# Metal rows remain in the checked-in case-study artifacts and are not rerun
# by this driver.
$PY scripts/bench_mlx_inference_stack.py \
  --model gemma-4-12b-it-ffn4 --model-dir "$FFN4_DIR" \
  --prompt-tokens 128,512,2048 \
  --generation-tokens 128 \
  --repetitions 5 \
  --cooldown 15 \
  --ax-direct --skip-mlx-lm --no-build-ax-engine \
  --output "$DIRECT_DIR/gemma-4-12b-it-4bit.json"

echo "===== Phase 2/2: same-artifact direct/MTP/MTP+n-gram ablation ====="
$PY scripts/bench_gemma4_assistant_mtp.py \
  --models 12b-4bit-ffn4 \
  --profiles direct,assistant_mtp_default,assistant_mtp_ngram_default,utility_gate_candidate,assistant_mtp_gate095,assistant_mtp_gate090,assistant_mtp_gate085,assistant_mtp_deep099,assistant_mtp_deep095,assistant_mtp_depth1,assistant_mtp_depth2,assistant_mtp_gpu_confidence,assistant_mtp_ngram_gpu_confidence,assistant_mtp_ngram_ctx2_support1,assistant_mtp_ngram_ctx4_support4,assistant_mtp_ngram_confidence050,assistant_mtp_ngram_safety_disable_all \
  --suites flappy,long_code,python_modules_long \
  --model-dir "12b-4bit-ffn4=$FFN4_MTP_DIR" \
  --max-tokens 1000 --repetitions 5 --cooldown 10 --inter-case-cooldown 5 \
  --no-build-ax-engine \
  --output-dir "$MTP_DIR"

echo "===== DONE. Direct: $DIRECT_DIR  MTP: $MTP_DIR"
