#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/Users/akiralam/code/ax-engine}"
AX_BENCH="${AX_BENCH:-$REPO_DIR/target/release/ax-engine-bench}"
PROMPT_TOKENS="${PROMPT_TOKENS:-256}"
WARMUP_ITERS="${WARMUP_ITERS:-1}"
MEASURE_ITERS="${MEASURE_ITERS:-3}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/automatosx/tmp}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
TSV_OUT="${TSV_OUT:-}"

MODELS=(
  "qwen3-0.6B|$REPO_DIR/models/Qwen3-0.6B-Q4_K_M.gguf"
  "qwen35-9b|$REPO_DIR/models/Qwen3.5-9B-Q4_K_M.gguf"
  "llama3-8b|$REPO_DIR/models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf"
  "gemma3-4b|$REPO_DIR/models/gemma-3-4b-it-Q4_K_M.gguf"
)

if [[ ! -x "$AX_BENCH" ]]; then
  echo "error: AX bench not found or not executable: $AX_BENCH" >&2
  exit 1
fi

if ! mkdir -p "$OUT_DIR" 2>/dev/null || [[ ! -w "$OUT_DIR" ]]; then
  OUT_DIR="/tmp/ax-engine-bench-prefill"
fi
mkdir -p "$OUT_DIR"
TSV_OUT="${TSV_OUT:-$OUT_DIR/prefill-v2-ab-$TIMESTAMP.tsv}"

run_bench() {
  local model_path=$1
  local env_prefix=$2
  local suffix=$3
  local log_path="$OUT_DIR/prefill-${suffix}-${TIMESTAMP}.log"

  # shellcheck disable=SC2086
  if [[ -n "$env_prefix" ]]; then
    env $env_prefix "$AX_BENCH" bench \
      --model "$model_path" \
      --prompt-tokens "$PROMPT_TOKENS" \
      --decode-tokens 0 \
      --warmup-iters "$WARMUP_ITERS" \
      --measure-iters "$MEASURE_ITERS" \
      --intent latency \
      >"$log_path" 2>&1
  else
    "$AX_BENCH" bench \
      --model "$model_path" \
      --prompt-tokens "$PROMPT_TOKENS" \
      --decode-tokens 0 \
      --warmup-iters "$WARMUP_ITERS" \
      --measure-iters "$MEASURE_ITERS" \
      --intent latency \
      >"$log_path" 2>&1
  fi

  local prefill
  prefill="$(sed -n 's/^.*Prefill:.*median \([0-9][0-9.]*\) tok\/s.*/\1/p' "$log_path" | tail -1)"
  if [[ -z "$prefill" ]]; then
    echo "error: failed to parse prefill output in $log_path" >&2
    exit 1
  fi

  echo "$prefill"
}

printf "model\tvariant\tprompt_tokens\tprefill_tok_per_s\n" >"$TSV_OUT"
echo "Model | Variant | Prefill tok/s |"
echo "---|---|---:|"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name path <<<"$entry"
  [[ -f "$path" ]] || { echo "error: missing model file: $path" >&2; exit 1; }

  default_prefill="$(run_bench "$path" "" "default-$name")"
  v2off_prefill="$(run_bench "$path" "AX_METAL_BATCH_Q4K_V2=0" "v2off-$name")"
  v2on_prefill="$(run_bench "$path" "AX_METAL_BATCH_Q4K_V2=1" "v2on-$name")"

  printf "%s\t%s\t%s\t%s\n" "$name" "default" "$PROMPT_TOKENS" "$default_prefill" >>"$TSV_OUT"
  printf "%s\t%s\t%s\t%s\n" "$name" "v2=0" "$PROMPT_TOKENS" "$v2off_prefill" >>"$TSV_OUT"
  printf "%s\t%s\t%s\t%s\n" "$name" "v2=1" "$PROMPT_TOKENS" "$v2on_prefill" >>"$TSV_OUT"

  printf "%s | %s | %s |\\n" "$name" "default" "$default_prefill"
  printf "%s | %s | %s |\\n" "$name" "v2=0" "$v2off_prefill"
  printf "%s | %s | %s |\\n" "$name" "v2=1" "$v2on_prefill"
done

echo "results: $TSV_OUT"
