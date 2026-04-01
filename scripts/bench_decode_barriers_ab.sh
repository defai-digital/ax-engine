#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
usage() {
    cat <<'EOF'
Usage:
  scripts/bench_decode_barriers_ab.sh

Environment overrides:
  AX_BENCH_BIN     Benchmark binary path (default: target/release/ax-engine-bench)
  PROMPT_TOKENS    Prompt token count (default: 64)
  DECODE_TOKENS    Decode token count (default: 128)
  WARMUP_ITERS     Warmup iterations (default: 1)
  MEASURE_ITERS    Measurement iterations (default: 1)
  COOLDOWN         Cooldown between runs in seconds (default: 15)
  OUT_PREFIX       Explicit output prefix (default: automatosx/tmp/barrier-ab-<timestamp>)

Runs two configs on the default profile-backed path for:
  - Qwen3 8B
  - Gemma3 4B
  - LLaMA 3 8B

Configs:
  profile      profile/default env
  no_barrier   AX_METAL_DECODE_BARRIERS=0
EOF
}

if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

AX_BENCH_BIN=${AX_BENCH_BIN:-$REPO_DIR/target/release/ax-engine-bench}
PROMPT_TOKENS=${PROMPT_TOKENS:-64}
DECODE_TOKENS=${DECODE_TOKENS:-128}
WARMUP_ITERS=${WARMUP_ITERS:-1}
MEASURE_ITERS=${MEASURE_ITERS:-1}
COOLDOWN=${COOLDOWN:-15}

if [[ ! -x "$AX_BENCH_BIN" ]]; then
    echo "error: benchmark binary not found or not executable: $AX_BENCH_BIN" >&2
    exit 1
fi

STAMP=$(date +%Y%m%d-%H%M%S)
OUT_PREFIX=${OUT_PREFIX:-$REPO_DIR/automatosx/tmp/barrier-ab-${STAMP}}
OUT_TSV="${OUT_PREFIX}.tsv"
OUT_MD="${OUT_PREFIX}.md"
LOG_DIR="${OUT_PREFIX}-logs"
mkdir -p "$(dirname "$OUT_PREFIX")" "$LOG_DIR"

MODELS=(
  "qwen3-8b|$REPO_DIR/models/Qwen3-8B-Q4_K_M.gguf"
  "gemma3-4b|$REPO_DIR/models/gemma-3-4b-it-Q4_K_M.gguf"
  "llama3-8b|$REPO_DIR/models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf"
)

printf "timestamp\tmodel\tconfig\tprofile\tprefill_tok_s\tdecode_tok_s\texit_code\tlog_path\n" >"$OUT_TSV"

extract_profile() {
    local log_path=$1
    sed -n 's/.*Loaded kernel profile path=\([^ ]*\).*/\1/p' "$log_path" | tail -1
}

extract_prefill() {
    local log_path=$1
    sed -n 's/^Prefill:.* median \([0-9][0-9.]*\) tok\/s.*/\1/p' "$log_path" | tail -1
}

extract_decode() {
    local log_path=$1
    sed -n 's/^Decode:.* median \([0-9][0-9.]*\) tok\/s.*/\1/p' "$log_path" | tail -1
}

run_case() {
    local model_name=$1
    local model_path=$2
    local case_name=$3
    shift 3

    if [[ ! -f "$model_path" ]]; then
        echo "warning: skipping missing model $model_path" >&2
        return 0
    fi

    local log_path="${LOG_DIR}/${model_name}-${case_name}.log"
    local exit_code profile prefill decode ts
    ts=$(date +%Y-%m-%dT%H:%M:%S)

    if [[ "$COOLDOWN" != "0" ]]; then
        sleep "$COOLDOWN"
    fi

    if env "$@" "$AX_BENCH_BIN" bench \
        --model "$model_path" \
        --prompt-tokens "$PROMPT_TOKENS" \
        --decode-tokens "$DECODE_TOKENS" \
        --warmup-iters "$WARMUP_ITERS" \
        --measure-iters "$MEASURE_ITERS" \
        >"$log_path" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi

    profile=$(extract_profile "$log_path")
    prefill=$(extract_prefill "$log_path")
    decode=$(extract_decode "$log_path")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ts" \
        "$model_name" \
        "$case_name" \
        "${profile:-N/A}" \
        "${prefill:-N/A}" \
        "${decode:-N/A}" \
        "$exit_code" \
        "$log_path" >>"$OUT_TSV"

    echo "${model_name} ${case_name}: profile=${profile:-N/A} prefill=${prefill:-N/A} decode=${decode:-N/A} exit=${exit_code}"
}

for entry in "${MODELS[@]}"; do
    model_name=${entry%%|*}
    model_path=${entry#*|}
    run_case "$model_name" "$model_path" "profile"
    run_case "$model_name" "$model_path" "no_barrier" AX_METAL_DECODE_BARRIERS=0
done

awk -F'\t' '
BEGIN {
  print "# Decode Barrier A/B";
  print "";
  print "| Model | Profile prefill | No-barrier prefill | Δ prefill | Profile decode | No-barrier decode | Δ decode |";
  print "|---|---:|---:|---:|---:|---:|---:|";
}
NR > 1 {
  key = $2;
  if ($3 == "profile") {
    pp[key] = $5 + 0.0;
    pd[key] = $6 + 0.0;
  } else if ($3 == "no_barrier") {
    np[key] = $5 + 0.0;
    nd[key] = $6 + 0.0;
  }
}
END {
  n = split("qwen3-8b gemma3-4b llama3-8b", order, " ");
  for (i = 1; i <= n; ++i) {
    key = order[i];
    if (!(key in pp) || !(key in np)) {
      continue;
    }
    dpp = (np[key] - pp[key]) / pp[key] * 100.0;
    ddd = (nd[key] - pd[key]) / pd[key] * 100.0;
    printf "| %s | %.1f | %.1f | %+.1f%% | %.1f | %.1f | %+.1f%% |\n",
      key, pp[key], np[key], dpp, pd[key], nd[key], ddd;
  }
}
' "$OUT_TSV" >"$OUT_MD"

echo "TSV: $OUT_TSV"
echo "MD:  $OUT_MD"
