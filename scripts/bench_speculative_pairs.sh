#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/bench_speculative_pairs.sh

Environment overrides:
  AX_BENCH_BIN     Benchmark binary path (default: target/release/ax-engine-bench)
  PROMPT_TOKENS    Prompt token count (default: 16)
  DECODE_TOKENS    Decode token count (default: 8)
  WARMUP_ITERS     Warmup iterations (default: 0)
  MEASURE_ITERS    Measurement iterations (default: 1)
  SPEC_K           Speculative lookahead K (default: 4)
  COOLDOWN         Cooldown between runs in seconds (default: 15)
  OUT_PREFIX       Explicit output prefix (default: automatosx/tmp/spec-pairs-<timestamp>)

Runs the locally plausible draft/target pairs and records:
  - whether the pair is valid
  - decode throughput
  - accepted draft tokens per step
  - draft / verify / accept timing
EOF
}

if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

AX_BENCH_BIN=${AX_BENCH_BIN:-target/release/ax-engine-bench}
PROMPT_TOKENS=${PROMPT_TOKENS:-16}
DECODE_TOKENS=${DECODE_TOKENS:-8}
WARMUP_ITERS=${WARMUP_ITERS:-0}
MEASURE_ITERS=${MEASURE_ITERS:-1}
SPEC_K=${SPEC_K:-4}
COOLDOWN=${COOLDOWN:-15}

if [[ ! -x "$AX_BENCH_BIN" ]]; then
    echo "error: benchmark binary not found or not executable: $AX_BENCH_BIN" >&2
    exit 1
fi

STAMP=$(date +%Y%m%d-%H%M%S)
OUT_PREFIX=${OUT_PREFIX:-automatosx/tmp/spec-pairs-${STAMP}}
OUT_TSV="${OUT_PREFIX}.tsv"
OUT_MD="${OUT_PREFIX}.md"
LOG_DIR="${OUT_PREFIX}-logs"
mkdir -p "$(dirname "$OUT_PREFIX")" "$LOG_DIR"

PAIRS=(
  "llama3-8b|models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf|llama3.1-8b-q8|models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
  "llama3-8b|models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf|deepseek-llama-8b-q4|models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
  "qwen3-8b|models/Qwen3-8B-Q4_K_M.gguf|deepseek-qwen-7b-q4|models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
)

printf "timestamp\ttarget\tdraft\tvalid\tdecode_tok_s\taccepted_per_step\tdraft_ms_per_step\tverify_ms_per_step\taccept_ms_per_step\texit_code\tlog_path\n" >"$OUT_TSV"

extract_decode() {
    local log_path=$1
    sed -n 's/^Decode:.* median \([0-9][0-9.]*\) tok\/s.*/\1/p' "$log_path" | tail -1
}

extract_accepted() {
    local log_path=$1
    sed -n 's/^Accepted: *\([0-9][0-9.]*\) draft tokens\/step/\1/p' "$log_path" | tail -1
}

extract_step_ms() {
    local label=$1
    local log_path=$2
    sed -n "s/^${label}: *\([0-9][0-9.]*\) ms\/step.*/\1/p" "$log_path" | tail -1
}

is_vocab_mismatch() {
    local log_path=$1
    rg -q "matching draft/target vocab sizes" "$log_path"
}

run_pair() {
    local target_name=$1
    local target_model=$2
    local draft_name=$3
    local draft_model=$4

    local ts log_path exit_code valid decode accepted draft_ms verify_ms accept_ms
    ts=$(date +%Y-%m-%dT%H:%M:%S)
    log_path="${LOG_DIR}/${target_name}__${draft_name}.log"

    if [[ ! -f "$target_model" || ! -f "$draft_model" ]]; then
        echo "warning: skipping missing pair target=$target_model draft=$draft_model" >&2
        return 0
    fi

    if [[ "$COOLDOWN" != "0" ]]; then
        sleep "$COOLDOWN"
    fi

    if "$AX_BENCH_BIN" speculative \
        --model "$target_model" \
        --draft-model "$draft_model" \
        --prompt-tokens "$PROMPT_TOKENS" \
        --decode-tokens "$DECODE_TOKENS" \
        --warmup-iters "$WARMUP_ITERS" \
        --measure-iters "$MEASURE_ITERS" \
        --speculative-k "$SPEC_K" \
        >"$log_path" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi

    if is_vocab_mismatch "$log_path"; then
        valid=no
    elif [[ $exit_code -eq 0 ]]; then
        valid=yes
    else
        valid=error
    fi

    decode=$(extract_decode "$log_path")
    accepted=$(extract_accepted "$log_path")
    draft_ms=$(extract_step_ms "Draft" "$log_path")
    verify_ms=$(extract_step_ms "Verify" "$log_path")
    accept_ms=$(extract_step_ms "Accept" "$log_path")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ts" \
        "$target_name" \
        "$draft_name" \
        "$valid" \
        "${decode:-N/A}" \
        "${accepted:-N/A}" \
        "${draft_ms:-N/A}" \
        "${verify_ms:-N/A}" \
        "${accept_ms:-N/A}" \
        "$exit_code" \
        "$log_path" >>"$OUT_TSV"

    echo "${target_name} <- ${draft_name}: valid=${valid} decode=${decode:-N/A} accepted=${accepted:-N/A} exit=${exit_code}"
}

for entry in "${PAIRS[@]}"; do
    IFS='|' read -r target_name target_model draft_name draft_model <<<"$entry"
    run_pair "$target_name" "$target_model" "$draft_name" "$draft_model"
done

awk -F'\t' '
BEGIN {
  print "# Speculative Pair Check";
  print "";
  print "| Target | Draft | Valid | Decode tok/s | Accepted/step | Draft ms/step | Verify ms/step |";
  print "|---|---|---|---:|---:|---:|---:|";
}
NR > 1 {
  printf "| %s | %s | %s | %s | %s | %s | %s |\n", $2, $3, $4, $5, $6, $7, $8;
}
' "$OUT_TSV" >"$OUT_MD"

echo "TSV: $OUT_TSV"
echo "MD:  $OUT_MD"
