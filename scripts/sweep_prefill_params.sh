#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/sweep_prefill_params.sh <model> <model_name> <prompt_tokens> [cooldown_seconds]

Environment overrides:
  AX_BENCH_BIN       Benchmark binary path (default: target/release/ax-engine-bench)
  DECODE_TOKENS      Decode token count (default: 128)
  WARMUP_ITERS       Warmup iterations (default: 3)
  MEASURE_ITERS      Measurement iterations (default: 5)
  SWEEP_CONFIGS      Comma-separated config names to run (default: all)
  SWEEP_OUTPUT       Explicit TSV output path

Configs:
  P1_defaults
  P2_pair_enabled
  P3_fa2_auto
  P4_fa2_plus_pair
  P5_no_f16_io
  P6_bn32_off
EOF
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
    usage >&2
    exit 2
fi

MODEL=$1
MODEL_NAME=$2
PROMPT_TOKENS=$3
COOLDOWN=${4:-30}

AX_BENCH_BIN=${AX_BENCH_BIN:-$REPO_DIR/target/release/ax-engine-bench}
DECODE_TOKENS=${DECODE_TOKENS:-128}
WARMUP_ITERS=${WARMUP_ITERS:-3}
MEASURE_ITERS=${MEASURE_ITERS:-5}
OUT_DIR="${OUT_DIR:-$REPO_DIR/automatosx/tmp}"

if [[ ! -x "$AX_BENCH_BIN" ]]; then
    echo "error: benchmark binary not found or not executable: $AX_BENCH_BIN" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    if [[ -f "$REPO_DIR/$MODEL" ]]; then
        MODEL="$REPO_DIR/$MODEL"
    else
        echo "error: model not found: $MODEL" >&2
        exit 1
    fi
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="$OUT_DIR/prefill-sweep-logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

OUT=${SWEEP_OUTPUT:-"$OUT_DIR/prefill-sweep-${MODEL_NAME}-p${PROMPT_TOKENS}-${TIMESTAMP}.tsv"}
BENCH_ARGS=(
    bench
    --model "$MODEL"
    --prompt-tokens "$PROMPT_TOKENS"
    --decode-tokens "$DECODE_TOKENS"
    --warmup-iters "$WARMUP_ITERS"
    --measure-iters "$MEASURE_ITERS"
)

printf "timestamp\tmodel_name\tprompt_tokens\tconfig\tprofile_path\tprefill_toks\tdecode_toks\texit_code\tlog_path\n" >"$OUT"

declare -a SELECTED_CONFIGS=()
if [[ -n "${SWEEP_CONFIGS:-}" ]]; then
    IFS=',' read -r -a SELECTED_CONFIGS <<<"$SWEEP_CONFIGS"
fi

config_enabled() {
    local candidate=$1
    if [[ ${#SELECTED_CONFIGS[@]} -eq 0 ]]; then
        return 0
    fi

    local selected
    for selected in "${SELECTED_CONFIGS[@]}"; do
        if [[ "$selected" == "$candidate" ]]; then
            return 0
        fi
    done
    return 1
}

extract_profile_path() {
    local log_path=$1
    sed -n 's/.*Loaded kernel profile path=\([^ ]*\).*/\1/p' "$log_path" | tail -1
}

extract_prefill_toks() {
    local log_path=$1
    sed -n 's/^Prefill:.* median \([0-9][0-9.]*\) tok\/s.*/\1/p' "$log_path" | tail -1
}

extract_decode_toks() {
    local log_path=$1
    sed -n 's/^Decode:.* median \([0-9][0-9.]*\) tok\/s.*/\1/p' "$log_path" | tail -1
}

run_config() {
    local name=$1
    shift

    if ! config_enabled "$name"; then
        return 0
    fi

    local run_ts log_path exit_code profile_path prefill_toks decode_toks
    run_ts=$(date +%Y-%m-%dT%H:%M:%S)
    log_path="$LOG_DIR/${MODEL_NAME}-p${PROMPT_TOKENS}-${name}-${TIMESTAMP}.log"

    echo "--- Config: $name (${MODEL_NAME} P=${PROMPT_TOKENS}) ---"
    if [[ "$COOLDOWN" != "0" ]]; then
        echo "cooldown: ${COOLDOWN}s"
        sleep "$COOLDOWN"
    fi

    if env "$@" "$AX_BENCH_BIN" "${BENCH_ARGS[@]}" >"$log_path" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi

    profile_path=$(extract_profile_path "$log_path")
    prefill_toks=$(extract_prefill_toks "$log_path")
    decode_toks=$(extract_decode_toks "$log_path")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$run_ts" \
        "$MODEL_NAME" \
        "$PROMPT_TOKENS" \
        "$name" \
        "${profile_path:-N/A}" \
        "${prefill_toks:-N/A}" \
        "${decode_toks:-N/A}" \
        "$exit_code" \
        "$log_path" >>"$OUT"

    echo "profile=${profile_path:-N/A} prefill=${prefill_toks:-N/A} decode=${decode_toks:-N/A} exit=${exit_code}"
    if [[ $exit_code -ne 0 ]]; then
        echo "warning: benchmark failed for config=$name, see $log_path" >&2
    fi
}

run_config "P1_defaults"
run_config "P2_pair_enabled" \
    AX_METAL_BATCH_F16_PAIR=1
run_config "P3_fa2_auto" \
    AX_METAL_PREFILL_FA2_HD128_MODE=auto \
    AX_METAL_PREFILL_FA2_MODE=auto
run_config "P4_fa2_plus_pair" \
    AX_METAL_BATCH_F16_PAIR=1 \
    AX_METAL_PREFILL_FA2_HD128_MODE=auto \
    AX_METAL_PREFILL_FA2_MODE=auto
run_config "P5_no_f16_io" \
    AX_METAL_BATCH_F16_IO=0
run_config "P6_bn32_off" \
    AX_METAL_F16IN_BN32=0

echo "Results written to $OUT"
