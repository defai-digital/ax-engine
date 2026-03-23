#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/sweep_kernel_params.sh <model> <model_name> <prompt_tokens> [cooldown_seconds]

Environment overrides:
  AX_BENCH_BIN       Benchmark binary path (default: target/release/ax-bench)
  DECODE_TOKENS      Decode token count (default: 128)
  WARMUP_ITERS       Warmup iterations (default: 3)
  MEASURE_ITERS      Measurement iterations (default: 5)
  SWEEP_CONFIGS      Comma-separated config names to run (default: all applicable)
  SWEEP_OUTPUT       Explicit TSV output path

Configs:
  A_profile
  B_base
  C_q4k_nr2_only
  D_q6k_nr2_only
  E_nr2_splitk_off   (Gemma-only)
  F_nr2_splitk_on    (Gemma-only)
  G_nr2_no_barrier
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

AX_BENCH_BIN=${AX_BENCH_BIN:-target/release/ax-bench}
DECODE_TOKENS=${DECODE_TOKENS:-128}
WARMUP_ITERS=${WARMUP_ITERS:-3}
MEASURE_ITERS=${MEASURE_ITERS:-5}

if [[ ! -x "$AX_BENCH_BIN" ]]; then
    echo "error: benchmark binary not found or not executable: $AX_BENCH_BIN" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "error: model not found: $MODEL" >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="automatosx/tmp"
LOG_DIR="$OUT_DIR/sweep-logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

OUT=${SWEEP_OUTPUT:-"$OUT_DIR/sweep-${MODEL_NAME}-p${PROMPT_TOKENS}-${TIMESTAMP}.tsv"}
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

is_gemma_model() {
    [[ "$MODEL_NAME" == *gemma* || "$MODEL_NAME" == *Gemma* ]]
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

run_config "A_profile"
run_config "B_base" \
    AX_METAL_MATVEC_Q4K_TG=128 AX_METAL_MATVEC_Q4K_NR=1 \
    AX_METAL_MATVEC_Q6K_TG=128 AX_METAL_MATVEC_Q6K_NR=1
run_config "C_q4k_nr2_only" \
    AX_METAL_MATVEC_Q4K_TG=64 AX_METAL_MATVEC_Q4K_NR=2 \
    AX_METAL_MATVEC_Q6K_TG=128 AX_METAL_MATVEC_Q6K_NR=1
run_config "D_q6k_nr2_only" \
    AX_METAL_MATVEC_Q4K_TG=128 AX_METAL_MATVEC_Q4K_NR=1 \
    AX_METAL_MATVEC_Q6K_TG=64 AX_METAL_MATVEC_Q6K_NR=2
if is_gemma_model; then
    run_config "E_nr2_splitk_off" \
        AX_METAL_MATVEC_Q4K_TG=64 AX_METAL_MATVEC_Q4K_NR=2 \
        AX_METAL_MATVEC_Q6K_TG=64 AX_METAL_MATVEC_Q6K_NR=2 \
        AX_METAL_DECODE_SPLITK_MODE=off
    run_config "F_nr2_splitk_on" \
        AX_METAL_MATVEC_Q4K_TG=64 AX_METAL_MATVEC_Q4K_NR=2 \
        AX_METAL_MATVEC_Q6K_TG=64 AX_METAL_MATVEC_Q6K_NR=2 \
        AX_METAL_DECODE_SPLITK_MODE=on
fi
run_config "G_nr2_no_barrier" \
    AX_METAL_MATVEC_Q4K_TG=64 AX_METAL_MATVEC_Q4K_NR=2 \
    AX_METAL_MATVEC_Q6K_TG=64 AX_METAL_MATVEC_Q6K_NR=2 \
    AX_METAL_DECODE_BARRIERS=0

echo "Results written to $OUT"
