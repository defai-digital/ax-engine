#!/usr/bin/env bash
# =============================================================================
# ax-engine autotuner
#
# Sweeps kernel parameters for a given model, finds the best config, and writes
# a kernel profile JSON to perfs/.
#
# Usage:
#   ./scripts/autotune.sh --model ./models/Qwen3.5-9B-Q4_K_M.gguf
#   ./scripts/autotune.sh --model ./models/Qwen3-8B-Q4_K_M.gguf --phase decode
#   ./scripts/autotune.sh --model ./models/Qwen3.5-9B-Q4_K_M.gguf --prompt-tokens 512 --decode-tokens 128
#
# Writes results to: automatosx/tmp/autotune-<model>-<date>/
# Final artifacts:
#   - tuned.env: env-only overrides
#   - tuned-profile.json: slim kernel profile override for schema-backed knobs
# =============================================================================
set -euo pipefail

# ─── defaults ───────────────────────────────────────────────────────────────
MODEL=""
PROMPT_TOKENS="256"
DECODE_TOKENS="64"
SAMPLES=3
COOLDOWN_MS=20000
WARMUP=1
MEASURE=3
PHASE="all"         # all | prefill | decode | attention
APPLY=false
DRY_RUN=false
BENCH_BIN="./target/release/ax-engine-bench"

# ─── parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2;;
        --prompt-tokens) PROMPT_TOKENS="$2"; shift 2;;
        --decode-tokens) DECODE_TOKENS="$2"; shift 2;;
        --samples)     SAMPLES="$2"; shift 2;;
        --cooldown-ms) COOLDOWN_MS="$2"; shift 2;;
        --phase)       PHASE="$2"; shift 2;;
        --apply)       APPLY=true; shift;;
        --dry-run)     DRY_RUN=true; shift;;
        --bench-bin)   BENCH_BIN="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 --model <path-to-gguf> [--phase all|prefill|decode|attention]"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Model not found: $MODEL"
    exit 1
fi

# ─── setup ──────────────────────────────────────────────────────────────────
MODEL_BASENAME="$(basename "$MODEL" .gguf | tr '[:upper:]' '[:lower:]' | tr ' ' '-')"
DATE="$(date +%Y%m%d-%H%M%S)"
OUTDIR="automatosx/tmp/autotune-${MODEL_BASENAME}-${DATE}"
mkdir -p "$OUTDIR"

# Build if needed
if [[ ! -x "$BENCH_BIN" ]]; then
    echo "Building ax-engine-bench (release)..."
    cargo build -p ax-engine-bench --release
fi

echo "═══════════════════════════════════════════════════════════════"
echo " ax-engine autotuner"
echo " Model:    $MODEL"
echo " Tokens:   prefill=$PROMPT_TOKENS  decode=$DECODE_TOKENS"
echo " Samples:  $SAMPLES  cooldown=${COOLDOWN_MS}ms"
echo " Phase:    $PHASE"
echo " Output:   $OUTDIR/"
echo "═══════════════════════════════════════════════════════════════"

# ─── helpers ────────────────────────────────────────────────────────────────

# Run a single benchmark with given env vars, return JSON path
# Usage: run_bench "label" "ENV1=val1 ENV2=val2"
run_bench() {
    local label="$1"
    local envs="$2"
    local json_path="${OUTDIR}/${label}.json"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[dry-run] $label: $envs"
        return 0
    fi

    echo -n "  ► $label ... "

    # Build the env command
    local cmd="env"
    for kv in $envs; do
        cmd="$cmd $kv"
    done
    cmd="$cmd $BENCH_BIN bench"
    cmd="$cmd --model $MODEL"
    cmd="$cmd --prompt-tokens $PROMPT_TOKENS"
    cmd="$cmd --decode-tokens $DECODE_TOKENS"
    cmd="$cmd --deterministic"
    cmd="$cmd --samples $SAMPLES"
    cmd="$cmd --cooldown-ms $COOLDOWN_MS"
    cmd="$cmd --warmup-iters $WARMUP"
    cmd="$cmd --measure-iters $MEASURE"
    cmd="$cmd --json-output $json_path"

    # Run, capture stderr for the human summary
    if eval "$cmd" 2>"${OUTDIR}/${label}.log" ; then
        # Extract key metrics
        if [[ -f "$json_path" ]]; then
            local prefill decode
            prefill=$(python3 -c "import json; d=json.load(open('$json_path')); print(f\"{d['prefill_tok_per_sec_median']:.1f}\")" 2>/dev/null || echo "?")
            decode=$(python3 -c "import json; d=json.load(open('$json_path')); print(f\"{d['decode_tok_per_sec_median']:.1f}\")" 2>/dev/null || echo "?")
            echo "prefill=${prefill} tok/s  decode=${decode} tok/s"
        else
            echo "no JSON output"
        fi
    else
        echo "FAILED (see ${label}.log)"
    fi
}

# Extract a metric from a JSON result file
# Usage: get_metric "file.json" "prefill_tok_per_sec_median"
get_metric() {
    python3 -c "import json; d=json.load(open('$1')); print(d.get('$2', 0))" 2>/dev/null || echo "0"
}

is_qwen35_model() {
    local lowered
    lowered="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
    [[ "$lowered" == *"qwen3.5"* || "$lowered" == *"qwen35"* ]]
}

run_prefill_profile() {
    local label="$1"
    local envs="$2"
    local json_path="${OUTDIR}/${label}.json"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[dry-run] $label: $envs"
        return 0
    fi

    echo -n "  ► $label ... "
    local cmd="env"
    for kv in $envs; do
        cmd="$cmd $kv"
    done
    cmd="$cmd $BENCH_BIN prefill-profile"
    cmd="$cmd --model $MODEL"
    cmd="$cmd --prompt-tokens $PROMPT_TOKENS"
    cmd="$cmd --json-output $json_path"

    if eval "$cmd" 2>"${OUTDIR}/${label}.log" ; then
        python3 -c "
import json
d=json.load(open('$json_path'))
print(
    f\"prefill={d.get('tok_per_sec',0):.1f} tok/s  \"
    f\"handoff={d.get('recurrent_qkv_handoff_layers',0)}  \"
    f\"fast={d.get('recurrent_qkv_fast_path_eligible_layers',0)}  \"
    f\"batchfast={d.get('recurrent_batch_qkv_handoff_ms',0):.1f}ms\"
)
" 2>/dev/null || echo "ok"
    else
        echo "FAILED (see ${label}.log)"
    fi
}

qwen35_prefill_candidate_ok() {
    local label="$1"
    local envs="$2"
    if ! is_qwen35_model; then
        return 0
    fi

    local profile_label="${label}-qwen35-prefill"
    run_prefill_profile "$profile_label" "$envs"
    local handoff
    handoff=$(get_metric "$OUTDIR/${profile_label}.json" "recurrent_qkv_handoff_layers")
    if python3 -c "import sys; sys.exit(0 if float('$handoff') > 0 else 1)"; then
        return 0
    fi

    echo "    ✗ rejected: qkv-handoff disabled"
    return 1
}

# Compare two results, return "better" if candidate beats baseline by >1%
compare() {
    local baseline_val="$1"
    local candidate_val="$2"
    python3 -c "
b, c = float('$baseline_val'), float('$candidate_val')
if b > 0 and c > b * 1.01:
    print('better')
elif b > 0 and c < b * 0.99:
    print('worse')
else:
    print('same')
" 2>/dev/null || echo "same"
}

# ─── Phase 0: Baseline ─────────────────────────────────────────────────────
echo ""
echo "Phase 0: Baseline measurement"
echo "─────────────────────────────"
run_bench "baseline" ""
if [[ "$DRY_RUN" != "true" && "$(is_qwen35_model && echo 1 || echo 0)" == "1" ]]; then
    run_prefill_profile "baseline-qwen35-prefill" ""
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run mode — exiting]"
    exit 0
fi

BASELINE_PREFILL=$(get_metric "$OUTDIR/baseline.json" "prefill_tok_per_sec_median")
BASELINE_DECODE=$(get_metric "$OUTDIR/baseline.json" "decode_tok_per_sec_median")
echo "  Baseline: prefill=${BASELINE_PREFILL} tok/s  decode=${BASELINE_DECODE} tok/s"

# Track winners (env var assignments that improved over baseline)
declare -a PREFILL_WINNERS=()
declare -a DECODE_WINNERS=()

# ─── Phase 1: Decode matvec sweep ──────────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "decode" ]]; then
echo ""
echo "Phase 1: Decode matvec kernel sweep"
echo "────────────────────────────────────"

# Q4_K threadgroup size
for tg in 64 128; do
    for nr in 1 2; do
        label="decode-q4k-tg${tg}-nr${nr}"
        run_bench "$label" "AX_METAL_MATVEC_Q4K_TG=$tg AX_METAL_MATVEC_Q4K_NR=$nr"
        val=$(get_metric "$OUTDIR/${label}.json" "decode_tok_per_sec_median")
        result=$(compare "$BASELINE_DECODE" "$val")
        if [[ "$result" == "better" ]]; then
            DECODE_WINNERS+=("AX_METAL_MATVEC_Q4K_TG=$tg" "AX_METAL_MATVEC_Q4K_NR=$nr")
            echo "    ✓ winner (+$(python3 -c "print(f'{(float($val)/float($BASELINE_DECODE)-1)*100:.1f}')")%)"
        fi
    done
done

# Q6_K threadgroup size
for tg in 64 128; do
    for nr in 1 2; do
        label="decode-q6k-tg${tg}-nr${nr}"
        run_bench "$label" "AX_METAL_MATVEC_Q6K_TG=$tg AX_METAL_MATVEC_Q6K_NR=$nr"
        val=$(get_metric "$OUTDIR/${label}.json" "decode_tok_per_sec_median")
        result=$(compare "$BASELINE_DECODE" "$val")
        if [[ "$result" == "better" ]]; then
            DECODE_WINNERS+=("AX_METAL_MATVEC_Q6K_TG=$tg" "AX_METAL_MATVEC_Q6K_NR=$nr")
            echo "    ✓ winner (+$(python3 -c "print(f'{(float($val)/float($BASELINE_DECODE)-1)*100:.1f}')")%)"
        fi
    done
done

# Fused SiLU+down
for v in 0 1; do
    label="decode-fused-silu-down-${v}"
    run_bench "$label" "AX_METAL_DECODE_FUSED_SILU_DOWN=$v"
    val=$(get_metric "$OUTDIR/${label}.json" "decode_tok_per_sec_median")
    result=$(compare "$BASELINE_DECODE" "$val")
    if [[ "$result" == "better" ]]; then
        DECODE_WINNERS+=("AX_METAL_DECODE_FUSED_SILU_DOWN=$v")
        echo "    ✓ winner"
    fi
done

# Pair matvec decode
for v in 0 1; do
    label="decode-pair-matvec-${v}"
    run_bench "$label" "AX_METAL_DECODE_PAIR_MATVEC=$v"
    val=$(get_metric "$OUTDIR/${label}.json" "decode_tok_per_sec_median")
    result=$(compare "$BASELINE_DECODE" "$val")
    if [[ "$result" == "better" ]]; then
        DECODE_WINNERS+=("AX_METAL_DECODE_PAIR_MATVEC=$v")
        echo "    ✓ winner"
    fi
done

# Fused QKV decode
for v in 0 1; do
    label="decode-fused-qkv-${v}"
    run_bench "$label" "AX_METAL_DECODE_FUSED_QKV=$v"
    val=$(get_metric "$OUTDIR/${label}.json" "decode_tok_per_sec_median")
    result=$(compare "$BASELINE_DECODE" "$val")
    if [[ "$result" == "better" ]]; then
        DECODE_WINNERS+=("AX_METAL_DECODE_FUSED_QKV=$v")
        echo "    ✓ winner"
    fi
done

fi  # decode phase

# ─── Phase 2: Decode attention sweep ───────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "attention" ]]; then
echo ""
echo "Phase 2: Decode attention kernel sweep"
echo "───────────────────────────────────────"

# SplitK modes
for mode in off on auto; do
    label="decode-splitk-${mode}"
    run_bench "$label" "AX_METAL_DECODE_SPLITK_MODE=$mode"
    val=$(get_metric "$OUTDIR/${label}.json" "decode_tok_per_sec_median")
    result=$(compare "$BASELINE_DECODE" "$val")
    if [[ "$result" == "better" ]]; then
        DECODE_WINNERS+=("AX_METAL_DECODE_SPLITK_MODE=$mode")
        echo "    ✓ winner"
    fi
done

# SplitK chunk sizes (only if splitk helps)
for chunk in 128 256 512; do
    label="decode-splitk-chunk-${chunk}"
    run_bench "$label" "AX_METAL_DECODE_SPLITK_MODE=on AX_METAL_DECODE_SPLITK_CHUNK_SIZE=$chunk"
    val=$(get_metric "$OUTDIR/${label}.json" "decode_tok_per_sec_median")
    result=$(compare "$BASELINE_DECODE" "$val")
    if [[ "$result" == "better" ]]; then
        DECODE_WINNERS+=("AX_METAL_DECODE_SPLITK_MODE=on" "AX_METAL_DECODE_SPLITK_CHUNK_SIZE=$chunk")
        echo "    ✓ winner"
    fi
done

# HD128 N2 decode
for v in 0 1; do
    label="decode-hd128-n2-${v}"
    run_bench "$label" "AX_METAL_DECODE_HD128_N2=$v"
    val=$(get_metric "$OUTDIR/${label}.json" "decode_tok_per_sec_median")
    result=$(compare "$BASELINE_DECODE" "$val")
    if [[ "$result" == "better" ]]; then
        DECODE_WINNERS+=("AX_METAL_DECODE_HD128_N2=$v")
        echo "    ✓ winner"
    fi
done

# SDPA decode
for v in 0 1; do
    label="decode-sdpa-${v}"
    run_bench "$label" "AX_METAL_DECODE_SDPA=$v"
    val=$(get_metric "$OUTDIR/${label}.json" "decode_tok_per_sec_median")
    result=$(compare "$BASELINE_DECODE" "$val")
    if [[ "$result" == "better" ]]; then
        DECODE_WINNERS+=("AX_METAL_DECODE_SDPA=$v")
        echo "    ✓ winner"
    fi
done

fi  # attention phase

# ─── Phase 3: Prefill batch matmul sweep ───────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "prefill" ]]; then
echo ""
echo "Phase 3: Prefill batch kernel sweep"
echo "────────────────────────────────────"

# F16 I/O
for v in 0 1; do
    label="prefill-f16io-${v}"
    run_bench "$label" "AX_METAL_BATCH_F16_IO=$v"
    val=$(get_metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    result=$(compare "$BASELINE_PREFILL" "$val")
    if [[ "$result" == "better" ]]; then
        if qwen35_prefill_candidate_ok "$label" "AX_METAL_BATCH_F16_IO=$v"; then
            PREFILL_WINNERS+=("AX_METAL_BATCH_F16_IO=$v")
            echo "    ✓ winner (+$(python3 -c "print(f'{(float($val)/float($BASELINE_PREFILL)-1)*100:.1f}')")%)"
        fi
    fi
done

# F16 pair kernel
for v in 0 1; do
    label="prefill-f16pair-${v}"
    run_bench "$label" "AX_METAL_BATCH_F16_PAIR=$v"
    val=$(get_metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    result=$(compare "$BASELINE_PREFILL" "$val")
    if [[ "$result" == "better" ]]; then
        if qwen35_prefill_candidate_ok "$label" "AX_METAL_BATCH_F16_PAIR=$v"; then
            PREFILL_WINNERS+=("AX_METAL_BATCH_F16_PAIR=$v")
            echo "    ✓ winner"
        fi
    fi
done

# Blocked layout variants
for quant in Q4K Q5K Q6K Q8; do
    for v in 0 1; do
        label="prefill-blocked-${quant,,}-${v}"
        run_bench "$label" "AX_METAL_BATCH_${quant}_BLOCKED=$v"
        val=$(get_metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
        result=$(compare "$BASELINE_PREFILL" "$val")
        if [[ "$result" == "better" ]]; then
            if qwen35_prefill_candidate_ok "$label" "AX_METAL_BATCH_${quant}_BLOCKED=$v"; then
                PREFILL_WINNERS+=("AX_METAL_BATCH_${quant}_BLOCKED=$v")
                echo "    ✓ winner"
            fi
        fi
    done
done

# BK32 vs BK64
for v in 0 1; do
    label="prefill-bk32-${v}"
    run_bench "$label" "AX_METAL_F16IN_BK32=$v"
    val=$(get_metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    result=$(compare "$BASELINE_PREFILL" "$val")
    if [[ "$result" == "better" ]]; then
        if qwen35_prefill_candidate_ok "$label" "AX_METAL_F16IN_BK32=$v"; then
            PREFILL_WINNERS+=("AX_METAL_F16IN_BK32=$v")
            echo "    ✓ winner"
        fi
    fi
done

# BN32 full-tile
for v in 0 1; do
    label="prefill-bn32-${v}"
    run_bench "$label" "AX_METAL_F16IN_BN32=$v"
    val=$(get_metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    result=$(compare "$BASELINE_PREFILL" "$val")
    if [[ "$result" == "better" ]]; then
        if qwen35_prefill_candidate_ok "$label" "AX_METAL_F16IN_BN32=$v"; then
            PREFILL_WINNERS+=("AX_METAL_F16IN_BN32=$v")
            echo "    ✓ winner"
        fi
    fi
done

# Fused QKV prefill
for v in 0 1; do
    label="prefill-fused-qkv-${v}"
    run_bench "$label" "AX_METAL_FUSED_QKV=$v"
    val=$(get_metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    result=$(compare "$BASELINE_PREFILL" "$val")
    if [[ "$result" == "better" ]]; then
        if qwen35_prefill_candidate_ok "$label" "AX_METAL_FUSED_QKV=$v"; then
            PREFILL_WINNERS+=("AX_METAL_FUSED_QKV=$v")
            echo "    ✓ winner"
        fi
    fi
done

# Prefill attention: FA2 modes
for mode in off on auto; do
    label="prefill-fa2-${mode}"
    run_bench "$label" "AX_METAL_PREFILL_FA2_MODE=$mode"
    val=$(get_metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    result=$(compare "$BASELINE_PREFILL" "$val")
    if [[ "$result" == "better" ]]; then
        if qwen35_prefill_candidate_ok "$label" "AX_METAL_PREFILL_FA2_MODE=$mode"; then
            PREFILL_WINNERS+=("AX_METAL_PREFILL_FA2_MODE=$mode")
            echo "    ✓ winner"
        fi
    fi
done

# Prefill attention: FA2 HD128
for mode in off on auto; do
    label="prefill-fa2-hd128-${mode}"
    run_bench "$label" "AX_METAL_PREFILL_FA2_HD128_MODE=$mode"
    val=$(get_metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    result=$(compare "$BASELINE_PREFILL" "$val")
    if [[ "$result" == "better" ]]; then
        if qwen35_prefill_candidate_ok "$label" "AX_METAL_PREFILL_FA2_HD128_MODE=$mode"; then
            PREFILL_WINNERS+=("AX_METAL_PREFILL_FA2_HD128_MODE=$mode")
            echo "    ✓ winner"
        fi
    fi
done

# Graph IR scheduling
for v in 0 1; do
    label="prefill-graph-ir-${v}"
    run_bench "$label" "AX_METAL_PREFILL_GRAPH_IR=$v"
    val=$(get_metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    result=$(compare "$BASELINE_PREFILL" "$val")
    if [[ "$result" == "better" ]]; then
        if qwen35_prefill_candidate_ok "$label" "AX_METAL_PREFILL_GRAPH_IR=$v"; then
            PREFILL_WINNERS+=("AX_METAL_PREFILL_GRAPH_IR=$v")
            echo "    ✓ winner"
        fi
    fi
done

# Multi command buffer prefill
for v in 0 1; do
    label="prefill-multi-cb-${v}"
    run_bench "$label" "AX_METAL_PREFILL_MULTI_CB=$v"
    val=$(get_metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    result=$(compare "$BASELINE_PREFILL" "$val")
    if [[ "$result" == "better" ]]; then
        if qwen35_prefill_candidate_ok "$label" "AX_METAL_PREFILL_MULTI_CB=$v"; then
            PREFILL_WINNERS+=("AX_METAL_PREFILL_MULTI_CB=$v")
            echo "    ✓ winner"
        fi
    fi
done

fi  # prefill phase

# ─── Phase 4: Combine winners and verify ───────────────────────────────────
echo ""
echo "Phase 4: Combine winners and verify"
echo "────────────────────────────────────"

# Deduplicate winners (last write wins for same env var)
declare -A COMBINED_ENV=()
for w in "${PREFILL_WINNERS[@]+"${PREFILL_WINNERS[@]}"}"; do
    key="${w%%=*}"
    COMBINED_ENV["$key"]="$w"
done
for w in "${DECODE_WINNERS[@]+"${DECODE_WINNERS[@]}"}"; do
    key="${w%%=*}"
    COMBINED_ENV["$key"]="$w"
done

if [[ ${#COMBINED_ENV[@]} -eq 0 ]]; then
    echo "  No improvements found over baseline. Current profile is already optimal."
    echo ""
    echo "  Baseline results:"
    echo "    Prefill: ${BASELINE_PREFILL} tok/s"
    echo "    Decode:  ${BASELINE_DECODE} tok/s"
    exit 0
fi

COMBINED_STR=""
echo "  Combined winners:"
for key in "${!COMBINED_ENV[@]}"; do
    echo "    ${COMBINED_ENV[$key]}"
    COMBINED_STR="$COMBINED_STR ${COMBINED_ENV[$key]}"
done

run_bench "combined" "$COMBINED_STR"
COMBINED_PREFILL=$(get_metric "$OUTDIR/combined.json" "prefill_tok_per_sec_median")
COMBINED_DECODE=$(get_metric "$OUTDIR/combined.json" "decode_tok_per_sec_median")
if [[ "$(is_qwen35_model && echo 1 || echo 0)" == "1" ]]; then
    run_prefill_profile "combined-qwen35-prefill" "$COMBINED_STR"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Results Summary"
echo "─────────────────────────────────────────────────────────────"
echo "  Baseline:  prefill=${BASELINE_PREFILL} tok/s  decode=${BASELINE_DECODE} tok/s"
echo "  Tuned:     prefill=${COMBINED_PREFILL} tok/s  decode=${COMBINED_DECODE} tok/s"

PREFILL_DELTA=$(python3 -c "
b, t = float('$BASELINE_PREFILL'), float('$COMBINED_PREFILL')
print(f'+{(t/b-1)*100:.1f}%' if t > b else f'{(t/b-1)*100:.1f}%')
" 2>/dev/null || echo "?")
DECODE_DELTA=$(python3 -c "
b, t = float('$BASELINE_DECODE'), float('$COMBINED_DECODE')
print(f'+{(t/b-1)*100:.1f}%' if t > b else f'{(t/b-1)*100:.1f}%')
" 2>/dev/null || echo "?")
echo "  Delta:     prefill=${PREFILL_DELTA}  decode=${DECODE_DELTA}"
if [[ "$(is_qwen35_model && echo 1 || echo 0)" == "1" && -f "$OUTDIR/combined-qwen35-prefill.json" ]]; then
    QWEN35_HANDOFF=$(get_metric "$OUTDIR/combined-qwen35-prefill.json" "recurrent_qkv_handoff_layers")
    QWEN35_FAST=$(get_metric "$OUTDIR/combined-qwen35-prefill.json" "recurrent_qkv_fast_path_eligible_layers")
    QWEN35_BATCHFAST=$(get_metric "$OUTDIR/combined-qwen35-prefill.json" "recurrent_batch_qkv_handoff_ms")
    echo "  Qwen3.5:   handoff=${QWEN35_HANDOFF}  fast=${QWEN35_FAST}  batchfast=${QWEN35_BATCHFAST}ms"
    if python3 -c "import sys; sys.exit(0 if float('${QWEN35_HANDOFF}') == 0 else 1)"; then
        echo "  WARNING:   qkv-handoff did not activate; this tune may not help the current Qwen3.5 recurrent fast path"
    fi
fi
echo "═══════════════════════════════════════════════════════════════"

# ─── Phase 5: Generate profile JSON ────────────────────────────────────────
echo ""
echo "Phase 5: Generate kernel profile"
echo "─────────────────────────────────"

PROFILE_PATH="${OUTDIR}/tuned-profile.json"

# Generate profile from the combined env vars
python3 - "$OUTDIR" "$MODEL_BASENAME" "$COMBINED_STR" <<'PYEOF'
import json, os, shlex, sys
from datetime import date

outdir = sys.argv[1]
model_name = sys.argv[2]
combined_env = sys.argv[3].strip()

profile = {}

profile["model"] = model_name
profile["source"] = f"autotune-{date.today().isoformat()}"
profile["generated"] = date.today().isoformat()

ENV_TO_PROFILE_PATH = {
    "AX_METAL_MATVEC_Q4K_TG": ("decode_matvec", "q4_k", "threadgroup_size"),
    "AX_METAL_MATVEC_Q4K_NR": ("decode_matvec", "q4_k", "rows_per_simdgroup"),
    "AX_METAL_MATVEC_Q5K_TG": ("decode_matvec", "q5_k", "threadgroup_size"),
    "AX_METAL_MATVEC_Q5K_NR": ("decode_matvec", "q5_k", "rows_per_simdgroup"),
    "AX_METAL_MATVEC_Q6K_TG": ("decode_matvec", "q6_k", "threadgroup_size"),
    "AX_METAL_MATVEC_Q6K_NR": ("decode_matvec", "q6_k", "rows_per_simdgroup"),
    "AX_METAL_MATVEC_Q80_TG": ("decode_matvec", "q8_0", "threadgroup_size"),
    "AX_METAL_MATVEC_Q80_NR": ("decode_matvec", "q8_0", "rows_per_simdgroup"),
    "AX_METAL_BATCH_F16_IO": ("batch_prefill", "prefer_f16_io"),
    "AX_METAL_BATCH_F16_PAIR": ("batch_prefill", "prefer_pair_kernel"),
    "AX_METAL_F16IN_BK32": ("batch_prefill", "use_bk32"),
    "AX_METAL_F16IN_BN32": ("batch_prefill", "use_bn32"),
    "AX_METAL_DECODE_SPLITK_CHUNK_SIZE": ("attention_decode", "splitk_chunk_size"),
    "AX_METAL_DECODE_HD128_N2": ("attention_decode", "hd128_n2_default"),
    "AX_METAL_DECODE_SDPA": ("attention_decode", "sdpa_default"),
    "AX_METAL_PREFILL_FA2_MODE": ("attention_prefill", "fa2_mode"),
    "AX_METAL_PREFILL_FA2_HD128_MODE": ("attention_prefill", "fa2_hd128_mode"),
}


def coerce_value(raw: str):
    lower = raw.lower()
    if lower in {"0", "1"}:
        return lower == "1"
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"off", "on", "auto"}:
        return lower
    try:
        return int(raw)
    except ValueError:
        return raw


def set_path(tree, path, value):
    node = tree
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value


def prune_empty(node):
    if isinstance(node, dict):
        keys = list(node.keys())
        for key in keys:
            prune_empty(node[key])
            if isinstance(node[key], dict) and not node[key]:
                del node[key]


representable = []
env_only = []
for token in shlex.split(combined_env):
    if "=" not in token:
        continue
    key, value = token.split("=", 1)
    coerced = coerce_value(value)
    if key in ENV_TO_PROFILE_PATH:
        set_path(profile, ENV_TO_PROFILE_PATH[key], coerced)
        representable.append({"key": key, "value": coerced})
    else:
        env_only.append({"key": key, "value": value})

prune_empty(profile)

# Write a summary of all runs
all_results = []
for fname in sorted(os.listdir(outdir)):
    if fname.endswith('.json') and fname != 'tuned-profile.json':
        fpath = os.path.join(outdir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
            all_results.append({
                "label": fname.replace('.json', ''),
                "prefill_tok_s": data.get("prefill_tok_per_sec_median", 0),
                "decode_tok_s": data.get("decode_tok_per_sec_median", 0),
            })
        except (json.JSONDecodeError, KeyError):
            continue

# Sort by prefill throughput
all_results.sort(key=lambda x: x["prefill_tok_s"], reverse=True)

summary_path = os.path.join(outdir, "summary.json")
with open(summary_path, 'w') as f:
    json.dump(
        {
            "runs": all_results,
            "representable_overrides": representable,
            "env_only_overrides": env_only,
        },
        f,
        indent=2,
    )

# Write profile
profile_path = os.path.join(outdir, "tuned-profile.json")
with open(profile_path, 'w') as f:
    json.dump(profile, f, indent=2)

# Write env file for easy sourcing
env_path = os.path.join(outdir, "tuned.env")
with open(env_path, 'w') as f:
    for token in shlex.split(combined_env):
        if "=" in token:
            f.write(f"export {token}\n")

print(f"Summary:  {summary_path}")
print(f"Profile:  {profile_path}")
print(f"Env:      {env_path}")
PYEOF

# ─── Phase 6: Multi-context sweep (optional) ──────────────────────────────
echo ""
echo "Phase 6: Multi-context validation"
echo "──────────────────────────────────"
echo "  Running tuned config at multiple context lengths..."

for ptok in 64 256 512 1024; do
    label="validate-p${ptok}"
    PROMPT_TOKENS_ORIG=$PROMPT_TOKENS
    PROMPT_TOKENS=$ptok
    run_bench "$label" "$COMBINED_STR"
    PROMPT_TOKENS=$PROMPT_TOKENS_ORIG
done

# Generate final summary table
python3 - "$OUTDIR" <<'PYEOF'
import json, os, sys

outdir = sys.argv[1]

print("\n╔═══════════════════════════════════════════════════════════╗")
print("║  Multi-context validation results                       ║")
print("╠══════════╦══════════════════╦═══════════════════════════╣")
print("║ Prompt   ║ Prefill (tok/s)  ║ Decode (tok/s)            ║")
print("╠══════════╬══════════════════╬═══════════════════════════╣")

for ptok in [64, 256, 512, 1024]:
    fpath = os.path.join(outdir, f"validate-p{ptok}.json")
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        prefill = d.get("prefill_tok_per_sec_median", 0)
        decode = d.get("decode_tok_per_sec_median", 0)
        print(f"║ {ptok:>6}   ║ {prefill:>14.1f}   ║ {decode:>14.1f}              ║")

print("╚══════════╩══════════════════╩═══════════════════════════╝")
PYEOF

echo ""
echo "All results saved to: $OUTDIR/"
echo "Profile override: $OUTDIR/tuned-profile.json"
echo "Env override:     $OUTDIR/tuned.env"
if [[ "$APPLY" == "true" ]]; then
    cp "$OUTDIR/tuned-profile.json" "perfs/${MODEL_BASENAME}.json"
    echo "Applied! Profile written to perfs/${MODEL_BASENAME}.json"
fi
