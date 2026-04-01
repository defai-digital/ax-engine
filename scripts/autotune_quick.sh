#!/usr/bin/env bash
# =============================================================================
# ax-engine quick autotuner — targeted sweep of highest-ROI knobs only
#
# Based on empirical findings from ax-engine optimization history:
#   - Decode: TG size × NR is the dominant factor
#   - Prefill: f16io, blocked layout, bk32 are the big movers
#   - Attention: splitk mode/threshold matters at depth
#
# ~30 runs instead of 100+. Takes ~5-10 min depending on model size.
#
# Usage:
#   ./scripts/autotune_quick.sh --model ./models/Qwen3.5-9B-Q4_K_M.gguf
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL=""
BENCH_BIN="$REPO_DIR/target/release/ax-engine-bench"
SAMPLES=3
COOLDOWN_MS=20000
WARMUP=1
MEASURE=2
PROFILE_PROMPT_TOKENS=512

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     MODEL="$2"; shift 2;;
        --samples)   SAMPLES="$2"; shift 2;;
        --bench-bin) BENCH_BIN="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 --model <path-to-gguf>"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    if [[ -f "$REPO_DIR/$MODEL" ]]; then
        MODEL="$REPO_DIR/$MODEL"
    else
        echo "Model not found: $MODEL"
        exit 1
    fi
fi

MODEL_BASENAME="$(basename "$MODEL" .gguf | tr '[:upper:]' '[:lower:]' | tr ' ' '-')"
DATE="$(date +%Y%m%d-%H%M%S)"
OUTDIR="$REPO_DIR/automatosx/tmp/quicktune-${MODEL_BASENAME}-${DATE}"
mkdir -p "$OUTDIR"

if [[ ! -x "$BENCH_BIN" ]]; then
    echo "Building..."
    pushd "$REPO_DIR" >/dev/null
    cargo build -p ax-engine-bench --release
    popd >/dev/null
fi

echo "═══════════════════════════════════════════"
echo " Quick Autotuner: $MODEL_BASENAME"
echo " Output: $OUTDIR/"
echo "═══════════════════════════════════════════"

# ─── helpers ────────────────────────────────────────────────────────────────

bench() {
    local label="$1" envs="$2" ptok="$3" dtok="$4"
    local json_path="${OUTDIR}/${label}.json"

    echo -n "  $label ... "
    if env $envs "$BENCH_BIN" bench \
        --model "$MODEL" \
        --prompt-tokens "$ptok" --decode-tokens "$dtok" \
        --deterministic --samples "$SAMPLES" --cooldown-ms "$COOLDOWN_MS" \
        --warmup-iters "$WARMUP" --measure-iters "$MEASURE" \
        --json-output "$json_path" 2>"${OUTDIR}/${label}.log"; then
        python3 -c "
import json
d=json.load(open('$json_path'))
print(f\"P={d['prefill_tok_per_sec_median']:.0f}  D={d['decode_tok_per_sec_median']:.1f}\")
" 2>/dev/null || echo "ok"
    else
        echo "FAIL"
    fi
}

metric() {
    python3 -c "import json; print(json.load(open('$1')).get('$2',0))" 2>/dev/null || echo 0
}

is_qwen35_model() {
    local lowered
    lowered="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
    [[ "$lowered" == *"qwen3.5"* || "$lowered" == *"qwen35"* ]]
}

prefill_profile() {
    local label="$1" envs="$2"
    local json_path="${OUTDIR}/${label}.json"

    echo -n "  $label ... "
    if env $envs "$BENCH_BIN" prefill-profile \
        --model "$MODEL" \
        --prompt-tokens "$PROFILE_PROMPT_TOKENS" \
        --json-output "$json_path" 2>"${OUTDIR}/${label}.log"; then
        python3 -c "
import json
d=json.load(open('$json_path'))
print(
    f\"P={d.get('tok_per_sec',0):.1f}  handoff={d.get('recurrent_qkv_handoff_layers',0)}  \"
    f\"fast={d.get('recurrent_qkv_fast_path_eligible_layers',0)}  \"
    f\"batchfast={d.get('recurrent_batch_qkv_handoff_ms',0):.1f}ms\"
)
" 2>/dev/null || echo "ok"
    else
        echo "FAIL"
    fi
}

qwen35_prefill_candidate_ok() {
    local label="$1" envs="$2"
    if ! is_qwen35_model; then
        return 0
    fi

    local profile_label="${label}-qwen35-prefill"
    prefill_profile "$profile_label" "$envs"
    local handoff
    handoff=$(metric "$OUTDIR/${profile_label}.json" "recurrent_qkv_handoff_layers")
    if python3 -c "import sys; sys.exit(0 if float('$handoff') > 0 else 1)"; then
        return 0
    fi

    echo "    ✗ rejected: qkv-handoff disabled"
    return 1
}

# ─── Sweep strategy: test at 2 context lengths ─────────────────────────────
# Short context (64 tok) → decode-dominated
# Medium context (512 tok) → prefill-dominated

echo ""
echo "▸ Baseline"
bench "base-p64"  "" 64  64
bench "base-p512" "" 512 64
if is_qwen35_model; then
    echo ""
    echo "▸ Qwen3.5 recurrent prefill baseline"
    prefill_profile "base-qwen35-prefill" ""
fi

BASE_D64=$(metric "$OUTDIR/base-p64.json" "decode_tok_per_sec_median")
BASE_P512=$(metric "$OUTDIR/base-p512.json" "prefill_tok_per_sec_median")

echo ""
echo "  Baseline: decode@64=${BASE_D64} tok/s  prefill@512=${BASE_P512} tok/s"

# ─── Decode matvec: TG × NR grid ───────────────────────────────────────────
echo ""
echo "▸ Decode matvec (Q4_K TG × NR)"

BEST_DECODE_LABEL="base-p64"
BEST_DECODE_VAL=$BASE_D64
BEST_DECODE_ENV=""

for tg in 64 128; do
    for nr in 1 2; do
        label="d-tg${tg}-nr${nr}"
        bench "$label" "AX_METAL_MATVEC_Q4K_TG=$tg AX_METAL_MATVEC_Q4K_NR=$nr" 64 64
        val=$(metric "$OUTDIR/${label}.json" "decode_tok_per_sec_median")
        better=$(python3 -c "print('y' if float('$val')>float('$BEST_DECODE_VAL')*1.005 else 'n')")
        if [[ "$better" == "y" ]]; then
            BEST_DECODE_LABEL=$label
            BEST_DECODE_VAL=$val
            BEST_DECODE_ENV="AX_METAL_MATVEC_Q4K_TG=$tg AX_METAL_MATVEC_Q4K_NR=$nr"
        fi
    done
done
echo "  Best decode: $BEST_DECODE_LABEL = $BEST_DECODE_VAL tok/s"

# ─── Decode: fused ops ─────────────────────────────────────────────────────
echo ""
echo "▸ Decode fused ops"

for combo in "1,1" "1,0" "0,1" "0,0"; do
    IFS=',' read -r silu pair <<< "$combo"
    label="d-silu${silu}-pair${pair}"
    bench "$label" "$BEST_DECODE_ENV AX_METAL_DECODE_FUSED_SILU_DOWN=$silu AX_METAL_DECODE_PAIR_MATVEC=$pair" 64 64
    val=$(metric "$OUTDIR/${label}.json" "decode_tok_per_sec_median")
    better=$(python3 -c "print('y' if float('$val')>float('$BEST_DECODE_VAL')*1.005 else 'n')")
    if [[ "$better" == "y" ]]; then
        BEST_DECODE_VAL=$val
        BEST_DECODE_ENV="$BEST_DECODE_ENV AX_METAL_DECODE_FUSED_SILU_DOWN=$silu AX_METAL_DECODE_PAIR_MATVEC=$pair"
        echo "    ✓ new best: $val tok/s"
    fi
done

# ─── Decode: attention at depth ────────────────────────────────────────────
echo ""
echo "▸ Decode attention (splitK at depth=512)"

for mode in off on; do
    for chunk in 128 256; do
        label="d-splitk-${mode}-c${chunk}"
        bench "$label" "$BEST_DECODE_ENV AX_METAL_DECODE_SPLITK_MODE=$mode AX_METAL_DECODE_SPLITK_CHUNK_SIZE=$chunk" 512 64
        val=$(metric "$OUTDIR/${label}.json" "decode_tok_per_sec_median")
        echo "    splitk=$mode chunk=$chunk → decode=$val tok/s"
    done
done

# ─── Prefill: batch kernel routing ─────────────────────────────────────────
echo ""
echo "▸ Prefill batch kernels"

BEST_PREFILL_VAL=$BASE_P512
BEST_PREFILL_ENV=""

# f16io on/off
for f16io in 0 1; do
    label="p-f16io-${f16io}"
    bench "$label" "AX_METAL_BATCH_F16_IO=$f16io" 512 0
    val=$(metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    better=$(python3 -c "print('y' if float('$val')>float('$BEST_PREFILL_VAL')*1.005 else 'n')")
    if [[ "$better" == "y" ]]; then
        if qwen35_prefill_candidate_ok "$label" "AX_METAL_BATCH_F16_IO=$f16io"; then
            BEST_PREFILL_VAL=$val
            BEST_PREFILL_ENV="AX_METAL_BATCH_F16_IO=$f16io"
            echo "    ✓ f16io=$f16io → $val tok/s"
        fi
    fi
done

# bk32 on/off (K-tile size)
for bk in 0 1; do
    label="p-bk32-${bk}"
    bench "$label" "$BEST_PREFILL_ENV AX_METAL_F16IN_BK32=$bk" 512 0
    val=$(metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    better=$(python3 -c "print('y' if float('$val')>float('$BEST_PREFILL_VAL')*1.005 else 'n')")
    if [[ "$better" == "y" ]]; then
        candidate_env="$BEST_PREFILL_ENV AX_METAL_F16IN_BK32=$bk"
        if qwen35_prefill_candidate_ok "$label" "$candidate_env"; then
            BEST_PREFILL_VAL=$val
            BEST_PREFILL_ENV="$candidate_env"
            echo "    ✓ bk32=$bk → $val tok/s"
        fi
    fi
done

# Fused QKV prefill
for fqkv in 0 1; do
    label="p-fqkv-${fqkv}"
    bench "$label" "$BEST_PREFILL_ENV AX_METAL_FUSED_QKV=$fqkv" 512 0
    val=$(metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    better=$(python3 -c "print('y' if float('$val')>float('$BEST_PREFILL_VAL')*1.005 else 'n')")
    if [[ "$better" == "y" ]]; then
        candidate_env="$BEST_PREFILL_ENV AX_METAL_FUSED_QKV=$fqkv"
        if qwen35_prefill_candidate_ok "$label" "$candidate_env"; then
            BEST_PREFILL_VAL=$val
            BEST_PREFILL_ENV="$candidate_env"
            echo "    ✓ fused_qkv=$fqkv → $val tok/s"
        fi
    fi
done

# F16 pair kernel
for fp in 0 1; do
    label="p-f16pair-${fp}"
    bench "$label" "$BEST_PREFILL_ENV AX_METAL_BATCH_F16_PAIR=$fp" 512 0
    val=$(metric "$OUTDIR/${label}.json" "prefill_tok_per_sec_median")
    better=$(python3 -c "print('y' if float('$val')>float('$BEST_PREFILL_VAL')*1.005 else 'n')")
    if [[ "$better" == "y" ]]; then
        candidate_env="$BEST_PREFILL_ENV AX_METAL_BATCH_F16_PAIR=$fp"
        if qwen35_prefill_candidate_ok "$label" "$candidate_env"; then
            BEST_PREFILL_VAL=$val
            BEST_PREFILL_ENV="$candidate_env"
            echo "    ✓ f16pair=$fp → $val tok/s"
        fi
    fi
done

echo ""
echo "  Best prefill: $BEST_PREFILL_VAL tok/s"

# ─── Final combined validation ─────────────────────────────────────────────
echo ""
echo "▸ Final validation (combined best decode + prefill)"

FINAL_ENV="$BEST_DECODE_ENV $BEST_PREFILL_ENV"

for ptok in 64 256 512 1024; do
    bench "final-p${ptok}" "$FINAL_ENV" $ptok 64
done

if is_qwen35_model; then
    echo ""
    echo "▸ Qwen3.5 recurrent prefill validation"
    prefill_profile "final-qwen35-prefill" "$FINAL_ENV"
fi

# ─── Summary ────────────────────────────────────────────────────────────────
python3 - "$OUTDIR" "$BASE_D64" "$BASE_P512" "$FINAL_ENV" "$(is_qwen35_model && echo 1 || echo 0)" <<'PYEOF'
import json, os, sys

outdir = sys.argv[1]
base_d = float(sys.argv[2])
base_p = float(sys.argv[3])
final_env = sys.argv[4]
is_qwen35 = sys.argv[5] == "1"

print("\n" + "═" * 65)
print(" AUTOTUNER RESULTS")
print("═" * 65)
print(f"\n  Winning env vars:")
for kv in final_env.strip().split():
    if kv:
        print(f"    export {kv}")

print(f"\n  {'Context':<10} {'Prefill':>12} {'Decode':>12}")
print(f"  {'─'*10} {'─'*12} {'─'*12}")

for ptok in [64, 256, 512, 1024]:
    fpath = os.path.join(outdir, f"final-p{ptok}.json")
    if os.path.exists(fpath):
        d = json.load(open(fpath))
        p = d.get("prefill_tok_per_sec_median", 0)
        dec = d.get("decode_tok_per_sec_median", 0)
        print(f"  {ptok:<10} {p:>10.0f}   {dec:>10.1f}")

print(f"\n  vs baseline: decode {base_d:.1f} tok/s → see above")
print(f"               prefill {base_p:.0f} tok/s → see above")

if is_qwen35:
    fpath = os.path.join(outdir, "final-qwen35-prefill.json")
    if os.path.exists(fpath):
        d = json.load(open(fpath))
        print("\n  Qwen3.5 recurrent prefill:")
        print(
            f"    tok/s={d.get('tok_per_sec',0):.1f}  "
            f"handoff={d.get('recurrent_qkv_handoff_layers',0)}  "
            f"fast={d.get('recurrent_qkv_fast_path_eligible_layers',0)}  "
            f"batchfast={d.get('recurrent_batch_qkv_handoff_ms',0):.1f}ms"
        )
        if d.get("recurrent_qkv_handoff_layers", 0) == 0:
            print("    WARNING: qkv-handoff did not activate; tune result may not help current Qwen3.5 fast path")

# Write env file for easy sourcing
env_path = os.path.join(outdir, "tuned.env")
with open(env_path, 'w') as f:
    for kv in final_env.strip().split():
        if kv and '=' in kv:
            f.write(f"export {kv}\n")
print(f"\n  Source the config:  source {env_path}")
print("═" * 65)
PYEOF

echo ""
echo "Done. All results in $OUTDIR/"
