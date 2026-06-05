#!/usr/bin/env bash
# Scratch: run pure-MTP rows under a given AX_MLX_MTP_DRAFT_MIN_CONFIDENCE and
# report accept rate + decode tok/s per row. NOT committed.
# Usage: _gate_run.sh <conf> <tag> <max_tokens> <reps> "<model:suite> [model:suite ...]"
set -u
cd /Users/akiralam/code/ax-engine_v5
CONF="$1"; TAG="$2"; MAXTOK="$3"; REPS="$4"; PAIRS="$5"
OUT="/tmp/ax-gate/$TAG"
rm -rf "$OUT"; mkdir -p "$OUT"
for pair in $PAIRS; do
  MODEL="${pair%%:*}"; SUITE="${pair##*:}"
  D="$OUT/${MODEL}_${SUITE}"
  env AX_MLX_MTP_DRAFT_MIN_CONFIDENCE="$CONF" \
    python3 scripts/bench_qwen36_mtp_fair.py \
      --engines ax --modes mtp --models "$MODEL" --suites "$SUITE" \
      --max-tokens "$MAXTOK" --repetitions "$REPS" --cooldown 5 --inter-case-cooldown 3 \
      --no-build-ax-engine --output-dir "$D" > "$D.log" 2>&1
  python3 - "$D/summary.json" "$CONF" "$MODEL" "$SUITE" <<'PY'
import json,sys
p,conf,model,suite=sys.argv[1:5]
try:
    d=json.load(open(p)); r=d["rows"][0]["engines"]["ax_engine"]
    ar=r.get("accept_rate"); tok=r.get("decode_tok_s")
    flag="" if (ar is not None and ar>=0.99) else "  <99"
    print(f"GATE conf={conf} {model}/{suite}: accept={ar*100:.2f}%  decode={tok:.1f} tok/s{flag}")
except Exception as e:
    print(f"GATE conf={conf} {model}/{suite}: PARSE-ERROR {e}")
PY
done
echo "GATE_DONE tag=$TAG"
