#!/usr/bin/env bash
# Scratch: verify pure-MTP accept + decode across ALL six model/suite rows
# under the temp-1.0 confidence gate (greedy draft via draft temp 0).
# NOT committed.
# Usage: _verify_all_rows.sh <min_confidence> <max_tokens> <reps>
set -u
cd /Users/akiralam/code/ax-engine_v5
CONF="$1"; MAXTOK="${2:-384}"; REPS="${3:-3}"
OUT="/tmp/ax-verify/conf_${CONF}"
rm -rf "$OUT"; mkdir -p "$OUT"
PAIRS="27b-4bit:flappy 27b-4bit:long_code 27b-4bit:python_modules_long 35b-a3b-4bit:flappy 35b-a3b-4bit:long_code 35b-a3b-4bit:python_modules_long"
worst=100
for pair in $PAIRS; do
  MODEL="${pair%%:*}"; SUITE="${pair##*:}"
  D="$OUT/${MODEL}_${SUITE}"
  env AX_MLX_MTP_DRAFT_TEMPERATURE=0 AX_MLX_MTP_DRAFT_MIN_CONFIDENCE="$CONF" \
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
    flag="" if (ar and ar>=0.99) else "  <<< BELOW 99"
    print(f"VERIFY conf={conf} {model}/{suite}: accept={ar*100:.2f}%  decode={tok:.1f} tok/s{flag}")
except Exception as e:
    print(f"VERIFY conf={conf} {model}/{suite}: PARSE-ERROR {e}")
PY
done
echo "VERIFY_DONE conf=$CONF"
