#!/usr/bin/env bash
# Scratch ablation: pure-MTP accept rate on the worst row under different
# acceptance-mode / draft-sampler env configs. NOT committed.
set -u
cd /Users/akiralam/code/ax-engine_v5
OUT=/tmp/ax-ablation
rm -rf "$OUT"; mkdir -p "$OUT"
MODEL=27b-4bit
SUITE=python_modules_long
COMMON=(--engines ax --modes mtp --models "$MODEL" --suites "$SUITE"
        --max-tokens 256 --repetitions 2 --cooldown 5 --inter-case-cooldown 3
        --no-build-ax-engine)

run_cfg () {
  local name="$1"; shift
  echo "############## CONFIG: $name ##############"
  env "$@" python3 scripts/bench_qwen36_mtp_fair.py "${COMMON[@]}" \
      --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  python3 - "$OUT/$name/summary.json" "$name" <<'PY'
import json,sys
p,name=sys.argv[1],sys.argv[2]
try:
    d=json.load(open(p))
    r=d["rows"][0]["engines"]["ax_engine"]
    ar=r.get("accept_rate"); tok=r.get("decode_tok_s")
    print(f"RESULT {name}: accept_rate={ar*100:.2f}%  decode={tok:.1f} tok/s  status={r.get('status')}")
except Exception as e:
    print(f"RESULT {name}: PARSE-ERROR {e}")
PY
}

run_cfg baseline
run_cfg rejection            AX_MLX_MTP_MODEL_ACCEPTANCE_MODE=rejection
run_cfg rejection_dt06       AX_MLX_MTP_MODEL_ACCEPTANCE_MODE=rejection AX_MLX_MTP_DRAFT_TEMPERATURE=0.6
run_cfg rejection_dt06_stoch AX_MLX_MTP_MODEL_ACCEPTANCE_MODE=rejection AX_MLX_MTP_DRAFT_TEMPERATURE=0.6 AX_MLX_MTP_DRAFT_MODE=stochastic
echo "==================== SUMMARY ===================="
grep -h '^RESULT' "$OUT"/*.log 2>/dev/null || true
grep -rh '^RESULT' "$OUT" 2>/dev/null || true
echo "ABLATION_DONE"
