#!/usr/bin/env bash
# Full gated pure-MTP run for README refresh. Pre-seeds MTPLX + AX MTP+n-gram
# artifacts from the 2026-06-05 run, then re-runs ONLY gated pure MTP (ax_engine)
# at 1000 tokens / 5 reps across all six rows. The gate is default-on in the
# rebuilt binary, so no env override is needed. NOT committed.
# Usage: _full_gated_mtp_run.sh <new_dir_name>
set -u
cd /Users/akiralam/code/ax-engine_v5
NEWNAME="${1:-2026-06-05-ax-mtp-accept-gate}"
PRIOR=benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh
NEW=benchmarks/results/mtp-fair/$NEWNAME
rm -rf "$NEW"; cp -r "$PRIOR" "$NEW"
# Keep mtplx.json + ax_engine_ngram.json (+ their prompts); drop pure-MTP outputs
# and top-level aggregates so only gated ax_engine regenerates.
find "$NEW" -name 'ax_engine.json' -delete
find "$NEW" -type d -name 'ax_engine-prompts' -exec rm -rf {} +
rm -f "$NEW"/summary.json "$NEW"/summary.md "$NEW"/*.svg "$NEW"/prefill-ttft-report.* "$NEW"/bench-run.log
echo "Pre-seeded $NEW (kept mtplx + ax_engine_ngram):"
find "$NEW" -mindepth 3 -maxdepth 3 -name '*.json' | sort
python3 scripts/bench_qwen36_mtp_fair.py \
  --engines mtplx ax --modes mtp mtp-ngram \
  --models 27b-4bit 35b-a3b-4bit \
  --suites flappy long_code python_modules_long \
  --max-tokens 1000 --repetitions 5 --cooldown 30 \
  --skip-existing --no-build-ax-engine \
  --output-dir "$NEW" > "$NEW/bench-run.log" 2>&1
echo "EXIT=$? full run done -> $NEW"
echo "=== gated pure-MTP accept (AX MTP column) ==="
python3 - "$NEW/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for r in d["rows"]:
    e=r["engines"].get("ax_engine",{})
    ar=e.get("accept_rate"); tok=e.get("decode_tok_s")
    flag="" if (ar and ar>=0.99) else "  <<< BELOW 99"
    arp=f"{ar*100:.2f}%" if ar is not None else "n/a"
    print(f"  {r['model']}/{r['suite']}: accept={arp} decode={tok:.1f} tok/s{flag}")
PY
echo "FULL_RUN_DONE"
