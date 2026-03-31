#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/Users/akiralam/code/ax-engine}"
AX_BENCH="${AX_BENCH:-}"
MODEL="${MODEL:-$REPO_DIR/models/Qwen3.5-9B-Q4_K_M.gguf}"
PROMPT_TOKENS="${PROMPT_TOKENS:-64}"
WARMUP_ITERS="${WARMUP_ITERS:-0}"
COOLDOWN_S="${COOLDOWN_S:-1}"
SAMPLES="${SAMPLES:-3}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/automatosx/tmp}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)-$$}"
STATE_MODE="${STATE_MODE:-auto}"

if [[ ! -f "$MODEL" ]]; then
  echo "error: missing model file: $MODEL" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
RUN_DIR="$OUT_DIR/qwen35-prefill-same-kv-carryover-ab-$TIMESTAMP"
mkdir -p "$RUN_DIR"
TSV_OUT="$RUN_DIR/summary.tsv"
MD_OUT="$RUN_DIR/summary.md"

run_ax_bench() {
  if [[ -n "$AX_BENCH" && -x "$AX_BENCH" ]]; then
    "$AX_BENCH" "$@"
  else
    cargo run --release -p ax-engine-bench -- "$@"
  fi
}

run_sample() {
  local prewarm=$1
  local suffix=$2
  local sample_idx=$3
  local json_out="$RUN_DIR/${suffix}-sample${sample_idx}.json"
  local log_out="$RUN_DIR/${suffix}-sample${sample_idx}.log"
  if [[ "$prewarm" == "on" ]]; then
    run_ax_bench prefill-profile \
      --model "$MODEL" \
      --prompt-tokens "$PROMPT_TOKENS" \
      --warmup-iters "$WARMUP_ITERS" \
      --qwen35-recurrent-state-mode "$STATE_MODE" \
      --qwen35-prewarm-prefill-same-kv \
      --json-output "$json_out" \
      >"$log_out" 2>&1
  else
    run_ax_bench prefill-profile \
      --model "$MODEL" \
      --prompt-tokens "$PROMPT_TOKENS" \
      --warmup-iters "$WARMUP_ITERS" \
      --qwen35-recurrent-state-mode "$STATE_MODE" \
      --json-output "$json_out" \
      >"$log_out" 2>&1
  fi

  python3 - "$json_out" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
audit = data.get("qwen35_dtype_audit") or {}
print(
    "\t".join(
        [
            str(data.get("tok_per_sec", 0.0)),
            str(data.get("recurrent_batch_qkv_handoff_ms", 0.0)),
            str(data.get("recurrent_qkv_handoff_layers", 0)),
            str(data.get("recurrent_qkv_handoff_backend_carryover_layers", 0)),
            str(data.get("recurrent_qkv_handoff_backend_zero_init_layers", 0)),
            str(data.get("recurrent_qkv_handoff_cpu_materialization_layers", 0)),
            str(audit.get("effective_same_kv_prewarm", False)).lower(),
            str(audit.get("recurrent_handoff_observed_state_path", "")),
            str(audit.get("recurrent_handoff_observed_state_owner", "")),
            str(audit.get("recurrent_state_batch_kind", "")),
        ]
    )
)
PY
}

printf "same_kv_prewarm\tsamples\tprefill_tok_per_s\tbatchfast_ms\tqkv_handoff_layers\tbackend_carryover_layers\tbackend_zero_init_layers\tcpu_materialization_layers\teffective_same_kv_prewarm\tobserved_path\tobserved_owner\tbatch_kind\n" >"$TSV_OUT"

for prewarm in off on; do
  echo "--- qwen35 same-kv prewarm A/B state_mode=$STATE_MODE prewarm=$prewarm ---" >&2
  RAW_TSV="$RUN_DIR/${prewarm}-raw.tsv"
  : >"$RAW_TSV"
  for sample_idx in $(seq 1 "$SAMPLES"); do
    IFS=$'\t' read -r tok_s batchfast qkv_layers backend_carryover_layers backend_zero_init_layers cpu_materialization_layers effective_same_kv_prewarm observed_path observed_owner batch_kind < <(
      run_sample "$prewarm" "$prewarm" "$sample_idx"
    )
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$tok_s" "$batchfast" "$qkv_layers" "$backend_carryover_layers" "$backend_zero_init_layers" "$cpu_materialization_layers" "$effective_same_kv_prewarm" "$observed_path" "$observed_owner" "$batch_kind" \
      >>"$RAW_TSV"
    if [[ "$sample_idx" -lt "$SAMPLES" ]]; then
      sleep "$COOLDOWN_S"
    fi
  done
  python3 - "$RAW_TSV" "$prewarm" "$SAMPLES" >>"$TSV_OUT" <<'PY'
import statistics
import sys
from collections import Counter

rows = [line.strip().split("\t") for line in open(sys.argv[1], "r", encoding="utf-8") if line.strip()]
prewarm = sys.argv[2]
samples = sys.argv[3]

print(
    "\t".join(
        [
            prewarm,
            samples,
            str(statistics.median(float(row[0]) for row in rows)),
            str(statistics.median(float(row[1]) for row in rows)),
            str(int(round(statistics.median(float(row[2]) for row in rows)))),
            str(int(round(statistics.median(float(row[3]) for row in rows)))),
            str(int(round(statistics.median(float(row[4]) for row in rows)))),
            str(int(round(statistics.median(float(row[5]) for row in rows)))),
            Counter(row[6] for row in rows).most_common(1)[0][0],
            Counter(row[7] for row in rows).most_common(1)[0][0],
            Counter(row[8] for row in rows).most_common(1)[0][0],
            Counter(row[9] for row in rows).most_common(1)[0][0],
        ]
    )
)
PY
done

python3 - "$TSV_OUT" "$MD_OUT" <<'PY'
import csv
import pathlib
import sys

tsv_path = pathlib.Path(sys.argv[1])
md_path = pathlib.Path(sys.argv[2])
rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))

with md_path.open("w", encoding="utf-8") as out:
    out.write("# Qwen3.5 Prefill Same-KV Carryover A/B\n\n")
    out.write("| Same-KV prewarm | Samples | Prefill tok/s | BatchFast ms | Handoff layers | Carryover | Zero-init | Materialized | Effective prewarm | Observed path | Observed owner | Batch kind |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|\n")
    for row in rows:
        out.write(
            f"| {row['same_kv_prewarm']} | {row['samples']} | {float(row['prefill_tok_per_s']):.1f} | "
            f"{float(row['batchfast_ms']):.1f} | {row['qkv_handoff_layers']} | "
            f"{row['backend_carryover_layers']} | {row['backend_zero_init_layers']} | {row['cpu_materialization_layers']} | "
            f"{row['effective_same_kv_prewarm']} | {row['observed_path']} | {row['observed_owner']} | {row['batch_kind']} |\n"
        )
PY

echo "TSV: $TSV_OUT"
echo "MD:  $MD_OUT"
