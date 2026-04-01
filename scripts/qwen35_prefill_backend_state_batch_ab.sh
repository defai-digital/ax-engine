#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
AX_BENCH="${AX_BENCH:-$REPO_DIR/target/release/ax-engine-bench}"
MODEL="${MODEL:-$REPO_DIR/models/Qwen3.5-9B-Q4_K_M.gguf}"
PROMPT_TOKENS="${PROMPT_TOKENS:-64}"
WARMUP_ITERS="${WARMUP_ITERS:-0}"
COOLDOWN_S="${COOLDOWN_S:-1}"
SAMPLES="${SAMPLES:-3}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/automatosx/tmp}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)-$$}"
STATE_MODE="${STATE_MODE:-backend-owned}"

if [[ ! -f "$MODEL" ]]; then
  if [[ -f "$REPO_DIR/$MODEL" ]]; then
    MODEL="$REPO_DIR/$MODEL"
  else
    echo "error: missing model file: $MODEL" >&2
    exit 1
  fi
fi

mkdir -p "$OUT_DIR"
RUN_DIR="$OUT_DIR/qwen35-prefill-backend-state-batch-ab-$TIMESTAMP"
mkdir -p "$RUN_DIR"
TSV_OUT="$RUN_DIR/summary.tsv"
MD_OUT="$RUN_DIR/summary.md"

run_ax_bench() {
  if [[ -x "$AX_BENCH" ]]; then
    "$AX_BENCH" "$@"
    return
  fi

  echo "error: benchmark binary not found or not executable: $AX_BENCH" >&2
  echo "build it with: cargo build --release -p ax-engine-bench" >&2
  exit 1
}

run_sample() {
  local mode=$1
  local sample_idx=$2
  local json_out="$RUN_DIR/${mode}-sample${sample_idx}.json"
  local log_out="$RUN_DIR/${mode}-sample${sample_idx}.log"
  if [[ "$mode" == "force" ]]; then
    run_ax_bench prefill-profile \
      --model "$MODEL" \
      --prompt-tokens "$PROMPT_TOKENS" \
      --warmup-iters "$WARMUP_ITERS" \
      --qwen35-recurrent-state-mode "$STATE_MODE" \
      --qwen35-force-backend-state-batch \
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
            str(audit.get("effective_force_backend_state_batch", False)).lower(),
            str(audit.get("recurrent_handoff_observed_state_path", "")),
            str(audit.get("recurrent_handoff_observed_state_owner", "")),
            str(audit.get("recurrent_state_batch_kind", "")),
            str(data.get("recurrent_state_batch_backend_native_layers", 0)),
            str(data.get("recurrent_state_batch_cpu_direct_layers", 0)),
            str(data.get("recurrent_state_batch_cpu_gathered_layers", 0)),
            str(data.get("recurrent_state_batch_cpu_gathered_materialized_from_backend_layers", 0)),
        ]
    )
)
PY
}

printf "backend_state_batch_mode\tsamples\tprefill_tok_per_s\tbatchfast_ms\tqkv_handoff_layers\teffective_force_backend_state_batch\tobserved_path\tobserved_owner\tbatch_kind\tbackend_native_layers\tcpu_direct_layers\tcpu_gathered_layers\tcpu_gathered_from_backend_layers\n" >"$TSV_OUT"

for mode in off force; do
  echo "--- qwen35 backend-state-batch A/B state_mode=$STATE_MODE mode=$mode ---" >&2
  RAW_TSV="$RUN_DIR/${mode}-raw.tsv"
  : >"$RAW_TSV"
  for sample_idx in $(seq 1 "$SAMPLES"); do
    IFS=$'\t' read -r tok_s batchfast qkv_layers effective_force observed_path observed_owner batch_kind backend_native_layers cpu_direct_layers cpu_gathered_layers cpu_gathered_from_backend_layers < <(
      run_sample "$mode" "$sample_idx"
    )
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$tok_s" "$batchfast" "$qkv_layers" "$effective_force" "$observed_path" "$observed_owner" "$batch_kind" "$backend_native_layers" "$cpu_direct_layers" "$cpu_gathered_layers" "$cpu_gathered_from_backend_layers" \
      >>"$RAW_TSV"
    if [[ "$sample_idx" -lt "$SAMPLES" ]]; then
      sleep "$COOLDOWN_S"
    fi
  done
  python3 - "$RAW_TSV" "$mode" "$SAMPLES" >>"$TSV_OUT" <<'PY'
import statistics
import sys
from collections import Counter

rows = [line.strip().split("\t") for line in open(sys.argv[1], "r", encoding="utf-8") if line.strip()]
mode = sys.argv[2]
samples = sys.argv[3]

print(
    "\t".join(
        [
            mode,
            samples,
            str(statistics.median(float(row[0]) for row in rows)),
            str(statistics.median(float(row[1]) for row in rows)),
            str(int(round(statistics.median(float(row[2]) for row in rows)))),
            Counter(row[3] for row in rows).most_common(1)[0][0],
            Counter(row[4] for row in rows).most_common(1)[0][0],
            Counter(row[5] for row in rows).most_common(1)[0][0],
            Counter(row[6] for row in rows).most_common(1)[0][0],
            str(int(round(statistics.median(float(row[7]) for row in rows)))),
            str(int(round(statistics.median(float(row[8]) for row in rows)))),
            str(int(round(statistics.median(float(row[9]) for row in rows)))),
            str(int(round(statistics.median(float(row[10]) for row in rows)))),
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
    out.write("# Qwen3.5 Prefill Backend-State-Batch A/B\n\n")
    out.write("| Mode | Samples | Prefill tok/s | BatchFast ms | Handoff layers | Effective force | Observed path | Observed owner | Batch kind | Backend-native | CPU direct | CPU gathered | CPU gathered from backend |\n")
    out.write("|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|\n")
    for row in rows:
        out.write(
            f"| {row['backend_state_batch_mode']} | {row['samples']} | {float(row['prefill_tok_per_s']):.1f} | "
            f"{float(row['batchfast_ms']):.1f} | {row['qkv_handoff_layers']} | {row['effective_force_backend_state_batch']} | "
            f"{row['observed_path']} | {row['observed_owner']} | {row['batch_kind']} | {row['backend_native_layers']} | "
            f"{row['cpu_direct_layers']} | {row['cpu_gathered_layers']} | {row['cpu_gathered_from_backend_layers']} |\n"
        )
PY

echo "TSV: $TSV_OUT"
echo "MD:  $MD_OUT"
