#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
AX_BENCH="${AX_BENCH:-$REPO_DIR/target/release/ax-engine-bench}"
MODEL="${MODEL:-$REPO_DIR/models/Qwen3.5-9B-Q4_K_M.gguf}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
WARMUP_ITERS="${WARMUP_ITERS:-1}"
COOLDOWN_S="${COOLDOWN_S:-20}"
SAMPLES="${SAMPLES:-3}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/automatosx/tmp}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)-$$}"

MODES=(
  "auto"
  "cpu-alias"
  "slot-buffer"
  "backend-owned"
)

if [[ ! -f "$MODEL" ]]; then
  if [[ -f "$REPO_DIR/$MODEL" ]]; then
    MODEL="$REPO_DIR/$MODEL"
  else
    echo "error: missing model file: $MODEL" >&2
    exit 1
  fi
fi

mkdir -p "$OUT_DIR"
RUN_DIR="$OUT_DIR/qwen35-prefill-dtype-ab-$TIMESTAMP"
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
  local suffix=${mode//-/_}
  local json_out="$RUN_DIR/${suffix}-sample${sample_idx}.json"
  local log_out="$RUN_DIR/${suffix}-sample${sample_idx}.log"

  run_ax_bench prefill-profile \
    --model "$MODEL" \
    --prompt-tokens "$PROMPT_TOKENS" \
    --warmup-iters "$WARMUP_ITERS" \
    --qwen35-recurrent-state-mode "$mode" \
    --json-output "$json_out" \
    >"$log_out" 2>&1

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
            str(data.get("recurrent_qkv_handoff_cpu_alias_layers", 0)),
            str(data.get("recurrent_qkv_handoff_slot_buffer_layers", 0)),
            str(data.get("recurrent_qkv_handoff_backend_carryover_layers", 0)),
            str(data.get("recurrent_qkv_handoff_backend_zero_init_layers", 0)),
            str(data.get("recurrent_qkv_handoff_cpu_materialization_layers", 0)),
            str(audit.get("recurrent_handoff_observed_state_path", "")),
            str(audit.get("recurrent_handoff_observed_state_owner", "")),
            str(audit.get("recurrent_state_batch_kind", "")),
        ]
    )
)
PY
}

printf "mode\tsamples\tprefill_tok_per_s\tbatchfast_ms\tqkv_handoff_layers\tcpu_alias_layers\tslot_buffer_layers\tbackend_carryover_layers\tbackend_zero_init_layers\tcpu_materialization_layers\tobserved_path\tobserved_owner\tbatch_kind\n" >"$TSV_OUT"

for mode in "${MODES[@]}"; do
  echo "--- qwen35 prefill dtype A/B mode=$mode ---" >&2
  RAW_TSV="$RUN_DIR/${mode//-/_}-raw.tsv"
  : >"$RAW_TSV"
  for sample_idx in $(seq 1 "$SAMPLES"); do
    IFS=$'\t' read -r tok_s batchfast qkv_layers cpu_alias_layers slot_buffer_layers backend_carryover_layers backend_zero_init_layers cpu_materialization_layers observed_path observed_owner batch_kind < <(
      run_sample "$mode" "$sample_idx"
    )
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$tok_s" "$batchfast" "$qkv_layers" "$cpu_alias_layers" "$slot_buffer_layers" "$backend_carryover_layers" "$backend_zero_init_layers" "$cpu_materialization_layers" "$observed_path" "$observed_owner" "$batch_kind" \
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

def median_col(idx):
    return statistics.median(float(row[idx]) for row in rows)

def mode_col(idx):
    return Counter(row[idx] for row in rows).most_common(1)[0][0]

print(
    "\t".join(
        [
            mode,
            samples,
            str(median_col(0)),
            str(median_col(1)),
            str(int(round(median_col(2)))),
            str(int(round(median_col(3)))),
            str(int(round(median_col(4)))),
            str(int(round(median_col(5)))),
            str(int(round(median_col(6)))),
            str(int(round(median_col(7)))),
            mode_col(8),
            mode_col(9),
            mode_col(10),
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
    out.write("# Qwen3.5 Prefill DType / State-Mode A/B\n\n")
    out.write("| Mode | Samples | Prefill tok/s | BatchFast ms | Handoff layers | CPU alias | Slot buffer | Carryover | Zero-init | Materialized | Observed path | Observed owner | Batch kind |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|\n")
    for row in rows:
        out.write(
            f"| {row['mode']} | {row['samples']} | {float(row['prefill_tok_per_s']):.1f} | "
            f"{float(row['batchfast_ms']):.1f} | {row['qkv_handoff_layers']} | "
            f"{row['cpu_alias_layers']} | {row['slot_buffer_layers']} | "
            f"{row['backend_carryover_layers']} | {row['backend_zero_init_layers']} | {row['cpu_materialization_layers']} | "
            f"{row['observed_path']} | {row['observed_owner']} | {row['batch_kind']} |\n"
        )
PY

echo "TSV: $TSV_OUT"
echo "MD:  $MD_OUT"
