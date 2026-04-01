#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLAMA_BENCH="${LLAMA_BENCH:-/opt/homebrew/bin/llama-bench}"
MODEL="${MODEL:-$REPO_DIR/models/Qwen3.5-9B-Q4_K_M.gguf}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
DECODE_DEPTH="${DECODE_DEPTH:-$PROMPT_TOKENS}"
SAMPLES="${SAMPLES:-5}"
COOLDOWN_S="${COOLDOWN_S:-20}"
THREADS="${THREADS:-12}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/automatosx/tmp}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)-$$}"

[[ -x "$LLAMA_BENCH" ]] || {
  echo "error: llama-bench not executable: $LLAMA_BENCH" >&2
  exit 1
}
[[ -f "$MODEL" ]] || {
  echo "error: missing model file: $MODEL" >&2
  exit 1
}

mkdir -p "$OUT_DIR"
RUN_DIR="$OUT_DIR/llama-serial-median-$TIMESTAMP"
mkdir -p "$RUN_DIR"

RAW_TSV="$RUN_DIR/raw.tsv"
SUMMARY_TSV="$RUN_DIR/summary.tsv"
SUMMARY_MD="$RUN_DIR/summary.md"
SUMMARY_JSON="$RUN_DIR/summary.json"

printf "phase\tsample\tavg_ts\n" >"$RAW_TSV"

run_one() {
  local phase=$1
  local sample_idx=$2
  local out_json="$RUN_DIR/${phase}-sample${sample_idx}.json"
  if [[ "$phase" == "prefill" ]]; then
    "$LLAMA_BENCH" \
      -m "$MODEL" \
      -p "$PROMPT_TOKENS" \
      -n 0 \
      -d 0 \
      -r 1 \
      --no-warmup \
      -ngl 99 \
      -b 2048 \
      -ub 512 \
      -ctk f16 \
      -ctv f16 \
      -fa 1 \
      -t "$THREADS" \
      -o json \
      >"$out_json"
    python3 - "$out_json" "$PROMPT_TOKENS" <<'PY'
import json, sys
text = open(sys.argv[1], "r", encoding="utf-8").read()
rows = json.loads(text[text.find("["):])
prompt = int(sys.argv[2])
for row in rows:
    if row.get("n_prompt") == prompt and row.get("n_gen") == 0:
        print(row["avg_ts"])
        raise SystemExit(0)
raise SystemExit(1)
PY
  else
    "$LLAMA_BENCH" \
      -m "$MODEL" \
      -p 0 \
      -n "$DECODE_TOKENS" \
      -d "$DECODE_DEPTH" \
      -r 1 \
      --no-warmup \
      -ngl 99 \
      -b 2048 \
      -ub 512 \
      -ctk f16 \
      -ctv f16 \
      -fa 1 \
      -t "$THREADS" \
      -o json \
      >"$out_json"
    python3 - "$out_json" "$DECODE_TOKENS" "$DECODE_DEPTH" <<'PY'
import json, sys
text = open(sys.argv[1], "r", encoding="utf-8").read()
rows = json.loads(text[text.find("["):])
n_gen = int(sys.argv[2])
depth = int(sys.argv[3])
for row in rows:
    if row.get("n_prompt") == 0 and row.get("n_gen") == n_gen and row.get("n_depth") == depth:
        print(row["avg_ts"])
        raise SystemExit(0)
raise SystemExit(1)
PY
  fi
}

for phase in prefill decode; do
  for sample_idx in $(seq 1 "$SAMPLES"); do
    echo "--- llama serial median phase=$phase sample=$sample_idx/$SAMPLES ---" >&2
    value="$(run_one "$phase" "$sample_idx")"
    printf "%s\t%s\t%s\n" "$phase" "$sample_idx" "$value" >>"$RAW_TSV"
    if [[ "$sample_idx" -lt "$SAMPLES" ]]; then
      sleep "$COOLDOWN_S"
    fi
  done
  sleep "$COOLDOWN_S"
done

python3 - "$RAW_TSV" "$SUMMARY_TSV" "$SUMMARY_MD" "$SUMMARY_JSON" "$MODEL" "$PROMPT_TOKENS" "$DECODE_TOKENS" "$DECODE_DEPTH" "$SAMPLES" "$COOLDOWN_S" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

raw_path = pathlib.Path(sys.argv[1])
summary_tsv = pathlib.Path(sys.argv[2])
summary_md = pathlib.Path(sys.argv[3])
summary_json = pathlib.Path(sys.argv[4])
model = sys.argv[5]
prompt_tokens = int(sys.argv[6])
decode_tokens = int(sys.argv[7])
decode_depth = int(sys.argv[8])
samples = int(sys.argv[9])
cooldown_s = int(sys.argv[10])

rows = list(csv.DictReader(raw_path.open(), delimiter="\t"))
prefill = [float(r["avg_ts"]) for r in rows if r["phase"] == "prefill"]
decode = [float(r["avg_ts"]) for r in rows if r["phase"] == "decode"]

payload = {
    "model": model,
    "prompt_tokens": prompt_tokens,
    "decode_tokens": decode_tokens,
    "decode_depth": decode_depth,
    "samples": samples,
    "cooldown_s": cooldown_s,
    "prefill_median_tok_per_s": statistics.median(prefill),
    "prefill_mean_tok_per_s": statistics.mean(prefill),
    "decode_median_tok_per_s": statistics.median(decode),
    "decode_mean_tok_per_s": statistics.mean(decode),
    "prefill_samples_tok_per_s": prefill,
    "decode_samples_tok_per_s": decode,
}

summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

with summary_tsv.open("w", encoding="utf-8") as out:
    out.write("model\tprompt_tokens\tdecode_tokens\tdecode_depth\tsamples\tcooldown_s\tprefill_median_tok_per_s\tprefill_mean_tok_per_s\tdecode_median_tok_per_s\tdecode_mean_tok_per_s\n")
    out.write(
        "\t".join(
            [
                model,
                str(prompt_tokens),
                str(decode_tokens),
                str(decode_depth),
                str(samples),
                str(cooldown_s),
                str(payload["prefill_median_tok_per_s"]),
                str(payload["prefill_mean_tok_per_s"]),
                str(payload["decode_median_tok_per_s"]),
                str(payload["decode_mean_tok_per_s"]),
            ]
        )
        + "\n"
    )

with summary_md.open("w", encoding="utf-8") as out:
    out.write("# llama.cpp Serial Median Benchmark\n\n")
    out.write(f"- Model: `{model}`\n")
    out.write(f"- Prompt: `{prompt_tokens}`\n")
    out.write(f"- Decode: `{decode_tokens}` @ depth `{decode_depth}`\n")
    out.write(f"- Samples: `{samples}`\n")
    out.write(f"- Cooldown: `{cooldown_s}s`\n\n")
    out.write("| Phase | Median tok/s | Mean tok/s |\n")
    out.write("|---|---:|---:|\n")
    out.write(f"| Prefill | {payload['prefill_median_tok_per_s']:.1f} | {payload['prefill_mean_tok_per_s']:.1f} |\n")
    out.write(f"| Decode | {payload['decode_median_tok_per_s']:.1f} | {payload['decode_mean_tok_per_s']:.1f} |\n")
PY

echo "TSV:  $SUMMARY_TSV"
echo "MD:   $SUMMARY_MD"
echo "JSON: $SUMMARY_JSON"
