#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/Users/akiralam/code/ax-engine-v2"
AX_BENCH="${AX_BENCH:-$REPO_DIR/target/release/ax-bench}"
LLAMA_BENCH="${LLAMA_BENCH:-/opt/homebrew/bin/llama-bench}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/automatosx/tmp}"

DECODE_TOKENS="${DECODE_TOKENS:-128}"
WARMUP_ITERS="${WARMUP_ITERS:-3}"
MEASURE_ITERS="${MEASURE_ITERS:-5}"
LLAMA_REPS="${LLAMA_REPS:-5}"
COOLDOWN_S="${COOLDOWN_S:-60}"
PROMPTS=(${PROMPTS:-64 256 512})

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
TSV_OUT="$OUT_DIR/phase3-cross-validate-defaults-$TIMESTAMP.tsv"
MD_OUT="$OUT_DIR/phase3-cross-validate-defaults-$TIMESTAMP.md"

MODELS=(
  "qwen3-8b|$REPO_DIR/models/Qwen3-8B-Q4_K_M.gguf|perfs/qwen3-8b.json"
  "gemma3-4b|$REPO_DIR/models/gemma-3-4b-it-Q4_K_M.gguf|perfs/gemma3-4b.json"
  "llama3-8b|$REPO_DIR/models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf|perfs/llama3-8b.json"
)

MODEL_FILTER="${MODEL_FILTER:-}"

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_exec() {
  [[ -x "$1" ]] || die "missing executable: $1"
}

cooldown() {
  echo "cooldown: ${COOLDOWN_S}s" >&2
  sleep "$COOLDOWN_S"
}

extract_ax_profile() {
  local log_path=$1
  sed -n 's/.*Loaded kernel profile path=\([^ ]*\).*/\1/p' "$log_path" | tail -1
}

extract_ax_prefill() {
  local log_path=$1
  sed -n 's/^Prefill:.* median \([0-9][0-9.]*\) tok\/s.*/\1/p' "$log_path" | tail -1
}

extract_ax_decode() {
  local log_path=$1
  sed -n 's/^Decode:.* median \([0-9][0-9.]*\) tok\/s.*/\1/p' "$log_path" | tail -1
}

extract_llama_prefill() {
  local log_path=$1
  local prompt_tokens=$2
  python3 - "$log_path" "$prompt_tokens" <<'PY'
import json
import sys

log_path = sys.argv[1]
prompt_tokens = int(sys.argv[2])

text = open(log_path, "r", encoding="utf-8").read()
start = text.find("[")
if start == -1:
    raise SystemExit(f"no json payload in {log_path}")

rows = json.loads(text[start:])
for row in rows:
    if (
        row.get("n_prompt") == prompt_tokens
        and row.get("n_gen") == 0
        and row.get("avg_ts") is not None
    ):
        print(row["avg_ts"])
        raise SystemExit(0)

raise SystemExit(1)
PY
}

extract_llama_decode() {
  local log_path=$1
  local decode_tokens=$2
  python3 - "$log_path" "$decode_tokens" <<'PY'
import json
import sys

log_path = sys.argv[1]
decode_tokens = int(sys.argv[2])

text = open(log_path, "r", encoding="utf-8").read()
start = text.find("[")
if start == -1:
    raise SystemExit(f"no json payload in {log_path}")

rows = json.loads(text[start:])
for row in rows:
    if (
        row.get("n_prompt") == 0
        and row.get("n_gen") == decode_tokens
        and row.get("avg_ts") is not None
    ):
        print(row["avg_ts"])
        raise SystemExit(0)

raise SystemExit(1)
PY
}

run_ax() {
  local model_path=$1
  local prompt_tokens=$2
  local log_path=$3

  "$AX_BENCH" bench \
    --model "$model_path" \
    --prompt-tokens "$prompt_tokens" \
    --decode-tokens "$DECODE_TOKENS" \
    --warmup-iters "$WARMUP_ITERS" \
    --measure-iters "$MEASURE_ITERS" \
    >"$log_path" 2>&1
}

run_llama_prefill() {
  local model_path=$1
  local prompt_tokens=$2

  "$LLAMA_BENCH" \
    -m "$model_path" \
    -p "$prompt_tokens" \
    -n 0 \
    -d 0 \
    -r "$LLAMA_REPS" \
    -ngl 99 \
    -b 2048 \
    -ub 512 \
    -fa 1 \
    -t 12 \
    -o json
}

run_llama_decode() {
  local model_path=$1
  local prompt_tokens=$2

  "$LLAMA_BENCH" \
    -m "$model_path" \
    -p 0 \
    -n "$DECODE_TOKENS" \
    -d "$prompt_tokens" \
    -r "$LLAMA_REPS" \
    -ngl 99 \
    -b 2048 \
    -ub 512 \
    -fa 1 \
    -t 12 \
    -o json
}

mkdir -p "$OUT_DIR"
require_exec "$AX_BENCH"
require_exec "$LLAMA_BENCH"

printf "model_name\tprompt_tokens\tprofile_path\tax_prefill\tax_decode\tllama_prefill\tllama_decode\n" >"$TSV_OUT"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model_name model_path expected_profile <<<"$entry"
  if [[ -n "$MODEL_FILTER" && "$model_name" != "$MODEL_FILTER" ]]; then
    continue
  fi
  require_file "$model_path"

  for prompt_tokens in "${PROMPTS[@]}"; do
    ax_log="$OUT_DIR/phase3-${model_name}-p${prompt_tokens}-${TIMESTAMP}.log"

    echo "--- AX ${model_name} P=${prompt_tokens} ---" >&2
    run_ax "$model_path" "$prompt_tokens" "$ax_log"
    ax_profile="$(extract_ax_profile "$ax_log")"
    ax_prefill="$(extract_ax_prefill "$ax_log")"
    ax_decode="$(extract_ax_decode "$ax_log")"
    [[ -n "$ax_prefill" && -n "$ax_decode" ]] || die "failed to parse AX output for $model_name P=$prompt_tokens"
    [[ "$ax_profile" == "$expected_profile" ]] || die "unexpected AX profile for $model_name P=$prompt_tokens: got '$ax_profile', expected '$expected_profile'"
    cooldown

    echo "--- llama.cpp prefill ${model_name} P=${prompt_tokens} ---" >&2
    llama_prefill_log="$OUT_DIR/phase3-${model_name}-llama-prefill-${TIMESTAMP}.json"
    run_llama_prefill "$model_path" "$prompt_tokens" >"$llama_prefill_log"
    llama_prefill="$(extract_llama_prefill "$llama_prefill_log" "$prompt_tokens")"
    [[ -n "$llama_prefill" ]] || die "failed to parse llama prefill for $model_name P=$prompt_tokens"
    cooldown

    echo "--- llama.cpp decode ${model_name} D=${prompt_tokens} ---" >&2
    llama_decode_log="$OUT_DIR/phase3-${model_name}-llama-decode-${TIMESTAMP}.json"
    run_llama_decode "$model_path" "$prompt_tokens" >"$llama_decode_log"
    llama_decode="$(extract_llama_decode "$llama_decode_log" "$DECODE_TOKENS")"
    [[ -n "$llama_decode" ]] || die "failed to parse llama decode for $model_name P=$prompt_tokens"
    cooldown

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$model_name" \
      "$prompt_tokens" \
      "$ax_profile" \
      "$ax_prefill" \
      "$ax_decode" \
      "$llama_prefill" \
      "$llama_decode" >>"$TSV_OUT"
  done
done

python3 - "$TSV_OUT" "$MD_OUT" <<'PY'
import csv
import pathlib
import sys

tsv_path = pathlib.Path(sys.argv[1])
md_path = pathlib.Path(sys.argv[2])

rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))

with md_path.open("w") as out:
    out.write("# PRD-0004 Phase 3 — Cross-Validation of Shipped Defaults\n")
    out.write("**Method:** current `perfs/*.json` defaults vs local `llama.cpp` on the same machine.\n\n")
    out.write("| Model | P | AX prefill | llama.cpp prefill | AX/llama | AX decode | llama.cpp decode | AX/llama |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        ax_pp = float(row["ax_prefill"])
        ax_tg = float(row["ax_decode"])
        ll_pp = float(row["llama_prefill"])
        ll_tg = float(row["llama_decode"])
        out.write(
            f"| {row['model_name']} | {row['prompt_tokens']} | "
            f"{ax_pp:.1f} | {ll_pp:.1f} | {ax_pp / ll_pp * 100:.1f}% | "
            f"{ax_tg:.1f} | {ll_tg:.1f} | {ax_tg / ll_tg * 100:.1f}% |\n"
        )
PY

echo "TSV: $TSV_OUT"
echo "MD:  $MD_OUT"
