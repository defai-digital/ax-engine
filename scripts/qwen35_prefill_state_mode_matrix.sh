#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/Users/akiralam/code/ax-engine}"
MODEL="${MODEL:-$REPO_DIR/models/Qwen3.5-9B-Q4_K_M.gguf}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-32 64 128}"
WARMUP_ITERS="${WARMUP_ITERS:-0}"
COOLDOWN_S="${COOLDOWN_S:-1}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/automatosx/tmp}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)-$$}"

mkdir -p "$OUT_DIR"
RUN_DIR="$OUT_DIR/qwen35-prefill-state-matrix-$TIMESTAMP"
mkdir -p "$RUN_DIR"
TSV_OUT="$RUN_DIR/summary.tsv"
MD_OUT="$RUN_DIR/summary.md"

printf "prompt_tokens\tbest_state_mode\tbest_effective_state_path\tbest_effective_state_owner\tbest_state_tok_per_s\tstate_delta_vs_auto_tok_per_s\tbest_scratch_mode\tbest_scratch_tok_per_s\tscratch_delta_vs_f32_tok_per_s\tbest_prime_mode\tbest_prime_tok_per_s\tslot_buffer_priming_on_minus_off_tok_per_s\tbest_same_kv_carryover_mode\tbest_same_kv_carryover_tok_per_s\tsame_kv_prewarm_on_minus_off_tok_per_s\tbest_backend_state_batch_mode\tbest_backend_state_batch_tok_per_s\tbackend_state_batch_force_on_minus_off_tok_per_s\treport_json\n" >"$TSV_OUT"

for prompt_tokens in $PROMPT_LENGTHS; do
  echo "--- qwen35 state matrix prompt=$prompt_tokens ---" >&2
  output=$(PROMPT_TOKENS="$prompt_tokens" WARMUP_ITERS="$WARMUP_ITERS" COOLDOWN_S="$COOLDOWN_S" "$REPO_DIR/scripts/qwen35_prefill_recovery_report.sh")
  report_json=$(printf "%s\n" "$output" | awk '/^JSON: /{print $2}' | tail -n1)
  python3 - "$prompt_tokens" "$report_json" >>"$TSV_OUT" <<'PY'
import json
import sys

prompt_tokens = sys.argv[1]
report_json = sys.argv[2]
d = json.load(open(report_json, "r", encoding="utf-8"))
print(
    "\t".join(
        [
            prompt_tokens,
            d["best_state_mode"],
            (
                next(
                    (
                        row.get("observed_path", "")
                        for row in d.get("state_modes", [])
                        if row.get("mode") == d["best_state_mode"]
                    ),
                    "",
                )
            ),
            (
                next(
                    (
                        row.get("observed_owner", "")
                        for row in d.get("state_modes", [])
                        if row.get("mode") == d["best_state_mode"]
                    ),
                    "",
                )
            ),
            str(d["best_state_tok_per_s"]),
            str(d["state_delta_vs_auto_tok_per_s"]),
            d["best_scratch_mode"],
            str(d["best_scratch_tok_per_s"]),
            str(d["scratch_delta_vs_f32_tok_per_s"]),
            d["best_prime_mode"],
            str(d["best_prime_tok_per_s"]),
            str(d["slot_buffer_priming_on_minus_off_tok_per_s"]),
            d["best_same_kv_carryover_mode"],
            str(d["best_same_kv_carryover_tok_per_s"]),
            str(d["same_kv_prewarm_on_minus_off_tok_per_s"]),
            d["best_backend_state_batch_mode"],
            str(d["best_backend_state_batch_tok_per_s"]),
            str(d["backend_state_batch_force_on_minus_off_tok_per_s"]),
            report_json,
        ]
    )
)
PY
done

python3 - "$TSV_OUT" "$MD_OUT" <<'PY'
import csv
import pathlib
import sys

rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
md_path = pathlib.Path(sys.argv[2])

all_cpu_alias_path = all(row["best_effective_state_path"] == "cpu_alias_only" for row in rows)
all_backend_owned = all(row["best_effective_state_owner"] == "backend_owned" for row in rows)
some_backend_owned = any(row["best_effective_state_owner"] == "backend_owned" for row in rows)
all_scratch_f32 = all(row["best_scratch_mode"] == "f32" for row in rows)
all_prime_nonpositive = all(
    float(row["slot_buffer_priming_on_minus_off_tok_per_s"]) <= 0.0 for row in rows
)
some_same_kv_carryover_win = any(
    float(row["same_kv_prewarm_on_minus_off_tok_per_s"]) > 5.0 for row in rows
)
all_backend_batch_negative = all(
    float(row["backend_state_batch_force_on_minus_off_tok_per_s"]) < 0.0 for row in rows
)

with md_path.open("w", encoding="utf-8") as out:
    out.write("# Qwen3.5 Prefill State-Mode Matrix\n\n")
    out.write("| Prompt tokens | Best state mode | Effective path | Effective owner | Best state tok/s | Delta vs auto | Best scratch mode | Best scratch tok/s | Delta vs f32 | Best prime mode | Best prime tok/s | Prime on-off delta | Best same-KV mode | Best same-KV tok/s | Same-KV on-off delta | Best backend-batch mode | Best backend-batch tok/s | Backend-batch force delta |\n")
    out.write("|---:|---|---|---|---:|---:|---|---:|---:|---|---:|---:|---|---:|---:|---|---:|---:|\n")
    for row in rows:
        out.write(
            f"| {row['prompt_tokens']} | {row['best_state_mode']} | {row['best_effective_state_path']} | {row['best_effective_state_owner']} | {float(row['best_state_tok_per_s']):.1f} | "
            f"{float(row['state_delta_vs_auto_tok_per_s']):+.1f} | {row['best_scratch_mode']} | "
            f"{float(row['best_scratch_tok_per_s']):.1f} | {float(row['scratch_delta_vs_f32_tok_per_s']):+.1f} | "
            f"{row['best_prime_mode']} | {float(row['best_prime_tok_per_s']):.1f} | {float(row['slot_buffer_priming_on_minus_off_tok_per_s']):+.1f} | "
            f"{row['best_same_kv_carryover_mode']} | {float(row['best_same_kv_carryover_tok_per_s']):.1f} | {float(row['same_kv_prewarm_on_minus_off_tok_per_s']):+.1f} | "
            f"{row['best_backend_state_batch_mode']} | {float(row['best_backend_state_batch_tok_per_s']):.1f} | {float(row['backend_state_batch_force_on_minus_off_tok_per_s']):+.1f} |\n"
        )
    out.write("\n## Recommendation\n\n")
    if all_backend_owned:
        out.write("- Current matrix supports `backend_owned` recurrent carryover as the default effective state owner across all tested prompt lengths.\n")
    elif all_cpu_alias_path:
        out.write("- Current matrix supports `cpu_alias_only` as the default effective state path across all tested prompt lengths.\n")
    else:
        out.write("- Current matrix shows mixed best effective state paths; keep `auto` heuristic prompt-aware until more data is collected.\n")
    if some_backend_owned:
        out.write("- Some prompt lengths already prefer true `backend_owned` carryover; continue shifting default lifecycle toward persistent backend ownership rather than adding more slot-buffer toggles.\n")
    out.write("- Use `Effective owner` to distinguish CPU-materialized wins from backend-owned or pre-synced slot-buffer wins.\n")
    if all_scratch_f32:
        out.write("- Current matrix supports keeping `alpha/beta` scratch storage at `f32` by default.\n")
    else:
        out.write("- Current matrix shows scratch mode variance; keep `alpha/beta` storage configurable for more sampling.\n")
    if all_prime_nonpositive:
        out.write("- Current matrix does not justify slot-buffer priming as a default-on optimization.\n")
    else:
        out.write("- Current matrix shows priming wins in some prompt lengths; keep slot-buffer priming as an explicit experiment until more stable data is collected.\n")
    if some_same_kv_carryover_win:
        out.write("- Same-KV carryover already shows material wins in part of the matrix; prioritize normal lifecycle promotion toward backend-owned carryover over more scratch tuning.\n")
    if all_backend_batch_negative:
        out.write("- Forced backend-native state batch is still consistently slower than model-side handoff in this matrix; Module B still needs GPU-QKV ingress and backend batch execution-shape work, not just ownership cleanup.\n")
    elif all(
        abs(float(row["backend_state_batch_force_on_minus_off_tok_per_s"])) <= 5.0
        for row in rows
    ):
        out.write("- Backend-native state batch is now near parity with model-side handoff across this matrix; next work should focus on making backend-native ingress/lifecycle default without regressing same-KV carryover wins.\n")
    else:
        out.write("- Backend-native state batch is becoming competitive in part of the matrix; next work should remove the explicit force toggle and converge normal lifecycle onto backend-native batch.\n")
PY

echo "TSV: $TSV_OUT"
echo "MD:  $MD_OUT"
