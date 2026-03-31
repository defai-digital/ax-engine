#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/Users/akiralam/code/ax-engine}"
MODEL="${MODEL:-$REPO_DIR/models/Qwen3.5-9B-Q4_K_M.gguf}"
PROMPT_TOKENS="${PROMPT_TOKENS:-64}"
WARMUP_ITERS="${WARMUP_ITERS:-0}"
COOLDOWN_S="${COOLDOWN_S:-1}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/automatosx/tmp}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)-$$}"

mkdir -p "$OUT_DIR"
RUN_DIR="$OUT_DIR/qwen35-prefill-recovery-report-$TIMESTAMP"
mkdir -p "$RUN_DIR"

STATE_TIMEST="${TIMESTAMP}-state"
SCRATCH_TIMEST="${TIMESTAMP}-scratch"
PRIME_TIMEST="${TIMESTAMP}-prime"
CARRY_TIMEST="${TIMESTAMP}-carry"
BACKEND_BATCH_TIMEST="${TIMESTAMP}-backend-batch"

STATE_OUTDIR="$OUT_DIR/qwen35-prefill-dtype-ab-$STATE_TIMEST"
SCRATCH_OUTDIR="$OUT_DIR/qwen35-prefill-scratch-ab-$SCRATCH_TIMEST"
PRIME_OUTDIR="$OUT_DIR/qwen35-prefill-slot-buffer-prime-ab-$PRIME_TIMEST"
CARRY_OUTDIR="$OUT_DIR/qwen35-prefill-same-kv-carryover-ab-$CARRY_TIMEST"
BACKEND_BATCH_OUTDIR="$OUT_DIR/qwen35-prefill-backend-state-batch-ab-$BACKEND_BATCH_TIMEST"

env \
  REPO_DIR="$REPO_DIR" \
  MODEL="$MODEL" \
  PROMPT_TOKENS="$PROMPT_TOKENS" \
  WARMUP_ITERS="$WARMUP_ITERS" \
  COOLDOWN_S="$COOLDOWN_S" \
  OUT_DIR="$OUT_DIR" \
  TIMESTAMP="$STATE_TIMEST" \
  "$REPO_DIR/scripts/qwen35_prefill_dtype_ab.sh"

env \
  REPO_DIR="$REPO_DIR" \
  MODEL="$MODEL" \
  PROMPT_TOKENS="$PROMPT_TOKENS" \
  WARMUP_ITERS="$WARMUP_ITERS" \
  COOLDOWN_S="$COOLDOWN_S" \
  STATE_MODE="auto" \
  OUT_DIR="$OUT_DIR" \
  TIMESTAMP="$SCRATCH_TIMEST" \
  "$REPO_DIR/scripts/qwen35_prefill_scratch_ab.sh"

env \
  REPO_DIR="$REPO_DIR" \
  MODEL="$MODEL" \
  PROMPT_TOKENS="$PROMPT_TOKENS" \
  WARMUP_ITERS="$WARMUP_ITERS" \
  COOLDOWN_S="$COOLDOWN_S" \
  OUT_DIR="$OUT_DIR" \
  TIMESTAMP="$PRIME_TIMEST" \
  "$REPO_DIR/scripts/qwen35_prefill_slot_buffer_prime_ab.sh"

env \
  REPO_DIR="$REPO_DIR" \
  MODEL="$MODEL" \
  PROMPT_TOKENS="$PROMPT_TOKENS" \
  WARMUP_ITERS="$WARMUP_ITERS" \
  COOLDOWN_S="$COOLDOWN_S" \
  STATE_MODE="auto" \
  OUT_DIR="$OUT_DIR" \
  TIMESTAMP="$CARRY_TIMEST" \
  "$REPO_DIR/scripts/qwen35_prefill_same_kv_carryover_ab.sh"

env \
  REPO_DIR="$REPO_DIR" \
  MODEL="$MODEL" \
  PROMPT_TOKENS="$PROMPT_TOKENS" \
  WARMUP_ITERS="$WARMUP_ITERS" \
  COOLDOWN_S="$COOLDOWN_S" \
  STATE_MODE="backend-owned" \
  OUT_DIR="$OUT_DIR" \
  TIMESTAMP="$BACKEND_BATCH_TIMEST" \
  "$REPO_DIR/scripts/qwen35_prefill_backend_state_batch_ab.sh"

REPORT_MD="$RUN_DIR/report.md"
REPORT_JSON="$RUN_DIR/report.json"

python3 - "$STATE_OUTDIR/summary.tsv" "$SCRATCH_OUTDIR/summary.tsv" "$PRIME_OUTDIR/summary.tsv" "$CARRY_OUTDIR/summary.tsv" "$BACKEND_BATCH_OUTDIR/summary.tsv" "$REPORT_MD" "$REPORT_JSON" <<'PY'
import csv
import json
import pathlib
import sys

state_path = pathlib.Path(sys.argv[1])
scratch_path = pathlib.Path(sys.argv[2])
prime_path = pathlib.Path(sys.argv[3])
carry_path = pathlib.Path(sys.argv[4])
backend_batch_path = pathlib.Path(sys.argv[5])
report_md = pathlib.Path(sys.argv[6])
report_json = pathlib.Path(sys.argv[7])

state_rows = list(csv.DictReader(state_path.open(), delimiter="\t"))
scratch_rows = list(csv.DictReader(scratch_path.open(), delimiter="\t"))
prime_rows = list(csv.DictReader(prime_path.open(), delimiter="\t"))
carry_rows = list(csv.DictReader(carry_path.open(), delimiter="\t"))
backend_batch_rows = list(csv.DictReader(backend_batch_path.open(), delimiter="\t"))

def f(x):
    return float(x)

state_best = max(state_rows, key=lambda r: f(r["prefill_tok_per_s"]))
scratch_best = max(scratch_rows, key=lambda r: f(r["prefill_tok_per_s"]))
prime_best = max(prime_rows, key=lambda r: f(r["prefill_tok_per_s"]))
carry_best = max(carry_rows, key=lambda r: f(r["prefill_tok_per_s"]))
backend_batch_best = max(backend_batch_rows, key=lambda r: f(r["prefill_tok_per_s"]))

state_baseline = next((r for r in state_rows if r["mode"] == "auto"), state_rows[0])
scratch_baseline = next((r for r in scratch_rows if r["alpha_beta_mode"] == "f32"), scratch_rows[0])
prime_baseline = next((r for r in prime_rows if r["prime_mode"] == "off"), prime_rows[0])
carry_baseline = next((r for r in carry_rows if r["same_kv_prewarm"] == "off"), carry_rows[0])
backend_batch_baseline = next(
    (r for r in backend_batch_rows if r["backend_state_batch_mode"] == "off"),
    backend_batch_rows[0],
)

payload = {
    "state_modes": state_rows,
    "scratch_modes": scratch_rows,
    "slot_buffer_priming_modes": prime_rows,
    "same_kv_carryover_modes": carry_rows,
    "backend_state_batch_modes": backend_batch_rows,
    "best_state_mode": state_best["mode"],
    "best_state_tok_per_s": f(state_best["prefill_tok_per_s"]),
    "best_scratch_mode": scratch_best["alpha_beta_mode"],
    "best_scratch_tok_per_s": f(scratch_best["prefill_tok_per_s"]),
    "best_prime_mode": prime_best["prime_mode"],
    "best_prime_tok_per_s": f(prime_best["prefill_tok_per_s"]),
    "best_same_kv_carryover_mode": carry_best["same_kv_prewarm"],
    "best_same_kv_carryover_tok_per_s": f(carry_best["prefill_tok_per_s"]),
    "best_backend_state_batch_mode": backend_batch_best["backend_state_batch_mode"],
    "best_backend_state_batch_tok_per_s": f(backend_batch_best["prefill_tok_per_s"]),
    "state_delta_vs_auto_tok_per_s": f(state_best["prefill_tok_per_s"]) - f(state_baseline["prefill_tok_per_s"]),
    "scratch_delta_vs_f32_tok_per_s": f(scratch_best["prefill_tok_per_s"]) - f(scratch_baseline["prefill_tok_per_s"]),
    "prime_delta_vs_unprimed_tok_per_s": f(prime_best["prefill_tok_per_s"]) - f(prime_baseline["prefill_tok_per_s"]),
    "same_kv_carryover_delta_vs_cold_tok_per_s": f(carry_best["prefill_tok_per_s"]) - f(carry_baseline["prefill_tok_per_s"]),
    "backend_state_batch_delta_vs_model_side_tok_per_s": (
        f(backend_batch_best["prefill_tok_per_s"]) - f(backend_batch_baseline["prefill_tok_per_s"])
    ),
    "slot_buffer_priming_on_minus_off_tok_per_s": (
        f(next(r for r in prime_rows if r["prime_mode"] == "on")["prefill_tok_per_s"])
        - f(next(r for r in prime_rows if r["prime_mode"] == "off")["prefill_tok_per_s"])
    ),
    "same_kv_prewarm_on_minus_off_tok_per_s": (
        f(next(r for r in carry_rows if r["same_kv_prewarm"] == "on")["prefill_tok_per_s"])
        - f(next(r for r in carry_rows if r["same_kv_prewarm"] == "off")["prefill_tok_per_s"])
    ),
    "backend_state_batch_force_on_minus_off_tok_per_s": (
        f(next(r for r in backend_batch_rows if r["backend_state_batch_mode"] == "force")["prefill_tok_per_s"])
        - f(next(r for r in backend_batch_rows if r["backend_state_batch_mode"] == "off")["prefill_tok_per_s"])
    ),
}

report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

with report_md.open("w", encoding="utf-8") as out:
    out.write("# Qwen3.5 Prefill Recovery Report\n\n")
    out.write("## State Mode A/B\n\n")
    out.write("| Mode | Prefill tok/s | BatchFast ms | Handoff layers | CPU alias | Slot buffer | Carryover | Zero-init | Materialized | Observed path | Observed owner | Batch kind |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|\n")
    for row in state_rows:
        out.write(
            f"| {row['mode']} | {f(row['prefill_tok_per_s']):.1f} | {f(row['batchfast_ms']):.1f} | "
            f"{row['qkv_handoff_layers']} | {row['cpu_alias_layers']} | {row['slot_buffer_layers']} | "
            f"{row['backend_carryover_layers']} | {row.get('backend_zero_init_layers', '0')} | {row['cpu_materialization_layers']} | "
            f"{row['observed_path']} | {row['observed_owner']} | {row['batch_kind']} |\n"
        )
    out.write("\n")
    out.write(
        f"Best state mode: `{state_best['mode']}` ({f(state_best['prefill_tok_per_s']):.1f} tok/s, "
        f"delta vs auto {payload['state_delta_vs_auto_tok_per_s']:+.1f} tok/s)\n\n"
    )

    out.write("## Scratch A/B\n\n")
    out.write("| Alpha/Beta mode | Prefill tok/s | BatchFast ms | Handoff layers | Carryover | Zero-init | Materialized | Effective alpha/beta dtype | Observed path | Observed owner | Batch kind |\n")
    out.write("|---|---:|---:|---:|---:|---:|---|---|---|---|---|\n")
    for row in scratch_rows:
        out.write(
            f"| {row['alpha_beta_mode']} | {f(row['prefill_tok_per_s']):.1f} | {f(row['batchfast_ms']):.1f} | "
            f"{row['qkv_handoff_layers']} | {row['backend_carryover_layers']} | {row.get('backend_zero_init_layers', '0')} | {row['cpu_materialization_layers']} | "
            f"{row['effective_alpha_beta_dtype']} | {row['observed_path']} | {row['observed_owner']} | {row['batch_kind']} |\n"
        )
    out.write("\n")
    out.write(
        f"Best scratch mode: `{scratch_best['alpha_beta_mode']}` ({f(scratch_best['prefill_tok_per_s']):.1f} tok/s, "
        f"delta vs f32 {payload['scratch_delta_vs_f32_tok_per_s']:+.1f} tok/s)\n\n"
    )

    out.write("## Slot-Buffer Priming A/B\n\n")
    out.write("| Prime | Prefill tok/s | BatchFast ms | Handoff layers | Slot-buffer layers | Carryover | Zero-init | Materialized | Effective prime | Observed path | Observed owner | Batch kind |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|\n")
    for row in prime_rows:
        out.write(
            f"| {row['prime_mode']} | {f(row['prefill_tok_per_s']):.1f} | {f(row['batchfast_ms']):.1f} | "
            f"{row['qkv_handoff_layers']} | {row['slot_buffer_layers']} | {row['backend_carryover_layers']} | "
            f"{row.get('backend_zero_init_layers', '0')} | {row['cpu_materialization_layers']} | {row['effective_prime']} | {row['observed_path']} | {row['observed_owner']} | {row['batch_kind']} |\n"
        )
    out.write("\n")
    out.write(
        f"Best priming mode: `{prime_best['prime_mode']}` ({f(prime_best['prefill_tok_per_s']):.1f} tok/s, "
        f"on-off delta {payload['slot_buffer_priming_on_minus_off_tok_per_s']:+.1f} tok/s)\n\n"
    )

    out.write("## Same-KV Carryover A/B\n\n")
    out.write("| Same-KV prewarm | Prefill tok/s | BatchFast ms | Handoff layers | Carryover | Zero-init | Materialized | Effective prewarm | Observed path | Observed owner | Batch kind |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|---|---|---|---|\n")
    for row in carry_rows:
        out.write(
            f"| {row['same_kv_prewarm']} | {f(row['prefill_tok_per_s']):.1f} | {f(row['batchfast_ms']):.1f} | "
            f"{row['qkv_handoff_layers']} | {row['backend_carryover_layers']} | {row['backend_zero_init_layers']} | "
            f"{row['cpu_materialization_layers']} | {row['effective_same_kv_prewarm']} | {row['observed_path']} | {row['observed_owner']} | {row['batch_kind']} |\n"
        )
    out.write("\n")
    out.write(
        f"Best same-KV carryover mode: `{carry_best['same_kv_prewarm']}` ({f(carry_best['prefill_tok_per_s']):.1f} tok/s, "
        f"on-off delta {payload['same_kv_prewarm_on_minus_off_tok_per_s']:+.1f} tok/s)\n\n"
    )

    out.write("## Backend-State-Batch A/B\n\n")
    out.write("| Backend-state-batch | Prefill tok/s | BatchFast ms | Handoff layers | Effective force | Observed path | Observed owner | Batch kind | Backend-native | CPU direct | CPU gathered | CPU gathered from backend |\n")
    out.write("|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|\n")
    for row in backend_batch_rows:
        out.write(
            f"| {row['backend_state_batch_mode']} | {f(row['prefill_tok_per_s']):.1f} | {f(row['batchfast_ms']):.1f} | "
            f"{row['qkv_handoff_layers']} | {row['effective_force_backend_state_batch']} | {row['observed_path']} | "
            f"{row['observed_owner']} | {row['batch_kind']} | {row['backend_native_layers']} | "
            f"{row['cpu_direct_layers']} | {row['cpu_gathered_layers']} | {row['cpu_gathered_from_backend_layers']} |\n"
        )
    out.write("\n")
    out.write(
        f"Best backend-state-batch mode: `{backend_batch_best['backend_state_batch_mode']}` ({f(backend_batch_best['prefill_tok_per_s']):.1f} tok/s, "
        f"force-on delta {payload['backend_state_batch_force_on_minus_off_tok_per_s']:+.1f} tok/s)\n\n"
    )

    state_mode = state_best["mode"]
    scratch_mode = scratch_best["alpha_beta_mode"]
    prime_mode = prime_best["prime_mode"]
    carry_mode = carry_best["same_kv_prewarm"]
    backend_batch_mode = backend_batch_best["backend_state_batch_mode"]
    carry_owner = next(
        (
            row.get("observed_owner", "")
            for row in carry_rows
            if row.get("same_kv_prewarm") == carry_mode
        ),
        "",
    )
    backend_batch_kind = next(
        (
            row.get("batch_kind", "")
            for row in backend_batch_rows
            if row.get("backend_state_batch_mode") == backend_batch_mode
        ),
        "",
    )
    state_owner = next(
        (row.get("observed_owner", "") for row in state_rows if row.get("mode") == state_mode),
        "",
    )
    out.write("## Recommendation\n\n")
    out.write(
        f"- Current best state-mode for this prompt length is `{state_mode}`.\n"
        f"- Best state-mode currently resolves to observed owner `{state_owner}`.\n"
        f"- Current best alpha/beta scratch mode is `{scratch_mode}`.\n"
        f"- Current best slot-buffer priming mode is `{prime_mode}`.\n"
        f"- Current best same-KV carryover mode is `{carry_mode}`.\n"
        f"- Current best backend-state-batch mode is `{backend_batch_mode}`.\n"
        "- Use the `Observed owner`, `Carryover`, `Zero-init`, `Materialized`, and `Batch kind` columns to distinguish true backend residency from pristine-zero initialization or CPU-materialized slot-buffer runs.\n"
        "- If scratch delta stays near zero while state-mode delta is material, prioritize persistent recurrent state/runtime work over more scratch compression.\n"
    )
    if state_owner == "backend_owned":
        out.write("- This prompt length already benefits from backend-owned recurrent carryover; next optimization work should target making this lifecycle default and removing remaining model-side handoff overhead.\n")
    elif state_owner in {"backend_zero_initialized", "already_synced"}:
        out.write("- Best path is still relying on zero-init or pre-synced slot buffers; next optimization work should push mutated recurrent state into persistent backend ownership.\n")
    else:
        out.write("- Best path is still CPU-materialized; next optimization work should prioritize device-primary recurrent ownership before more kernel tuning.\n")
    if payload["same_kv_prewarm_on_minus_off_tok_per_s"] > 5.0 and carry_owner == "backend_owned":
        out.write("- Same-KV prewarm materially improves throughput and now lands on backend-owned carryover; this is the strongest current signal for promoting normal prefill lifecycle toward persistent backend ownership by default.\n")
    elif payload["same_kv_prewarm_on_minus_off_tok_per_s"] > 5.0 and carry_owner == "cpu_materialized":
        out.write("- Same-KV prewarm materially improves throughput without changing owner semantics; this points to warm lifecycle / execution-shape effects still dominating over persistent backend ownership in the current contract.\n")
    if payload["backend_state_batch_force_on_minus_off_tok_per_s"] < -5.0:
        out.write("- Forcing backend-native state batch is currently slower than model-side handoff; the remaining Module B gap is not only ownership, but also GPU-QKV ingress and backend batch execution shape.\n")
    elif abs(payload["backend_state_batch_force_on_minus_off_tok_per_s"]) <= 5.0:
        out.write("- Backend-native state batch is now near parity with model-side handoff on this prompt length; the next step is to make backend-native ingress and lifecycle the default path without regressing current carryover wins.\n")
    elif backend_batch_kind == "backend_native":
        out.write("- Backend-native state batch is now competitive on this prompt length; the next step is to remove the explicit force toggle and make backend-native recurrent batch the normal lifecycle.\n")
PY

echo "MD:   $REPORT_MD"
echo "JSON: $REPORT_JSON"
