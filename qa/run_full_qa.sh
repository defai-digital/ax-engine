#!/usr/bin/env bash
# Thin wrapper: materialize a direct+ngram inventory and run the unified matrix.
#
# Preferred entrypoint for multi-model local QA is still:
#   python3 scripts/run_qa_matrix.py --matrix … [--surface]
#
# This script keeps a convenient default model list for HF hub cache layouts.

set -euo pipefail

QA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$QA_DIR/.." && pwd)"
SERVER_BIN="${SERVER_BIN:-${QA_SERVER_BIN:-}}"
if [[ -z "$SERVER_BIN" ]]; then
  if [[ -x "$REPO_ROOT/target/release/ax-engine-server" ]]; then
    SERVER_BIN="$REPO_ROOT/target/release/ax-engine-server"
  else
    SERVER_BIN="$REPO_ROOT/target/debug/ax-engine-server"
  fi
fi
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface/hub}"
REPORT_DIR="${REPORT_DIR:-${QA_SCRATCH:-$QA_DIR/reports/matrix}}"
PORT="${PORT:-${QA_PORT:-8080}}"
TIMEOUT="${TIMEOUT:-${QA_TIMEOUT:-120}}"
QA_SAMPLE="${QA_SAMPLE:-12}"
QA_SEED="${QA_SEED:-42}"
# Product-surface probes on by default for full suite (set QA_SURFACE=0 to skip).
QA_SURFACE="${QA_SURFACE:-1}"
STREAMS="${QA_STREAMS:-both}"

mkdir -p "$REPORT_DIR"
INVENTORY="$REPORT_DIR/qa-matrix-full.txt"

# model_id|artifacts_dir  (modes expanded to direct + ngram)
MODELS=(
  "glm4_moe_lite|$HF_CACHE/models--mlx-community--GLM-4.7-Flash-4bit/snapshots/1454cffb1a21737e162f508e5bc70be9def89276"
  "gemma4-e2b|$HF_CACHE/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd"
  "qwen3.5-27b|$HF_CACHE/models--mlx-community--Qwen3.6-27B-4bit/snapshots/c000ac2c2057d94be3fa931000c31723aac53282"
  "qwen3.6-35b|$HF_CACHE/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46"
)

: >"$INVENTORY"
for model_config in "${MODELS[@]}"; do
  IFS='|' read -r model_id artifacts_dir <<< "$model_config"
  if [[ ! -f "$artifacts_dir/config.json" ]]; then
    echo "SKIP (no artifacts): $model_id @ $artifacts_dir"
    continue
  fi
  echo "OK|direct|$model_id|$artifacts_dir" >>"$INVENTORY"
  echo "OK|ngram|$model_id|$artifacts_dir" >>"$INVENTORY"
done

if [[ ! -s "$INVENTORY" ]]; then
  echo "ERROR: no models with artifacts found under HF cache" >&2
  exit 2
fi

echo "AX Engine Full QA (unified matrix)"
echo "  Server:    $SERVER_BIN"
echo "  Inventory: $INVENTORY"
echo "  Reports:   $REPORT_DIR"
echo "  Sample:    $QA_SAMPLE seed=$QA_SEED surface=$QA_SURFACE streams=$STREAMS"
echo ""

SURFACE_FLAG=()
if [[ "$QA_SURFACE" == "1" ]]; then
  SURFACE_FLAG=(--surface)
fi

export QA_SCRATCH="$REPORT_DIR"
export QA_SERVER_BIN="$SERVER_BIN"
export QA_PORT="$PORT"
export QA_TIMEOUT="$TIMEOUT"
export QA_SAMPLE
export QA_SEED

python3 "$REPO_ROOT/scripts/run_qa_matrix.py" \
  --matrix "$INVENTORY" \
  --scratch "$REPORT_DIR" \
  --server-bin "$SERVER_BIN" \
  --port "$PORT" \
  --timeout "$TIMEOUT" \
  --sample "$QA_SAMPLE" \
  --seed "$QA_SEED" \
  --streams "$STREAMS" \
  --modes direct ngram \
  "${SURFACE_FLAG[@]}"

echo ""
echo "=== Generating HTML summary (if reports present) ==="
python3 "$QA_DIR/generate_summary.py" "$REPORT_DIR" || true
echo "Summary: $REPORT_DIR/summary.html (when HTML reports exist)"
echo "Matrix:  $REPORT_DIR/qa-summary.md"
