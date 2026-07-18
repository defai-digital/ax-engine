#!/usr/bin/env bash
# Live-model QA + product-surface gate (requires mounted MLX artifacts).
#
# Starts ax-engine-server against $AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR, then runs:
#   1) product-surface probes (concurrency, stream, cancel, tools, multimodal)
#   2) a small stratified QA bank sample (JSON report; partial model quality OK)
#
# Used by CI model-smoke when artifacts are mounted. Offline gate is check-qa.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"
HOST="${HOST:-127.0.0.1}"
ax_require_env AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR \
  "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR is required for QA model smoke"
MODEL_ID="${QA_MODEL_ID:-qwen3_5_9b_q4}"
SAMPLE="${QA_SAMPLE:-4}"
SEED="${QA_SEED:-20260716}"
TIMEOUT="${QA_TIMEOUT:-180}"

LOG_FILE="$(ax_tmp_file ax-engine-qa-model .log)"
OUT_DIR="$(ax_tmp_dir ax-qa-model-out)"
METAL_OUTPUT_DIR="$(ax_tmp_dir ax-metal-qa-model)"
PORT="$(ax_allocate_port)"
SERVER_PID=""

cleanup() {
    ax_kill_pid "$SERVER_PID"
    ax_rm_rf "$LOG_FILE" "$METAL_OUTPUT_DIR"
    # Keep OUT_DIR if AX_CI_LOG_DIR set so CI can upload reports.
    if [[ -z "${AX_CI_LOG_DIR:-}" ]]; then
        ax_rm_rf "$OUT_DIR"
    fi
}

trap 'ax_run_cleanup "$?" cleanup' EXIT

cd "$ROOT_DIR"

if [[ ! -f "$AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR/config.json" ]]; then
  echo "missing config.json under $AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR" >&2
  exit 1
fi

echo "==> Building Metal kernels + starting server on port $PORT"
AX_METAL_OUTPUT_DIR="$METAL_OUTPUT_DIR" bash scripts/build-metal-kernels.sh >/dev/null
AX_ENGINE_METAL_BUILD_DIR="$METAL_OUTPUT_DIR" \
  cargo run -p ax-engine-server -- \
    --host "$HOST" \
    --port "$PORT" \
    --model-id "$MODEL_ID" \
    --mlx \
    --mlx-model-artifacts-dir "$AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR" \
    --disable-ngram-acceleration \
    >"$LOG_FILE" 2>&1 &
SERVER_PID="$!"

BASE_URL="http://${HOST}:${PORT}"
echo "==> Waiting for $BASE_URL/v1/models"
AX_ENGINE_SERVER_URL="$BASE_URL" "$PYTHON_BIN" - <<'PY'
from __future__ import annotations
import os, time, urllib.error, urllib.request
url = os.environ["AX_ENGINE_SERVER_URL"].rstrip("/") + "/v1/models"
deadline = time.time() + 420
last = ""
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status == 200:
                print("server ready")
                raise SystemExit(0)
    except Exception as exc:
        last = str(exc)
    time.sleep(1)
print("server not ready:", last)
raise SystemExit(1)
PY

echo "==> Surface probes"
"$PYTHON_BIN" "$ROOT_DIR/qa/surface_probes.py" \
  --base-url "$BASE_URL" \
  --model "$MODEL_ID" \
  --timeout "$TIMEOUT" \
  --json-output "$OUT_DIR/surface.json"

# When the mounted package (or live model card) looks multimodal, run the
# multimodal smoke tier hard (policy + image path). Text-only packages skip.
echo "==> Multimodal smoke (capability/package aware)"
export ROOT_DIR BASE_URL MODEL_ID TIMEOUT OUT_DIR
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
sys.path.insert(0, str(root / "qa"))
from multimodal_probes import package_looks_like_multimodal, run_multimodal_probes
from surface_probes import fetch_model_card, model_advertises_image

base = os.environ["BASE_URL"]
model = os.environ["MODEL_ID"]
timeout = float(os.environ["TIMEOUT"])
out = Path(os.environ["OUT_DIR"]) / "multimodal-smoke.json"
artifacts = Path(os.environ["AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR"])
card = fetch_model_card(base, model, timeout=min(timeout, 30.0))
pkg = package_looks_like_multimodal(artifacts)
adv = model_advertises_image(card)
if not pkg and not adv:
    payload = {
        "schema_version": 1,
        "kind": "multimodal_probes",
        "skipped": True,
        "reason": "text_only_package_and_card",
        "hard_passed": True,
    }
    out.write_text(json.dumps(payload, indent=2))
    print("multimodal smoke skipped (text-only)")
    raise SystemExit(0)
report = run_multimodal_probes(
    base,
    model,
    timeout=timeout,
    tier="smoke",
    artifacts=artifacts if pkg else None,
    require_image=True if (pkg or adv) else None,
)
out.write_text(json.dumps(report.as_dict(), indent=2))
print(report.summary_line)
raise SystemExit(0 if report.hard_passed else 1)
PY

echo "==> QA bank sample (seed=$SEED sample=$SAMPLE)"
"$PYTHON_BIN" "$ROOT_DIR/qa/run_qa.py" \
  --base-url "$BASE_URL" \
  --model "$MODEL_ID" \
  --mode direct \
  --streams false \
  --sample "$SAMPLE" \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  --max-tokens 128 \
  --allow-partial \
  --output "$OUT_DIR/qa-sample.html" \
  --json-output "$OUT_DIR/qa-sample.json"

if [[ -n "${AX_CI_LOG_DIR:-}" ]]; then
  mkdir -p "$AX_CI_LOG_DIR"
  cp -f "$OUT_DIR"/* "$AX_CI_LOG_DIR/" 2>/dev/null || true
  cp -f "$LOG_FILE" "$AX_CI_LOG_DIR/qa-model-server.log" 2>/dev/null || true
fi

echo "==> QA model gate OK (surface hard pass; multimodal smoke; bank sample written)"
