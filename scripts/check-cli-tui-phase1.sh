#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
TMP_DIR="$(ax_tmp_dir ax-cli-tui-phase1)"

cleanup() {
    ax_rm_rf "$TMP_DIR"
}

trap cleanup EXIT

cd "$ROOT_DIR"

cat >"$TMP_DIR/doctor.json" <<'JSON'
{
  "schema_version": "ax.engine_bench.doctor.v1",
  "status": "ready",
  "mlx_runtime_ready": true,
  "bringup_allowed": true,
  "workflow": {
    "mode": "source_checkout",
    "cwd": "/repo",
    "source_root": "/repo",
    "doctor": {
      "argv": ["cargo", "run", "-p", "ax-engine-bench", "--", "doctor", "--json"],
      "cwd": "/repo"
    },
    "server": {
      "argv": ["cargo", "run", "-p", "ax-engine-server", "--"],
      "cwd": "/repo"
    },
    "generate_manifest": {
      "argv": ["cargo", "run", "-p", "ax-engine-bench", "--", "generate-manifest", "<model-dir>", "--json"],
      "cwd": "/repo"
    },
    "benchmark": {
      "argv": ["cargo", "run", "-p", "ax-engine-bench", "--", "scenario", "--manifest", "<manifest>", "--output-root", "<output-root>", "--json"],
      "cwd": "/repo"
    },
    "download_model": {
      "argv": ["python3", "scripts/download_model.py", "<repo-id>", "--json"],
      "cwd": "/repo"
    }
  },
  "model_artifacts": {
    "selected": true,
    "status": "ready",
    "path": "/models/qwen",
    "exists": true,
    "is_dir": true,
    "config_present": true,
    "manifest_present": true,
    "safetensors_present": true,
    "model_type": "qwen3",
    "quantization": {
      "mode": "affine",
      "group_size": 64,
      "bits": 4
    },
    "issues": []
  },
  "issues": [],
  "notes": [],
  "performance_advice": []
}
JSON

cat >"$TMP_DIR/benchmark-artifact.json" <<JSON
{
  "schema_version": "ax.benchmark_artifact.v1",
  "command": "scenario",
  "manifest": "$TMP_DIR/manifest.json",
  "output_root": "$TMP_DIR/results",
  "result_dir": "$TMP_DIR/results/run-1",
  "status": "contract_failure"
}
JSON

mkdir -p "$TMP_DIR/results/run-1"
printf '# Summary\n' >"$TMP_DIR/results/run-1/summary.md"
printf '{}\n' >"$TMP_DIR/results/run-1/contract_failure.json"

cargo test --quiet -p ax-engine-tui
cargo run --quiet -p ax-engine-tui --bin ax-engine-manager -- \
  --check \
  --doctor-json "$TMP_DIR/doctor.json" \
  --benchmark-json "$TMP_DIR/benchmark-artifact.json" \
  --artifact-root "$TMP_DIR/results" \
  >"$TMP_DIR/check.out"

grep -q "ax-engine-manager check" "$TMP_DIR/check.out"
grep -q "doctor=ready status=ready" "$TMP_DIR/check.out"
grep -q "workflow=source_checkout" "$TMP_DIR/check.out"
grep -q "model_artifacts=ready" "$TMP_DIR/check.out"

echo "CLI TUI Phase 1 read-only cockpit verified."
