#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
TMP_DIR="$(ax_tmp_dir ax-cli-tui-phase3)"

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
    "path": "/private/secret-model/model.safetensors",
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

cargo build --release -p ax-engine-server -p ax-engine-bench -p ax-engine-tui

for bin in ax-engine-server ax-engine-bench ax-engine-manager; do
  test -x "target/release/$bin"
done

ARCHIVE="$TMP_DIR/ax-engine-phase3-smoke.tar.gz"
tar -czf "$ARCHIVE" -C target/release ax-engine-server ax-engine-bench ax-engine-manager
tar -tzf "$ARCHIVE" >"$TMP_DIR/archive.txt"
grep -qx "ax-engine-server" "$TMP_DIR/archive.txt"
grep -qx "ax-engine-bench" "$TMP_DIR/archive.txt"
grep -qx "ax-engine-manager" "$TMP_DIR/archive.txt"

target/release/ax-engine-manager --help >"$TMP_DIR/help.out"
grep -q "support bundles write redacted diagnostics" "$TMP_DIR/help.out"

target/release/ax-engine-manager --check --doctor-json "$TMP_DIR/doctor.json" >"$TMP_DIR/check.out"
grep -q "ax-engine-manager check" "$TMP_DIR/check.out"
grep -q "doctor=ready status=ready" "$TMP_DIR/check.out"

target/release/ax-engine-manager \
  --doctor-json "$TMP_DIR/doctor.json" \
  --support-bundle "$TMP_DIR/support" \
  >"$TMP_DIR/support.out"

grep -q "support_bundle=$TMP_DIR/support/support-bundle.json" "$TMP_DIR/support.out"
test -f "$TMP_DIR/support/support-bundle.json"
grep -q '"schema_version": "ax.engine_manager.support_bundle.v1"' "$TMP_DIR/support/support-bundle.json"
grep -q '"model_weights_copied": false' "$TMP_DIR/support/support-bundle.json"
grep -q '"environment_copied": false' "$TMP_DIR/support/support-bundle.json"
grep -q '"path_present": true' "$TMP_DIR/support/support-bundle.json"
if grep -qE 'model\.safetensors|secret-model|API_KEY|TOKEN|PASSWORD' "$TMP_DIR/support/support-bundle.json"; then
  echo "support bundle leaked model path or secret-like content" >&2
  cat "$TMP_DIR/support/support-bundle.json" >&2
  exit 1
fi

echo "CLI TUI Phase 3 release integration verified."
