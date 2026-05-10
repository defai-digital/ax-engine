#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
TMP_DIR="$(ax_tmp_dir ax-manager-phase2)"

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

cargo test --quiet -p ax-engine-tui
cargo run --quiet -p ax-engine-tui --bin ax-engine-manager -- \
  --phase2-check \
  --doctor-json "$TMP_DIR/doctor.json" \
  --profile-dir "$TMP_DIR/profiles" \
  >"$TMP_DIR/phase2.out"

grep -q "ax-engine-manager phase2-check" "$TMP_DIR/phase2.out"
grep -q "jobs=5" "$TMP_DIR/phase2.out"
grep -q "job=server-launch kind=server_launch evidence=route_contract owns_process=true" "$TMP_DIR/phase2.out"
grep -q "job=benchmark-scenario kind=benchmark_scenario evidence=workload_contract owns_process=true" "$TMP_DIR/phase2.out"
grep -q "profile=ready path=$TMP_DIR/profiles/phase2-check.json" "$TMP_DIR/phase2.out"
grep -q "fake_job=succeeded log_tail=1" "$TMP_DIR/phase2.out"
grep -q "fake_server=canceled startup_observed=true" "$TMP_DIR/phase2.out"
grep -q "benchmark_display_guard=true" "$TMP_DIR/phase2.out"
test -f "$TMP_DIR/profiles/phase2-check.json"
echo "Manager Phase 2 local job runner verified."
