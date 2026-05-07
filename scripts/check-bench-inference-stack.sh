#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

cleanup() {
    ax_rm_rf "$ROOT_DIR/scripts/__pycache__"
}

trap cleanup EXIT

cd "$ROOT_DIR"

"$PYTHON_BIN" -m unittest \
  scripts/test_bench_mlx_inference_stack.py \
  scripts/test_build_mlx_prefill_scaling_artifact.py \
  scripts/test_mlx_prefill_scaling_artifact.py \
  scripts/test_render_mlx_prefill_scaling_report.py \
  scripts/test_mlx_prefill_scaling_campaign.py \
  scripts/test_mlx_startup_latency_artifact.py \
  scripts/test_mlx_concurrent_prefill_artifact.py \
  scripts/test_render_mlx_p2_latency_report.py \
  scripts/test_run_mlx_p2_latency_artifacts.py \
  scripts/test_gateddelta_prefill_profile_artifact.py \
  scripts/test_render_gateddelta_prefill_profile_report.py \
  scripts/test_gateddelta_prefill_model.py \
  scripts/test_run_gateddelta_prefill_profile.py \
  scripts/test_readme_performance_artifacts.py \
  scripts/test_turboquant_quality_artifact.py \
  scripts/test_turboquant_microbench_artifact.py \
  scripts/test_probe_mlx_model_support.py \
  -v
"$PYTHON_BIN" scripts/check_readme_performance_artifacts.py
