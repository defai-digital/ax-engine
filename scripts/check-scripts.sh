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

bash -n scripts/*.sh scripts/lib/common.sh
"$PYTHON_BIN" -m py_compile \
  scripts/bench_ax_serving.py \
  scripts/test_bench_ax_serving.py \
  scripts/bench_mlx_inference_stack.py \
  scripts/test_bench_mlx_inference_stack.py \
  scripts/build_mlx_prefill_scaling_artifact.py \
  scripts/test_build_mlx_prefill_scaling_artifact.py \
  scripts/check_mlx_prefill_scaling_artifact.py \
  scripts/test_mlx_prefill_scaling_artifact.py \
  scripts/render_mlx_prefill_scaling_report.py \
  scripts/test_render_mlx_prefill_scaling_report.py \
  scripts/check_mlx_prefill_scaling_campaign.py \
  scripts/test_mlx_prefill_scaling_campaign.py \
  scripts/check_mlx_startup_latency_artifact.py \
  scripts/test_mlx_startup_latency_artifact.py \
  scripts/check_mlx_concurrent_prefill_artifact.py \
  scripts/test_mlx_concurrent_prefill_artifact.py \
  scripts/render_mlx_p2_latency_report.py \
  scripts/test_render_mlx_p2_latency_report.py \
  scripts/run_mlx_p2_latency_artifacts.py \
  scripts/test_run_mlx_p2_latency_artifacts.py \
  scripts/check_gateddelta_prefill_profile_artifact.py \
  scripts/test_gateddelta_prefill_profile_artifact.py \
  scripts/render_gateddelta_prefill_profile_report.py \
  scripts/test_render_gateddelta_prefill_profile_report.py \
  scripts/render_mlx_forward_profile_report.py \
  scripts/test_render_mlx_forward_profile_report.py \
  scripts/check_gateddelta_prefill_model.py \
  scripts/test_gateddelta_prefill_model.py \
  scripts/test_run_gateddelta_prefill_profile.py \
  scripts/check_readme_performance_artifacts.py \
  scripts/test_readme_performance_artifacts.py \
  scripts/build_turboquant_decode_outputs.py \
  scripts/build_turboquant_quality_metrics.py \
  scripts/build_turboquant_quality_artifact.py \
  scripts/check_turboquant_quality_artifact.py \
  scripts/check_turboquant_microbench_artifact.py \
  scripts/check_turboquant_public_docs.py \
  scripts/check_turboquant_promotion_readiness.py \
  scripts/test_turboquant_quality_artifact.py \
  scripts/test_turboquant_microbench_artifact.py \
  scripts/probe_mlx_model_support.py \
  scripts/test_probe_mlx_model_support.py \
  scripts/diagnose_server_rss.py \
  scripts/verify_prefix_reuse_equivalence.py \
  scripts/profile_kv_long_context_evidence.py
"$PYTHON_BIN" -m unittest scripts/test_bench_ax_serving.py
bash scripts/check-bench-inference-stack.sh
bash scripts/check-turboquant-quality-gate.sh
bash scripts/check-turboquant-microbench-gate.sh
"$PYTHON_BIN" scripts/check_turboquant_public_docs.py
