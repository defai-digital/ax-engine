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
  scripts/bench_embedding_models.py \
  scripts/verify_embedding_models.py \
  scripts/test_embedding_server_ports.py \
  scripts/check_ax_serving_benchmark_artifact.py \
  scripts/test_ax_serving_benchmark_artifact.py \
  scripts/openwebui_e2e.py \
  scripts/test_openwebui_e2e.py \
  scripts/test_qa_checkers.py \
  scripts/render_ax_serving_benchmark_report.py \
  scripts/test_render_ax_serving_benchmark_report.py \
  scripts/bench_mlx_inference_stack.py \
  scripts/test_bench_mlx_inference_stack.py \
  scripts/bench_mtp_differential.py \
  scripts/check_mtp_sidecar_provenance.py \
  scripts/prepare_qwen36_mtp_sidecar.py \
  scripts/test_prepare_qwen36_mtp_sidecar.py \
  scripts/prepare_mtp_sidecar.py \
  scripts/test_prepare_mtp_sidecar.py \
  scripts/prepare_gemma4_assistant_mtp.py \
  scripts/test_prepare_gemma4_assistant_mtp.py \
  scripts/bench_gemma4_assistant_mtp.py \
  scripts/test_bench_gemma4_assistant_mtp.py \
  scripts/bench_lightning_mlx_raw.py \
  scripts/bench_qwen36_mtp_fair.py \
  scripts/bench_rapid_mlx_prompt_suites.py \
  scripts/test_bench_lightning_mlx_raw.py \
  scripts/test_bench_mtp_differential.py \
  scripts/test_check_mtp_sidecar_provenance.py \
  scripts/test_bench_qwen36_mtp_fair.py \
  scripts/test_bench_rapid_mlx_prompt_suites.py \
  scripts/bench_ax_only_sweep.py \
  scripts/test_bench_ax_only_sweep.py \
  scripts/bench_llama_cpp_metal_sweep.py \
  scripts/test_bench_llama_cpp_metal_sweep.py \
  scripts/build_mlx_prefill_scaling_artifact.py \
  scripts/test_build_mlx_prefill_scaling_artifact.py \
  scripts/check_mlx_prefill_scaling_artifact.py \
  scripts/test_mlx_prefill_scaling_artifact.py \
  scripts/render_mlx_prefill_scaling_report.py \
  scripts/test_render_mlx_prefill_scaling_report.py \
  scripts/build_long_context_comparison_artifact.py \
  scripts/check_long_context_comparison_artifact.py \
  scripts/render_long_context_comparison_report.py \
  scripts/test_long_context_comparison_artifact.py \
  scripts/build_long_context_decode_at_depth_artifact.py \
  scripts/check_long_context_decode_at_depth_artifact.py \
  scripts/render_long_context_decode_at_depth_report.py \
  scripts/test_long_context_decode_at_depth_artifact.py \
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
  scripts/test_run_mlx_artifact_wrappers.py \
  scripts/check_gateddelta_prefill_profile_artifact.py \
  scripts/test_gateddelta_prefill_profile_artifact.py \
  scripts/render_gateddelta_prefill_profile_report.py \
  scripts/test_render_gateddelta_prefill_profile_report.py \
  scripts/render_mlx_decode_profile_report.py \
  scripts/test_render_mlx_decode_profile_report.py \
  scripts/render_mlx_forward_profile_report.py \
  scripts/test_render_mlx_forward_profile_report.py \
  scripts/check_mlx_forward_profile_artifact.py \
  scripts/test_mlx_forward_profile_artifact.py \
  scripts/check_linear_attention_pack_promotion.py \
  scripts/test_check_linear_attention_pack_promotion.py \
  scripts/check_qwen_post_input_route_promotion.py \
  scripts/test_check_qwen_post_input_route_promotion.py \
  scripts/check_mlx_prefill_claim_cycle.py \
  scripts/test_mlx_prefill_claim_cycle.py \
  scripts/update_readme_from_bench.py \
  scripts/test_update_readme_from_bench.py \
  scripts/update_readme_from_results.py \
  scripts/test_update_readme_from_results.py \
  scripts/update_readme_inject_llama_cpp.py \
  scripts/test_update_readme_inject_llama_cpp.py \
  scripts/check_gateddelta_prefill_model.py \
  scripts/test_gateddelta_prefill_model.py \
  scripts/test_run_gateddelta_prefill_profile.py \
  scripts/check_readme_performance_artifacts.py \
  scripts/test_readme_performance_artifacts.py \
  scripts/render_readme_performance_charts.py \
  scripts/build_turboquant_decode_outputs.py \
  scripts/build_turboquant_quality_metrics.py \
  scripts/build_turboquant_quality_artifact.py \
  scripts/check_turboquant_quality_artifact.py \
  scripts/check_turboquant_microbench_artifact.py \
  scripts/check_turboquant_prd_completion.py \
  scripts/check_direct_mlx_hotpath_probe_artifact.py \
  scripts/check_direct_gemma4_ffn_route_promotion.py \
  scripts/check_direct_mlx_no_production_route.py \
  scripts/check_mlx_fastpath_env_controls.py \
  scripts/check_mla_prefix_restore_evidence.py \
  scripts/check_mla_prefix_restore_retirement.py \
  scripts/check_turboquant_public_docs.py \
  scripts/check_turboquant_promotion_readiness.py \
  scripts/test_turboquant_quality_artifact.py \
  scripts/test_run_turboquant_quality_artifact.py \
  scripts/test_turboquant_microbench_artifact.py \
  scripts/test_turboquant_prd_completion.py \
  scripts/test_check_direct_mlx_hotpath_probe_artifact.py \
  scripts/test_check_direct_gemma4_ffn_route_promotion.py \
  scripts/test_check_direct_mlx_no_production_route.py \
  scripts/test_check_mlx_fastpath_env_controls.py \
  scripts/test_check_qwen_post_input_route_promotion.py \
  scripts/test_check_linear_attention_pack_promotion.py \
  scripts/test_check_mla_prefix_restore_evidence.py \
  scripts/test_check_mla_prefix_restore_retirement.py \
  scripts/check_offline_policy_search_artifact.py \
  scripts/build_offline_policy_search_artifact.py \
  scripts/search_turboquant_kv_policy.py \
  scripts/test_offline_policy_search_artifact.py \
  scripts/test_build_offline_policy_search_artifact.py \
  scripts/test_search_turboquant_kv_policy.py \
  scripts/probe_mlx_model_support.py \
  scripts/test_probe_mlx_model_support.py \
  scripts/diagnose_server_rss.py \
  scripts/verify_prefix_reuse_equivalence.py \
  scripts/test_verify_prefix_reuse_equivalence.py \
  scripts/profile_kv_long_context_evidence.py \
  scripts/profile_kv_multiturn_chat_evidence.py \
  scripts/test_profile_kv_multiturn_chat_evidence.py
"$PYTHON_BIN" -m unittest \
  scripts/test_bench_ax_serving.py \
  scripts/test_embedding_server_ports.py \
  scripts/test_ax_serving_benchmark_artifact.py \
  scripts/test_openwebui_e2e.py \
  scripts/test_qa_checkers.py \
  scripts/test_render_ax_serving_benchmark_report.py \
  scripts/test_update_readme_from_bench.py \
  scripts/test_update_readme_from_results.py \
  scripts/test_update_readme_inject_llama_cpp.py \
  scripts/test_bench_ax_only_sweep.py \
  scripts/test_bench_llama_cpp_metal_sweep.py \
  scripts/test_bench_mtp_differential.py \
  scripts/test_check_mtp_sidecar_provenance.py \
  scripts/test_prepare_qwen36_mtp_sidecar.py \
  scripts/test_prepare_mtp_sidecar.py \
  scripts/test_prepare_gemma4_assistant_mtp.py \
  scripts/test_bench_gemma4_assistant_mtp.py \
  scripts/test_bench_lightning_mlx_raw.py \
  scripts/test_bench_qwen36_mtp_fair.py \
  scripts/test_bench_rapid_mlx_prompt_suites.py \
  scripts/test_offline_policy_search_artifact.py \
  scripts/test_build_offline_policy_search_artifact.py \
  scripts/test_search_turboquant_kv_policy.py \
  scripts/test_check_direct_mlx_hotpath_probe_artifact.py \
  scripts/test_check_direct_gemma4_ffn_route_promotion.py \
  scripts/test_check_direct_mlx_no_production_route.py \
  scripts/test_check_mlx_fastpath_env_controls.py \
  scripts/test_check_mla_prefix_restore_evidence.py \
  scripts/test_run_turboquant_quality_artifact.py \
  scripts/test_turboquant_prd_completion.py \
  scripts/test_long_context_comparison_artifact.py \
  scripts/test_long_context_decode_at_depth_artifact.py \
  scripts/test_run_mlx_artifact_wrappers.py \
  scripts/test_verify_prefix_reuse_equivalence.py \
  scripts/test_profile_kv_multiturn_chat_evidence.py
bash scripts/check-bench-inference-stack.sh
bash scripts/check-turboquant-quality-gate.sh
bash scripts/check-turboquant-microbench-gate.sh
bash scripts/check-offline-policy-search-artifacts.sh
"$PYTHON_BIN" scripts/check_turboquant_public_docs.py
"$PYTHON_BIN" scripts/check_direct_mlx_no_production_route.py
"$PYTHON_BIN" scripts/check_mlx_fastpath_env_controls.py
"$PYTHON_BIN" scripts/check_linear_attention_pack_promotion.py
"$PYTHON_BIN" scripts/check_qwen_post_input_route_promotion.py
"$PYTHON_BIN" scripts/check_mla_prefix_restore_evidence.py
"$PYTHON_BIN" scripts/check_mla_prefix_restore_retirement.py
"$PYTHON_BIN" scripts/render_readme_performance_charts.py --check
