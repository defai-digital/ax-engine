#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"
TMP_DIR="$(ax_tmp_dir ax-engine-bench-mlx-check)"
METAL_BUILD_DIR="${AX_ENGINE_METAL_BUILD_DIR:-${AX_METAL_OUTPUT_DIR:-$ROOT_DIR/build/metal}}"
: "${AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR:?AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR is required for MLX benchmark smoke}"

cleanup() {
    ax_rm_rf "$TMP_DIR"
}

trap cleanup EXIT

cd "$ROOT_DIR"

AX_METAL_OUTPUT_DIR="$METAL_BUILD_DIR" \
bash "$ROOT_DIR/scripts/build-metal-kernels.sh"

AX_BENCH_MLX_METAL_BUILD_DIR="$METAL_BUILD_DIR" \
AX_BENCH_MLX_TMP_DIR="$TMP_DIR" \
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


root = Path(os.environ["AX_BENCH_MLX_TMP_DIR"])
repo = Path.cwd()
build_report_path = Path(os.environ["AX_BENCH_MLX_METAL_BUILD_DIR"]) / "build_report.json"
compiled_mlx_artifacts = False

if build_report_path.is_file():
    build_report = json.loads(build_report_path.read_text())
    compiled_mlx_artifacts = build_report.get("status") == "compiled"

bench_env = os.environ.copy()
bench_env.pop("AX_ENGINE_METAL_BUILD_DIR", None)
if compiled_mlx_artifacts:
    bench_env["AX_ENGINE_METAL_BUILD_DIR"] = str(build_report_path.parent)
real_model_expected = True

scenarios = [
    {
        "name": "qwen",
        "manifest": repo / "benchmarks/manifests/scenario/chat_qwen_short.json",
        "model_family": "qwen3_dense",
        "prefill_plan": "phase1.qwen3_dense.dense_prefill",
        "decode_plan": "phase1.qwen3_dense.paged_decode",
        "prefill_route": "qwen3_dense_prefill",
        "decode_route": "qwen3_dense_paged_decode",
    },
    {
        "name": "gemma",
        "manifest": repo / "benchmarks/manifests/scenario/chat_gemma_short.json",
        "model_family": "gemma-4-27b-it",
        "prefill_plan": "phase1.gemma_4_27b_it.dense_prefill",
        "decode_plan": "phase1.gemma_4_27b_it.paged_decode",
        "prefill_route": "gemma_4_27b_it_prefill",
        "decode_route": "gemma_4_27b_it_paged_decode",
    },
]


def load_single_run(output_root: Path) -> tuple[Path, dict, dict, dict, dict]:
    runs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(runs) == 1, f"expected exactly one run directory under {output_root}, found {runs}"
    run_dir = runs[0]
    environment = json.loads((run_dir / "environment.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    routes = json.loads((run_dir / "routes.json").read_text())
    trace = json.loads((run_dir / "trace.json").read_text())
    summary = (run_dir / "summary.md").read_text()
    return run_dir, environment, metrics, routes, {"trace": trace, "summary": summary}


for scenario in scenarios:
    output_root = root / f"{scenario['name']}-results"
    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "ax-engine-bench",
            "--",
            "scenario",
            "--manifest",
            str(scenario["manifest"]),
            "--output-root",
            str(output_root),
        ],
        check=True,
        cwd=repo,
        env=bench_env,
    )

    run_dir, environment, metrics, routes, artifacts = load_single_run(output_root)
    trace = artifacts["trace"]
    summary = artifacts["summary"]

    expected_tool_mode = "engine_bringup_runtime"
    assert environment["software"]["tool_mode"] == expected_tool_mode
    assert environment["runtime"]["selected_backend"] == "mlx"
    assert environment["runtime"]["support_tier"] == "mlx_preview"
    assert environment["runtime"]["resolution_policy"] == "mlx_only"
    mlx_runtime = environment["runtime"]["mlx_runtime"]
    assert environment["route"]["prefix_cache_path"] == "metadata_lookup"
    assert environment["route"]["prefix_cache_evidence"] == "none_observed"
    assert environment["route"]["prefix_reuse_provenance"] == "none_observed"
    assert environment["benchmark"]["subcommand"] == "scenario"

    route = routes["route"]
    assert route["execution_plan"] == "mixed_step_plans"
    assert route["attention_route"] == "mixed_attention_routes"
    assert route["kv_mode"] == "paged_metadata"
    assert route["barrier_mode"] == "serial"
    assert route["prefix_cache_path"] == "metadata_lookup"
    assert route["prefix_cache_evidence"] == "none_observed"
    assert route["prefix_reuse_provenance"] == "none_observed"
    assert route["crossover_decisions"]["execution_plan_variants"] == 2
    assert route["crossover_decisions"]["attention_route_variants"] == 2
    assert route["crossover_decisions"]["kv_mode_variants"] == 1
    assert route["crossover_decisions"]["barrier_mode_variants"] == 1
    assert route["crossover_decisions"]["prefix_reused_requests"] == 0

    assert metrics["runtime"]["selected_backend"] == "mlx"
    assert metrics["runtime"]["support_tier"] == "mlx_preview"
    assert metrics["correctness"]["passed"] is True
    assert metrics["determinism"]["passed"] is True
    assert metrics["replay_status"] == "not_applicable"
    assert metrics["churn_status"] == "pass"
    assert metrics["step_count"] > 1
    assert metrics["metrics"]["ttft_ms"] > 0.0
    assert metrics["metrics"]["decode_tok_s"] > 0.0

    steps = trace["steps"]
    assert len(steps) == metrics["step_count"]
    assert any(
        step.get("route", {}).get("execution_plan") == scenario["prefill_plan"]
        and step.get("route", {}).get("attention_route") == scenario["prefill_route"]
        for step in steps
    )
    assert any(
        step.get("route", {}).get("execution_plan") == scenario["decode_plan"]
        and step.get("route", {}).get("attention_route") == scenario["decode_route"]
        for step in steps
    )
    assert compiled_mlx_artifacts
    assert mlx_runtime["runner"] == "metal_bringup"
    assert mlx_runtime["artifacts_source"] == "explicit_env"
    assert (
        route["crossover_decisions"].get("metal_dispatch_completed", 0) > 0
    ), "compiled Metal artifacts should surface aggregate dispatch evidence"
    assert any(
        step.get("route", {})
        .get("crossover_decisions", {})
        .get("metal_dispatch_completed")
        == 1
        for step in steps
    ), "compiled Metal artifacts should surface per-step dispatch evidence"
    assert real_model_expected
    mlx_model = environment["runtime"].get("mlx_model")
    assert mlx_model is not None
    assert mlx_model["artifacts_source"] == "explicit_env"
    assert route["metal_model_artifacts_validated"] is True
    assert route["metal_model_conditioned_inputs"] is True
    assert route["metal_real_model_tensor_inputs"] is True
    assert route["execution_semantics"] in {
        "metal_real_model_tensor_inputs",
        "metal_real_model_forward",
    }
    assert summary.count(expected_tool_mode) == 1
    assert scenario["model_family"] in (run_dir / "manifest.json").read_text()
PY
