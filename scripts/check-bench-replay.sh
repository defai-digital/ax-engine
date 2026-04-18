#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ax-bench-replay-check.XXXXXX")"

cleanup() {
    rm -rf "$TMP_DIR"
}

trap cleanup EXIT

cd "$ROOT_DIR"

AX_BENCH_REPLAY_TMP_DIR="$TMP_DIR" \
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


root = Path(os.environ["AX_BENCH_REPLAY_TMP_DIR"])
repo = Path.cwd()

replays = [
    {
        "name": "shared_prefix_long_churn",
        "manifest": repo / "benchmarks/manifests/replay/shared_prefix_long_churn.json",
        "route": "live_request_share",
        "replay_status": "pass",
    },
    {
        "name": "retained_prefix_after_cleanup",
        "manifest": repo / "benchmarks/manifests/replay/retained_prefix_after_cleanup.json",
        "route": "retained_prompt_prefix_cache",
        "replay_status": "not_applicable",
    },
    {
        "name": "mixed_live_and_retained_prefix_paths",
        "manifest": repo / "benchmarks/manifests/replay/mixed_live_and_retained_prefix_paths.json",
        "route": "mixed_live_and_retained",
        "replay_status": "not_applicable",
    },
    {
        "name": "full_prefix_to_decode_branch",
        "manifest": repo / "benchmarks/manifests/replay/full_prefix_to_decode_branch.json",
        "route": "live_request_share",
        "replay_status": "not_applicable",
    },
    {
        "name": "memory_blocked_prefix_recovery",
        "manifest": repo / "benchmarks/manifests/replay/memory_blocked_prefix_recovery.json",
        "route": "live_request_share",
        "replay_status": "not_applicable",
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


def route_decision(routes: dict, key: str) -> int:
    return int(routes["route"]["crossover_decisions"].get(key, 0))


for replay in replays:
    output_root = root / f"{replay['name']}-results"
    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "ax-bench",
            "--",
            "replay",
            "--manifest",
            str(replay["manifest"]),
            "--output-root",
            str(output_root),
        ],
        check=True,
        cwd=repo,
    )

    run_dir, environment, metrics, routes, artifacts = load_single_run(output_root)
    trace = artifacts["trace"]
    summary = artifacts["summary"]

    assert environment["software"]["tool_mode"] == "engine_deterministic_runtime"
    assert environment["runtime"]["selected_backend"] == "ax_native"
    assert environment["runtime"]["support_tier"] == "native_preview"
    assert environment["runtime"]["resolution_policy"] == "strict_native"
    assert environment["benchmark"]["subcommand"] == "replay"

    assert metrics["runtime"]["selected_backend"] == "ax_native"
    assert metrics["runtime"]["support_tier"] == "native_preview"
    assert metrics["correctness"]["passed"] is True
    assert metrics["determinism"]["passed"] is True
    assert metrics["replay_status"] == replay["replay_status"]
    assert metrics["step_count"] > 0

    route = routes["route"]
    assert route["prefix_cache_path"] == replay["route"]
    assert route["execution_plan"] == "mixed_step_plans"
    assert route["attention_route"] == "mixed_attention_routes"
    assert route["kv_mode"] == "paged_metadata"
    assert route["barrier_mode"] == "serial"
    assert route["crossover_decisions"]["execution_plan_variants"] >= 1
    assert route["crossover_decisions"]["attention_route_variants"] >= 1
    assert route["crossover_decisions"]["kv_mode_variants"] == 1
    assert route["crossover_decisions"]["barrier_mode_variants"] == 1

    steps = trace["steps"]
    assert len(steps) == metrics["step_count"]
    assert "ax-bench replay" in summary
    assert replay["name"] in (run_dir / "manifest.json").read_text()

    if replay["name"] == "shared_prefix_long_churn":
        assert route_decision(routes, "live_share_hits") > 0
        assert route_decision(routes, "retained_cache_hits") == 0
        assert route_decision(routes, "prefix_reused_tokens") > 0

    if replay["name"] == "retained_prefix_after_cleanup":
        assert route_decision(routes, "retained_cache_hits") > 0
        assert route_decision(routes, "live_share_hits") == 0
        assert route_decision(routes, "branch_decode_requests") == 0

    if replay["name"] == "mixed_live_and_retained_prefix_paths":
        assert route_decision(routes, "retained_cache_hits") > 0
        assert route_decision(routes, "live_share_hits") > 0
        assert route_decision(routes, "branch_prefill_requests") > 0

    if replay["name"] == "full_prefix_to_decode_branch":
        assert route_decision(routes, "live_share_hits") > 0
        assert route_decision(routes, "retained_cache_hits") == 0
        assert route_decision(routes, "branch_decode_requests") > 0
        assert any(
            step.get("route", {}).get("prefix_cache_path") == "live_request_share"
            and any(
                item.get("mode") == "Decode"
                and int(item.get("prefix_tokens_reused", 0)) > 0
                for item in step.get("items", [])
            )
            for step in steps
        )

    if replay["name"] == "memory_blocked_prefix_recovery":
        assert metrics["memory_blocked_steps"] > 0
        assert metrics["memory_blocked_request_events"] > 0
        assert metrics["runtime"]["kv_total_blocks"] == 17
        assert route_decision(routes, "live_share_hits") > 0
        assert route_decision(routes, "retained_cache_hits") == 0
        assert route_decision(routes, "prefix_reused_tokens") > 0
        assert route_decision(routes, "blocked_prefix_reuse_requests") > 0
        assert route_decision(routes, "blocked_prefix_reuse_tokens") > 0
        assert any(step.get("memory_blocked_request_ids") for step in steps)
        assert any(
            step.get("route", {}).get("prefix_cache_path") == "live_request_share"
            and int(
                step.get("route", {})
                .get("crossover_decisions", {})
                .get("blocked_prefix_reuse_requests", 0)
            )
            > 0
            and step.get("memory_blocked_request_ids")
            for step in steps
        )
        assert any(
            any(item.get("request_id") == 2 and item.get("mode") == "Decode" for item in step.get("items", []))
            for step in steps
        )
PY
