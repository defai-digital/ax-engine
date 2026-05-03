#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ax-bench-baseline-check.XXXXXX")"

cleanup() {
    rm -rf "$TMP_DIR"
}

trap cleanup EXIT

cd "$ROOT_DIR"

AX_BENCH_BASELINE_TMP_DIR="$TMP_DIR" \
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


root = Path(os.environ["AX_BENCH_BASELINE_TMP_DIR"])
repo = Path.cwd()
manifest = repo / "benchmarks/manifests/scenario/chat_qwen_short.json"

baseline_results = root / "baseline-results"
candidate_results = root / "candidate-results"
baseline_store = root / "trusted-baselines"
compare_results = root / "compare-results"

for path in (baseline_results, candidate_results, baseline_store, compare_results):
    path.mkdir(parents=True, exist_ok=True)


def single_run(output_root: Path) -> Path:
    runs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(runs) == 1, f"expected exactly one run under {output_root}, found {runs}"
    return runs[0]


subprocess.run(
    [
        "cargo",
        "run",
        "-p",
        "ax-bench",
        "--",
        "scenario",
        "--manifest",
        str(manifest),
        "--output-root",
        str(baseline_results),
    ],
    check=True,
    cwd=repo,
)

baseline_run = single_run(baseline_results)

subprocess.run(
    [
        "cargo",
        "run",
        "-p",
        "ax-bench",
        "--",
        "baseline",
        "--source",
        str(baseline_run),
        "--name",
        "Dense Qwen Trusted",
        "--output-root",
        str(baseline_store),
    ],
    check=True,
    cwd=repo,
)

trusted_baseline_dir = baseline_store / "Dense-Qwen-Trusted"
assert trusted_baseline_dir.is_dir()
metadata = json.loads((trusted_baseline_dir / "trusted_baseline.json").read_text())
assert metadata["name"] == "Dense Qwen Trusted"
assert metadata["slug"] == "Dense-Qwen-Trusted"
assert metadata["source_result_dir"] == str(baseline_run)
assert metadata["manifest"]["id"] == "chat_qwen_short"
assert metadata["runtime"]["selected_backend"] == "mlx"
assert metadata["route"]["prefix_cache_path"] == "metadata_lookup"
assert (trusted_baseline_dir / "trusted_baseline.md").is_file()
assert (trusted_baseline_dir / "metrics.json").is_file()
assert (trusted_baseline_dir / "summary.md").is_file()

duplicate = subprocess.run(
    [
        "cargo",
        "run",
        "-p",
        "ax-bench",
        "--",
        "baseline",
        "--source",
        str(baseline_run),
        "--name",
        "Dense Qwen Trusted",
        "--output-root",
        str(baseline_store),
    ],
    check=False,
    cwd=repo,
    capture_output=True,
    text=True,
)
assert duplicate.returncode != 0
assert "trusted baseline already exists" in duplicate.stderr

subprocess.run(
    [
        "cargo",
        "run",
        "-p",
        "ax-bench",
        "--",
        "scenario",
        "--manifest",
        str(manifest),
        "--output-root",
        str(candidate_results),
    ],
    check=True,
    cwd=repo,
)

candidate_run = single_run(candidate_results)
subprocess.run(
    [
        "cargo",
        "run",
        "-p",
        "ax-bench",
        "--",
        "compare",
        "--baseline",
        str(trusted_baseline_dir),
        "--candidate",
        str(candidate_run),
        "--output-root",
        str(compare_results),
    ],
    check=True,
    cwd=repo,
)

compare_run = single_run(compare_results)
regression = json.loads((compare_run / "regression.json").read_text())
summary = (compare_run / "comparison.md").read_text()
assert regression["trusted_baseline"]["name"] == "Dense Qwen Trusted"
assert regression["trusted_baseline"]["source_run_id"] == metadata["source_run_id"]
assert "trusted_baseline: `Dense Qwen Trusted`" in summary
PY
