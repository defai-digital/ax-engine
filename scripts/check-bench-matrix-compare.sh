#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ax-bench-matrix-compare-check.XXXXXX")"

cleanup() {
    rm -rf "$TMP_DIR"
}

trap cleanup EXIT

cd "$ROOT_DIR"

AX_BENCH_MATRIX_COMPARE_TMP_DIR="$TMP_DIR" \
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


root = Path(os.environ["AX_BENCH_MATRIX_COMPARE_TMP_DIR"])
repo = Path.cwd()
matrix_manifest = root / "subset-matrix.json"
matrix_manifest.write_text(
    json.dumps(
        {
            "schema_version": "ax.bench.matrix.v1",
            "id": "subset_native_dense_phase7",
            "class": "scenario_matrix",
            "members": [
                {
                    "manifest": str(repo / "benchmarks/manifests/scenario/chat_qwen_short.json"),
                    "label": "Chat Qwen Short",
                },
                {
                    "manifest": str(repo / "benchmarks/manifests/scenario/concurrent_qwen_dual.json"),
                    "label": "Concurrent Qwen Dual",
                },
            ],
        },
        indent=2,
    )
)

baseline_output = root / "baseline-matrix"
candidate_output = root / "candidate-matrix"
compare_output = root / "matrix-compare"
for path in (baseline_output, candidate_output, compare_output):
    path.mkdir(parents=True, exist_ok=True)


def single_run(output_root: Path) -> Path:
    runs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(runs) == 1, f"expected one run under {output_root}, found {runs}"
    return runs[0]


for output_root in (baseline_output, candidate_output):
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "ax-bench",
            "--",
            "matrix",
            "--manifest",
            str(matrix_manifest),
            "--output-root",
            str(output_root),
        ],
        check=True,
        cwd=repo,
    )

baseline_run = single_run(baseline_output)
candidate_run = single_run(candidate_output)

subprocess.run(
    [
        "cargo",
        "run",
        "-p",
        "ax-bench",
        "--",
        "matrix-compare",
        "--baseline",
        str(baseline_run),
        "--candidate",
        str(candidate_run),
        "--output-root",
        str(compare_output),
    ],
    check=True,
    cwd=repo,
)

compare_run = single_run(compare_output)
matrix_regression = json.loads((compare_run / "matrix_regression.json").read_text())
summary = (compare_run / "summary.md").read_text()

assert matrix_regression["id"] == "subset_native_dense_phase7"
assert matrix_regression["summary"]["member_count"] == 2
assert len(matrix_regression["members"]) == 2
labels = {member["label"] for member in matrix_regression["members"]}
assert labels == {"Chat Qwen Short", "Concurrent Qwen Dual"}

for member in matrix_regression["members"]:
    assert Path(member["compare_result_dir"]).is_dir()
    assert "comparison" in member
    assert "ttft_ms_pct" in member["comparison"]
    assert "decode_tok_s_pct" in member["comparison"]

assert "Benchmark Matrix Compare" in summary
assert "Chat Qwen Short" in summary
assert "Concurrent Qwen Dual" in summary
PY
