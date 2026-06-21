#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"
TMP_DIR="$(ax_tmp_dir ax-engine-bench-matrix-compare-check)"

cleanup() {
    ax_rm_rf "$TMP_DIR"
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


def plumbing_manifest(source: Path) -> Path:
    manifest = json.loads(source.read_text())
    manifest.setdefault("checks", {})["expect_deterministic"] = False
    dest = root / source.name
    dest.write_text(json.dumps(manifest, indent=2) + "\n")
    return dest


chat_qwen_short = plumbing_manifest(repo / "benchmarks/manifests/scenario/chat_qwen_short.json")
chat_qwen35_short = plumbing_manifest(
    repo / "benchmarks/manifests/scenario/chat_qwen3_5_9b_short.json"
)
matrix_manifest.write_text(
    json.dumps(
        {
            "schema_version": "ax.engine_bench.matrix.v1",
            "id": "subset_mlx_dense_phase7",
            "class": "scenario_matrix",
            "members": [
                {
                    "manifest": str(chat_qwen_short),
                    "label": "Chat Qwen Short",
                },
                {
                    "manifest": str(chat_qwen35_short),
                    "label": "Chat Qwen3.5 9B Short",
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
            "ax-engine-bench",
            "--bin",
            "ax-engine-bench",
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
        "ax-engine-bench",
        "--bin",
        "ax-engine-bench",
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

assert matrix_regression["id"] == "subset_mlx_dense_phase7"
assert matrix_regression["summary"]["member_count"] == 2
assert len(matrix_regression["members"]) == 2
labels = {member["label"] for member in matrix_regression["members"]}
assert labels == {"Chat Qwen Short", "Chat Qwen3.5 9B Short"}

for member in matrix_regression["members"]:
    assert Path(member["compare_result_dir"]).is_dir()
    assert "comparison" in member
    assert "ttft_ms_pct" in member["comparison"]
    assert "decode_tok_s_pct" in member["comparison"]

assert "Benchmark Matrix Compare" in summary
assert "Chat Qwen Short" in summary
assert "Chat Qwen3.5 9B Short" in summary
PY
