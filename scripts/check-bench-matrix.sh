#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ax-engine-bench-matrix-check.XXXXXX")"
METAL_BUILD_DIR="${AX_ENGINE_METAL_BUILD_DIR:-${AX_METAL_OUTPUT_DIR:-$ROOT_DIR/build/metal}}"
: "${AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR:?AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR is required for MLX matrix smoke}"

cleanup() {
    rm -rf "$TMP_DIR"
}

trap cleanup EXIT

cd "$ROOT_DIR"

AX_METAL_OUTPUT_DIR="$METAL_BUILD_DIR" \
bash "$ROOT_DIR/scripts/build-metal-kernels.sh"

AX_ENGINE_METAL_BUILD_DIR="$METAL_BUILD_DIR" \
AX_BENCH_MATRIX_TMP_DIR="$TMP_DIR" \
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


root = Path(os.environ["AX_BENCH_MATRIX_TMP_DIR"])
repo = Path.cwd()
matrix_manifest = repo / "benchmarks/manifests/matrix/mlx_dense_phase7.json"
output_root = root / "matrix-results"
output_root.mkdir(parents=True, exist_ok=True)

subprocess.run(
    [
        "cargo",
        "run",
        "-p",
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

runs = [path for path in output_root.iterdir() if path.is_dir()]
assert len(runs) == 1, f"expected one matrix result, found {runs}"
run_dir = runs[0]

matrix_json = json.loads((run_dir / "matrix.json").read_text())
summary = (run_dir / "summary.md").read_text()

assert matrix_json["id"] == "mlx_dense_phase7"
assert matrix_json["status"] == "ok"
assert matrix_json["summary"]["member_count"] == 6
assert matrix_json["summary"]["ok_count"] == 6
assert matrix_json["summary"]["contract_failure_count"] == 0
assert len(matrix_json["members"]) == 6

labels = {member["label"] for member in matrix_json["members"]}
assert labels == {
    "Chat Qwen Short",
    "Chat Gemma Short",
    "Coding Qwen Medium",
    "Long Context Qwen 8K",
    "Concurrent Qwen Dual",
    "Shared Prefix Qwen Enterprise",
}

for member in matrix_json["members"]:
    assert member["status"] == "ok"
    assert member["selected_backend"] == "mlx"
    assert member["support_tier"] == "mlx_preview"
    assert Path(member["result_dir"]).is_dir()
    assert "ttft_ms" in member
    assert "decode_tok_s" in member
    assert "prefix_hit_rate" in member

assert "Benchmark Matrix" in summary
assert "Shared Prefix Qwen Enterprise" in summary
assert "Tier 2 dense-path benchmarking" in summary
PY
