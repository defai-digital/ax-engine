#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"
TMP_DIR="$(ax_tmp_dir ax-metal-check)"

cleanup() {
    ax_rm_rf "$TMP_DIR"
}

trap cleanup EXIT

cd "$ROOT_DIR"

AX_METAL_OUTPUT_DIR="$TMP_DIR" bash scripts/build-metal-kernels.sh

AX_METAL_TMP_DIR="$TMP_DIR" \
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path


root = Path.cwd()
tmp_dir = Path(os.environ["AX_METAL_TMP_DIR"])
report = json.loads((tmp_dir / "build_report.json").read_text())
summary = (tmp_dir / "summary.md").read_text()
doctor = json.loads((tmp_dir / "doctor.json").read_text())
manifest = json.loads((root / "metal/phase1-kernels.json").read_text())

assert report["schema_version"] == "ax.metal.build_report.v1"
assert report["library_name"] == manifest["library_name"]
assert report["mlx_target"] == manifest["mlx_target"]
assert report["metal_language_standard"] == "metal3.1"
assert report["default_block_size_tokens"] == 16
assert report["supported_block_size_tokens"] == [16]
assert report["doctor"]["status"] == doctor["status"]
assert report["doctor"]["bringup_allowed"] == doctor["bringup_allowed"]
assert report["doctor"]["metal_toolchain_fully_available"] == doctor["metal_toolchain"]["fully_available"]
assert len(report["source_sha256"]) == 64

kernel_names = [kernel["name"] for kernel in report["kernels"]]
manifest_kernel_names = [kernel["name"] for kernel in manifest["kernels"]]
assert kernel_names == manifest_kernel_names
assert [kernel["tier"] for kernel in report["kernels"]] == [
    kernel["tier"] for kernel in manifest["kernels"]
]

assert "# AX Metal Kernel Build" in summary
assert f"- status: `{report['status']}`" in summary
assert f"- library: `{report['library_name']}`" in summary
assert "- default_block_size_tokens: `16`" in summary
assert "- supported_block_size_tokens: `16`" in summary

if report["status"] == "compiled":
    air_path = Path(report["outputs"]["air"])
    metalar_path = Path(report["outputs"]["metalar"])
    metallib_path = Path(report["outputs"]["metallib"])
    assert air_path.is_file()
    assert metalar_path.is_file()
    assert metallib_path.is_file()
    assert air_path.stat().st_size > 0
    assert metalar_path.stat().st_size > 0
    assert metallib_path.stat().st_size > 0
    assert len(report["outputs"]["air_sha256"]) == 64
    assert len(report["outputs"]["metalar_sha256"]) == 64
    assert len(report["outputs"]["metallib_sha256"]) == 64
    assert len(report["compile_commands"]) == 3
else:
    assert report["status"] in {
        "skipped_toolchain_unavailable",
        "skipped_not_ready",
    }
    assert report["reason"]
    assert report["outputs"]["air"] is None
    assert report["outputs"]["metalar"] is None
    assert report["outputs"]["metallib"] is None
PY
