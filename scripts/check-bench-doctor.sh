#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

cd "$ROOT_DIR"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import subprocess
from pathlib import Path


repo = Path.cwd()

doctor_json = subprocess.check_output(
    ["cargo", "run", "-p", "ax-engine-bench", "--", "doctor", "--json"],
    cwd=repo,
    text=True,
)
report = json.loads(doctor_json)

assert report["schema_version"] == "ax.engine_bench.doctor.v1"
assert report["mlx_target"] == "apple_m2_or_newer_macos_aarch64"

host = report["host"]
toolchain = report["metal_toolchain"]

expected_ready = host["supported_mlx_runtime"] and toolchain["fully_available"]
expected_bringup_allowed = toolchain["fully_available"] and (
    host["supported_mlx_runtime"] or host["unsupported_host_override_active"]
)
expected_status = (
    "ready"
    if expected_ready
    else "bringup_only"
    if expected_bringup_allowed
    else "not_ready"
)
expected_status_label = {
    "ready": "ready",
    "bringup_only": "bring-up only",
    "not_ready": "not ready",
}[expected_status]

assert report["mlx_runtime_ready"] is expected_ready
assert report["bringup_allowed"] is expected_bringup_allowed
assert report["status"] == expected_status
assert isinstance(report["issues"], list)
assert isinstance(report["notes"], list)
assert any(
    note == "llama.cpp backends do not widen supported host scope"
    for note in report["notes"]
)

doctor_text = subprocess.check_output(
    ["cargo", "run", "-p", "ax-engine-bench", "--", "doctor"],
    cwd=repo,
    text=True,
)

assert "AX Engine v6 doctor" in doctor_text
assert f"Status: {expected_status_label}" in doctor_text
assert "Summary:" in doctor_text
assert "Workflow:" in doctor_text
assert "Model artifacts:" in doctor_text
assert "Issues:" in doctor_text
assert "Notes:" in doctor_text
assert "llama.cpp backends do not widen supported host scope" in doctor_text
PY
