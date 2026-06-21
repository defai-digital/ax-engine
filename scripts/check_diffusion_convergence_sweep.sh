#!/usr/bin/env bash

set -euo pipefail

# Validate the sweep_diffusion_convergence.py artifact schema.
#
# Usage:
#   bash scripts/check_diffusion_convergence_sweep.sh
#   bash scripts/check_diffusion_convergence_sweep.sh path/to/sweep_results.json
#
# When invoked without arguments, creates a synthetic artifact matching the
# schema produced by sweep_diffusion_convergence.py and validates it.
# When invoked with a path, validates the actual artifact file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"
TMP_DIR="$(ax_tmp_dir ax-diffusion-sweep-check)"

cleanup() {
    ax_rm_rf "$TMP_DIR"
}

trap 'ax_run_cleanup "$?" cleanup' EXIT

cd "$ROOT_DIR"

EXPECTED_SCHEMA="ax.diffusion_convergence_sweep.v1"

validate_artifact() {
    local artifact_path="$1"

    if [ ! -f "$artifact_path" ]; then
        echo "FAIL: artifact not found at $artifact_path"
        exit 1
    fi

    "$PYTHON_BIN" - "$artifact_path" "$EXPECTED_SCHEMA" <<'PYEOF'
import json
import sys

REQUIRED_TOP_KEYS = [
    "schema",
    "model_dir",
    "prompt_tokens",
    "grid_dimensions",
    "warmup_runs",
    "measure_runs",
    "results",
]

REQUIRED_GRID_KEYS = [
    "entropy_threshold",
    "acceptance_rate_threshold",
    "entropy_plateau_delta",
]

REQUIRED_RESULT_KEYS = [
    "entropy_threshold",
    "acceptance_rate_threshold",
    "entropy_plateau_delta",
    "prompt_tokens",
    "denoise_steps",
    "convergence_rate",
    "converged_strict_count",
    "converged_acceptance_count",
    "converged_plateau_count",
    "min_entropy",
    "min_acceptance_rate",
    "block_decode_tok_s",
    "block_wall_us",
    "measure_runs",
]

REQUIRED_STAT_KEYS = ["median"]

artifact_path = sys.argv[1]
expected_schema = sys.argv[2]

with open(artifact_path) as f:
    artifact = json.load(f)

errors = []

# Top-level keys.
for key in REQUIRED_TOP_KEYS:
    if key not in artifact:
        errors.append(f"missing top-level key: {key}")

# Schema match.
if artifact.get("schema") != expected_schema:
    errors.append(
        f"schema mismatch: expected {expected_schema!r}, "
        f"got {artifact.get('schema')!r}"
    )

# Grid dimensions.
grid = artifact.get("grid_dimensions", {})
for key in REQUIRED_GRID_KEYS:
    if key not in grid:
        errors.append(f"missing grid_dimensions key: {key}")
    elif not isinstance(grid[key], list) or len(grid[key]) == 0:
        errors.append(f"grid_dimensions.{key} must be a non-empty list")

# Results.
results = artifact.get("results", [])
if not isinstance(results, list) or len(results) == 0:
    errors.append("results must be a non-empty list")
else:
    for i, row in enumerate(results):
        prefix = f"results[{i}]"
        for key in REQUIRED_RESULT_KEYS:
            if key not in row:
                errors.append(f"{prefix}: missing key {key}")
        for stat_key in ("denoise_steps", "min_entropy", "min_acceptance_rate",
                         "block_decode_tok_s", "block_wall_us"):
            stat = row.get(stat_key, {})
            for sk in REQUIRED_STAT_KEYS:
                if sk not in stat:
                    errors.append(f"{prefix}.{stat_key}: missing {sk}")
        # convergence_rate must be a float in [0, 1].
        cr = row.get("convergence_rate")
        if not isinstance(cr, (int, float)) or cr < 0 or cr > 1:
            errors.append(f"{prefix}: convergence_rate must be float in [0, 1]")

if errors:
    print(f"FAIL: {len(errors)} validation error(s):")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)

print(f"OK: artifact has {len(results)} results, schema={expected_schema}")
PYEOF
}

if [ $# -ge 1 ]; then
    echo "Validating provided artifact: $1"
    validate_artifact "$1"
    exit 0
fi

# Generate a synthetic sweep artifact for schema validation.
echo "Generating synthetic sweep artifact..."
SYNTHETIC="$TMP_DIR/sweep_results.json"

"$PYTHON_BIN" - "$SYNTHETIC" <<'PYEOF'
import json
import sys

artifact = {
    "schema": "ax.diffusion_convergence_sweep.v1",
    "model_dir": "/tmp/test-model-dir",
    "prompt_tokens": [128, 512],
    "grid_dimensions": {
        "entropy_threshold": [0.005, 0.01],
        "acceptance_rate_threshold": [0.01, 0.05],
        "entropy_plateau_delta": [0.001, 0.005],
    },
    "warmup_runs": 2,
    "measure_runs": 5,
    "results": [],
}

# Generate synthetic results for each grid point x prompt_tokens.
import itertools
for pt in artifact["prompt_tokens"]:
    for ent, acc, plateau in itertools.product(
        artifact["grid_dimensions"]["entropy_threshold"],
        artifact["grid_dimensions"]["acceptance_rate_threshold"],
        artifact["grid_dimensions"]["entropy_plateau_delta"],
    ):
        artifact["results"].append({
            "entropy_threshold": ent,
            "acceptance_rate_threshold": acc,
            "entropy_plateau_delta": plateau,
            "prompt_tokens": pt,
            "denoise_steps": {"median": 48.0, "min": 44, "max": 48},
            "convergence_rate": 0.0,
            "converged_strict_count": 0,
            "converged_acceptance_count": 0,
            "converged_plateau_count": 0,
            "min_entropy": {"median": 0.35},
            "min_acceptance_rate": {"median": 0.85},
            "block_decode_tok_s": {"median": 2800.0},
            "block_wall_us": {"median": 91000.0},
            "measure_runs": 5,
        })

with open(sys.argv[1], "w") as f:
    json.dump(artifact, f, indent=2)

print(f"Generated {len(artifact['results'])} synthetic results")
PYEOF

echo "Validating synthetic artifact..."
validate_artifact "$SYNTHETIC"

echo ""
echo "check_diffusion_convergence_sweep: PASS"
