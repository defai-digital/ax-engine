#!/usr/bin/env bash
#
# Serving-stress agent-workload runner (PRD §10 acceptance).
#
# Invokes `cargo run -p ax-engine-bench -- serving-stress --workload <name>`
# for every Phase 1 / Phase 5 fixture and aggregates the JSON artifacts under
# a per-run output directory. Fixtures that require an MLX model artifact
# directory will skip cleanly when AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR is
# unset; the session-free `post_restart_cache_safety` fixture runs always.
#
# Usage:
#   bash scripts/run-serving-stress.sh [--output-dir DIR] [--seed N] \
#     [--model-id ID] [--mlx-model-artifacts-dir PATH] [--fixtures LIST]
#
# Environment variables (honored when corresponding flag is absent):
#   AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR   MLX model artifact dir (skip target)
#   AX_SERVING_STRESS_OUTPUT_DIR        Default output directory
#   AX_SERVING_STRESS_SEED              Default seed (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"

DEFAULT_FIXTURES=(
    long_prefill_vs_decode
    partial_prefix_hit
    tool_output_repetition
    cancellation_during_prefill
    post_restart_cache_safety
    concurrent_short_inserts
)

usage() {
    cat <<'EOF'
Usage: scripts/run-serving-stress.sh [options]

Runs the agent-workload serving-stress fixtures and aggregates artifacts.
Each fixture's JSON artifact lands at <output-dir>/<fixture-name>.json plus a
combined summary at <output-dir>/summary.json.

Options:
  --output-dir DIR           Output directory for artifacts (default: a
                             timestamped dir under target/serving-stress).
  --seed N                   Seed forwarded to each fixture (default: 0).
  --model-id ID              Model id forwarded to each fixture (default:
                             fixture's documented default).
  --mlx-model-artifacts-dir P
                             Forwarded to each fixture; falls back to
                             AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR env var.
  --fixtures LIST            Comma-separated subset of fixtures to run
                             (default: every fixture).
  -h, --help                 Show this help.
EOF
}

OUTPUT_DIR=""
SEED="${AX_SERVING_STRESS_SEED:-0}"
MODEL_ID=""
ARTIFACTS_DIR="${AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR:-}"
SELECTED_FIXTURES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --mlx-model-artifacts-dir)
            ARTIFACTS_DIR="$2"
            shift 2
            ;;
        --fixtures)
            SELECTED_FIXTURES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown flag: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ -n "${AX_SERVING_STRESS_OUTPUT_DIR:-}" ]]; then
        OUTPUT_DIR="$AX_SERVING_STRESS_OUTPUT_DIR"
    else
        ts="$(date -u +'%Y-%m-%dT%H%M%SZ')"
        OUTPUT_DIR="$ROOT_DIR/target/serving-stress/$ts"
    fi
fi
mkdir -p "$OUTPUT_DIR"

if [[ -n "$SELECTED_FIXTURES" ]]; then
    IFS=',' read -r -a fixtures <<< "$SELECTED_FIXTURES"
else
    fixtures=("${DEFAULT_FIXTURES[@]}")
fi

echo "==> serving-stress run"
echo "    output_dir: $OUTPUT_DIR"
echo "    seed:       $SEED"
if [[ -n "$ARTIFACTS_DIR" ]]; then
    echo "    artifacts:  $ARTIFACTS_DIR"
else
    echo "    artifacts:  (unset; session-requiring fixtures will skip)"
fi
echo "    fixtures:   ${fixtures[*]}"

summary_path="$OUTPUT_DIR/summary.json"
: > "$summary_path.partial"
echo "{" > "$summary_path.partial"
echo '  "schema": "ax.serving_stress.summary.v1",' >> "$summary_path.partial"
echo "  \"seed\": $SEED," >> "$summary_path.partial"
echo "  \"fixtures\": {" >> "$summary_path.partial"

first_entry=1
exit_status=0
for fixture in "${fixtures[@]}"; do
    artifact_path="$OUTPUT_DIR/$fixture.json"
    echo "--> $fixture"
    cmd=(
        cargo run --quiet -p ax-engine-bench --
        serving-stress
        --workload "$fixture"
        --seed "$SEED"
        --json
        --output-path "$artifact_path"
    )
    if [[ -n "$MODEL_ID" ]]; then
        cmd+=(--model-id "$MODEL_ID")
    fi
    if [[ -n "$ARTIFACTS_DIR" ]]; then
        cmd+=(--mlx-model-artifacts-dir "$ARTIFACTS_DIR")
    fi

    # The bench binary already writes the artifact JSON to --output-path even
    # for Skipped outcomes; we capture its stdout only for inline display.
    fixture_status="ok"
    if ! "${cmd[@]}" >/dev/null; then
        fixture_status="failed"
        exit_status=1
    fi

    # Append a summary entry: {"<fixture>": {"status": "<...>", "artifact": "<path>"}}.
    if [[ -f "$artifact_path" ]]; then
        outcome_label="$("$AX_PYTHON_BIN" - "$artifact_path" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path) as f:
        doc = json.load(f)
    print(doc.get("status", "unknown"))
except Exception:
    print("unparsed")
PY
)"
    else
        outcome_label="$fixture_status"
    fi

    if [[ $first_entry -eq 1 ]]; then
        first_entry=0
    else
        echo "    ," >> "$summary_path.partial"
    fi
    printf '    "%s": { "status": "%s", "artifact": "%s" }\n' \
        "$fixture" "$outcome_label" "$artifact_path" >> "$summary_path.partial"
done

echo "  }" >> "$summary_path.partial"
echo "}" >> "$summary_path.partial"
mv "$summary_path.partial" "$summary_path"

echo "==> summary written to $summary_path"
exit "$exit_status"
