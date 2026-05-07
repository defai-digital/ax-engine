#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

MODEL_DIR="${AX_GATEDDELTA_PROFILE_MODEL_DIR:-}"
OUTPUT_ROOT="$ROOT_DIR/benchmarks/results/mlx-inference/gateddelta-prefill-runs"
GENERATION_TOKENS="${AX_GATEDDELTA_PROFILE_GENERATION_TOKENS:-128}"
REPETITIONS="${AX_GATEDDELTA_PROFILE_REPETITIONS:-3}"
COOLDOWN="${AX_GATEDDELTA_PROFILE_COOLDOWN:-5}"
PREFILL_STEP_SIZE="${AX_GATEDDELTA_PROFILE_PREFILL_STEP_SIZE:-2048}"
MODEL_ID="${AX_GATEDDELTA_PROFILE_MODEL_ID:-}"
RUN_LABEL="${AX_GATEDDELTA_PROFILE_RUN_LABEL:-}"
AXENGINE_PORT="${AX_GATEDDELTA_PROFILE_AXENGINE_PORT:-8091}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  scripts/run-gateddelta-prefill-profile.sh --model-dir /path/to/qwen-linear-attention-mlx-model [options]

Options:
  --model-dir PATH          Local Qwen/GatedDelta MLX model artifact directory. Required unless AX_GATEDDELTA_PROFILE_MODEL_DIR is set.
  --output-root PATH        Output root. Defaults to benchmarks/results/mlx-inference/gateddelta-prefill-runs.
  --generation-tokens N     Generated token count. Defaults to 128.
  --repetitions N           Timed repetitions. Defaults to 3.
  --cooldown SECONDS        Cooldown between repetitions. Defaults to 5.
  --prefill-step-size N     MLX prefill step size. Defaults to 2048.
  --model-id ID             Artifact model id. Defaults to the model directory path used by the harness.
  --run-label LABEL         Human-readable output directory label.
  --axengine-port N         AX server port. Defaults to 8091.
  --dry-run                 Print planned commands without running them.
  -h, --help                Show this help.

The runner executes the MLX inference-stack benchmark in
--gateddelta-prefill-profile mode. It always uses the 512,2048,8192,32768
prompt-token matrix, writes the raw ax.mlx_inference_stack.v2 artifact, validates
the GatedDelta profile contract, and renders a Markdown stage-profile report.
It is pre-kernel evidence for scan/fusion planning, not headline throughput.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="${2:?missing value for --model-dir}"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="${2:?missing value for --output-root}"
            shift 2
            ;;
        --generation-tokens)
            GENERATION_TOKENS="${2:?missing value for --generation-tokens}"
            shift 2
            ;;
        --repetitions)
            REPETITIONS="${2:?missing value for --repetitions}"
            shift 2
            ;;
        --cooldown)
            COOLDOWN="${2:?missing value for --cooldown}"
            shift 2
            ;;
        --prefill-step-size)
            PREFILL_STEP_SIZE="${2:?missing value for --prefill-step-size}"
            shift 2
            ;;
        --model-id)
            MODEL_ID="${2:?missing value for --model-id}"
            shift 2
            ;;
        --run-label)
            RUN_LABEL="${2:?missing value for --run-label}"
            shift 2
            ;;
        --axengine-port)
            AXENGINE_PORT="${2:?missing value for --axengine-port}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$MODEL_DIR" ]]; then
    echo "ERROR: --model-dir is required unless AX_GATEDDELTA_PROFILE_MODEL_DIR is set." >&2
    usage >&2
    exit 2
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model directory does not exist: $MODEL_DIR" >&2
    exit 2
fi

if [[ -z "$RUN_LABEL" ]]; then
    RUN_LABEL="$(basename "$MODEL_DIR")-$(date +%Y%m%d-%H%M%S)"
fi

OUTPUT_DIR="$OUTPUT_ROOT/$RUN_LABEL"
STACK_OUTPUT="$OUTPUT_DIR/gateddelta-prefill-profile.json"
REPORT_OUTPUT="$OUTPUT_DIR/gateddelta-prefill-profile.md"

cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/scripts/bench_mlx_inference_stack.py"
    --model-dir "$MODEL_DIR"
    --gateddelta-prefill-profile
    --generation-tokens "$GENERATION_TOKENS"
    --repetitions "$REPETITIONS"
    --cooldown "$COOLDOWN"
    --prefill-step-size "$PREFILL_STEP_SIZE"
    --axengine-port "$AXENGINE_PORT"
    --output "$STACK_OUTPUT"
    --gateddelta-prefill-profile-report-output "$REPORT_OUTPUT"
)

if [[ -n "$MODEL_ID" ]]; then
    cmd+=(--model "$MODEL_ID")
fi

echo "Output directory: $OUTPUT_DIR"
printf 'Preflight:'
printf ' %q' "$PYTHON_BIN" "$ROOT_DIR/scripts/check_gateddelta_prefill_model.py" "$MODEL_DIR"
printf '\n'
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" "$ROOT_DIR/scripts/check_gateddelta_prefill_model.py" "$MODEL_DIR"
cargo build -p ax-engine-server --release
"${cmd[@]}"

echo "GatedDelta profile artifact: $STACK_OUTPUT"
echo "GatedDelta profile report: $REPORT_OUTPUT"
