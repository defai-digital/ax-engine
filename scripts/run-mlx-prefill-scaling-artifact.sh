#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

MODEL_DIR="${AX_MLX_PREFILL_MODEL_DIR:-}"
OUTPUT_ROOT="$ROOT_DIR/benchmarks/results/mlx-inference/prefill-scaling-runs"
PROMPT_TOKENS="${AX_MLX_PREFILL_PROMPT_TOKENS:-1024,2048,4096,8192}"
GENERATION_TOKENS="${AX_MLX_PREFILL_GENERATION_TOKENS:-1}"
REPETITIONS="${AX_MLX_PREFILL_REPETITIONS:-3}"
COOLDOWN="${AX_MLX_PREFILL_COOLDOWN:-5}"
PREFILL_STEP_SIZE="${AX_MLX_PREFILL_STEP_SIZE:-2048}"
MODEL_ID="${AX_MLX_PREFILL_MODEL_ID:-}"
RUN_LABEL="${AX_MLX_PREFILL_RUN_LABEL:-}"
AXENGINE_PORT="${AX_MLX_PREFILL_AXENGINE_PORT:-8091}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  scripts/run-mlx-prefill-scaling-artifact.sh --model-dir /path/to/mlx-model [options]

Options:
  --model-dir PATH          Local MLX model artifact directory. Required unless AX_MLX_PREFILL_MODEL_DIR is set.
  --output-root PATH        Output root. Defaults to benchmarks/results/mlx-inference/prefill-scaling-runs.
  --prompt-tokens LIST      Comma-separated context sizes. Defaults to 1024,2048,4096,8192.
  --generation-tokens N     Generated token count. Defaults to 1 for prefill/TTFT evidence.
  --repetitions N           Timed repetitions. Defaults to 3.
  --cooldown SECONDS        Cooldown between repetitions. Defaults to 5.
  --prefill-step-size N     MLX prefill step size. Defaults to 2048.
  --model-id ID             Artifact model id. Defaults to the model directory path used by the harness.
  --run-label LABEL         Human-readable output directory label.
  --axengine-port N         AX server port. Defaults to 8091.
  --dry-run                 Print planned commands without running them.
  -h, --help                Show this help.

The runner executes the MLX inference-stack benchmark with direct AX rows,
builds an ax.mlx_prefill_scaling.v1 artifact, and validates it. It is intended
for P1 long-context prefill/TTFT scaling evidence, not n-gram decode claims.
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
        --prompt-tokens)
            PROMPT_TOKENS="${2:?missing value for --prompt-tokens}"
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
    echo "ERROR: --model-dir is required unless AX_MLX_PREFILL_MODEL_DIR is set." >&2
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
STACK_OUTPUT="$OUTPUT_DIR/inference-stack.json"
SCALING_OUTPUT="$OUTPUT_DIR/prefill-scaling.json"
REPORT_OUTPUT="$OUTPUT_DIR/prefill-scaling.md"

cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/scripts/bench_mlx_inference_stack.py"
    --model-dir "$MODEL_DIR"
    --prompt-tokens "$PROMPT_TOKENS"
    --generation-tokens "$GENERATION_TOKENS"
    --repetitions "$REPETITIONS"
    --cooldown "$COOLDOWN"
    --prefill-step-size "$PREFILL_STEP_SIZE"
    --ax-direct
    --axengine-port "$AXENGINE_PORT"
    --output "$STACK_OUTPUT"
    --prefill-scaling-output "$SCALING_OUTPUT"
)

if [[ -n "$MODEL_ID" ]]; then
    cmd+=(--model "$MODEL_ID")
fi

echo "Output directory: $OUTPUT_DIR"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
"${cmd[@]}"
"$PYTHON_BIN" "$ROOT_DIR/scripts/render_mlx_prefill_scaling_report.py" \
    "$SCALING_OUTPUT" \
    --output "$REPORT_OUTPUT"

echo "Inference-stack artifact: $STACK_OUTPUT"
echo "Prefill-scaling artifact: $SCALING_OUTPUT"
echo "Prefill-scaling report: $REPORT_OUTPUT"
