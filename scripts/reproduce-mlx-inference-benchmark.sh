#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

MODEL_DIR="${AX_ENGINE_MLX_MODEL_DIR:-}"
OUTPUT_ROOT="$ROOT_DIR/benchmarks/community-results/local"
PROMPT_TOKENS="${AX_BENCH_PROMPT_TOKENS:-128,512}"
GENERATION_TOKENS="${AX_BENCH_GENERATION_TOKENS:-128}"
REPETITIONS="${AX_BENCH_REPETITIONS:-3}"
COOLDOWN="${AX_BENCH_COOLDOWN:-3}"
PREFILL_STEP_SIZE="${AX_BENCH_PREFILL_STEP_SIZE:-2048}"
MLX_SWIFT_LM_COMMAND="${AX_BENCH_MLX_SWIFT_LM_COMMAND:-}"
RUN_LABEL="${AX_BENCH_RUN_LABEL:-}"
AX_COMPARE_POLICIES=1

usage() {
    cat <<'EOF'
Usage:
  scripts/reproduce-mlx-inference-benchmark.sh --model-dir /path/to/mlx-model [options]

Options:
  --model-dir PATH              Local MLX model artifact directory. Required unless AX_ENGINE_MLX_MODEL_DIR is set.
  --output-root PATH            Output root. Defaults to benchmarks/community-results/local.
  --prompt-tokens LIST          Comma-separated prompt lengths. Defaults to 128,512.
  --generation-tokens N         Generated token count. Defaults to 128.
  --repetitions N               Timed repetitions. Defaults to 3.
  --cooldown SECONDS            Cooldown between repetitions. Defaults to 3.
  --prefill-step-size N         MLX prefill step size. Defaults to 2048.
  --mlx-swift-lm-command CMD    Optional admitted mlx-swift-lm adapter command template.
  --run-label LABEL             Human-readable output directory label.
  --direct-only                 Emit only the direct same-policy AX row.
  -h, --help                    Show this help.

The script records doctor output, command logs, prompt-token artifacts, and raw
benchmark JSON. It reproduces the benchmark procedure, not identical throughput
across different Apple Silicon chips, memory sizes, or thermal conditions.
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
        --mlx-swift-lm-command)
            MLX_SWIFT_LM_COMMAND="${2:?missing value for --mlx-swift-lm-command}"
            shift 2
            ;;
        --run-label)
            RUN_LABEL="${2:?missing value for --run-label}"
            shift 2
            ;;
        --direct-only)
            AX_COMPARE_POLICIES=0
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
    echo "ERROR: --model-dir is required unless AX_ENGINE_MLX_MODEL_DIR is set." >&2
    usage >&2
    exit 2
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model directory does not exist: $MODEL_DIR" >&2
    exit 2
fi

cd "$ROOT_DIR"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$RUN_LABEL" ]]; then
    RUN_LABEL="$(basename "$MODEL_DIR")"
fi
RUN_SLUG="$("$PYTHON_BIN" - "$RUN_LABEL" <<'PY'
import re
import sys

label = sys.argv[1].strip().lower()
slug = re.sub(r"[^a-z0-9._-]+", "-", label).strip("-")
print(slug or "mlx-model")
PY
)"
RUN_DIR="$OUTPUT_ROOT/$TIMESTAMP-$RUN_SLUG"
PROMPT_ARTIFACT_ROOT="$RUN_DIR/prompts"
RESULT_JSON="$RUN_DIR/result.json"
COMMAND_LOG="$RUN_DIR/command.log"
COMMAND_TXT="$RUN_DIR/command.txt"

mkdir -p "$RUN_DIR" "$PROMPT_ARTIFACT_ROOT"

echo "AX Engine MLX benchmark reproduction"
echo "  model_dir: $MODEL_DIR"
echo "  output: $RUN_DIR"

echo "Collecting readiness report..."
cargo run -p ax-engine-bench -- doctor --json > "$RUN_DIR/doctor.json"
cargo run -p ax-engine-bench -- doctor > "$RUN_DIR/doctor.txt"

echo "Building release server binary..."
cargo build -p ax-engine-server --release

BENCH_CMD=(
    "$PYTHON_BIN"
    "scripts/bench_mlx_inference_stack.py"
    "--model-dir" "$MODEL_DIR"
    "--prompt-tokens" "$PROMPT_TOKENS"
    "--generation-tokens" "$GENERATION_TOKENS"
    "--repetitions" "$REPETITIONS"
    "--cooldown" "$COOLDOWN"
    "--prefill-step-size" "$PREFILL_STEP_SIZE"
    "--prompt-artifact-root" "$PROMPT_ARTIFACT_ROOT"
    "--output" "$RESULT_JSON"
)

if [[ "$AX_COMPARE_POLICIES" -eq 1 ]]; then
    BENCH_CMD+=("--ax-compare-policies")
else
    BENCH_CMD+=("--ax-direct")
fi

if [[ -n "$MLX_SWIFT_LM_COMMAND" ]]; then
    BENCH_CMD+=("--mlx-swift-lm-command" "$MLX_SWIFT_LM_COMMAND")
fi

printf '%q ' "${BENCH_CMD[@]}" > "$COMMAND_TXT"
printf '\n' >> "$COMMAND_TXT"

{
    echo "timestamp_utc=$TIMESTAMP"
    echo "repo=$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "model_dir=$MODEL_DIR"
    echo "prompt_tokens=$PROMPT_TOKENS"
    echo "generation_tokens=$GENERATION_TOKENS"
    echo "repetitions=$REPETITIONS"
    echo "cooldown=$COOLDOWN"
    echo "prefill_step_size=$PREFILL_STEP_SIZE"
    echo
    if command -v sw_vers >/dev/null 2>&1; then
        sw_vers
    fi
    uname -a
    sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    sysctl -n hw.memsize 2>/dev/null || true
} > "$RUN_DIR/environment.txt"

echo "Running benchmark..."
"${BENCH_CMD[@]}" 2>&1 | tee "$COMMAND_LOG"

cat > "$RUN_DIR/README.md" <<EOF
# AX Engine MLX Benchmark Reproduction

Generated: $TIMESTAMP

- Model directory: \`$MODEL_DIR\`
- Result JSON: \`result.json\`
- Prompt artifacts: \`prompts/\`
- Readiness: \`doctor.json\` and \`doctor.txt\`
- Command: \`command.txt\`
- Log: \`command.log\`
- Environment: \`environment.txt\`

Compare this bundle against published artifacts only when the model artifact,
prompt shape, runtime route, host class, and benchmark command are explicit.
EOF

echo
echo "Benchmark bundle written to:"
echo "  $RUN_DIR"
