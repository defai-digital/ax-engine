#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

MODEL_DIR="${AX_TURBOQUANT_MODEL_DIR:-}"
OUTPUT_ROOT="$ROOT_DIR/benchmarks/results/turboquant/quality-runs"
CONTEXT_TOKENS="${AX_TURBOQUANT_CONTEXT_TOKENS:-8192}"
GENERATION_TOKENS="${AX_TURBOQUANT_GENERATION_TOKENS:-256}"
REPETITIONS="${AX_TURBOQUANT_REPETITIONS:-3}"
COOLDOWN="${AX_TURBOQUANT_COOLDOWN:-3}"
PREFILL_STEP_SIZE="${AX_TURBOQUANT_PREFILL_STEP_SIZE:-2048}"
MODEL_ID="${AX_TURBOQUANT_MODEL_ID:-}"
MODEL_FAMILY="${AX_TURBOQUANT_MODEL_FAMILY:-}"
MODEL_REVISION="${AX_TURBOQUANT_MODEL_REVISION:-local-quality-run}"
HEAD_DIM="${AX_TURBOQUANT_HEAD_DIM:-}"
RUN_LABEL="${AX_TURBOQUANT_RUN_LABEL:-}"
HOT_WINDOW_TOKENS="${AX_TURBOQUANT_HOT_WINDOW_TOKENS:-}"
MIN_CONTEXT_TOKENS="${AX_TURBOQUANT_MIN_CONTEXT_TOKENS:-}"
ALLOW_SHORT_OUTPUT=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  scripts/run-turboquant-quality-artifact.sh --model-dir /path/to/mlx-model [options]

Options:
  --model-dir PATH              Local MLX model artifact directory. Required unless AX_TURBOQUANT_MODEL_DIR is set.
  --output-root PATH            Output root. Defaults to benchmarks/results/turboquant/quality-runs.
  --context-tokens N            Prompt token count. Defaults to 8192.
  --generation-tokens N         Generated token count. Defaults to 256.
  --repetitions N               Timed AX repetitions. Defaults to 3; must be at least 2 for promotion readiness.
  --cooldown SECONDS            Cooldown between repetitions. Defaults to 3.
  --prefill-step-size N         MLX prefill step size. Defaults to 2048.
  --model-id ID                 Artifact model id. Defaults to model directory basename.
  --model-family FAMILY         Artifact model family. Defaults to model-manifest.json model_family.
  --model-revision REVISION     Artifact model revision. Defaults to local-quality-run.
  --head-dim N                  Artifact fused attention head dimension. Defaults to global_head_dim when present, else attention_head_dim.
  --run-label LABEL             Human-readable output directory label.
  --hot-window-tokens N         Pass fused experimental hot-window token count.
  --min-context-tokens N        Pass fused experimental minimum context token count.
  --allow-short-output          Allow early-stop output vectors shorter than --generation-tokens.
  --dry-run                     Print inferred metadata and planned commands without running them.
  -h, --help                    Show this help.

The runner builds a real-model TurboQuant quality artifact by running AX MLX
baseline and turboquant-fused-experimental candidate rows with output-token
capture, extracting same-shaped decode vectors, building quality metrics, and
validating the final artifact. Any failed promotion contract stops the run.
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
        --context-tokens)
            CONTEXT_TOKENS="${2:?missing value for --context-tokens}"
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
        --model-family)
            MODEL_FAMILY="${2:?missing value for --model-family}"
            shift 2
            ;;
        --model-revision)
            MODEL_REVISION="${2:?missing value for --model-revision}"
            shift 2
            ;;
        --head-dim)
            HEAD_DIM="${2:?missing value for --head-dim}"
            shift 2
            ;;
        --run-label)
            RUN_LABEL="${2:?missing value for --run-label}"
            shift 2
            ;;
        --hot-window-tokens)
            HOT_WINDOW_TOKENS="${2:?missing value for --hot-window-tokens}"
            shift 2
            ;;
        --min-context-tokens)
            MIN_CONTEXT_TOKENS="${2:?missing value for --min-context-tokens}"
            shift 2
            ;;
        --allow-short-output)
            ALLOW_SHORT_OUTPUT=1
            shift
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
    echo "ERROR: --model-dir is required unless AX_TURBOQUANT_MODEL_DIR is set." >&2
    usage >&2
    exit 2
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model directory does not exist: $MODEL_DIR" >&2
    exit 2
fi

IFS=$'\t' read -r INFERRED_MODEL_ID INFERRED_MODEL_FAMILY INFERRED_HEAD_DIM < <(
    "$PYTHON_BIN" - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
manifest = {}
config = {}
for path, target in (
    (model_dir / "model-manifest.json", "manifest"),
    (model_dir / "config.json", "config"),
):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        payload = {}
    if target == "manifest":
        manifest = payload
    else:
        config = payload

model_id = model_dir.name
family = manifest.get("model_family") or config.get("model_type") or ""
head_dim = (
    manifest.get("global_head_dim")
    or manifest.get("attention_head_dim")
    or config.get("global_head_dim")
    or config.get("attention_head_dim")
    or ""
)
print(f"{model_id}\t{family}\t{head_dim}")
PY
)

MODEL_ID="${MODEL_ID:-$INFERRED_MODEL_ID}"
MODEL_FAMILY="${MODEL_FAMILY:-$INFERRED_MODEL_FAMILY}"
HEAD_DIM="${HEAD_DIM:-$INFERRED_HEAD_DIM}"

if [[ -z "$MODEL_ID" || -z "$MODEL_FAMILY" || -z "$HEAD_DIM" ]]; then
    echo "ERROR: model-id, model-family, and head-dim must be provided or inferable." >&2
    exit 2
fi

if [[ ! "$REPETITIONS" =~ ^[0-9]+$ || "$REPETITIONS" -lt 2 ]]; then
    echo "ERROR: --repetitions must be an integer >= 2 for promotion readiness." >&2
    exit 2
fi

cd "$ROOT_DIR"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$RUN_LABEL" ]]; then
    RUN_LABEL="$MODEL_ID"
fi
RUN_SLUG="$("$PYTHON_BIN" - "$RUN_LABEL" <<'PY'
import re
import sys

label = sys.argv[1].strip().lower()
slug = re.sub(r"[^a-z0-9._-]+", "-", label).strip("-")
print(slug or "turboquant-quality")
PY
)"
RUN_DIR="$OUTPUT_ROOT/$TIMESTAMP-$RUN_SLUG"
PROMPT_ARTIFACT_ROOT="$RUN_DIR/prompts"
BASELINE_JSON="$RUN_DIR/baseline.json"
CANDIDATE_JSON="$RUN_DIR/candidate.json"
BASELINE_OUTPUTS_JSON="$RUN_DIR/baseline-decode-outputs.json"
CANDIDATE_OUTPUTS_JSON="$RUN_DIR/candidate-decode-outputs.json"
QUALITY_METRICS_JSON="$RUN_DIR/quality-metrics.json"
QUALITY_GATE_JSON="$RUN_DIR/quality-gate.json"
READINESS_JSON="$RUN_DIR/promotion-readiness.json"
COMMAND_LOG="$RUN_DIR/command.log"
COMMAND_TXT="$RUN_DIR/commands.txt"
PORT="$(ax_allocate_port)"

mkdir -p "$RUN_DIR" "$PROMPT_ARTIFACT_ROOT"

format_cmd() {
    printf '%q ' "$@"
}

run_cmd() {
    format_cmd "$@" >> "$COMMAND_TXT"
    printf '\n' >> "$COMMAND_TXT"
    "$@" 2>&1 | tee -a "$COMMAND_LOG"
}

echo "TurboQuant real-model quality artifact run"
echo "  model_dir: $MODEL_DIR"
echo "  model_id: $MODEL_ID"
echo "  model_family: $MODEL_FAMILY"
echo "  head_dim: $HEAD_DIM"
echo "  output: $RUN_DIR"

{
    echo "timestamp_utc=$TIMESTAMP"
    echo "repo=$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "model_dir=$MODEL_DIR"
    echo "model_id=$MODEL_ID"
    echo "model_family=$MODEL_FAMILY"
    echo "head_dim=$HEAD_DIM"
    echo "context_tokens=$CONTEXT_TOKENS"
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

BUILD_CMD=(cargo build -p ax-engine-server --release)

BASELINE_CMD=(
    "$PYTHON_BIN" scripts/bench_mlx_inference_stack.py
    --model-dir "$MODEL_DIR"
    --prompt-tokens "$CONTEXT_TOKENS"
    --generation-tokens "$GENERATION_TOKENS"
    --repetitions "$REPETITIONS"
    --cooldown "$COOLDOWN"
    --prefill-step-size "$PREFILL_STEP_SIZE"
    --prompt-artifact-root "$PROMPT_ARTIFACT_ROOT"
    --ax-direct
    --capture-output-token-ids
    --axengine-port "$PORT"
    --output "$BASELINE_JSON"
)

CANDIDATE_CMD=(
    "$PYTHON_BIN" scripts/bench_mlx_inference_stack.py
    --model-dir "$MODEL_DIR"
    --prompt-tokens "$CONTEXT_TOKENS"
    --generation-tokens "$GENERATION_TOKENS"
    --repetitions "$REPETITIONS"
    --cooldown "$COOLDOWN"
    --prefill-step-size "$PREFILL_STEP_SIZE"
    --prompt-artifact-root "$PROMPT_ARTIFACT_ROOT"
    --reuse-reference-results-from "$BASELINE_JSON"
    --ax-direct
    --experimental-mlx-kv-compression turboquant-fused-experimental
    --capture-output-token-ids
    --axengine-port "$PORT"
    --output "$CANDIDATE_JSON"
)
if [[ -n "$HOT_WINDOW_TOKENS" ]]; then
    CANDIDATE_CMD+=(--experimental-mlx-kv-compression-hot-window-tokens "$HOT_WINDOW_TOKENS")
fi
if [[ -n "$MIN_CONTEXT_TOKENS" ]]; then
    CANDIDATE_CMD+=(--experimental-mlx-kv-compression-min-context-tokens "$MIN_CONTEXT_TOKENS")
fi

DECODE_BASELINE_CMD=(
    "$PYTHON_BIN" scripts/build_turboquant_decode_outputs.py
    --benchmark "$BASELINE_JSON"
    --context-tokens "$CONTEXT_TOKENS"
    --generation-tokens "$GENERATION_TOKENS"
    --compression-mode disabled
    --output "$BASELINE_OUTPUTS_JSON"
)
DECODE_CANDIDATE_CMD=(
    "$PYTHON_BIN" scripts/build_turboquant_decode_outputs.py
    --benchmark "$CANDIDATE_JSON"
    --context-tokens "$CONTEXT_TOKENS"
    --generation-tokens "$GENERATION_TOKENS"
    --compression-mode turboquant-fused-experimental
    --output "$CANDIDATE_OUTPUTS_JSON"
)
if [[ "$ALLOW_SHORT_OUTPUT" -eq 1 ]]; then
    DECODE_BASELINE_CMD+=(--allow-short-output)
    DECODE_CANDIDATE_CMD+=(--allow-short-output)
fi

QUALITY_METRICS_CMD=(
    "$PYTHON_BIN" scripts/build_turboquant_quality_metrics.py
    --baseline-outputs "$BASELINE_OUTPUTS_JSON"
    --candidate-outputs "$CANDIDATE_OUTPUTS_JSON"
    --output "$QUALITY_METRICS_JSON"
)

QUALITY_ARTIFACT_CMD=(
    "$PYTHON_BIN" scripts/build_turboquant_quality_artifact.py
    --baseline-benchmark "$BASELINE_JSON"
    --candidate-benchmark "$CANDIDATE_JSON"
    --quality-metrics "$QUALITY_METRICS_JSON"
    --output "$QUALITY_GATE_JSON"
    --manifest "$ROOT_DIR/benchmarks/manifests/scenario/long_context_qwen_8k.json"
    --model-id "$MODEL_ID"
    --model-family "$MODEL_FAMILY"
    --model-revision "$MODEL_REVISION"
    --head-dim "$HEAD_DIM"
    --context-tokens "$CONTEXT_TOKENS"
    --generation-tokens "$GENERATION_TOKENS"
    --root "$ROOT_DIR"
)

CHECK_ARTIFACT_CMD=(
    "$PYTHON_BIN" scripts/check_turboquant_quality_artifact.py "$QUALITY_GATE_JSON"
)

READINESS_CMD=(
    "$PYTHON_BIN" scripts/check_turboquant_promotion_readiness.py
    --artifact "$QUALITY_GATE_JSON"
    --output "$READINESS_JSON"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
    {
        echo "dry_run=true"
        echo "run_dir=$RUN_DIR"
        echo
        echo "# Commands"
        format_cmd "${BUILD_CMD[@]}"; printf '\n'
        format_cmd "${BASELINE_CMD[@]}"; printf '\n'
        format_cmd "${CANDIDATE_CMD[@]}"; printf '\n'
        format_cmd "${DECODE_BASELINE_CMD[@]}"; printf '\n'
        format_cmd "${DECODE_CANDIDATE_CMD[@]}"; printf '\n'
        format_cmd "${QUALITY_METRICS_CMD[@]}"; printf '\n'
        format_cmd "${QUALITY_ARTIFACT_CMD[@]}"; printf '\n'
        format_cmd "${CHECK_ARTIFACT_CMD[@]}"; printf '\n'
        format_cmd "${READINESS_CMD[@]}"; printf '\n'
    } | tee "$COMMAND_TXT"
    echo
    echo "Dry run complete; commands written to:"
    echo "  $COMMAND_TXT"
    exit 0
fi

echo "Building release server binary..."
run_cmd "${BUILD_CMD[@]}"

echo "Running full-precision AX MLX baseline..."
run_cmd "${BASELINE_CMD[@]}"

echo "Running TurboQuant fused experimental AX MLX candidate..."
run_cmd "${CANDIDATE_CMD[@]}"

echo "Extracting captured decode output vectors..."
run_cmd "${DECODE_BASELINE_CMD[@]}"
run_cmd "${DECODE_CANDIDATE_CMD[@]}"

echo "Building quality metrics..."
run_cmd "${QUALITY_METRICS_CMD[@]}"

echo "Building and validating TurboQuant quality artifact..."
run_cmd "${QUALITY_ARTIFACT_CMD[@]}"

run_cmd "${CHECK_ARTIFACT_CMD[@]}"

echo "Writing promotion readiness report for this artifact..."
run_cmd "${READINESS_CMD[@]}"

cat > "$RUN_DIR/README.md" <<EOF
# TurboQuant Quality Artifact Run

Generated: $TIMESTAMP

- Model directory: \`$MODEL_DIR\`
- Model id: \`$MODEL_ID\`
- Model family: \`$MODEL_FAMILY\`
- Head dim: \`$HEAD_DIM\`
- Baseline benchmark: \`baseline.json\`
- Candidate benchmark: \`candidate.json\`
- Decode outputs: \`baseline-decode-outputs.json\`, \`candidate-decode-outputs.json\`
- Quality metrics: \`quality-metrics.json\`
- Quality gate artifact: \`quality-gate.json\`
- Promotion readiness: \`promotion-readiness.json\`
- Command log: \`command.log\`
- Commands: \`commands.txt\`
- Environment: \`environment.txt\`

This artifact is promotion evidence only if \`quality-gate.json\` validates and
\`promotion-readiness.json\` reports no blockers. Public support docs remain
experimental until that readiness report is clean.
EOF

echo
echo "TurboQuant quality artifact bundle written to:"
echo "  $RUN_DIR"
