#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

MODEL_DIR="${AX_MLX_P2_MODEL_DIR:-}"
OUTPUT_ROOT="$ROOT_DIR/benchmarks/results/mlx-inference/p2-latency-runs"
CONTEXT_TOKENS="${AX_MLX_P2_CONTEXT_TOKENS:-8192}"
STARTUP_GENERATION_TOKENS="${AX_MLX_P2_STARTUP_GENERATION_TOKENS:-128}"
CONCURRENT_GENERATION_TOKENS="${AX_MLX_P2_CONCURRENT_GENERATION_TOKENS:-1}"
CONCURRENCY_LEVELS="${AX_MLX_P2_CONCURRENCY_LEVELS:-1,2,4}"
REPETITIONS="${AX_MLX_P2_REPETITIONS:-3}"
COOLDOWN="${AX_MLX_P2_COOLDOWN:-5}"
MODEL_ID="${AX_MLX_P2_MODEL_ID:-}"
RUN_LABEL="${AX_MLX_P2_RUN_LABEL:-}"
HOST_LABEL="${AX_MLX_P2_HOST_LABEL:-unknown_apple_silicon}"
AXENGINE_PORT="${AX_MLX_P2_AXENGINE_PORT:-8092}"
SKIP_STARTUP=0
SKIP_CONCURRENT=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  scripts/run-mlx-p2-latency-artifacts.sh --model-dir /path/to/mlx-model [options]

Options:
  --model-dir PATH                  Local MLX model artifact directory. Required unless AX_MLX_P2_MODEL_DIR is set.
  --output-root PATH                Output root. Defaults to benchmarks/results/mlx-inference/p2-latency-runs.
  --context-tokens N                Prompt/context token count. Defaults to 8192.
  --startup-generation-tokens N     Generated token count for cold/warm startup rows. Defaults to 128.
  --concurrent-generation-tokens N  Generated token count for concurrent-prefill rows. Defaults to 1.
  --concurrency-levels LIST         Comma-separated concurrency levels. Defaults to 1,2,4.
  --repetitions N                   Timed repetitions. Defaults to 3.
  --cooldown SECONDS                Cooldown between repetitions. Defaults to 5.
  --model-id ID                     Artifact model id. Defaults to the model directory path used by the runner.
  --run-label LABEL                 Human-readable output directory label.
  --host-label LABEL                Host label recorded in artifacts. Defaults to unknown_apple_silicon.
  --axengine-port N                 AX server port. Defaults to 8092.
  --skip-startup                    Only capture concurrent-prefill artifact/report section.
  --skip-concurrent                 Only capture startup artifact/report section.
  --dry-run                         Print planned command without building or starting the server.
  -h, --help                        Show this help.

The runner captures P2 latency evidence with direct AX MLX policy:
startup-latency.json, concurrent-prefill.json, and p2-latency.md. These are
server-path artifacts for cold/warm and concurrency claims, not raw throughput
or n-gram decode acceleration evidence.
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
        --startup-generation-tokens)
            STARTUP_GENERATION_TOKENS="${2:?missing value for --startup-generation-tokens}"
            shift 2
            ;;
        --concurrent-generation-tokens)
            CONCURRENT_GENERATION_TOKENS="${2:?missing value for --concurrent-generation-tokens}"
            shift 2
            ;;
        --concurrency-levels)
            CONCURRENCY_LEVELS="${2:?missing value for --concurrency-levels}"
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
        --model-id)
            MODEL_ID="${2:?missing value for --model-id}"
            shift 2
            ;;
        --run-label)
            RUN_LABEL="${2:?missing value for --run-label}"
            shift 2
            ;;
        --host-label)
            HOST_LABEL="${2:?missing value for --host-label}"
            shift 2
            ;;
        --axengine-port)
            AXENGINE_PORT="${2:?missing value for --axengine-port}"
            shift 2
            ;;
        --skip-startup)
            SKIP_STARTUP=1
            shift
            ;;
        --skip-concurrent)
            SKIP_CONCURRENT=1
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
    echo "ERROR: --model-dir is required unless AX_MLX_P2_MODEL_DIR is set." >&2
    usage >&2
    exit 2
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model directory does not exist: $MODEL_DIR" >&2
    exit 2
fi

if [[ "$SKIP_STARTUP" -eq 1 && "$SKIP_CONCURRENT" -eq 1 ]]; then
    echo "ERROR: --skip-startup and --skip-concurrent cannot both be set." >&2
    exit 2
fi

if [[ -z "$RUN_LABEL" ]]; then
    RUN_LABEL="$(basename "$MODEL_DIR")-$(date +%Y%m%d-%H%M%S)"
fi

OUTPUT_DIR="$OUTPUT_ROOT/$RUN_LABEL"

cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/scripts/run_mlx_p2_latency_artifacts.py"
    --model-dir "$MODEL_DIR"
    --output-dir "$OUTPUT_DIR"
    --context-tokens "$CONTEXT_TOKENS"
    --startup-generation-tokens "$STARTUP_GENERATION_TOKENS"
    --concurrent-generation-tokens "$CONCURRENT_GENERATION_TOKENS"
    --concurrency-levels "$CONCURRENCY_LEVELS"
    --repetitions "$REPETITIONS"
    --cooldown "$COOLDOWN"
    --host-label "$HOST_LABEL"
    --axengine-port "$AXENGINE_PORT"
)

if [[ -n "$MODEL_ID" ]]; then
    cmd+=(--model-id "$MODEL_ID")
fi
if [[ "$SKIP_STARTUP" -eq 1 ]]; then
    cmd+=(--skip-startup)
fi
if [[ "$SKIP_CONCURRENT" -eq 1 ]]; then
    cmd+=(--skip-concurrent)
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
    cmd+=(--dry-run)
fi

echo "Output directory: $OUTPUT_DIR"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
    "${cmd[@]}"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
cargo build -p ax-engine-server --release
"${cmd[@]}"

if [[ "$SKIP_STARTUP" -eq 0 ]]; then
    echo "Startup artifact: $OUTPUT_DIR/startup-latency.json"
fi
if [[ "$SKIP_CONCURRENT" -eq 0 ]]; then
    echo "Concurrent artifact: $OUTPUT_DIR/concurrent-prefill.json"
fi
echo "P2 latency report: $OUTPUT_DIR/p2-latency.md"
