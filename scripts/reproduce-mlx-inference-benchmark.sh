#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

MODEL_DIR="${AX_ENGINE_MLX_MODEL_DIR:-}"
MODEL_REPO_ID="${AX_ENGINE_MLX_MODEL_REPO_ID:-mlx-community/Qwen3.5-9B-MLX-4bit}"
HF_CACHE_ROOT="${AX_ENGINE_HF_CACHE_ROOT:-}"
OUTPUT_ROOT="$ROOT_DIR/benchmarks/community-results/local"
PROMPT_TOKENS="${AX_BENCH_PROMPT_TOKENS:-128,512,2048}"
GENERATION_TOKENS="${AX_BENCH_GENERATION_TOKENS:-128}"
REPETITIONS="${AX_BENCH_REPETITIONS:-5}"
COOLDOWN="${AX_BENCH_COOLDOWN:-15}"
PREFILL_STEP_SIZE="${AX_BENCH_PREFILL_STEP_SIZE:-2048}"
RUN_LABEL="${AX_BENCH_RUN_LABEL:-}"
AX_COMPARE_POLICIES=1

usage() {
    cat <<'EOF'
Usage:
  scripts/reproduce-mlx-inference-benchmark.sh [--model-repo-id mlx-community/Qwen3.5-9B-MLX-4bit] [options]

Options:
  --model-repo-id REPO          Hugging Face repo id resolved from the local cache when --model-dir is omitted.
  --model-dir PATH              Local MLX model artifact directory. Overrides Hugging Face cache resolution.
  --hf-cache-root PATH          Hugging Face Hub cache root. Defaults to HF_HUB_CACHE, HF_HOME, XDG_CACHE_HOME, or ~/.cache.
  --output-root PATH            Output root. Defaults to benchmarks/community-results/local.
  --prompt-tokens LIST          Comma-separated prompt lengths. Defaults to 128,512,2048.
  --generation-tokens N         Generated token count. Defaults to 128.
  --repetitions N               Timed repetitions. Defaults to 5.
  --cooldown SECONDS            Cooldown between repetitions. Defaults to 15.
  --prefill-step-size N         MLX prefill step size. Defaults to 2048.
  --run-label LABEL             Human-readable output directory label.
  --direct-only                 Emit only the direct same-policy AX row.
  -h, --help                    Show this help.

The script records doctor output, command logs, prompt-token artifacts, and raw
benchmark JSON. Model artifacts must already be AX-ready in the Hugging Face
cache: config.json, model-manifest.json, and safetensors. It reproduces the
benchmark procedure, not identical throughput across different Apple Silicon
chips, memory sizes, or thermal conditions.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-repo-id)
            MODEL_REPO_ID="${2:?missing value for --model-repo-id}"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="${2:?missing value for --model-dir}"
            shift 2
            ;;
        --hf-cache-root)
            HF_CACHE_ROOT="${2:?missing value for --hf-cache-root}"
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

MODEL_DIR_SOURCE="explicit"
if [[ -z "$MODEL_DIR" ]]; then
    MODEL_DIR_SOURCE="huggingface_cache"
    MODEL_DIR="$("$PYTHON_BIN" - "$MODEL_REPO_ID" "$HF_CACHE_ROOT" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

repo_id = sys.argv[1]
explicit_root = sys.argv[2] or None

def roots() -> list[Path]:
    if explicit_root:
        return [Path(explicit_root).expanduser()]
    out: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        out.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        out.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    if os.environ.get("XDG_CACHE_HOME"):
        out.append(Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub")
    out.append(Path.home() / ".cache" / "huggingface" / "hub")
    deduped: list[Path] = []
    for root in out:
        if root not in deduped:
            deduped.append(root)
    return deduped

repo_cache_name = "models--" + repo_id.replace("/", "--")
candidates: list[Path] = []
for root in roots():
    repo_cache = root / repo_cache_name
    refs_main = repo_cache / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text().strip()
        snapshot = repo_cache / "snapshots" / revision
        if revision and snapshot.is_dir():
            print(snapshot)
            raise SystemExit(0)
    snapshots = repo_cache / "snapshots"
    if snapshots.is_dir():
        candidates.extend(path for path in snapshots.iterdir() if path.is_dir())

if candidates:
    print(max(candidates, key=lambda path: path.stat().st_mtime))
    raise SystemExit(0)

print(
    f"ERROR: no Hugging Face cache snapshot found for {repo_id}; run "
    f"`python3 scripts/download_model.py {repo_id}` first or pass --model-dir.",
    file=sys.stderr,
)
raise SystemExit(2)
PY
)"
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model directory does not exist: $MODEL_DIR" >&2
    exit 2
fi

missing_model_files=()
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    missing_model_files+=("config.json")
fi
if [[ ! -f "$MODEL_DIR/model-manifest.json" ]]; then
    missing_model_files+=("model-manifest.json")
fi
safetensor_files=("$MODEL_DIR"/*.safetensors)
if [[ "${#safetensor_files[@]}" -eq 0 ]]; then
    missing_model_files+=("*.safetensors")
fi
if [[ "${#missing_model_files[@]}" -gt 0 ]]; then
    echo "ERROR: model directory is not AX-ready: $MODEL_DIR" >&2
    echo "Missing: ${missing_model_files[*]}" >&2
    echo "Run: ax-engine-bench generate-manifest \"$MODEL_DIR\"" >&2
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
echo "  model_repo_id: $MODEL_REPO_ID"
echo "  model_dir_source: $MODEL_DIR_SOURCE"
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
    "--model-repo-id" "$MODEL_REPO_ID"
    "--model-dir" "$MODEL_DIR"
    "--prompt-tokens" "$PROMPT_TOKENS"
    "--generation-tokens" "$GENERATION_TOKENS"
    "--repetitions" "$REPETITIONS"
    "--cooldown" "$COOLDOWN"
    "--prefill-step-size" "$PREFILL_STEP_SIZE"
    "--prompt-artifact-root" "$PROMPT_ARTIFACT_ROOT"
    "--output" "$RESULT_JSON"
)

if [[ "$MODEL_DIR_SOURCE" == "huggingface_cache" ]]; then
    BENCH_CMD+=("--model" "$MODEL_REPO_ID")
fi

if [[ "$AX_COMPARE_POLICIES" -eq 1 ]]; then
    BENCH_CMD+=("--ax-compare-policies")
else
    BENCH_CMD+=("--ax-direct")
fi

printf '%q ' "${BENCH_CMD[@]}" > "$COMMAND_TXT"
printf '\n' >> "$COMMAND_TXT"

{
    echo "timestamp_utc=$TIMESTAMP"
    echo "repo=$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "model_repo_id=$MODEL_REPO_ID"
    echo "model_dir_source=$MODEL_DIR_SOURCE"
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
- Model repo id: \`$MODEL_REPO_ID\`
- Model directory source: \`$MODEL_DIR_SOURCE\`
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
