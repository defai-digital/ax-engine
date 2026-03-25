#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${AX_ENGINE_PY_VENV:-/tmp/ax-engine-py-smoke-venv}"
MODEL_PATH="${1:-${AX_ENGINE_PY_MODEL:-$ROOT_DIR/models/Qwen3-8B-Q4_K_M.gguf}}"
BACKEND="${AX_ENGINE_PY_BACKEND:-cpu}"
GENERATE="${AX_ENGINE_PY_GENERATE:-0}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model not found: $MODEL_PATH" >&2
  exit 1
fi

python3 -m venv "$VENV_DIR"

env -u CONDA_PREFIX \
  VIRTUAL_ENV="$VENV_DIR" \
  PATH="$VENV_DIR/bin:$PATH" \
  maturin develop --manifest-path "$ROOT_DIR/crates/ax-engine-py/Cargo.toml"

SMOKE_ARGS=()
if [[ "$GENERATE" == "1" ]]; then
  SMOKE_ARGS+=(--generate)
fi

env -u CONDA_PREFIX \
  VIRTUAL_ENV="$VENV_DIR" \
  PATH="$VENV_DIR/bin:$PATH" \
  python "$ROOT_DIR/examples/python/smoke.py" "$MODEL_PATH" --backend "$BACKEND" ${SMOKE_ARGS+"${SMOKE_ARGS[@]}"}
