#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

cleanup() {
    ax_rm_rf "$ROOT_DIR/scripts/__pycache__"
}

trap cleanup EXIT

cd "$ROOT_DIR"

bash -n scripts/*.sh scripts/lib/common.sh
"$PYTHON_BIN" -m py_compile \
  scripts/bench_mlx_inference_stack.py \
  scripts/test_bench_mlx_inference_stack.py \
  scripts/diagnose_server_rss.py
bash scripts/check-bench-inference-stack.sh
