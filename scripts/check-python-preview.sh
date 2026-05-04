#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"
VENV_DIR="$(ax_tmp_dir ax-engine-py-check)"

cleanup() {
    ax_rm_rf "$VENV_DIR"
}

trap cleanup EXIT

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --quiet --upgrade pip
python -m pip install --quiet "maturin>=1.7,<2"

cd "$ROOT_DIR"

python -m maturin develop --quiet

python "$ROOT_DIR/examples/python/basic.py"
python "$ROOT_DIR/examples/python/stepwise.py"
python "$ROOT_DIR/examples/python/streaming.py"

AX_ENGINE_RUN_INSTALLED_TESTS=1 python -m unittest discover -s "$ROOT_DIR/python/tests" -v
