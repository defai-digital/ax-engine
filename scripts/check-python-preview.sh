#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ax-engine-py-check.XXXXXX")"

cleanup() {
    rm -rf "$VENV_DIR"
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
