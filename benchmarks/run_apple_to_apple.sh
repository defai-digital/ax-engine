#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/Users/akiralam/code/ax-engine}"

exec python3 "$REPO_DIR/benchmarks/run_apple_to_apple.py" "$@"
