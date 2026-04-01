#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

exec python3 "$REPO_DIR/benchmarks/run_apple_to_apple.py" "$@"
