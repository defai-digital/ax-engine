#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
OUTPUT_DIR="${AX_METAL_OUTPUT_DIR:-$ROOT_DIR/build/metal}"
MANIFEST_PATH="${AX_METAL_MANIFEST_PATH:-$ROOT_DIR/metal/phase1-kernels.json}"

cd "$ROOT_DIR"

cargo run --quiet -p ax-engine-bench -- metal-build \
  --manifest "$MANIFEST_PATH" \
  --output-dir "$OUTPUT_DIR"
