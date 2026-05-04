#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${AX_METAL_OUTPUT_DIR:-$ROOT_DIR/build/metal}"
MANIFEST_PATH="${AX_METAL_MANIFEST_PATH:-$ROOT_DIR/metal/phase1-kernels.json}"

cd "$ROOT_DIR"

cargo run --quiet -p ax-engine-bench -- metal-build \
  --manifest "$MANIFEST_PATH" \
  --output-dir "$OUTPUT_DIR"
