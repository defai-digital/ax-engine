#!/usr/bin/env bash
# Usage: bash scripts/cargo-pinned.sh test --workspace
set -euo pipefail
AX_PIN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$AX_PIN_SCRIPT_DIR/lib/rust-toolchain.sh"
ax_use_pinned_rust "$(cd "$AX_PIN_SCRIPT_DIR/.." && pwd)"
exec "$AX_CARGO_BIN" "$@"
