#!/usr/bin/env bash
# Build the ax-engine PyPI wheel for macOS arm64.
#
# The resulting wheel is self-contained: libmlxc.dylib and libmlx.dylib are
# bundled by delocate so the user does not need Homebrew at install time.
#
# Prerequisites (one-time):
#   pip install maturin delocate
#   brew install mlx mlx-c       # provides libmlx and libmlxc for the build
#
# Environment variables (optional):
#   MLX_LIB_DIR      — path to dir containing libmlxc.dylib / libmlx.dylib
#                       (default: resolved from `brew --prefix mlx-c`)
#   MLX_INCLUDE_DIR  — path to dir containing mlx/c/mlx.h
#                       (default: <MLX_LIB_DIR>/../include)
#
# Usage:
#   bash scripts/build-pypi-wheel.sh            # build only
#   bash scripts/build-pypi-wheel.sh --publish  # build + upload to PyPI
set -euo pipefail

WHEEL_OUT="target/wheels"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

# ── 1. Verify prerequisites ────────────────────────────────────────────────
for tool in maturin delocate-wheel delocate-listdeps; do
    if ! command -v "$tool" &>/dev/null; then
        echo "error: '$tool' not found — install with: pip install maturin delocate"
        exit 1
    fi
done

# ── 2. Build the wheel ─────────────────────────────────────────────────────
echo "==> Building wheel (release, stripped)..."
maturin build --release --strip --out "$WHEEL_OUT"

WHEEL="$(ls "${WHEEL_OUT}"/ax_engine-*.whl | sort -V | tail -1)"
if [[ -z "$WHEEL" ]]; then
    echo "error: no wheel found in $WHEEL_OUT after maturin build"
    exit 1
fi
echo "    built: $WHEEL"

# ── 3. Delocalize — bundle libmlxc + libmlx into the wheel ────────────────
echo "==> Delocalizing wheel (bundling dylibs)..."
# --require-archs ensures we only accept arm64 (Apple Silicon only)
delocate-wheel --require-archs arm64 -w "${WHEEL_OUT}/delocated" "$WHEEL"

DELOCATED="$(ls "${WHEEL_OUT}/delocated"/ax_engine-*.whl | sort -V | tail -1)"
echo "    delocated: $DELOCATED"

echo "==> Bundled dependencies:"
delocate-listdeps "$DELOCATED"

# ── 4. Optionally publish ──────────────────────────────────────────────────
if [[ "${1:-}" == "--publish" ]]; then
    echo "==> Publishing to PyPI..."
    maturin upload "$DELOCATED"
fi

echo ""
echo "Done. Wheel ready at: $DELOCATED"
