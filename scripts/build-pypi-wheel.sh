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
#   MLX_LIB_DIR          — path to dir containing libmlxc.dylib
#                           (default: resolved from `brew --prefix mlx-c`)
#   MLX_INCLUDE_DIR      — path to dir containing mlx/c/mlx.h
#                           (default: <MLX_LIB_DIR>/../include)
#   MLX_CPP_INCLUDE_DIR  — path to dir containing mlx/fast.h (C++ headers)
#                           (default: same as MLX_INCLUDE_DIR; override when
#                            mlx and mlx-c are installed under separate prefixes)
#
# Usage:
#   bash scripts/build-pypi-wheel.sh            # build only
#   bash scripts/build-pypi-wheel.sh --publish  # build + upload to PyPI
set -euo pipefail

WHEEL_OUT="target/wheels"
DELOCATED_OUT="${WHEEL_OUT}/delocated"
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

# ── 2. Clean output dirs so stale wheels can't be mistaken for new ones ───
rm -f "${WHEEL_OUT}"/ax_engine-*.whl
rm -f "${DELOCATED_OUT}"/ax_engine-*.whl
mkdir -p "$WHEEL_OUT" "$DELOCATED_OUT"

# ── 3. Build ax-engine-server binary and stage it into the wheel data dir ─
echo "==> Building ax-engine-server binary..."
cargo build --release -p ax-engine-server

SERVER_BIN="$REPO_ROOT/target/release/ax-engine-server"
if [[ ! -f "$SERVER_BIN" ]]; then
    echo "error: expected binary at $SERVER_BIN after cargo build"
    exit 1
fi

SCRIPTS_DIR="$REPO_ROOT/python/ax_engine.data/scripts"
mkdir -p "$SCRIPTS_DIR"
cp "$SERVER_BIN" "$SCRIPTS_DIR/ax-engine-server"
chmod +x "$SCRIPTS_DIR/ax-engine-server"
echo "    staged: $SCRIPTS_DIR/ax-engine-server"

# ── 4. Build the wheel ─────────────────────────────────────────────────────
echo "==> Building wheel (release, stripped)..."
maturin build --release --strip --out "$WHEEL_OUT"

# Use a glob expansion instead of ls+sort so we get exactly what was just built.
# After the clean above there should be exactly one match.
wheels=("${WHEEL_OUT}"/ax_engine-*.whl)
if [[ ${#wheels[@]} -ne 1 || ! -f "${wheels[0]}" ]]; then
    echo "error: expected exactly one wheel in $WHEEL_OUT after build, found: ${wheels[*]}"
    exit 1
fi
WHEEL="${wheels[0]}"
echo "    built: $WHEEL"

# ── 5. Delocalize — bundle libmlxc + libmlx into the wheel ────────────────
echo "==> Delocalizing wheel (bundling dylibs)..."
# --require-archs ensures we only accept arm64 (Apple Silicon only)
delocate-wheel --require-archs arm64 -w "$DELOCATED_OUT" "$WHEEL"

delocated_wheels=("${DELOCATED_OUT}"/ax_engine-*.whl)
if [[ ${#delocated_wheels[@]} -ne 1 || ! -f "${delocated_wheels[0]}" ]]; then
    echo "error: delocate-wheel did not produce a wheel in $DELOCATED_OUT"
    exit 1
fi
DELOCATED="${delocated_wheels[0]}"
echo "    delocated: $DELOCATED"

echo "==> Bundled dependencies:"
delocate-listdeps "$DELOCATED"

# ── 6. Optionally publish ──────────────────────────────────────────────────
if [[ "${1:-}" == "--publish" ]]; then
    echo "==> Publishing to PyPI..."
    maturin upload "$DELOCATED"
fi

echo ""
echo "Done. Wheel ready at: $DELOCATED"
