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

verify_wheel_member() {
    local wheel="$1"
    local member="$2"
    python3 - "$wheel" "$member" <<'PY'
import sys
import zipfile

wheel, member = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(wheel) as zf:
    try:
        info = zf.getinfo(member)
    except KeyError:
        raise SystemExit(f"error: wheel missing required member: {member}")
    if info.file_size == 0:
        raise SystemExit(f"error: wheel member is empty: {member}")
    print(f"    verified: {member} ({info.file_size} bytes)")
PY
}

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

# ── 3b. Stage mlx.metallib so the wheel ships MLX's Metal shader library. ──
# pyproject.toml includes python/ax_engine.dylibs/mlx.metallib in the wheel and
# the guard below verifies it post-delocate. Copy it from the MLX install so the
# build is self-contained on CI runners as well as local machines.
echo "==> Staging mlx.metallib into the wheel data dir..."
MLX_METALLIB="${MLX_METALLIB:-$(brew --prefix mlx 2>/dev/null)/lib/mlx.metallib}"
if [[ ! -f "$MLX_METALLIB" ]]; then
    echo "error: mlx.metallib not found at '$MLX_METALLIB'"
    echo "       install MLX (e.g. 'brew install mlx') or set MLX_METALLIB to its path"
    exit 1
fi
METALLIB_DIR="$REPO_ROOT/python/ax_engine.dylibs"
mkdir -p "$METALLIB_DIR"
cp -f "$MLX_METALLIB" "$METALLIB_DIR/mlx.metallib"
echo "    staged: $METALLIB_DIR/mlx.metallib ($(wc -c < "$METALLIB_DIR/mlx.metallib" | tr -d ' ') bytes)"

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

# MLX resolves mlx.metallib relative to libmlx.dylib's current binary dir.
# Keep this guard after delocate so a release cannot silently regress native pip inference.
echo "==> Verifying bundled MLX runtime assets..."
verify_wheel_member "$DELOCATED" "ax_engine.dylibs/mlx.metallib"

# ── 6. Optionally publish ──────────────────────────────────────────────────
if [[ "${1:-}" == "--publish" ]]; then
    echo "==> Publishing to PyPI..."
    maturin upload "$DELOCATED"
fi

echo ""
echo "Done. Wheel ready at: $DELOCATED"
