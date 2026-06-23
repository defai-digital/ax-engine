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

# ── macOS floor ────────────────────────────────────────────────────────────
# The bundled Homebrew MLX dylibs are built minos 26.0, so the wheel must declare
# a macOS 26 (Tahoe) floor. Exporting the deployment target makes maturin tag the
# wheel macosx_26_0_arm64 — pip then refuses it on macOS < 26 instead of installing
# a wheel whose libs fail to load at runtime. Override only if the MLX libs' minos
# changes (and update the docs' macOS requirement to match).
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-26.0}"
EXPECTED_PLAT_TAG="macosx_${MACOSX_DEPLOYMENT_TARGET//./_}_arm64"

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

# ── 3. Build ax-engine-server and ax-engine-bench; stage into wheel data dir ─
echo "==> Building ax-engine-server and ax-engine-bench binaries..."
cargo build --release -p ax-engine-server -p ax-engine-bench

SERVER_BIN="$REPO_ROOT/target/release/ax-engine-server"
if [[ ! -f "$SERVER_BIN" ]]; then
    echo "error: expected binary at $SERVER_BIN after cargo build"
    exit 1
fi

BENCH_BIN="$REPO_ROOT/target/release/ax-engine-bench"
if [[ ! -f "$BENCH_BIN" ]]; then
    echo "error: expected binary at $BENCH_BIN after cargo build"
    exit 1
fi

# The native `ax-engine` binary (second bin of the ax-engine-bench crate) hosts
# the `ax-engine tui` subcommand, which the Python CLI execs.
NATIVE_BIN="$REPO_ROOT/target/release/ax-engine"
if [[ ! -f "$NATIVE_BIN" ]]; then
    echo "error: expected binary at $NATIVE_BIN after cargo build"
    exit 1
fi

SCRIPTS_DIR="$REPO_ROOT/python/ax_engine/_bin"
mkdir -p "$SCRIPTS_DIR"
cp "$SERVER_BIN" "$SCRIPTS_DIR/ax-engine-server"
chmod +x "$SCRIPTS_DIR/ax-engine-server"
echo "    staged: $SCRIPTS_DIR/ax-engine-server"
cp "$BENCH_BIN" "$SCRIPTS_DIR/ax-engine-bench"
chmod +x "$SCRIPTS_DIR/ax-engine-bench"
echo "    staged: $SCRIPTS_DIR/ax-engine-bench"
cp "$NATIVE_BIN" "$SCRIPTS_DIR/ax-engine"
chmod +x "$SCRIPTS_DIR/ax-engine"
echo "    staged: $SCRIPTS_DIR/ax-engine"

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
echo "==> Building wheel (release, stripped, target $EXPECTED_PLAT_TAG)..."
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
verify_wheel_member "$DELOCATED" "ax_engine/_bin/ax-engine-server"
verify_wheel_member "$DELOCATED" "ax_engine/_bin/ax-engine-bench"
verify_wheel_member "$DELOCATED" "ax_engine/_bin/ax-engine"

# ── 5b. Guard the platform tag ─────────────────────────────────────────────
# Refuse to ship a wheel whose tag understates the macOS floor. The bundled MLX
# dylibs are minos 26.0; a lower tag (e.g. macosx_14_0) lets pip install on
# macOS < 26 where those libs fail to load at runtime ("installs but doesn't run").
echo "==> Verifying wheel platform tag..."
if [[ "$(basename "$DELOCATED")" != *"-${EXPECTED_PLAT_TAG}.whl" ]]; then
    echo "error: wheel is not tagged ${EXPECTED_PLAT_TAG}: $(basename "$DELOCATED")"
    echo "       the bundled MLX libs require macOS ${MACOSX_DEPLOYMENT_TARGET};"
    echo "       refusing to publish a tag that lies about the macOS floor"
    exit 1
fi
echo "    verified platform tag: ${EXPECTED_PLAT_TAG}"

# ── 6. Optionally publish ──────────────────────────────────────────────────
if [[ "${1:-}" == "--publish" ]]; then
    echo "==> Publishing to PyPI..."
    maturin upload "$DELOCATED"
fi

echo ""
echo "Done. Wheel ready at: $DELOCATED"
