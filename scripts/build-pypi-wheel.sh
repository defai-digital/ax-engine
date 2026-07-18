#!/usr/bin/env bash
# Build the ax-engine PyPI wheel for macOS arm64.
#
# The resulting wheel is self-contained: libmlx.dylib is
# bundled by delocate so the user does not need Homebrew at install time.
#
# Prerequisites (one-time):
#   pip install maturin delocate
#
# MLX source: this script pip-installs `mlx` itself (see step 0 below) and
# links/bundles from THAT build rather than Homebrew. Homebrew's `mlx`
# formula derives its build's MACOSX_DEPLOYMENT_TARGET from
# `MacOS.version.major.minor`, which structurally truncates to "26.0" on
# every macOS 26.x host (Homebrew stopped tracking minor OS versions after
# Big Sur). MLX's NAX (Neural Accelerator) GEMM/attention kernels require
# CMAKE_OSX_DEPLOYMENT_TARGET >= 26.2 (mlx PR #3622); anything below that
# silently compiles WITHOUT NAX — no build error, ~3-4x slower prefill, only
# discoverable by inspecting `otool -l libmlx.dylib`'s LC_BUILD_VERSION. The
# official PyPI `mlx` wheel is built by upstream's own CI with the correct
# target (verified: minos 26.2, and benchmarks on par with a from-source
# build that has the target set explicitly) — do not switch this back to
# `brew --prefix mlx` without re-verifying the Homebrew formula no longer has
# this gap.
#
# Environment variables (optional):
#   MLX_LIB_DIR          — path to dir containing libmlx.dylib
#                           (default: resolved from the pip-installed `mlx`)
#   MLX_INCLUDE_DIR      — path to dir containing mlx/*.h
#                           (default: <MLX_LIB_DIR>/../include)
#   MLX_METALLIB         — path to mlx.metallib
#                           (default: <MLX_LIB_DIR>/mlx.metallib)
#
# Usage:
#   bash scripts/build-pypi-wheel.sh            # build only
#   bash scripts/build-pypi-wheel.sh --publish  # build + upload to PyPI
set -euo pipefail

# True if $1 >= $2, comparing dotted-numeric versions (e.g. macOS minos
# strings) numerically rather than lexicographically.
version_ge() {
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | tail -1)" == "$1" ]]
}

WHEEL_OUT="target/wheels"
DELOCATED_OUT="${WHEEL_OUT}/delocated"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

# ── 0. Ensure a correctly-built MLX (pip wheel) is available ──────────────
# See the file header for why this must be pip's mlx, not Homebrew's.
echo "==> Ensuring a correctly-built MLX (pip wheel) is available..."
# Pinned in mlx.version (repo root); mlx-sys/build.rs enforces the same pin.
MLX_PIN="$(tr -d '[:space:]' < "$REPO_ROOT/mlx.version")"
python3 -m pip install --upgrade --quiet "mlx==${MLX_PIN}"
MLX_PIP_DIR="$(python3 -c 'import mlx, pathlib; print(pathlib.Path(list(mlx.__path__)[0]))')"
if [[ ! -f "$MLX_PIP_DIR/lib/libmlx.dylib" ]]; then
    echo "error: pip-installed mlx has no lib/libmlx.dylib at $MLX_PIP_DIR"
    exit 1
fi
export MLX_LIB_DIR="${MLX_LIB_DIR:-$MLX_PIP_DIR/lib}"
export MLX_INCLUDE_DIR="${MLX_INCLUDE_DIR:-$MLX_PIP_DIR/include}"
mlx_minos="$(otool -l "$MLX_LIB_DIR/libmlx.dylib" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')"
echo "    using MLX_LIB_DIR=$MLX_LIB_DIR (minos $mlx_minos)"
if ! version_ge "$mlx_minos" "26.2"; then
    echo "error: resolved MLX build has minos $mlx_minos (< 26.2) — NAX kernels are"
    echo "       silently disabled on this build (mlx PR #3622); refusing to ship a"
    echo "       regressed wheel. Set MLX_LIB_DIR to a correctly-built MLX, or check"
    echo "       whether the pip mlx wheel's own build regressed."
    exit 1
fi

# ── macOS floor ────────────────────────────────────────────────────────────
# The bundled pip MLX dylibs are built minos 26.2 (see step 0), so every
# compiled binary in this wheel must target macOS 26.2. Override only if the
# MLX libs' minos changes (and update the docs' macOS requirement to match).
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-26.2}"
# maturin's wheel platform tag comes from Python's own sysconfig.get_platform(),
# which reflects the interpreter's *compile-time* MACOSX_DEPLOYMENT_TARGET, not
# this shell's env var — on this host that's baked in as 26.0 (any
# Homebrew-built Python has the same MacOS.version.major.minor truncation as
# the mlx formula; see the file header). _PYTHON_HOST_PLATFORM is the
# standard override distutils/setuptools/maturin respect for exactly this;
# it makes maturin compile every binary with the correct deployment target
# even though the *final* wheel filename won't reflect it (see step 5b: delocate
# deliberately zeros the minor version for any macOS 11+ tag, so the wheel ends
# up named macosx_${major}_0 regardless — the real guarantee we need is that
# every bundled binary's actual minos is >= MACOSX_DEPLOYMENT_TARGET, verified
# directly in step 5b rather than trusting the filename).
export _PYTHON_HOST_PLATFORM="${_PYTHON_HOST_PLATFORM:-macosx-${MACOSX_DEPLOYMENT_TARGET}-arm64}"
DEPLOYMENT_MAJOR="${MACOSX_DEPLOYMENT_TARGET%%.*}"
EXPECTED_PLAT_TAG="macosx_${DEPLOYMENT_MAJOR}_0_arm64"

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
# the guard below verifies it post-delocate. Copy it from the same pip MLX
# resolved in step 0 (not Homebrew — see the file header) so the build is
# self-contained and consistent on CI runners as well as local machines.
echo "==> Staging mlx.metallib into the wheel data dir..."
MLX_METALLIB="${MLX_METALLIB:-$MLX_LIB_DIR/mlx.metallib}"
if [[ ! -f "$MLX_METALLIB" ]]; then
    echo "error: mlx.metallib not found at '$MLX_METALLIB'"
    echo "       expected it alongside the pip-installed mlx's libmlx.dylib "
    echo "       (MLX_LIB_DIR=$MLX_LIB_DIR); set MLX_METALLIB to override"
    exit 1
fi
METALLIB_DIR="$REPO_ROOT/python/ax_engine.dylibs"
mkdir -p "$METALLIB_DIR"
cp -f "$MLX_METALLIB" "$METALLIB_DIR/mlx.metallib"
echo "    staged: $METALLIB_DIR/mlx.metallib ($(wc -c < "$METALLIB_DIR/mlx.metallib" | tr -d ' ') bytes)"

# ── 3c. Stage AX Metal runtime assets into the wheel data dir. ─────────────
# Keep the package-owned copy self-contained: _setup_bundled_metal() writes a
# build_report.json that resolves paths under ax_engine/_metal after pip install.
echo "==> Building bundled AX Metal runtime assets..."
AX_METAL_PACKAGE_DIR="$REPO_ROOT/python/ax_engine/_metal"
AX_METAL_PACKAGE_MANIFEST_DIR="$AX_METAL_PACKAGE_DIR/metal"
AX_METAL_PACKAGE_KERNEL_DIR="$AX_METAL_PACKAGE_MANIFEST_DIR/kernels"
AX_METAL_PACKAGE_BUILD_DIR="$AX_METAL_PACKAGE_DIR/build"
mkdir -p "$AX_METAL_PACKAGE_KERNEL_DIR" "$AX_METAL_PACKAGE_BUILD_DIR"
cp -f "$REPO_ROOT/metal/phase1-kernels.json" "$AX_METAL_PACKAGE_MANIFEST_DIR/phase1-kernels.json"
cp -f "$REPO_ROOT/metal/kernels/phase1_dense_path.metal" "$AX_METAL_PACKAGE_KERNEL_DIR/phase1_dense_path.metal"
"$BENCH_BIN" metal-build \
    --manifest "$AX_METAL_PACKAGE_MANIFEST_DIR/phase1-kernels.json" \
    --output-dir "$AX_METAL_PACKAGE_BUILD_DIR"
echo "    staged: $AX_METAL_PACKAGE_BUILD_DIR/ax_phase1_dense_path.metallib ($(wc -c < "$AX_METAL_PACKAGE_BUILD_DIR/ax_phase1_dense_path.metallib" | tr -d ' ') bytes)"

# ── 4. Build the wheel ─────────────────────────────────────────────────────
# Use the release-pyext profile (Cargo.toml), not --release: it inherits the
# workspace release profile but keeps panic="unwind" instead of "abort", so
# PyO3's catch_unwind can actually turn a Rust panic into a catchable Python
# exception instead of aborting the whole embedding process. --release would
# silently defeat that safety net for every wheel this script produces.
echo "==> Building wheel (release-pyext, stripped, target $EXPECTED_PLAT_TAG)..."
maturin build --profile release-pyext --strip --out "$WHEEL_OUT"

# Use a glob expansion instead of ls+sort so we get exactly what was just built.
# After the clean above there should be exactly one match.
wheels=("${WHEEL_OUT}"/ax_engine-*.whl)
if [[ ${#wheels[@]} -ne 1 || ! -f "${wheels[0]}" ]]; then
    echo "error: expected exactly one wheel in $WHEEL_OUT after build, found: ${wheels[*]}"
    exit 1
fi
WHEEL="${wheels[0]}"
echo "    built: $WHEEL"

# ── 5. Delocalize — bundle libmlx into the wheel ──────────────────────────
echo "==> Delocalizing wheel (bundling dylibs)..."
# --require-archs ensures we only accept arm64 (Apple Silicon only)
# The pip mlx wheel's libmlx.dylib has no embedded LC_RPATH (unlike Homebrew's
# build), so its own @rpath/libjaccl.dylib reference can't be statically
# resolved by delocate's dependency walk without help. Export DYLD_LIBRARY_PATH
# so delocate finds libjaccl.dylib alongside libmlx.dylib and bundles both.
DYLD_LIBRARY_PATH="$MLX_LIB_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
DYLD_FALLBACK_LIBRARY_PATH="$MLX_LIB_DIR${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}" \
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

echo "==> Verifying bundled AX Metal runtime assets..."
verify_wheel_member "$DELOCATED" "ax_engine/_metal/build/ax_phase1_dense_path.metallib"
verify_wheel_member "$DELOCATED" "ax_engine/_metal/build/ax_phase1_dense_path.air"
verify_wheel_member "$DELOCATED" "ax_engine/_metal/metal/phase1-kernels.json"
verify_wheel_member "$DELOCATED" "ax_engine/_metal/metal/kernels/phase1_dense_path.metal"

echo "==> Verifying bundled command binaries..."
verify_wheel_member "$DELOCATED" "ax_engine/_bin/ax-engine-server"
verify_wheel_member "$DELOCATED" "ax_engine/_bin/ax-engine-bench"
verify_wheel_member "$DELOCATED" "ax_engine/_bin/ax-engine"

# ── 5b. Guard the platform tag ─────────────────────────────────────────────
# Refuse to ship a wheel whose tag major version is wrong (e.g. reverted to
# some ancient default). delocate deliberately zeros the minor version for
# any macOS 11+ tag (its own documented policy: point releases don't gate
# wheel-tag compatibility in general — pip's tagging convention has no way to
# express "macOS 26.2 specifically", only "macOS 26 or later"), so
# EXPECTED_PLAT_TAG is major-only; do not "fix" this by forcing a
# macosx_26_2 filename.
echo "==> Verifying wheel platform tag..."
if [[ "$(basename "$DELOCATED")" != *"-${EXPECTED_PLAT_TAG}.whl" ]]; then
    echo "error: wheel is not tagged ${EXPECTED_PLAT_TAG}: $(basename "$DELOCATED")"
    exit 1
fi
echo "    verified platform tag: ${EXPECTED_PLAT_TAG}"

# ── 5c. Guard every bundled binary's actual minos ──────────────────────────
# The wheel tag can't express our real requirement (see 5b), so verify it
# directly: every Mach-O file actually bundled in the wheel must have
# LC_BUILD_VERSION minos >= MACOSX_DEPLOYMENT_TARGET. This is what actually
# protects users — a regression here (e.g. a future dependency silently
# reintroducing a <26.2 build) means the wheel "installs but doesn't run" on
# older macOS 26.x, or worse, silently loses NAX acceleration with no error.
echo "==> Verifying bundled binaries' minos..."
INSPECT_DIR="$(mktemp -d)"
trap 'rm -rf "$INSPECT_DIR"' EXIT
unzip -q "$DELOCATED" -d "$INSPECT_DIR"
bad_binaries=()
while IFS= read -r -d '' f; do
    if file "$f" | grep -q "Mach-O"; then
        minos="$(otool -l "$f" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')"
        [[ -z "$minos" ]] && continue
        if ! version_ge "$minos" "$MACOSX_DEPLOYMENT_TARGET"; then
            bad_binaries+=("${f#"$INSPECT_DIR"/} (minos $minos)")
        fi
    fi
done < <(find "$INSPECT_DIR" -type f -print0)
if [[ ${#bad_binaries[@]} -gt 0 ]]; then
    echo "error: bundled binaries with minos below ${MACOSX_DEPLOYMENT_TARGET}:"
    printf '       %s\n' "${bad_binaries[@]}"
    echo "       refusing to ship a wheel that installs but silently loses NAX"
    echo "       acceleration (or fails to load) on some macOS 26.x hosts"
    exit 1
fi
echo "    verified: all bundled Mach-O binaries have minos >= ${MACOSX_DEPLOYMENT_TARGET}"

# ── 6. Optionally publish ──────────────────────────────────────────────────
if [[ "${1:-}" == "--publish" ]]; then
    echo "==> Publishing to PyPI..."
    maturin upload "$DELOCATED"
fi

echo ""
echo "Done. Wheel ready at: $DELOCATED"
