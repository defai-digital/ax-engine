#!/usr/bin/env bash
# Build MLX from source at a pinned commit so AX Engine can use kernels that
# are not on PyPI yet (default: 914732419, the 0.32.5 NAX verify kernels).
#
# Local use only. The repo pin (mlx.version) and every published install stay
# on the PyPI release; this build is linked with AX_MLX_VERSION_OVERRIDE=1.
#
# Usage: scripts/build-mlx-source.sh [--commit <sha>] [--prefix <dir>] [--jobs <n>]
set -euo pipefail

COMMIT="914732419f255f8b45ff1ee253ad33ab0a3ce623"
PREFIX=""
JOBS=""
REPO_URL="https://github.com/ml-explore/mlx.git"
MIN_TARGET="26.2"

usage() {
  sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit) COMMIT="${2:?--commit needs a value}"; shift 2 ;;
    --prefix) PREFIX="${2:?--prefix needs a value}"; shift 2 ;;
    --jobs) JOBS="${2:?--jobs needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "error: $*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] \
  || die "MLX source builds need macOS on Apple Silicon"

# NAX kernels compile only with a deployment target of 26.2 or later; a lower
# target builds a runtime that reports the same version with them compiled out.
os_version="$(sw_vers -productVersion)"
version_at_least() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}
version_at_least "$os_version" "$MIN_TARGET" \
  || die "macOS $os_version is below $MIN_TARGET; the NAX kernels would be compiled out"
xcrun --find metal >/dev/null 2>&1 \
  || die "the Metal compiler is missing; install Xcode (not only the Command Line Tools)"
command -v git >/dev/null || die "git is required"
command -v uv >/dev/null || die "uv is required (https://docs.astral.sh/uv/)"

short="${COMMIT:0:9}"
PREFIX="${PREFIX:-$HOME/.ax-engine/mlx-src-$short}"
SRC="$PREFIX/src"
VENV="$PREFIX/venv"
JOBS="${JOBS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)}"

echo "MLX commit: $COMMIT"
echo "prefix:     $PREFIX"

if [[ ! -d "$SRC/.git" ]]; then
  mkdir -p "$PREFIX"
  git clone --filter=blob:none "$REPO_URL" "$SRC"
fi
git -C "$SRC" fetch --quiet origin
git -C "$SRC" checkout --quiet --detach "$COMMIT"
resolved="$(git -C "$SRC" rev-parse HEAD)"
[[ "$resolved" == "$COMMIT"* ]] || die "checked out $resolved, expected $COMMIT"

[[ -x "$VENV/bin/python" ]] || uv venv --quiet --python 3.12 "$VENV"
uv pip install --quiet --python "$VENV/bin/python" \
  setuptools nanobind cmake ninja typing_extensions

# PYPI_RELEASE=1 builds the release version string (no dev suffix), which AX
# links with AX_MLX_VERSION_OVERRIDE=1.
(
  cd "$SRC"
  PATH="$VENV/bin:$PATH" PYPI_RELEASE=1 MACOSX_DEPLOYMENT_TARGET="$MIN_TARGET" \
    CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" \
    uv pip install --python "$VENV/bin/python" --no-build-isolation --reinstall-package mlx .
)

# Check the build loads from the venv (python -I would ignore PYTHONPATH and
# could silently load another MLX) and carries the NAX verify kernels.
pkg="$("$VENV/bin/python" -c 'import mlx.core, os; print(os.path.dirname(mlx.core.__file__))')"
version="$("$VENV/bin/python" -c 'import mlx.core as mx; print(mx.__version__)')"
[[ "$pkg" == "$VENV"/* ]] || die "mlx loaded from $pkg, not from $VENV"
metallib="$pkg/lib/mlx.metallib"
[[ -f "$metallib" && -f "$pkg/lib/libmlx.dylib" ]] || die "no libmlx.dylib / mlx.metallib under $pkg/lib"
for marker in qmv_nax sdpa_split_reduce; do
  LC_ALL=C grep -q -a "$marker" "$metallib" \
    || die "$metallib has no $marker kernels; the NAX verify hand-offs would stay off"
done

cat <<EOF

Built MLX $version ($short) with the NAX verify kernels.

Build AX Engine against it (from the ax-engine checkout):

  export MLX_LIB_DIR="$pkg/lib"
  export MLX_INCLUDE_DIR="$pkg/include"
  export AX_MLX_VERSION_OVERRIDE=1
  cargo build --release

Then check that the hand-offs are on (M5 or later, macOS 26.2+):

  target/release/ax-engine-bench doctor --json   # nax_verify_kernels.active and native_sdpa_active

Benchmarks and bit-exactness gates are certified against the mlx.version pin,
not this build.
EOF
