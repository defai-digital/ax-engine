#!/usr/bin/env bash
# Verify the MLX toolchain matches the admitted pin (mlx.version) before a
# build, benchmark, or release. Deployment correctness depends on which MLX
# gets linked: Homebrew's formula has lagged the pip wheel's Metal backend
# quality on macOS 26.x hosts, and version drift invalidates every certified
# benchmark and bit-exactness gate. mlx-sys/build.rs enforces the same pin and
# refuses Homebrew provenance at link time; this script gives operators/CI the
# same answer without compiling anything.
#
# Do NOT hard-fail on LC_BUILD_VERSION minos: the official PyPI mlx wheel for
# the admitted pin ships with minos 15.0 while still embedding NAX kernels.
# minos was never a reliable NAX proxy (Homebrew bottles can also report 26.0
# and still carry NAX symbol names). Mirror build.rs: pin + wheel layout +
# refuse Homebrew-resolved dylibs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"

cd "$ROOT_DIR"

# Mirror mlx-sys/build.rs resolution: the repo .venv is the canonical dev
# environment and is consulted even when the shell has not activated it.
if [[ -z "${PYTHON_BIN:-}" && -x "$ROOT_DIR/.venv/bin/python3" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python3"
else
    PYTHON_BIN="$AX_PYTHON_BIN"
fi

PIN="$(tr -d '[:space:]' < mlx.version)"
if [[ -z "$PIN" ]]; then
    echo "error: mlx.version is empty" >&2
    exit 1
fi
echo "==> admitted MLX pin: ${PIN}"

# 1. The Python that builds resolve from must have the pinned mlx wheel with
#    the bundled dylib (wheel layout, not a Homebrew site-packages shadow).
RESOLVED="$("$PYTHON_BIN" - <<'PY'
import pathlib
import sys

try:
    import mlx
    import mlx.core as core
except Exception as error:  # noqa: BLE001 - report any import failure
    print(f"import-failed: {error}")
    sys.exit(0)

root = pathlib.Path(list(mlx.__path__)[0])
dylib = root / "lib" / "libmlx.dylib"
version = getattr(core, "__version__", "unknown")
print(f"{version} {root} dylib={'yes' if dylib.is_file() else 'no'}")
PY
)"
echo "==> resolved: ${RESOLVED}"

case "$RESOLVED" in
    import-failed:*)
        echo "error: python cannot import mlx — activate the repo .venv or run" >&2
        echo "       python3 -m pip install mlx==${PIN}" >&2
        exit 1
        ;;
    "${PIN} "*)
        ;;
    *)
        echo "error: resolved MLX does not match the pin ${PIN}" >&2
        echo "       install it with: python3 -m pip install mlx==${PIN}" >&2
        exit 1
        ;;
esac
if [[ "$RESOLVED" != *"dylib=yes"* ]]; then
    echo "error: resolved mlx package has no bundled lib/libmlx.dylib (Homebrew" >&2
    echo "       site-packages shadow?) — install the pip wheel: mlx==${PIN}" >&2
    exit 1
fi

# 2. Refuse Homebrew-resolved dylibs (matches mlx-sys/build.rs). Report minos
#    for operators, but do not treat it as a NAX gate — pip wheels for the
#    admitted pin currently ship minos 15.0 with NAX present.
DYLIB="$("$PYTHON_BIN" -c 'import mlx, pathlib; print(pathlib.Path(list(mlx.__path__)[0]) / "lib" / "libmlx.dylib")')"
echo "==> libmlx.dylib: ${DYLIB}"
if command -v otool >/dev/null 2>&1; then
    MINOS="$(otool -l "$DYLIB" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')"
    echo "==> libmlx.dylib LC_BUILD_VERSION minos: ${MINOS:-unknown} (informational)"
fi

is_homebrew_mlx_path() {
    local path="$1"
    case "$path" in
        */Cellar/mlx/*|*/opt/mlx/*|*/homebrew/opt/mlx/*|*/linuxbrew/opt/mlx/*)
            return 0
            ;;
    esac
    if command -v brew >/dev/null 2>&1; then
        local brew_mlx
        brew_mlx="$(brew --prefix mlx 2>/dev/null || true)"
        if [[ -n "$brew_mlx" && "$path" == "$brew_mlx"* ]]; then
            return 0
        fi
    fi
    return 1
}

if is_homebrew_mlx_path "$DYLIB"; then
    echo "error: resolved libmlx.dylib is under Homebrew (${DYLIB})." >&2
    echo "       AX builds refuse Homebrew MLX (mlx-sys/build.rs); install the" >&2
    echo "       pip wheel instead: python3 -m pip install mlx==${PIN}" >&2
    exit 1
fi

# 3. Warn (not fail) when the Homebrew formula is present: mlx-sys refuses it
#    at link time, but its presence invites accidental use elsewhere.
if command -v brew >/dev/null 2>&1 && brew list --versions mlx >/dev/null 2>&1; then
    echo "warning: Homebrew mlx formula is installed ($(brew list --versions mlx));" >&2
    echo "         AX builds refuse it, but other tools may still pick it up." >&2
fi

echo "==> MLX toolchain OK (${PIN})"
