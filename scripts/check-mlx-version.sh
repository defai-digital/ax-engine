#!/usr/bin/env bash
# Verify the MLX toolchain matches the admitted pin (mlx.version) before a
# build, benchmark, or release. Deployment correctness depends on which MLX
# gets linked: Homebrew's formula silently compiles out the NAX kernels on
# macOS 26.x (deployment-target truncation), and version drift invalidates
# every certified benchmark and bit-exactness gate. mlx-sys/build.rs enforces
# the same pin at link time; this script gives operators/CI the same answer
# without compiling anything.

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

# 2. The wheel's dylib must carry a macOS 26.2+ build target or the NAX
#    GEMM/attention kernels are silently absent (mlx PR #3622).
DYLIB="$("$PYTHON_BIN" -c 'import mlx, pathlib; print(pathlib.Path(list(mlx.__path__)[0]) / "lib" / "libmlx.dylib")')"
if command -v otool >/dev/null 2>&1; then
    MINOS="$(otool -l "$DYLIB" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')"
    echo "==> libmlx.dylib LC_BUILD_VERSION minos: ${MINOS:-unknown}"
    if [[ -n "${MINOS:-}" ]]; then
        MAJOR="${MINOS%%.*}"
        MINOR_PART="${MINOS#*.}"
        MINOR="${MINOR_PART%%.*}"
        if (( MAJOR < 26 )) || { (( MAJOR == 26 )) && (( MINOR < 2 )); }; then
            echo "error: libmlx.dylib build target ${MINOS} < 26.2 — NAX kernels" >&2
            echo "       are silently compiled out; install the official pip wheel." >&2
            exit 1
        fi
    fi
fi

# 3. Warn (not fail) when the Homebrew formula is present: mlx-sys refuses it
#    at link time, but its presence invites accidental use elsewhere.
if command -v brew >/dev/null 2>&1 && brew list --versions mlx >/dev/null 2>&1; then
    echo "warning: Homebrew mlx formula is installed ($(brew list --versions mlx));" >&2
    echo "         AX builds refuse it, but other tools may still pick it up." >&2
fi

echo "==> MLX toolchain OK (${PIN})"
