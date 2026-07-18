#!/usr/bin/env bash

set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
    echo "error: Homebrew is required to install native build dependencies" >&2
    exit 127
fi

# GitHub-hosted macOS images can include third-party taps that AX Engine does
# not use. Remove them before installing official Homebrew dependencies so tap
# trust enforcement cannot make unrelated taps affect CI.
brew untap --force aws/tap azure/bicep >/dev/null 2>&1 || true

brew install protobuf

# MLX must come from pip, NOT Homebrew. The brew formula derives its build's
# MACOSX_DEPLOYMENT_TARGET from MacOS.version.major.minor, which truncates to
# 26.0 on every macOS 26.x host; MLX's NAX GEMM/attention kernels require a
# target of 26.2+ (mlx PR #3622) and silently compile out below that — no
# build error, ~3-4x slower prefill. The official pip wheel is built with the
# correct target, and mlx-sys/build.rs prefers a pip-installed mlx, so
# installing it here is what makes builds resolve the correct one. See
# scripts/build-pypi-wheel.sh's header for the full investigation.
# The exact admitted version is pinned in mlx.version at the repo root;
# mlx-sys/build.rs enforces the same pin at link time, so an unpinned
# install here would fail the build anyway. Bump the pin deliberately
# (qmm microbench parity + bit-exactness gates first).
MLX_PIN="$(cat "$(dirname "$0")/../mlx.version" | tr -d '[:space:]')"
if ! python3 -m pip install --upgrade "mlx==${MLX_PIN}"; then
    # Fallback for runner images whose default python3 is externally managed
    # (PEP 668) and rejects plain pip installs.
    python3 -m pip install --upgrade --break-system-packages "mlx==${MLX_PIN}"
fi

# Export resolved MLX dirs so cargo/maturin steps can find headers/lib even
# when a later step activates a fresh venv without reinstalling mlx. Prefer
# an already-set MLX_LIB_DIR (CI/local override) over the just-installed path.
MLX_PIP_DIR="$(python3 -c 'import mlx, pathlib; print(pathlib.Path(list(mlx.__path__)[0]))')"
if [[ ! -f "$MLX_PIP_DIR/lib/libmlx.dylib" ]]; then
    echo "error: pip-installed mlx has no lib/libmlx.dylib at $MLX_PIP_DIR" >&2
    exit 1
fi
export MLX_LIB_DIR="${MLX_LIB_DIR:-$MLX_PIP_DIR/lib}"
export MLX_INCLUDE_DIR="${MLX_INCLUDE_DIR:-$MLX_PIP_DIR/include}"
echo "Using MLX_LIB_DIR=$MLX_LIB_DIR"
if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
        echo "MLX_LIB_DIR=$MLX_LIB_DIR"
        echo "MLX_INCLUDE_DIR=$MLX_INCLUDE_DIR"
    } >>"$GITHUB_ENV"
fi
