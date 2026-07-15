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
if ! python3 -m pip install --upgrade mlx; then
    # Fallback for runner images whose default python3 is externally managed
    # (PEP 668) and rejects plain pip installs.
    python3 -m pip install --upgrade --break-system-packages mlx
fi
