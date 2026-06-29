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

brew install mlx protobuf
