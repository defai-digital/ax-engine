#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"

RUN_FULL_WORKSPACE=0
TARGET_TIMEOUT_SECS="${AX_TARGET_TIMEOUT_SECS:-900}"
FULL_TIMEOUT_SECS="${AX_FULL_TIMEOUT_SECS:-1800}"
CARGO_JOBS="${AX_CARGO_JOBS:-1}"

usage() {
    cat <<'EOF'
Usage: scripts/check-mlx-telemetry.sh [--full-workspace]

Runs the targeted gates for Gemma/AX MLX telemetry and decode-profile changes.

Options:
  --full-workspace  Also run a timeout-protected workspace compile gate followed
                    by crate-by-crate Rust tests with AX_CARGO_JOBS (default: 1).

Environment:
  AX_TARGET_TIMEOUT_SECS  Timeout for each targeted gate command (default: 900).
  AX_FULL_TIMEOUT_SECS    Timeout for each full-workspace command (default: 1800).
  AX_CARGO_JOBS           Cargo job count for full-workspace mode (default: 1).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full-workspace)
            RUN_FULL_WORKSPACE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

run_timed() {
    local timeout_secs="$1"
    shift
    echo "+ timeout ${timeout_secs}s $*"
    "$AX_PYTHON_BIN" - "$timeout_secs" "$@" <<'PY'
import os
import signal
import subprocess
import sys
import time

timeout = float(sys.argv[1])
argv = sys.argv[2:]

proc = subprocess.Popen(argv, start_new_session=True)
deadline = time.monotonic() + timeout

try:
    while True:
        code = proc.poll()
        if code is not None:
            raise SystemExit(code)
        if time.monotonic() >= deadline:
            break
        time.sleep(0.2)

    print(
        f"command timed out after {timeout:.0f}s: {' '.join(argv)}",
        file=sys.stderr,
    )
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    raise SystemExit(124)
except KeyboardInterrupt:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    raise
PY
}

cd "$ROOT_DIR"

run_timed "$TARGET_TIMEOUT_SECS" cargo clippy -p ax-engine-mlx --all-targets -- -D warnings
run_timed "$TARGET_TIMEOUT_SECS" cargo test -p ax-engine-mlx --quiet
run_timed "$TARGET_TIMEOUT_SECS" bash scripts/check-bench-inference-stack.sh
run_timed "$TARGET_TIMEOUT_SECS" bash scripts/check-scripts.sh

if [[ "$RUN_FULL_WORKSPACE" -eq 1 ]]; then
    full_cargo_test() {
        local package="$1"
        run_timed "$FULL_TIMEOUT_SECS" cargo test -p "$package" --quiet --jobs "$CARGO_JOBS"
    }

    run_timed "$FULL_TIMEOUT_SECS" cargo test --workspace --no-run --jobs "$CARGO_JOBS"
    full_cargo_test mlx-sys
    full_cargo_test ax-engine-core
    full_cargo_test ax-engine-sdk
    full_cargo_test ax-engine-mlx
    full_cargo_test ax-engine-bench
    full_cargo_test ax-engine-server
    full_cargo_test ax-engine-py
    full_cargo_test ax-engine-tui
fi
