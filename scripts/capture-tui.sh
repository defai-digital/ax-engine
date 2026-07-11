#!/usr/bin/env bash
# Capture the ax-engine TUI as GIF + PNG for README / docs (via Charm VHS).
#
# Usage (from anywhere):
#   bash scripts/capture-tui.sh
#   bash scripts/capture-tui.sh --skip-build
#   bash scripts/capture-tui.sh --tape docs/assets/tui-demo.tape
#
# Prerequisites (macOS):
#   brew install vhs
#   # VHS pulls ttyd + ffmpeg as deps on typical Homebrew installs
#
# Outputs (default tape):
#   docs/assets/tui-demo.gif
#   docs/assets/tui-demo.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"

SKIP_BUILD=0
TAPE_REL="docs/assets/tui-demo.tape"
PROFILE="debug"
LINKED_DEBUG=0

usage() {
    cat <<'EOF'
Usage: bash scripts/capture-tui.sh [options]

Capture the ax-engine TUI with VHS (GIF + PNG still).

Options:
  --skip-build     Do not rebuild ax-engine (use existing binary)
  --release        Build/use the release binary (symlinked for the tape)
  --tape <path>    Tape file relative to repo root (default: docs/assets/tui-demo.tape)
  -h, --help       Show this help

Requires: vhs (brew install vhs), cargo
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --release)
            PROFILE="release"
            shift
            ;;
        --tape)
            TAPE_REL="${2:?--tape requires a path}"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "$ROOT_DIR"

if ! command -v vhs >/dev/null 2>&1; then
    echo "error: vhs not found. Install with: brew install vhs" >&2
    exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found" >&2
    exit 1
fi

BIN_REL="target/${PROFILE}/ax-engine"
TAPE_BIN="target/debug/ax-engine"

cleanup() {
    if [[ "$LINKED_DEBUG" -eq 1 && -L "$ROOT_DIR/$TAPE_BIN" ]]; then
        rm -f "$ROOT_DIR/$TAPE_BIN"
    fi
}
trap 'ax_run_cleanup "$?" cleanup' EXIT

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    echo "building ax-engine (${PROFILE})…"
    if [[ "$PROFILE" == "release" ]]; then
        cargo build -p ax-engine-bench --bin ax-engine --release
    else
        cargo build -p ax-engine-bench --bin ax-engine
    fi
elif [[ ! -x "$BIN_REL" ]]; then
    echo "error: --skip-build but missing executable: $BIN_REL" >&2
    exit 1
fi

if [[ ! -x "$BIN_REL" ]]; then
    echo "error: binary not executable: $BIN_REL" >&2
    exit 1
fi

# Tape always launches ./target/debug/ax-engine; for --release, symlink it.
if [[ "$PROFILE" == "release" ]]; then
    mkdir -p "$ROOT_DIR/target/debug"
    if [[ -e "$ROOT_DIR/$TAPE_BIN" && ! -L "$ROOT_DIR/$TAPE_BIN" ]]; then
        echo "error: $TAPE_BIN exists and is not a symlink; refuse to overwrite" >&2
        exit 1
    fi
    ln -sfn "$ROOT_DIR/$BIN_REL" "$ROOT_DIR/$TAPE_BIN"
    LINKED_DEBUG=1
fi

if [[ ! -x "$TAPE_BIN" ]]; then
    echo "error: tape binary missing: $TAPE_BIN" >&2
    exit 1
fi

TAPE_PATH="$ROOT_DIR/$TAPE_REL"
if [[ ! -f "$TAPE_PATH" ]]; then
    echo "error: tape not found: $TAPE_REL" >&2
    exit 1
fi

mkdir -p "$ROOT_DIR/docs/assets"

# VHS may leave a directory named *.png if Output points at a .png path; clean
# any stale still before / after render.
STILL_PNG="docs/assets/tui-demo.png"
GIF_OUT="docs/assets/tui-demo.gif"
if [[ -d "$STILL_PNG" ]]; then
    rm -rf "$STILL_PNG"
fi

echo "rendering $TAPE_REL with vhs…"
vhs "$TAPE_PATH"

# Drop accidental frame dumps if VHS recreated a png directory.
if [[ -d "$STILL_PNG" ]]; then
    rm -rf "$STILL_PNG"
fi

if [[ -f "$GIF_OUT" ]]; then
    # Still from ~12s in (Home with chart trail, before tab switch).
    if command -v ffmpeg >/dev/null 2>&1; then
        echo "extracting still PNG from GIF…"
        ffmpeg -y -loglevel error -ss 12 -i "$GIF_OUT" -frames:v 1 "$STILL_PNG"
    elif command -v convert >/dev/null 2>&1; then
        echo "extracting still PNG from GIF (ImageMagick)…"
        convert "${GIF_OUT}[60]" "$STILL_PNG"
    else
        echo "warning: no ffmpeg/convert; skip still PNG (GIF only)" >&2
    fi
else
    echo "error: expected GIF missing: $GIF_OUT" >&2
    exit 1
fi

echo
echo "done."
for out in "$GIF_OUT" "$STILL_PNG"; do
    if [[ -f "$out" ]]; then
        ls -lh "$out"
    fi
done
