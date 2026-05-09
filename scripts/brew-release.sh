#!/usr/bin/env bash
# brew-release.sh — build, upload, and publish an ax-engine Homebrew release
# from the local machine. Replaces the GitHub Actions release workflow.
#
# Usage:
#   scripts/brew-release.sh v4.5.0
#   scripts/brew-release.sh v4.5.0 --dry-run
#   scripts/brew-release.sh v4.5.0 --skip-build
#   scripts/brew-release.sh v4.5.0 --skip-build --skip-upload
#
# Flags:
#   --dry-run       Build and package but do not upload or push anything
#   --skip-build    Use existing target/release binaries (skip cargo build)
#   --skip-upload   Skip uploading the archive to the GitHub release
#   --skip-tap      Skip cloning/updating the Homebrew tap
#   --skip-test     Skip the local brew install + test after updating the formula
#
# Prerequisites (checked at start):
#   gh        GitHub CLI, authenticated (gh auth status)
#   cargo     Rust toolchain
#   brew      Homebrew
#   ruby      For formula syntax validation (ruby -c)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MAIN_REPO="defai-digital/ax-engine"
TAP_REPO="defai-digital/homebrew-ax-engine"
TAP_FORMULA="Formula/ax-engine.rb"

# ── parse arguments ──────────────────────────────────────────────────────────

TAG=""
DRY_RUN=false
SKIP_BUILD=false
SKIP_UPLOAD=false
SKIP_TAP=false
SKIP_TEST=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)      DRY_RUN=true ;;
        --skip-build)   SKIP_BUILD=true ;;
        --skip-upload)  SKIP_UPLOAD=true ;;
        --skip-tap)     SKIP_TAP=true ;;
        --skip-test)    SKIP_TEST=true ;;
        v*.*)           TAG="$arg" ;;
        *)
            echo "error: unknown argument: $arg" >&2
            echo "usage: $0 <tag> [--dry-run] [--skip-build] [--skip-upload] [--skip-tap] [--skip-test]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$TAG" ]]; then
    echo "error: tag is required (e.g. v4.5.0)" >&2
    exit 1
fi

VERSION="${TAG#v}"

# ── prerequisite checks ───────────────────────────────────────────────────────

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "error: '$1' not found — $2" >&2
        exit 1
    fi
}

check_cmd gh    "install the GitHub CLI: brew install gh"
check_cmd cargo "install Rust: https://rustup.rs"
check_cmd brew  "install Homebrew: https://brew.sh"
check_cmd ruby  "ruby is required for formula syntax validation"

if ! gh auth status &>/dev/null; then
    echo "error: gh is not authenticated — run: gh auth login" >&2
    exit 1
fi

# Verify the GitHub release exists (or will exist) for the tag
if ! gh release view "$TAG" --repo "$MAIN_REPO" &>/dev/null; then
    echo "error: GitHub release '$TAG' does not exist in $MAIN_REPO" >&2
    echo "       create it first: gh release create $TAG --repo $MAIN_REPO" >&2
    exit 1
fi

# ── build ─────────────────────────────────────────────────────────────────────

cd "$ROOT_DIR"

if [[ "$SKIP_BUILD" = false ]]; then
    echo "▶ building release binaries…"
    cargo build --release -p ax-engine-server -p ax-engine-bench
else
    echo "▶ skipping build (--skip-build)"
    for bin in ax-engine-server ax-engine-bench; do
        if [[ ! -f "target/release/$bin" ]]; then
            echo "error: target/release/$bin not found — run without --skip-build first" >&2
            exit 1
        fi
    done
fi

# ── package ───────────────────────────────────────────────────────────────────

ARCHIVE="ax-engine-${TAG}-macos-arm64.tar.gz"
ARCHIVE_PATH="/tmp/${ARCHIVE}"

echo "▶ packaging ${ARCHIVE}…"
tar -czf "$ARCHIVE_PATH" -C target/release ax-engine-server ax-engine-bench
SHA256=$(shasum -a 256 "$ARCHIVE_PATH" | awk '{print $1}')
ARCHIVE_SIZE=$(du -sh "$ARCHIVE_PATH" | awk '{print $1}')

DOWNLOAD_URL="https://github.com/${MAIN_REPO}/releases/download/${TAG}/${ARCHIVE}"

echo ""
echo "  archive:  $ARCHIVE_PATH  ($ARCHIVE_SIZE)"
echo "  sha256:   $SHA256"
echo "  url:      $DOWNLOAD_URL"
echo ""

if [[ "$DRY_RUN" = true ]]; then
    echo "▶ dry-run: skipping upload and tap update"
    echo ""
    echo "  tag:      $TAG"
    echo "  version:  $VERSION"
    exit 0
fi

# ── upload to GitHub release ──────────────────────────────────────────────────

if [[ "$SKIP_UPLOAD" = false ]]; then
    echo "▶ uploading to GitHub release $TAG…"
    gh release upload "$TAG" "$ARCHIVE_PATH" \
        --repo "$MAIN_REPO" \
        --clobber
    echo "  ✓ uploaded"
else
    echo "▶ skipping upload (--skip-upload)"
fi

# ── update Homebrew tap ───────────────────────────────────────────────────────

if [[ "$SKIP_TAP" = true ]]; then
    echo "▶ skipping tap update (--skip-tap)"
    exit 0
fi

TAP_DIR="$(mktemp -d /tmp/ax-engine-tap.XXXXXX)"
trap 'rm -rf "$TAP_DIR"' EXIT

echo "▶ cloning tap ${TAP_REPO}…"
gh repo clone "$TAP_REPO" "$TAP_DIR" -- --depth=1 --quiet

cd "$TAP_DIR"
git config user.name "Automatosx"
git config user.email "automatosx@defai.digital"

FORMULA_PATH="$TAP_DIR/$TAP_FORMULA"

if [[ ! -f "$FORMULA_PATH" ]]; then
    echo "error: $TAP_FORMULA not found in cloned tap at $TAP_DIR" >&2
    exit 1
fi

echo "▶ updating formula to $VERSION…"

# Update version, url, and sha256 in place.
# The formula uses an on_macos / if Hardware::CPU.arm? block; the url and
# sha256 lines exist only once inside that block so these patterns are unambiguous.
sed -i '' \
    -e "s|version \"[^\"]*\"|version \"${VERSION}\"|" \
    -e "s|url \"[^\"]*\"|url \"${DOWNLOAD_URL}\"|" \
    -e "s|sha256 \"[^\"]*\"|sha256 \"${SHA256}\"|" \
    "$FORMULA_PATH"

# Verify the three substitutions actually landed
for pattern in "version \"${VERSION}\"" "url \"${DOWNLOAD_URL}\"" "sha256 \"${SHA256}\""; do
    if ! grep -qF "$pattern" "$FORMULA_PATH"; then
        echo "error: formula update failed — '$pattern' not found after sed" >&2
        echo "       check the formula manually: $FORMULA_PATH" >&2
        exit 1
    fi
done

echo "▶ validating formula syntax…"
ruby -c "$FORMULA_PATH" > /dev/null

echo ""
echo "  formula diff:"
git diff "$TAP_FORMULA"
echo ""

git add "$TAP_FORMULA"
if git diff --cached --quiet; then
    echo "▶ formula is already up to date — nothing to commit"
    exit 0
fi
git commit -m "chore: update ax-engine to ${TAG}"

# ── optional local brew test ──────────────────────────────────────────────────

if [[ "$SKIP_TEST" = false ]]; then
    echo "▶ testing formula with local brew install…"
    brew uninstall --ignore-dependencies ax-engine 2>/dev/null || true
    brew untap defai-digital/ax-engine 2>/dev/null || true
    HOMEBREW_NO_AUTO_UPDATE=1 brew tap defai-digital/ax-engine "$TAP_DIR"
    HOMEBREW_NO_AUTO_UPDATE=1 brew install defai-digital/ax-engine/ax-engine
    HOMEBREW_NO_AUTO_UPDATE=1 brew test defai-digital/ax-engine/ax-engine
    echo "  ✓ brew install and test passed"
else
    echo "▶ skipping local brew test (--skip-test)"
fi

# ── push ─────────────────────────────────────────────────────────────────────

echo "▶ pushing to ${TAP_REPO}…"
# gh repo clone sets up the remote with gh auth; push directly
git push origin main
echo "  ✓ tap updated"

echo ""
echo "✓ release complete: ax-engine ${TAG}"
echo "  brew tap defai-digital/ax-engine"
echo "  brew install defai-digital/ax-engine/ax-engine"
