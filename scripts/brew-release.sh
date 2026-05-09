#!/usr/bin/env bash
# brew-release.sh — build, upload, and publish an ax-engine Homebrew release
# from the local machine. Replaces the GitHub Actions release workflow.
#
# Usage:
#   scripts/brew-release.sh v4.5.0
#   scripts/brew-release.sh v4.5.0 --dry-run
#   scripts/brew-release.sh v4.5.0 --skip-build
#   scripts/brew-release.sh v4.5.0 --skip-build --skip-upload
#   scripts/brew-release.sh v4.5.0 --sign-identity "Developer ID Application: Your Name (TEAMID)"
#
# Flags:
#   --dry-run                Build and package but do not upload or push anything
#   --skip-build             Use existing target/release binaries (skip cargo build)
#   --skip-upload            Skip uploading the archive to the GitHub release
#   --skip-tap               Skip cloning/updating the Homebrew tap
#   --skip-test              Skip the local brew install + test after updating the formula
#   --sign-identity <id>     Codesign and notarize binaries with this Developer ID Application
#                            certificate identity. If omitted, binaries are left unsigned and
#                            users may see a Gatekeeper warning on first launch.
#
# Prerequisites (checked at start):
#   gh        GitHub CLI, authenticated (gh auth status)
#   cargo     Rust toolchain
#   brew      Homebrew
#   ruby      For formula syntax validation (ruby -c)
#   python3   For formula install stanza validation (checked just before tap update; not needed with --skip-tap)
#
# Additional prerequisites for --sign-identity:
#   codesign  Xcode Command Line Tools
#   xcrun     Xcode Command Line Tools (notarytool)
#   zip       Temporary notarization submission archive
#   Apple Developer ID Application certificate in the local Keychain

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MAIN_REPO="defai-digital/ax-engine"
TAP_REPO="defai-digital/homebrew-ax-engine"
TAP_FORMULA="Formula/ax-engine.rb"
RELEASE_BINS=(ax-engine-server ax-engine-bench ax-engine-manager)

# ── parse arguments ──────────────────────────────────────────────────────────

TAG=""
DRY_RUN=false
SKIP_BUILD=false
SKIP_UPLOAD=false
SKIP_TAP=false
SKIP_TEST=false
SIGN_IDENTITY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=true ;;
        --skip-build)     SKIP_BUILD=true ;;
        --skip-upload)    SKIP_UPLOAD=true ;;
        --skip-tap)       SKIP_TAP=true ;;
        --skip-test)      SKIP_TEST=true ;;
        --sign-identity)
            shift
            if [[ -z "${1:-}" ]]; then
                echo "error: --sign-identity requires an argument" >&2
                exit 1
            fi
            SIGN_IDENTITY="$1"
            ;;
        v*.*)             TAG="$1" ;;
        *)
            echo "error: unknown argument: $1" >&2
            echo "usage: $0 <tag> [--dry-run] [--skip-build] [--skip-upload] [--skip-tap] [--skip-test] [--sign-identity <identity>]" >&2
            exit 1
            ;;
    esac
    shift
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

if [[ -n "$SIGN_IDENTITY" ]]; then
    check_cmd codesign "codesign is required for --sign-identity"
    check_cmd xcrun    "xcrun notarytool is required for --sign-identity"
    check_cmd zip      "zip is required for notarization submission"
    if ! xcrun notarytool --help &>/dev/null; then
        echo "error: xcrun notarytool is not available — install current Xcode Command Line Tools" >&2
        exit 1
    fi
fi

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
    cargo build --release -p ax-engine-server -p ax-engine-bench -p ax-engine-tui
else
    echo "▶ skipping build (--skip-build)"
    for bin in "${RELEASE_BINS[@]}"; do
        if [[ ! -f "target/release/$bin" ]]; then
            echo "error: target/release/$bin not found — run without --skip-build first" >&2
            exit 1
        fi
    done
fi

# ── codesign + notarize ───────────────────────────────────────────────────────

if [[ -n "$SIGN_IDENTITY" ]]; then
    echo "▶ codesigning binaries with identity: $SIGN_IDENTITY"
    for bin in "${RELEASE_BINS[@]}"; do
        codesign \
            --sign "$SIGN_IDENTITY" \
            --options runtime \
            --timestamp \
            --force \
            "target/release/$bin"
        echo "  ✓ signed target/release/$bin"
    done

    echo "▶ verifying codesignatures…"
    for bin in "${RELEASE_BINS[@]}"; do
        codesign --verify --strict --verbose=2 "target/release/$bin"
    done
else
    echo "⚠️  no --sign-identity provided — binaries will remain unsigned"
    echo "   users on macOS 14+ may see a Gatekeeper warning on first launch"
    echo "   pass --sign-identity \"Developer ID Application: ...\" to notarize"
fi

# ── package ───────────────────────────────────────────────────────────────────

ARCHIVE="ax-engine-${TAG}-macos-arm64.tar.gz"
ARCHIVE_PATH="/tmp/${ARCHIVE}"

echo "▶ packaging ${ARCHIVE}…"
tar -czf "$ARCHIVE_PATH" -C target/release "${RELEASE_BINS[@]}"
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

# ── notarize ─────────────────────────────────────────────────────────────────
# Must run before upload so users cannot download an un-notarized binary.
# notarytool only accepts .zip, .pkg, or .dmg — not .tar.gz — so we create a
# temporary zip of the binaries solely for submission, then discard it.

if [[ -n "$SIGN_IDENTITY" ]]; then
    echo "▶ submitting binaries for notarization…"
    # notarytool reads credentials from the Keychain; the store-credentials
    # profile "ax-engine-notary" must be set up once with:
    #   xcrun notarytool store-credentials ax-engine-notary \
    #     --apple-id <email> --team-id <TEAMID> --password <app-specific-password>
    NOTARIZE_PROFILE="${AX_NOTARY_PROFILE:-ax-engine-notary}"
    NOTARIZE_ZIP="/tmp/ax-engine-${TAG}-notarize.zip"
    (cd target/release && zip -j "$NOTARIZE_ZIP" "${RELEASE_BINS[@]}")
    xcrun notarytool submit "$NOTARIZE_ZIP" \
        --keychain-profile "$NOTARIZE_PROFILE" \
        --wait
    rm -f "$NOTARIZE_ZIP"
    echo "  ✓ notarized"
else
    echo "▶ skipping notarization (no --sign-identity)"
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

check_cmd python3 "python3 is required for formula install stanza validation"

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

python3 - "$FORMULA_PATH" "${RELEASE_BINS[@]}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

formula_path = Path(sys.argv[1])
bins = sys.argv[2:]
text = formula_path.read_text(encoding="utf-8")
if all(f'"{binary}"' in text for binary in bins):
    raise SystemExit(0)

pattern = re.compile(r'^(?P<indent>\s*)bin\.install\s+.*ax-engine-server.*ax-engine-bench.*$', re.MULTILINE)
match = pattern.search(text)
if not match:
    raise SystemExit(
        "formula must install ax-engine-server and ax-engine-bench before ax-engine-manager can be added"
    )

replacement = f'{match.group("indent")}bin.install {", ".join(repr(binary).replace(chr(39), chr(34)) for binary in bins)}'
formula_path.write_text(pattern.sub(replacement, text, count=1), encoding="utf-8")
PY

for bin in "${RELEASE_BINS[@]}"; do
    if ! grep -qF "\"$bin\"" "$FORMULA_PATH"; then
        echo "error: formula does not install $bin after update" >&2
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
