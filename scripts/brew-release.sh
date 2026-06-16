#!/usr/bin/env bash
# brew-release.sh — build, upload, and publish an ax-engine Homebrew release
# from the local machine. Replaces the GitHub Actions release workflow.
#
# Usage:
#   scripts/brew-release.sh v4.5.0
#   scripts/brew-release.sh v4.5.0 --dry-run
#   scripts/brew-release.sh v4.5.0 --skip-build
#   scripts/brew-release.sh v4.5.0 --skip-build --skip-upload
#   scripts/brew-release.sh v4.5.0 --minisign
#   scripts/brew-release.sh v4.5.0 --sign-identity "Developer ID Application: Your Name (TEAMID)"
#
# Flags:
#   --dry-run                Build and package but do not upload or push anything
#   --skip-build             Use existing target/release binaries (skip cargo build)
#   --skip-upload            Skip uploading the archive to the GitHub release
#   --skip-tap               Skip cloning/updating the Homebrew tap
#   --skip-test              Skip the local brew install + test after updating the formula
#   --minisign               Sign the release archive with minisign before upload
#   --minisign-key <path>    Secret key path for --minisign
#                            (default: ~/signkey/ax-code.sec)
#   --minisign-pubkey <path> Public key path for --minisign verification
#                            (default: ~/signkey/ax-code.pub)
#   --minisign-public-key <key>
#                            Public key string for --minisign verification
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
# Additional prerequisites for --minisign:
#   minisign  Minisign signing tool
#   ~/signkey/ax-code.sec and ~/signkey/ax-code.pub (shared ax-code signing key)
#   Passphrase stored in macOS Keychain: security add-generic-password -U -a ax-code-release -s ax-code-minisign -w
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
RELEASE_BINS=(ax-engine ax-engine-server ax-engine-bench)
RELEASE_HELPER_SOURCES=(
    "scripts/download_model.py:ax-engine-download-model.py"
    "scripts/prepare_mtp_sidecar.py:ax-engine-prepare-mtp-sidecar.py"
    "scripts/check_mtp_sidecar_provenance.py:ax-engine-check-mtp-sidecar-provenance.py"
)

# ── parse arguments ──────────────────────────────────────────────────────────

TAG=""
DRY_RUN=false
SKIP_BUILD=false
SKIP_UPLOAD=false
SKIP_TAP=false
SKIP_TEST=false
MINISIGN=false
MINISIGN_SECRET_KEY="${AX_MINISIGN_SECRET_KEY:-$HOME/signkey/ax-code.sec}"
MINISIGN_PUBLIC_KEY="${AX_MINISIGN_PUBLIC_KEY:-$HOME/signkey/ax-code.pub}"
MINISIGN_PUBLIC_KEY_STRING="${AX_MINISIGN_PUBLIC_KEY_STRING:-}"
SIGN_IDENTITY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=true ;;
        --skip-build)     SKIP_BUILD=true ;;
        --skip-upload)    SKIP_UPLOAD=true ;;
        --skip-tap)       SKIP_TAP=true ;;
        --skip-test)      SKIP_TEST=true ;;
        --minisign)       MINISIGN=true ;;
        --minisign-key)
            shift
            if [[ -z "${1:-}" ]]; then
                echo "error: --minisign-key requires an argument" >&2
                exit 1
            fi
            MINISIGN_SECRET_KEY="$1"
            ;;
        --minisign-pubkey)
            shift
            if [[ -z "${1:-}" ]]; then
                echo "error: --minisign-pubkey requires an argument" >&2
                exit 1
            fi
            MINISIGN_PUBLIC_KEY="$1"
            ;;
        --minisign-public-key)
            shift
            if [[ -z "${1:-}" ]]; then
                echo "error: --minisign-public-key requires an argument" >&2
                exit 1
            fi
            MINISIGN_PUBLIC_KEY_STRING="$1"
            ;;
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
            echo "usage: $0 <tag> [--dry-run] [--skip-build] [--skip-upload] [--skip-tap] [--skip-test] [--minisign] [--minisign-key <path>] [--minisign-pubkey <path>] [--minisign-public-key <key>] [--sign-identity <identity>]" >&2
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

if [[ "$MINISIGN" = true ]]; then
    check_cmd minisign "install minisign: brew install minisign"
    check_cmd shasum "shasum is required for release archive comments"
    if [[ ! -f "$MINISIGN_SECRET_KEY" ]]; then
        echo "error: minisign secret key not found: $MINISIGN_SECRET_KEY" >&2
        echo "       create it with:" >&2
        echo "       mkdir -p \"$(dirname "$MINISIGN_SECRET_KEY")\" && chmod 700 \"$(dirname "$MINISIGN_SECRET_KEY")\"" >&2
        echo "       minisign -G -p \"$MINISIGN_PUBLIC_KEY\" -s \"$MINISIGN_SECRET_KEY\"" >&2
        exit 1
    fi
    if [[ -z "$MINISIGN_PUBLIC_KEY_STRING" && ! -f "$MINISIGN_PUBLIC_KEY" ]]; then
        echo "error: minisign public key not found: $MINISIGN_PUBLIC_KEY" >&2
        echo "       recreate it with: minisign -R -s \"$MINISIGN_SECRET_KEY\" -p \"$MINISIGN_PUBLIC_KEY\"" >&2
        echo "       or pass --minisign-public-key <base64-key>" >&2
        exit 1
    fi
fi

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
    cargo build --release -p ax-engine-server -p ax-engine-bench --bins
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
    echo "   users on macOS 26+ may see a Gatekeeper warning on first launch"
    echo "   pass --sign-identity \"Developer ID Application: ...\" to notarize"
fi

# ── package ───────────────────────────────────────────────────────────────────

ARCHIVE="ax-engine-${TAG}-macos-arm64.tar.gz"
ARCHIVE_PATH="/tmp/${ARCHIVE}"
STAGING_DIR="$(mktemp -d /tmp/ax-engine-release-payload.XXXXXX)"
TAP_DIR=""
trap 'rm -rf "${TAP_DIR:-}" "$STAGING_DIR"' EXIT

echo "▶ packaging ${ARCHIVE}…"
release_payload=()
for bin in "${RELEASE_BINS[@]}"; do
    cp "target/release/$bin" "$STAGING_DIR/$bin"
    chmod +x "$STAGING_DIR/$bin"
    release_payload+=("$bin")
done
for mapping in "${RELEASE_HELPER_SOURCES[@]}"; do
    source_path="${mapping%%:*}"
    install_name="${mapping#*:}"
    cp "$source_path" "$STAGING_DIR/$install_name"
    chmod +x "$STAGING_DIR/$install_name"
    release_payload+=("$install_name")
done
tar -czf "$ARCHIVE_PATH" -C "$STAGING_DIR" "${release_payload[@]}"
SHA256=$(shasum -a 256 "$ARCHIVE_PATH" | awk '{print $1}')
ARCHIVE_SIZE=$(du -sh "$ARCHIVE_PATH" | awk '{print $1}')

DOWNLOAD_URL="https://github.com/${MAIN_REPO}/releases/download/${TAG}/${ARCHIVE}"

echo ""
echo "  archive:  $ARCHIVE_PATH  ($ARCHIVE_SIZE)"
echo "  sha256:   $SHA256"
echo "  url:      $DOWNLOAD_URL"
echo ""

# ── minisign ─────────────────────────────────────────────────────────────────

MINISIGN_SIG_PATH="${ARCHIVE_PATH}.minisig"

if [[ "$MINISIGN" = true ]]; then
    echo "▶ signing archive with minisign…"
    TRUSTED_COMMENT="ax-engine ${TAG} macos-arm64 sha256 ${SHA256}"
    minisign_args=(
        --secret-key "$MINISIGN_SECRET_KEY"
        --public-key "$MINISIGN_PUBLIC_KEY"
        --trusted-comment "$TRUSTED_COMMENT"
        --force
        "$ARCHIVE_PATH"
    )
    if [[ -n "$MINISIGN_PUBLIC_KEY_STRING" ]]; then
        minisign_args+=(--public-key-string "$MINISIGN_PUBLIC_KEY_STRING")
    fi
    "$SCRIPT_DIR/minisign-artifact.sh" "${minisign_args[@]}"
    echo "  signature: $MINISIGN_SIG_PATH"
else
    echo "▶ skipping minisign archive signature (pass --minisign to enable)"
fi
echo ""

if [[ "$DRY_RUN" = true ]]; then
    echo "▶ dry-run: skipping upload and tap update"
    echo ""
    echo "  tag:      $TAG"
    echo "  version:  $VERSION"
    if [[ "$MINISIGN" = true ]]; then
        echo "  minisig:  $MINISIGN_SIG_PATH"
    fi
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
    echo "▶ uploading to GitHub release ${TAG}…"
    upload_paths=("$ARCHIVE_PATH")
    if [[ "$MINISIGN" = true ]]; then
        upload_paths+=("$MINISIGN_SIG_PATH")
    fi
    gh release upload "$TAG" "${upload_paths[@]}" --repo "$MAIN_REPO" --clobber
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

# If the local-test step taps TAP_DIR (see below), Homebrew records TAP_DIR as
# the tap's git origin. Once we remove TAP_DIR on exit, `brew update` against
# that tap fails with "does not appear to be a git repository". Before deleting
# TAP_DIR, repoint any installed tap whose origin matches TAP_DIR to the
# canonical GitHub URL so the developer's `brew update` keeps working.
cleanup_tap_dir() {
    local installed_tap_dir
    installed_tap_dir="$(brew --repository defai-digital/ax-engine 2>/dev/null || true)"
    if [[ -n "$installed_tap_dir" && -d "$installed_tap_dir/.git" ]]; then
        local current_origin
        current_origin="$(git -C "$installed_tap_dir" remote get-url origin 2>/dev/null || true)"
        if [[ "$current_origin" == "$TAP_DIR" ]]; then
            git -C "$installed_tap_dir" remote set-url origin \
                "https://github.com/${TAP_REPO}.git" 2>/dev/null || true
        fi
    fi
    rm -rf "$TAP_DIR" "$STAGING_DIR"
}
trap cleanup_tap_dir EXIT

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

echo "▶ updating formula to ${VERSION}…"

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

python3 - "$FORMULA_PATH" "${release_payload[@]}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

formula_path = Path(sys.argv[1])
bins = sys.argv[2:]
text = formula_path.read_text(encoding="utf-8")

# Match any bin.install line(s) — handles both combined and separate forms
bin_re = re.compile(r'^(?P<indent>[ \t]*)bin\.install\b(?P<rest>[^\n]*)$', re.MULTILINE)
matches = list(bin_re.finditer(text))
if not matches:
    raise SystemExit("formula has no bin.install lines — cannot update automatically")

indent = matches[0].group("indent")

# RELEASE_BINS is the source of truth. Replace all bin.install lines with one
# canonical line listing exactly RELEASE_BINS in order. This removes stale
# entries from prior releases when a binary is dropped from the build.
combined = indent + "bin.install " + ", ".join(f'"{b}"' for b in bins)

# Canonical already — single bin.install line, exact match
if len(matches) == 1 and matches[0].group(0) == combined:
    raise SystemExit(0)

# Replace the first bin.install with the canonical line; remove subsequent ones
counter: list[int] = [0]

def replacer(m: re.Match) -> str:
    counter[0] += 1
    return combined if counter[0] == 1 else ""

new_text = bin_re.sub(replacer, text)
# Clean up any extra blank lines left by removed lines
new_text = re.sub(r'\n{3,}', '\n\n', new_text)
formula_path.write_text(new_text, encoding="utf-8")
PY

# Verify the canonical bin.install line landed and no stale binary entries remain.
EXPECTED_BIN_LINE="bin.install $(printf '"%s", ' "${RELEASE_BINS[@]}" | sed 's/, $//')"
if ! grep -qF "$EXPECTED_BIN_LINE" "$FORMULA_PATH"; then
    echo "error: formula does not contain canonical line: $EXPECTED_BIN_LINE" >&2
    echo "       check the formula manually: $FORMULA_PATH" >&2
    exit 1
fi

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
