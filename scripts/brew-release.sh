#!/usr/bin/env bash
# Legacy local Homebrew release preview.
#
# The canonical publisher owns artifact assembly, MLX runtime staging, Mach-O
# rewrites, signing, and validation. This compatibility wrapper is intentionally
# dry-run-only so those security-sensitive steps cannot drift into a second
# implementation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
usage: scripts/brew-release.sh <vX.Y.Z> --dry-run [options]

Delegates a local, non-mutating preview to publish-github-release.sh.

Options:
  --dry-run                  Required.
  --skip-build               Reuse existing target/release binaries.
  --skip-upload              Accepted for compatibility; dry-run never uploads.
  --skip-tap                 Accepted for compatibility; dry-run never updates the tap.
  --skip-test                Accepted for compatibility; legacy previews skip release gates.
  --minisign                 Minisign the preview artifacts.
  --minisign-key <path>      Override the Minisign secret key.
  --minisign-pubkey <path>   Override the Minisign public key.
  --minisign-public-key <k>  Override the Minisign public key material.
  --sign-identity <id>       Developer ID-sign the preview payload.
  -h, --help                 Show this help.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

TAG=""
DRY_RUN=false
SKIP_BUILD=false
MINISIGN=false
MINISIGN_SECRET_KEY=""
MINISIGN_PUBLIC_KEY=""
MINISIGN_PUBLIC_KEY_STRING=""
SIGN_IDENTITY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            ;;
        --skip-build)
            SKIP_BUILD=true
            ;;
        --skip-upload|--skip-tap|--skip-test)
            # All are inherent in the delegated legacy preview.
            ;;
        --minisign)
            MINISIGN=true
            ;;
        --minisign-key)
            shift
            [[ -n "${1:-}" ]] || die "--minisign-key requires an argument"
            MINISIGN_SECRET_KEY="$1"
            ;;
        --minisign-pubkey)
            shift
            [[ -n "${1:-}" ]] || die "--minisign-pubkey requires an argument"
            MINISIGN_PUBLIC_KEY="$1"
            ;;
        --minisign-public-key)
            shift
            [[ -n "${1:-}" ]] || die "--minisign-public-key requires an argument"
            MINISIGN_PUBLIC_KEY_STRING="$1"
            ;;
        --sign-identity)
            shift
            [[ -n "${1:-}" ]] || die "--sign-identity requires an argument"
            SIGN_IDENTITY="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        v*.*)
            [[ -z "$TAG" ]] || die "tag specified more than once"
            TAG="$1"
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
    shift
done

[[ -n "$TAG" ]] || {
    usage >&2
    die "release tag is required"
}
if [[ "$DRY_RUN" = false ]]; then
    die "scripts/brew-release.sh is a legacy preview and may not publish releases"
fi

canonical_args=(
    "$TAG"
    --dry-run
    --skip-checks
    --allow-dirty
)
[[ "$SKIP_BUILD" = false ]] || canonical_args+=(--skip-build)
if [[ "$MINISIGN" = false ]]; then
    canonical_args+=(--no-minisign)
else
    [[ -z "$MINISIGN_SECRET_KEY" ]] || {
        canonical_args+=(--minisign-key "$MINISIGN_SECRET_KEY")
    }
    [[ -z "$MINISIGN_PUBLIC_KEY" ]] || {
        canonical_args+=(--minisign-pubkey "$MINISIGN_PUBLIC_KEY")
    }
    [[ -z "$MINISIGN_PUBLIC_KEY_STRING" ]] || {
        canonical_args+=(--minisign-public-key "$MINISIGN_PUBLIC_KEY_STRING")
    }
fi
[[ -z "$SIGN_IDENTITY" ]] || canonical_args+=(--sign-identity "$SIGN_IDENTITY")

echo "Delegating standalone Homebrew preview to the canonical release publisher."
exec "$SCRIPT_DIR/publish-github-release.sh" "${canonical_args[@]}"
