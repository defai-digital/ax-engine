#!/usr/bin/env bash
# Sign one or more release artifacts with minisign.

set -euo pipefail

SECRET_KEY="${AX_MINISIGN_SECRET_KEY:-$HOME/signkey/ax-engine.minisign.key}"
PUBLIC_KEY="${AX_MINISIGN_PUBLIC_KEY:-$HOME/signkey/ax-engine.minisign.pub}"
PUBLIC_KEY_STRING="${AX_MINISIGN_PUBLIC_KEY_STRING:-}"
TRUSTED_COMMENT="${AX_MINISIGN_TRUSTED_COMMENT:-}"
MINISIGN_PASSWORD="${AX_MINISIGN_PASSWORD:-}"
MINISIGN_KEYCHAIN_SERVICE="${AX_MINISIGN_KEYCHAIN_SERVICE:-ax-engine-minisign}"
MINISIGN_KEYCHAIN_ACCOUNT="${AX_MINISIGN_KEYCHAIN_ACCOUNT:-ax-engine-release}"
SIGNATURE_DIR=""
VERIFY=true
FORCE=false
FILES=()

usage() {
    cat <<'EOF'
usage: scripts/minisign-artifact.sh [options] <artifact> [artifact ...]

Options:
  --secret-key <path>       Minisign secret key. Default: ~/signkey/ax-engine.minisign.key
  --public-key <path>       Minisign public key. Default: ~/signkey/ax-engine.minisign.pub
  --public-key-string <key> Minisign public key string for verification.
  --signature-dir <dir>     Write signatures to this directory.
  --trusted-comment <text>  Trusted comment embedded in the signature.
                            Default: ax-engine artifact <name> sha256 <digest>
  --force                   Overwrite an existing signature file.
  --no-verify               Do not verify generated signatures after signing.
  --keychain-service <svc>  macOS Keychain service name for passphrase lookup.
                            Default: ax-engine-minisign
  --keychain-account <acct> macOS Keychain account name for passphrase lookup.
                            Default: ax-engine-release

Environment:
  AX_MINISIGN_SECRET_KEY         Overrides the default secret key path.
  AX_MINISIGN_PUBLIC_KEY         Overrides the default public key path.
  AX_MINISIGN_PUBLIC_KEY_STRING  Public key string for verification.
  AX_MINISIGN_TRUSTED_COMMENT    Default trusted comment.
  AX_MINISIGN_PASSWORD           Key passphrase (prefer Keychain over this).
  AX_MINISIGN_KEYCHAIN_SERVICE   Keychain service name (default: ax-engine-minisign).
  AX_MINISIGN_KEYCHAIN_ACCOUNT   Keychain account name (default: ax-engine-release).

Keychain setup (one-time, macOS):
  security add-generic-password -U -a ax-engine-release -s ax-engine-minisign -w
  # (you will be prompted for the passphrase)
EOF
}

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "error: '$1' not found" >&2
        exit 1
    fi
}

path_mode() {
    stat -f "%Lp" "$1" 2>/dev/null || stat -c "%a" "$1" 2>/dev/null || true
}

require_private_path() {
    local path="$1"
    local label="$2"
    local mode

    mode="$(path_mode "$path")"
    if [[ -z "$mode" ]]; then
        echo "error: could not inspect permissions for $label: $path" >&2
        exit 1
    fi

    if (( 10#$mode % 100 != 0 )); then
        echo "error: $label must not be group/world accessible: $path has mode $mode" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --secret-key)
            shift
            if [[ -z "${1:-}" ]]; then
                echo "error: --secret-key requires an argument" >&2
                exit 1
            fi
            SECRET_KEY="$1"
            ;;
        --public-key)
            shift
            if [[ -z "${1:-}" ]]; then
                echo "error: --public-key requires an argument" >&2
                exit 1
            fi
            PUBLIC_KEY="$1"
            ;;
        --public-key-string)
            shift
            if [[ -z "${1:-}" ]]; then
                echo "error: --public-key-string requires an argument" >&2
                exit 1
            fi
            PUBLIC_KEY_STRING="$1"
            ;;
        --signature-dir)
            shift
            if [[ -z "${1:-}" ]]; then
                echo "error: --signature-dir requires an argument" >&2
                exit 1
            fi
            SIGNATURE_DIR="$1"
            ;;
        --trusted-comment)
            shift
            if [[ -z "${1:-}" ]]; then
                echo "error: --trusted-comment requires an argument" >&2
                exit 1
            fi
            TRUSTED_COMMENT="$1"
            ;;
        --keychain-service)
            shift
            if [[ -z "${1:-}" ]]; then
                echo "error: --keychain-service requires an argument" >&2
                exit 1
            fi
            MINISIGN_KEYCHAIN_SERVICE="$1"
            ;;
        --keychain-account)
            shift
            if [[ -z "${1:-}" ]]; then
                echo "error: --keychain-account requires an argument" >&2
                exit 1
            fi
            MINISIGN_KEYCHAIN_ACCOUNT="$1"
            ;;
        --force)
            FORCE=true
            ;;
        --no-verify)
            VERIFY=false
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            FILES+=("$1")
            ;;
    esac
    shift
done

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "error: at least one artifact path is required" >&2
    usage >&2
    exit 1
fi

check_cmd minisign
check_cmd shasum

if [[ ! -f "$SECRET_KEY" ]]; then
    echo "error: minisign secret key not found: $SECRET_KEY" >&2
    echo "       create it with:" >&2
    echo "       mkdir -p \"$(dirname "$SECRET_KEY")\" && chmod 700 \"$(dirname "$SECRET_KEY")\"" >&2
    echo "       minisign -G -p \"$PUBLIC_KEY\" -s \"$SECRET_KEY\"" >&2
    exit 1
fi

if [[ ! -r "$SECRET_KEY" ]]; then
    echo "error: minisign secret key is not readable: $SECRET_KEY" >&2
    exit 1
fi

require_private_path "$SECRET_KEY" "minisign secret key"
require_private_path "$(dirname "$SECRET_KEY")" "minisign secret key directory"

if [[ "$VERIFY" = true && -z "$PUBLIC_KEY_STRING" && ! -f "$PUBLIC_KEY" ]]; then
    echo "error: minisign public key not found: $PUBLIC_KEY" >&2
    echo "       recreate it with: minisign -R -s \"$SECRET_KEY\" -p \"$PUBLIC_KEY\"" >&2
    echo "       or pass --public-key-string <base64-key>" >&2
    exit 1
fi

# Retrieve the minisign key passphrase: env var > macOS Keychain > "" (let minisign prompt).
_minisign_password_from_keychain() {
    [[ "$(uname -s)" == "Darwin" ]] || return 0
    security find-generic-password -w \
        -s "$MINISIGN_KEYCHAIN_SERVICE" \
        -a "$MINISIGN_KEYCHAIN_ACCOUNT" 2>/dev/null || true
}

_minisign_resolve_password() {
    if [[ -n "$MINISIGN_PASSWORD" ]]; then
        printf '%s' "$MINISIGN_PASSWORD"
    else
        _minisign_password_from_keychain
    fi
}

# Run minisign, piping the passphrase if one was resolved.
_run_minisign() {
    local pw
    pw="$(_minisign_resolve_password)"
    if [[ -n "$pw" ]]; then
        printf '%s\n' "$pw" | minisign "$@"
    else
        minisign "$@"
    fi
}

if [[ -n "$SIGNATURE_DIR" ]]; then
    mkdir -p "$SIGNATURE_DIR"
fi

for artifact in "${FILES[@]}"; do
    if [[ ! -f "$artifact" ]]; then
        echo "error: artifact not found: $artifact" >&2
        exit 1
    fi

    if [[ -n "$SIGNATURE_DIR" ]]; then
        sig="${SIGNATURE_DIR}/$(basename "$artifact").minisig"
    else
        sig="${artifact}.minisig"
    fi

    if [[ -e "$sig" && "$FORCE" = false ]]; then
        echo "error: signature already exists: $sig" >&2
        echo "       pass --force to overwrite it" >&2
        exit 1
    fi

    comment="$TRUSTED_COMMENT"
    if [[ -z "$comment" ]]; then
        sha256="$(shasum -a 256 "$artifact" | awk '{print $1}')"
        comment="ax-engine artifact $(basename "$artifact") sha256 $sha256"
    fi

    args=(-S -s "$SECRET_KEY" -m "$artifact" -x "$sig" -t "$comment")
    if [[ "$FORCE" = true ]]; then
        rm -f "$sig"
    fi

    _run_minisign "${args[@]}"
    echo "signed: $sig"

    if [[ "$VERIFY" = true ]]; then
        verify_args=(-V -m "$artifact" -x "$sig")
        if [[ -n "$PUBLIC_KEY_STRING" ]]; then
            verify_args+=(-P "$PUBLIC_KEY_STRING")
        else
            verify_args+=(-p "$PUBLIC_KEY")
        fi
        minisign "${verify_args[@]}"
        echo "verified: $sig"
    fi
done
