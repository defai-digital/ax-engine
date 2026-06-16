#!/usr/bin/env bash
# Sign one or more release artifacts with minisign.
#
# Hardened signer modeled on the ax-code-desktop release-signing process:
# optional pinned-public-key enforcement, dry-run, untrusted comment, portable
# SHA-256, and a UTC timestamp in the default trusted comment. The shared
# ax-code signing key is the default; operators may pin its public key string
# via AX_MINISIGN_PINNED_PUBLIC_KEY (CI secret / shell rc) so the signer fails
# closed if the local keypair ever drifts from the expected release key.

set -euo pipefail

SECRET_KEY="${AX_MINISIGN_SECRET_KEY:-$HOME/signkey/ax-code.sec}"
PUBLIC_KEY="${AX_MINISIGN_PUBLIC_KEY:-$HOME/signkey/ax-code.pub}"
PUBLIC_KEY_STRING="${AX_MINISIGN_PUBLIC_KEY_STRING:-}"
PINNED_PUBLIC_KEY="${AX_MINISIGN_PINNED_PUBLIC_KEY:-}"
TRUSTED_COMMENT="${AX_MINISIGN_TRUSTED_COMMENT:-}"
UNTRUSTED_COMMENT="${AX_MINISIGN_UNTRUSTED_COMMENT:-signature from ax-engine local signing key}"
MINISIGN_PASSWORD="${AX_MINISIGN_PASSWORD:-}"
MINISIGN_KEYCHAIN_SERVICE="${AX_MINISIGN_KEYCHAIN_SERVICE:-ax-code-minisign}"
MINISIGN_KEYCHAIN_ACCOUNT="${AX_MINISIGN_KEYCHAIN_ACCOUNT:-ax-code-release}"
SIGNATURE_DIR=""
VERIFY=true
FORCE=false
DRY_RUN=false
FILES=()

usage() {
    cat <<'EOF'
usage: scripts/minisign-artifact.sh [options] <artifact> [artifact ...]

Options:
  --secret-key <path>       Minisign secret key. Default: ~/signkey/ax-code.sec
  --public-key <path>       Minisign public key. Default: ~/signkey/ax-code.pub
  --public-key-string <key> Minisign public key string for verification.
  --pinned-public-key <key> Fail unless the public key file matches this key.
                             Default: AX_MINISIGN_PINNED_PUBLIC_KEY (empty =
                             unenforced).
  --signature-dir <dir>     Write signatures to this directory.
  --trusted-comment <text>  Trusted comment embedded in the signature.
                             Default: ax-engine artifact <name> sha256=<digest> signed=<utc>
  --untrusted-comment <text>
                            Untrusted comment embedded in the signature.
                             Default: "signature from ax-engine local signing key"
  --force                   Overwrite an existing signature file.
  --no-verify               Do not verify generated signatures after signing.
  --dry-run                 Print what would be signed; create nothing.
  --keychain-service <svc>  macOS Keychain service name for passphrase lookup.
                             Default: ax-code-minisign
  --keychain-account <acct> macOS Keychain account name for passphrase lookup.
                             Default: ax-code-release

Environment:
  AX_MINISIGN_SECRET_KEY         Overrides the default secret key path.
  AX_MINISIGN_PUBLIC_KEY         Overrides the default public key path.
  AX_MINISIGN_PUBLIC_KEY_STRING  Public key string for verification.
  AX_MINISIGN_PINNED_PUBLIC_KEY  Expected public key; signing fails on mismatch.
  AX_MINISIGN_TRUSTED_COMMENT    Default trusted comment.
  AX_MINISIGN_UNTRUSTED_COMMENT  Default untrusted comment.
  AX_MINISIGN_PASSWORD           Key passphrase (prefer Keychain over this).
  AX_MINISIGN_KEYCHAIN_SERVICE   Keychain service name (default: ax-code-minisign).
  AX_MINISIGN_KEYCHAIN_ACCOUNT   Keychain account name (default: ax-code-release).

Keychain setup (one-time, macOS):
  security add-generic-password -U -a ax-code-release -s ax-code-minisign -w
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

    if (( 8#$mode & 8#077 )); then
        echo "error: $label must not be group/world accessible: $path has mode $mode" >&2
        echo "       run: chmod 600 '$path'" >&2
        exit 1
    fi
}

# Public key material (the RW... line) from a minisign public key file.
public_key_material() {
    [[ -f "$PUBLIC_KEY" ]] || return 0
    awk '/^RW/ { print $1; exit }' "$PUBLIC_KEY"
}

# Public key id (the hex tail of the "untrusted comment" line).
public_key_id() {
    [[ -f "$PUBLIC_KEY" ]] || return 0
    awk '/^untrusted comment: minisign public key / { print $NF; exit }' "$PUBLIC_KEY"
}

require_pinned_public_key() {
    [[ -n "$PINNED_PUBLIC_KEY" ]] || return 0
    local actual
    actual="$(public_key_material)"
    if [[ -z "$actual" ]]; then
        echo "error: could not read minisign public key material from $PUBLIC_KEY" >&2
        echo "       (pinned key enforcement requires a readable public key file)" >&2
        exit 1
    fi
    if [[ "$actual" != "$PINNED_PUBLIC_KEY" ]]; then
        echo "error: minisign public key does not match the pinned ax-engine release key" >&2
        echo "       expected: $PINNED_PUBLIC_KEY" >&2
        echo "       actual:   $actual" >&2
        echo "       unset AX_MINISIGN_PINNED_PUBLIC_KEY only if you intentionally changed keys" >&2
        exit 1
    fi
}

# Portable SHA-256 hex digest: sha256sum (Linux) then shasum (macOS).
sha256_hex() {
    local file="$1"
    if command -v sha256sum &>/dev/null; then
        sha256sum "$file" | awk '{print $1}'
    else
        shasum -a 256 "$file" | awk '{print $1}'
    fi
}

trusted_comment_for_file() {
    local file="$1"
    local digest
    if [[ -n "$TRUSTED_COMMENT" ]]; then
        printf '%s\n' "$TRUSTED_COMMENT"
        return
    fi
    digest="$(sha256_hex "$file")"
    printf 'ax-engine artifact %s sha256=%s signed=%s\n' \
        "$(basename "$file")" "$digest" "$SIGNED_AT"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --secret-key)
            shift
            [[ -n "${1:-}" ]] || { echo "error: --secret-key requires an argument" >&2; exit 1; }
            SECRET_KEY="$1"
            ;;
        --public-key)
            shift
            [[ -n "${1:-}" ]] || { echo "error: --public-key requires an argument" >&2; exit 1; }
            PUBLIC_KEY="$1"
            ;;
        --public-key-string)
            shift
            [[ -n "${1:-}" ]] || { echo "error: --public-key-string requires an argument" >&2; exit 1; }
            PUBLIC_KEY_STRING="$1"
            ;;
        --pinned-public-key)
            shift
            [[ -n "${1:-}" ]] || { echo "error: --pinned-public-key requires an argument" >&2; exit 1; }
            PINNED_PUBLIC_KEY="$1"
            ;;
        --signature-dir)
            shift
            [[ -n "${1:-}" ]] || { echo "error: --signature-dir requires an argument" >&2; exit 1; }
            SIGNATURE_DIR="$1"
            ;;
        --trusted-comment)
            shift
            [[ -n "${1:-}" ]] || { echo "error: --trusted-comment requires an argument" >&2; exit 1; }
            TRUSTED_COMMENT="$1"
            ;;
        --untrusted-comment)
            shift
            [[ -n "${1:-}" ]] || { echo "error: --untrusted-comment requires an argument" >&2; exit 1; }
            UNTRUSTED_COMMENT="$1"
            ;;
        --keychain-service)
            shift
            [[ -n "${1:-}" ]] || { echo "error: --keychain-service requires an argument" >&2; exit 1; }
            MINISIGN_KEYCHAIN_SERVICE="$1"
            ;;
        --keychain-account)
            shift
            [[ -n "${1:-}" ]] || { echo "error: --keychain-account requires an argument" >&2; exit 1; }
            MINISIGN_KEYCHAIN_ACCOUNT="$1"
            ;;
        --force)
            FORCE=true
            ;;
        --no-verify)
            VERIFY=false
            ;;
        --dry-run)
            DRY_RUN=true
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

if [[ "$DRY_RUN" != true ]]; then
    check_cmd minisign
fi
# shasum/sha256sum is only needed to build the default trusted comment; require
# it unconditionally so the contract does not silently change under dry-run.
if command -v sha256sum &>/dev/null || command -v shasum &>/dev/null; then
    :
else
    echo "error: neither sha256sum nor shasum is available" >&2
    exit 1
fi

SIGNED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "$DRY_RUN" != true ]]; then
    if [[ ! -f "$SECRET_KEY" ]]; then
        echo "error: minisign secret key not found: $SECRET_KEY" >&2
        echo "       create it with: bash scripts/minisign-keygen.sh" >&2
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

    # Fail closed if a pin is configured and the local public key diverges.
    if [[ -n "$PINNED_PUBLIC_KEY" ]]; then
        require_pinned_public_key
    fi
fi

# Retrieve the minisign key passphrase: env var > macOS Keychain > "" (let minisign prompt).
_minisign_password_from_keychain() {
    [[ "$(uname -s)" == "Darwin" ]] || return 1
    command -v security &>/dev/null || return 1
    security find-generic-password -w \
        -s "$MINISIGN_KEYCHAIN_SERVICE" \
        -a "$MINISIGN_KEYCHAIN_ACCOUNT" 2>/dev/null || return 1
}

_minisign_resolve_password() {
    if [[ -n "$MINISIGN_PASSWORD" ]]; then
        printf '%s' "$MINISIGN_PASSWORD"
        return 0
    fi
    _minisign_password_from_keychain || true
}

# Run minisign, piping the passphrase if one was resolved.
_run_minisign() {
    local pw
    pw="$(_minisign_resolve_password || true)"
    if [[ -n "$pw" ]]; then
        printf '%s\n' "$pw" | minisign "$@"
    else
        minisign "$@"
    fi
}

if [[ -n "$SIGNATURE_DIR" ]]; then
    mkdir -p "$SIGNATURE_DIR"
fi

echo "Secret key: $SECRET_KEY"
if [[ "$VERIFY" = true && -z "$DRY_RUN" ]]; then
    if [[ -n "$PUBLIC_KEY_STRING" ]]; then
        echo "Public key (string): $PUBLIC_KEY_STRING"
    elif [[ -f "$PUBLIC_KEY" ]]; then
        echo "Public key: $PUBLIC_KEY"
        local_key_id="$(public_key_id)"
        [[ -n "$local_key_id" ]] && echo "Public key id: $local_key_id"
        if [[ -n "$PINNED_PUBLIC_KEY" ]]; then
            echo "Pinned public key: $PINNED_PUBLIC_KEY (enforced)"
        fi
    fi
fi
if [[ "$UNTRUSTED_COMMENT" != "signature from ax-engine local signing key" ]]; then
    echo "Untrusted comment: $UNTRUSTED_COMMENT"
fi

# Resolve output paths and validate existence up front so dry-run is accurate.
declare -a SIG_PATHS=()
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
    SIG_PATHS+=("$sig")
    echo "  $(basename "$artifact") -> $sig"
done

if [[ "$DRY_RUN" = true ]]; then
    echo "Dry run only; no signatures were created."
    exit 0
fi

for i in "${!FILES[@]}"; do
    artifact="${FILES[$i]}"
    sig="${SIG_PATHS[$i]}"
    comment="$(trusted_comment_for_file "$artifact")"

    args=(-S -s "$SECRET_KEY" -m "$artifact" -x "$sig" -t "$comment" -c "$UNTRUSTED_COMMENT")
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

echo "Created signatures:"
for sig in "${SIG_PATHS[@]}"; do
    echo "  $sig"
done
