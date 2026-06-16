#!/usr/bin/env bash
#
# Generate the local minisign keypair used for ax-engine release artifacts.
# The engine uses the shared ax-code signing key by default
# (~/signkey/ax-code.sec / ~/signkey/ax-code.pub).

set -euo pipefail

KEY_DIR="${AX_MINISIGN_KEY_DIR:-${SIGNKEY_DIR:-$HOME/signkey}}"
SECRET_KEY="${AX_MINISIGN_SECRET_KEY:-}"
PUBLIC_KEY="${AX_MINISIGN_PUBLIC_KEY:-}"
FORCE=false
NO_PASSWORD=false
DRY_RUN=false

usage() {
    cat <<'EOF'
usage: scripts/minisign-keygen.sh [options]

Generate the minisign keypair for signing ax-engine release artifacts.

Options:
  --key-dir <path>          Directory for generated keys (default: ~/signkey)
  --secret-key <path>       Secret key path (default: <key-dir>/ax-code.sec)
  --public-key <path>       Public key path (default: <key-dir>/ax-code.pub)
  --force                   Overwrite an existing keypair
  --allow-unencrypted-test-key
                            Generate an unencrypted secret key for short-lived
                            tests (alias: --no-password)
  --dry-run                 Print what would be done
  -h, --help                Show this help

Environment:
  AX_MINISIGN_KEY_DIR, SIGNKEY_DIR   Directory for generated keys.
  AX_MINISIGN_SECRET_KEY             Overrides the default secret key path.
  AX_MINISIGN_PUBLIC_KEY             Overrides the default public key path.
EOF
}

path_mode() {
    stat -f "%Lp" "$1" 2>/dev/null || stat -c "%a" "$1" 2>/dev/null || true
}

ensure_private_permissions() {
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

public_key_id() {
    [[ -f "$PUBLIC_KEY" ]] || return 0
    awk '/^untrusted comment: minisign public key / { print $NF; exit }' "$PUBLIC_KEY"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --key-dir)
            shift
            [[ -n "${1:-}" ]] || { echo "error: --key-dir requires an argument" >&2; exit 1; }
            KEY_DIR="$1"
            ;;
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
        --force)
            FORCE=true
            ;;
        --allow-unencrypted-test-key|--no-password)
            NO_PASSWORD=true
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
            echo "error: unexpected argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

SECRET_KEY="${SECRET_KEY:-$KEY_DIR/ax-code.sec}"
PUBLIC_KEY="${PUBLIC_KEY:-$KEY_DIR/ax-code.pub}"
SECRET_KEY_DIR="$(dirname "$SECRET_KEY")"
PUBLIC_KEY_DIR="$(dirname "$PUBLIC_KEY")"

if ! command -v minisign &>/dev/null; then
    echo "error: minisign is not installed (try: brew install minisign)" >&2
    exit 1
fi

if [[ "$FORCE" != true ]]; then
    if [[ -e "$SECRET_KEY" || -e "$PUBLIC_KEY" ]]; then
        echo "error: refusing to overwrite an existing keypair:" >&2
        echo "       $SECRET_KEY" >&2
        echo "       $PUBLIC_KEY" >&2
        echo "       pass --force only if you intentionally want to rotate the signing key" >&2
        exit 1
    fi
fi

minisign_args=(-G -s "$SECRET_KEY" -p "$PUBLIC_KEY")
if [[ "$FORCE" = true ]]; then
    minisign_args+=(-f)
fi
if [[ "$NO_PASSWORD" = true ]]; then
    minisign_args+=(-W)
fi

echo "Key directory: $KEY_DIR"
echo "Secret key:    $SECRET_KEY"
echo "Public key:    $PUBLIC_KEY"

if [[ "$DRY_RUN" = true ]]; then
    printf 'would ensure private secret-key directory: %s\n' "$SECRET_KEY_DIR"
    printf 'would ensure public-key directory exists: %s\n' "$PUBLIC_KEY_DIR"
    printf 'would run: minisign'
    printf ' %s' "${minisign_args[@]}"
    printf '\n'
    exit 0
fi

umask 077
if [[ -d "$SECRET_KEY_DIR" ]]; then
    ensure_private_permissions "$SECRET_KEY_DIR" "secret key directory"
else
    mkdir -p "$SECRET_KEY_DIR"
    chmod 700 "$SECRET_KEY_DIR"
fi
mkdir -p "$PUBLIC_KEY_DIR"

if [[ "$NO_PASSWORD" = true ]]; then
    echo "warning: generating an unencrypted test key; do not use it for releases" >&2
else
    echo "minisign will prompt for a secret-key password; keep it out of shell history and chat."
fi

minisign "${minisign_args[@]}"

chmod 600 "$SECRET_KEY"
chmod 644 "$PUBLIC_KEY"
ensure_private_permissions "$SECRET_KEY_DIR" "secret key directory"
ensure_private_permissions "$SECRET_KEY" "secret key"

echo ""
echo "Generated minisign keypair."
echo "Public key id: $(public_key_id)"
echo "Publish this public key for verification:"
cat "$PUBLIC_KEY"
