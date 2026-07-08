#!/usr/bin/env bash
# Publish an AX Engine GitHub release from the local checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MAIN_REPO="${AX_RELEASE_REPO:-defai-digital/ax-engine}"
RELEASE_BINS=(ax-engine ax-engine-server ax-engine-bench)
MACOS_RELEASE_ENTITLEMENTS="$ROOT_DIR/scripts/macos-release.entitlements.plist"
RELEASE_HELPER_SOURCES=(
    "scripts/download_model.py:ax-engine-download-model.py"
    "scripts/prepare_mtp_sidecar.py:ax-engine-prepare-mtp-sidecar.py"
    "scripts/prepare_gemma4_assistant_mtp.py:ax-engine-prepare-gemma4-assistant-mtp.py"
    "scripts/prepare_glm_mtp_sidecar.py:ax-engine-prepare-glm-mtp-sidecar.py"
    "scripts/check_mtp_sidecar_provenance.py:ax-engine-check-mtp-sidecar-provenance.py"
)

TAG=""
DRY_RUN=false
SKIP_CHECKS=false
SKIP_BUILD=false
SKIP_TAG_PUSH=false
ALLOW_DIRTY=false
MINISIGN=true
MINISIGN_SECRET_KEY="${AX_MINISIGN_SECRET_KEY:-$HOME/signkey/ax-code.sec}"
MINISIGN_PUBLIC_KEY="${AX_MINISIGN_PUBLIC_KEY:-$HOME/signkey/ax-code.pub}"
MINISIGN_PUBLIC_KEY_STRING="${AX_MINISIGN_PUBLIC_KEY_STRING:-}"
SIGN_IDENTITY="${AX_CODESIGN_IDENTITY:-}"
NOTARY_PROFILE="${AX_NOTARY_PROFILE:-ax-engine-notary}"
APPLE_API_KEY="${APPLE_API_KEY:-}"
APPLE_API_KEY_B64="${APPLE_API_KEY_B64:-}"
APPLE_API_KEY_ID="${APPLE_API_KEY_ID:-}"
APPLE_API_ISSUER="${APPLE_API_ISSUER:-}"
SKIP_NOTARIZATION=false
ARTIFACT_DIR=""
TITLE=""
NOTES_FILE=""
DRAFT=false
PRERELEASE=false
CLOBBER_ASSETS=false
APPLE_API_KEY_TEMP=""

cleanup() {
    if [[ -n "$APPLE_API_KEY_TEMP" ]]; then
        rm -f "$APPLE_API_KEY_TEMP"
    fi
}

trap cleanup EXIT

usage() {
    cat <<'EOF'
usage: scripts/publish-github-release.sh <vX.Y.Z> [options]

Publishes the macOS arm64 GitHub release assets for AX Engine:
  1. validate clean checkout and tag/version consistency
  2. run release gates
  3. build release binaries
  4. optionally Apple Developer ID sign and notarize binaries
  5. package tarball, sha256, and release manifest
  6. minisign artifacts by default
  7. push the tag, create/update the GitHub release, upload assets, verify assets

Options:
  --dry-run                  Run local checks/build/package/sign, but do not push or upload.
  --skip-checks              Skip release gates.
  --skip-build               Reuse existing target/release binaries.
  --skip-tag-push            Do not push the tag before creating/uploading the release.
  --allow-dirty              Allow a dirty git worktree.
  --artifact-dir <dir>       Output directory. Default: target/release-artifacts/<tag>
  --repo <owner/name>        GitHub repository. Default: defai-digital/ax-engine
  --title <text>             Release title. Default: <tag>
  --notes-file <path>        Release notes file. Default: gh --generate-notes.
  --draft                    Create the release as draft.
  --prerelease               Mark the release as prerelease.
  --clobber-assets           Overwrite existing release assets when uploading.
  --no-minisign              Do not sign release artifacts.
  --minisign-key <path>      Secret key path. Default: ~/signkey/ax-code.sec
  --minisign-pubkey <path>   Public key file path. Default: ~/signkey/ax-code.pub
  --minisign-public-key <k>  Public key string for verification.
  --sign-identity <id>       Developer ID Application identity for codesign.
                             Can also be set with AX_CODESIGN_IDENTITY.
  --notary-profile <name>    notarytool Keychain profile. Default: ax-engine-notary.
                             Ignored when Apple API key flags/env are provided.
  --apple-api-key <path>     App Store Connect API key path for notarytool.
                             Defaults to APPLE_API_KEY when set.
                             If unset, APPLE_API_KEY_B64 is decoded to a
                             temporary .p8 file, matching ax-code CI.
  --apple-api-key-id <id>    App Store Connect API key id for notarytool.
                             Defaults to APPLE_API_KEY_ID when set.
  --apple-api-issuer <uuid>  App Store Connect API issuer for notarytool.
                             Defaults to APPLE_API_ISSUER when set.
  --skip-notarization        Codesign only; do not submit to Apple notary service.
  -h, --help                 Show this help.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        die "'$1' not found - $2"
    fi
}

run() {
    echo ">> $*"
    "$@"
}

NOTARY_ARGS=()

resolve_notary_args() {
    NOTARY_ARGS=()
    if [[ "$SKIP_NOTARIZATION" = true || -z "$SIGN_IDENTITY" ]]; then
        return
    fi

    if [[ -n "$APPLE_API_KEY_B64" && -z "$APPLE_API_KEY" ]]; then
        APPLE_API_KEY_TEMP="$(mktemp "${TMPDIR:-/tmp}/ax-engine-apple-api-key.XXXXXX.p8")"
        printf '%s' "$APPLE_API_KEY_B64" | base64 --decode > "$APPLE_API_KEY_TEMP"
        chmod 600 "$APPLE_API_KEY_TEMP"
        APPLE_API_KEY="$APPLE_API_KEY_TEMP"
    fi

    if [[ -n "$APPLE_API_KEY" || -n "$APPLE_API_KEY_B64" || -n "$APPLE_API_KEY_ID" || -n "$APPLE_API_ISSUER" ]]; then
        [[ -n "$APPLE_API_KEY" ]] || die "--apple-api-key, APPLE_API_KEY, or APPLE_API_KEY_B64 is required when using App Store Connect notarization"
        [[ -n "$APPLE_API_KEY_ID" ]] || die "--apple-api-key-id or APPLE_API_KEY_ID is required when using App Store Connect notarization"
        [[ -n "$APPLE_API_ISSUER" ]] || die "--apple-api-issuer or APPLE_API_ISSUER is required when using App Store Connect notarization"
        [[ -f "$APPLE_API_KEY" ]] || die "Apple API key file not found: $APPLE_API_KEY"
        NOTARY_ARGS=(--key "$APPLE_API_KEY" --key-id "$APPLE_API_KEY_ID" --issuer "$APPLE_API_ISSUER")
    else
        [[ -n "$NOTARY_PROFILE" ]] || die "--notary-profile must not be empty"
        NOTARY_ARGS=(--keychain-profile "$NOTARY_PROFILE")
    fi
}

codesign_release_binaries() {
    if [[ -z "$SIGN_IDENTITY" ]]; then
        echo "warning: no --sign-identity provided; release binaries will not be Apple Developer ID signed" >&2
        return
    fi

    echo "Codesigning release binaries with identity: $SIGN_IDENTITY"
    [[ -f "$MACOS_RELEASE_ENTITLEMENTS" ]] || die "macOS release entitlements not found: $MACOS_RELEASE_ENTITLEMENTS"
    for bin in "${RELEASE_BINS[@]}"; do
        run codesign \
            --sign "$SIGN_IDENTITY" \
            --options runtime \
            --entitlements "$MACOS_RELEASE_ENTITLEMENTS" \
            --timestamp \
            --force \
            "target/release/$bin"
    done

    echo "Verifying codesignatures"
    for bin in "${RELEASE_BINS[@]}"; do
        run codesign --verify --strict --verbose=2 "target/release/$bin"
    done
}

notarize_release_binaries() {
    if [[ -z "$SIGN_IDENTITY" ]]; then
        return
    fi
    if [[ "$SKIP_NOTARIZATION" = true ]]; then
        echo "warning: skipping notarization (--skip-notarization)" >&2
        return
    fi
    if [[ "$DRY_RUN" = true ]]; then
        echo "dry-run: skipping Apple notarization submit"
        return
    fi

    local notarize_zip="$ARTIFACT_DIR/ax-engine-${TAG}-notarize.zip"
    rm -f "$notarize_zip"
    run zip -j "$notarize_zip" "${RELEASE_BINS[@]/#/target/release/}"
    run xcrun notarytool submit "$notarize_zip" "${NOTARY_ARGS[@]}" --wait
    rm -f "$notarize_zip"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true ;;
        --skip-checks) SKIP_CHECKS=true ;;
        --skip-build) SKIP_BUILD=true ;;
        --skip-tag-push) SKIP_TAG_PUSH=true ;;
        --allow-dirty) ALLOW_DIRTY=true ;;
        --artifact-dir)
            shift
            [[ -n "${1:-}" ]] || die "--artifact-dir requires an argument"
            ARTIFACT_DIR="$1"
            ;;
        --repo)
            shift
            [[ -n "${1:-}" ]] || die "--repo requires an argument"
            MAIN_REPO="$1"
            ;;
        --title)
            shift
            [[ -n "${1:-}" ]] || die "--title requires an argument"
            TITLE="$1"
            ;;
        --notes-file)
            shift
            [[ -n "${1:-}" ]] || die "--notes-file requires an argument"
            NOTES_FILE="$1"
            ;;
        --draft) DRAFT=true ;;
        --prerelease) PRERELEASE=true ;;
        --clobber-assets) CLOBBER_ASSETS=true ;;
        --no-minisign) MINISIGN=false ;;
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
        --notary-profile)
            shift
            [[ -n "${1:-}" ]] || die "--notary-profile requires an argument"
            NOTARY_PROFILE="$1"
            ;;
        --apple-api-key)
            shift
            [[ -n "${1:-}" ]] || die "--apple-api-key requires an argument"
            APPLE_API_KEY="$1"
            ;;
        --apple-api-key-id)
            shift
            [[ -n "${1:-}" ]] || die "--apple-api-key-id requires an argument"
            APPLE_API_KEY_ID="$1"
            ;;
        --apple-api-issuer)
            shift
            [[ -n "${1:-}" ]] || die "--apple-api-issuer requires an argument"
            APPLE_API_ISSUER="$1"
            ;;
        --skip-notarization) SKIP_NOTARIZATION=true ;;
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

VERSION="${TAG#v}"
TITLE="${TITLE:-$TAG}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT_DIR/target/release-artifacts/$TAG}"

cd "$ROOT_DIR"

check_cmd git "install Git"
check_cmd gh "install GitHub CLI: brew install gh"
check_cmd cargo "install Rust: https://rustup.rs"
check_cmd python3 "install Python 3"
check_cmd shasum "shasum is required for release checksums"
check_cmd tar "tar is required for release packaging"

if [[ "$MINISIGN" = true ]]; then
    check_cmd minisign "install minisign: brew install minisign"
fi
if [[ -n "$SIGN_IDENTITY" ]]; then
    check_cmd codesign "codesign is required for Apple Developer ID signing"
    if [[ "$SKIP_NOTARIZATION" = false ]]; then
        check_cmd xcrun "xcrun notarytool is required for notarization"
        check_cmd zip "zip is required for notarization submission"
        if ! xcrun notarytool --help &>/dev/null; then
            die "xcrun notarytool is not available - install current Xcode Command Line Tools"
        fi
    fi
fi
resolve_notary_args

if [[ "$DRY_RUN" = false ]]; then
    gh auth status >/dev/null || die "gh is not authenticated - run: gh auth login"
fi

repo_root="$(git rev-parse --show-toplevel)"
[[ "$repo_root" == "$ROOT_DIR" ]] || die "script must run from the ax-engine checkout"

if [[ "$ALLOW_DIRTY" = false ]]; then
    status="$(git status --short --untracked-files=all)"
    [[ -z "$status" ]] || {
        echo "$status" >&2
        die "working tree is dirty; commit or stash changes before publishing, or pass --allow-dirty"
    }
else
    echo "warning: publishing from a dirty working tree (--allow-dirty)" >&2
fi

python3 - "$TAG" <<'PY'
from __future__ import annotations

import json
import pathlib
import re
import sys
import tomllib

tag = sys.argv[1]
version = tag.removeprefix("v")
root = pathlib.Path.cwd()
cargo_version = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"]["version"]
pyproject_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]

# Check all independently-versioned artifacts stay in sync.
checks: dict[str, str] = {
    "Cargo.toml": cargo_version,
    "pyproject.toml": pyproject_version,
    "sdk/javascript/package.json": json.loads(
        (root / "sdk/javascript/package.json").read_text()
    )["version"],
}
version_rb = (root / "sdk/ruby/lib/ax_engine/version.rb").read_text()
m = re.search(r'VERSION\s*=\s*"([^"]+)"', version_rb)
if m:
    checks["sdk/ruby/lib/ax_engine/version.rb"] = m.group(1)

mismatches = {f: v for f, v in checks.items() if v != version}
if mismatches:
    for f, v in mismatches.items():
        print(f"Version mismatch: tag={version}, {f}={v}", file=sys.stderr)
    raise SystemExit(1)

print(f"Version verified: {version} (across {len(checks)} files)")
PY

head_commit="$(git rev-parse HEAD)"
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    tag_commit="$(git rev-list -n 1 "$TAG")"
    [[ "$tag_commit" == "$head_commit" ]] || {
        die "tag $TAG points at $tag_commit, not current HEAD $head_commit"
    }
    echo "Tag verified at HEAD: $TAG"
else
    if [[ "$DRY_RUN" = true ]]; then
        echo "dry-run: would create annotated tag $TAG at $head_commit"
    else
        run git tag -a "$TAG" -m "Release $TAG"
    fi
fi

if [[ "$SKIP_CHECKS" = false ]]; then
    run cargo fmt --check
    run cargo test --quiet --no-fail-fast
    run cargo clippy --all-targets --all-features -- \
        -D warnings \
        --force-warn clippy::unwrap-used \
        --force-warn clippy::expect-used \
        --force-warn clippy::panic \
        --force-warn clippy::dbg-macro \
        --force-warn clippy::large-enum-variant
    run bash scripts/check-scripts.sh
    run bash scripts/check-bench-doctor.sh
    run bash scripts/check-metal-kernel-contract.sh
    run bash scripts/check-python-preview.sh
else
    echo "warning: skipping release checks (--skip-checks)" >&2
fi

if [[ "$SKIP_BUILD" = false ]]; then
    run cargo build --release -p ax-engine-server -p ax-engine-bench --bins
else
    echo "warning: skipping build (--skip-build)" >&2
fi

for bin in "${RELEASE_BINS[@]}"; do
    [[ -x "target/release/$bin" ]] || die "missing executable target/release/$bin"
done

codesign_release_binaries

mkdir -p "$ARTIFACT_DIR"
STAGING_DIR="$ARTIFACT_DIR/payload"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

notarize_release_binaries

release_payload=()
for bin in "${RELEASE_BINS[@]}"; do
    cp "target/release/$bin" "$STAGING_DIR/$bin"
    chmod +x "$STAGING_DIR/$bin"
    release_payload+=("$bin")
done
for mapping in "${RELEASE_HELPER_SOURCES[@]}"; do
    source_path="${mapping%%:*}"
    install_name="${mapping#*:}"
    [[ -f "$source_path" ]] || die "missing release helper $source_path"
    cp "$source_path" "$STAGING_DIR/$install_name"
    chmod +x "$STAGING_DIR/$install_name"
    release_payload+=("$install_name")
done

ARCHIVE="ax-engine-${TAG}-macos-arm64.tar.gz"
ARCHIVE_PATH="$ARTIFACT_DIR/$ARCHIVE"
SHA256_PATH="$ARTIFACT_DIR/$ARCHIVE.sha256"
MANIFEST_PATH="$ARTIFACT_DIR/ax-engine-${TAG}-macos-arm64.manifest.json"
DOWNLOAD_URL="https://github.com/${MAIN_REPO}/releases/download/${TAG}/${ARCHIVE}"

run tar -czf "$ARCHIVE_PATH" -C "$STAGING_DIR" "${release_payload[@]}"
SHA256="$(shasum -a 256 "$ARCHIVE_PATH" | awk '{print $1}')"
printf '%s  %s\n' "$SHA256" "$ARCHIVE" > "$SHA256_PATH"

AX_RELEASE_CODESIGN_IDENTITY="$SIGN_IDENTITY" \
AX_RELEASE_NOTARIZED="$([[ -n "$SIGN_IDENTITY" && "$SKIP_NOTARIZATION" = false && "$DRY_RUN" = false ]] && echo true || echo false)" \
python3 - "$TAG" "$VERSION" "$MAIN_REPO" "$head_commit" "$ARCHIVE" "$SHA256" "$DOWNLOAD_URL" "$MANIFEST_PATH" "${RELEASE_BINS[@]}" "--" "${release_payload[@]}" <<'PY'
from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime, timezone

args = sys.argv[1:]
separator = args.index("--")
tag, version, repo, commit, archive, sha256, download_url, manifest_path, *bins = args[:separator]
payload = args[separator + 1 :]
manifest = {
    "schema_version": "ax.github_release_manifest.v1",
    "project": "ax-engine",
    "repository": repo,
    "tag": tag,
    "version": version,
    "git_commit": commit,
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "platform": "macos-arm64",
    "archive": {
        "name": archive,
        "sha256": sha256,
        "download_url": download_url,
    },
    "binaries": bins,
    "payload": payload,
    "code_signing": {
        "apple_developer_id": bool(os.environ.get("AX_RELEASE_CODESIGN_IDENTITY")),
        "identity": os.environ.get("AX_RELEASE_CODESIGN_IDENTITY") or None,
        "notarized": os.environ.get("AX_RELEASE_NOTARIZED") == "true",
        "disable_library_validation": bool(os.environ.get("AX_RELEASE_CODESIGN_IDENTITY")),
    },
}
path = pathlib.Path(manifest_path)
path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

assets=("$ARCHIVE_PATH" "$SHA256_PATH" "$MANIFEST_PATH")

if [[ "$MINISIGN" = true ]]; then
    minisign_args=(
        --secret-key "$MINISIGN_SECRET_KEY"
        --public-key "$MINISIGN_PUBLIC_KEY"
        --signature-dir "$ARTIFACT_DIR"
        --force
    )
    if [[ -n "$MINISIGN_PUBLIC_KEY_STRING" ]]; then
        minisign_args+=(--public-key-string "$MINISIGN_PUBLIC_KEY_STRING")
    fi
    run "$SCRIPT_DIR/minisign-artifact.sh" "${minisign_args[@]}" "${assets[@]}"
    assets+=(
        "$ARCHIVE_PATH.minisig"
        "$SHA256_PATH.minisig"
        "$MANIFEST_PATH.minisig"
    )
else
    echo "warning: minisign disabled (--no-minisign)" >&2
fi

echo
echo "Release assets:"
for asset in "${assets[@]}"; do
    echo "  $asset"
done
echo

if [[ "$DRY_RUN" = true ]]; then
    echo "dry-run: skipping tag push, GitHub release creation, upload, and remote asset verification"
    exit 0
fi

if [[ "$SKIP_TAG_PUSH" = false ]]; then
    remote_tag_commit="$(git ls-remote --tags origin "refs/tags/$TAG^{}" | awk '{print $1}')"
    remote_tag_ref="$(git ls-remote --tags origin "refs/tags/$TAG" | awk '{print $1}')"
    remote_tag="${remote_tag_commit:-$remote_tag_ref}"
    if [[ -n "$remote_tag" && "$remote_tag" != "$head_commit" ]]; then
        die "origin tag $TAG points at commit $remote_tag, not local HEAD $head_commit"
    fi
    if [[ -z "$remote_tag" ]]; then
        run git push origin "$TAG"
    else
        echo "Remote tag already exists at HEAD: $TAG"
    fi
else
    echo "warning: skipping tag push (--skip-tag-push)" >&2
fi

release_args=("$TAG" --repo "$MAIN_REPO" --title "$TITLE")
if [[ -n "$NOTES_FILE" ]]; then
    [[ -f "$NOTES_FILE" ]] || die "release notes file not found: $NOTES_FILE"
    release_args+=(--notes-file "$NOTES_FILE")
else
    release_args+=(--generate-notes)
fi
if [[ "$DRAFT" = true ]]; then
    release_args+=(--draft)
fi
if [[ "$PRERELEASE" = true ]]; then
    release_args+=(--prerelease)
fi

if gh release view "$TAG" --repo "$MAIN_REPO" >/dev/null 2>&1; then
    echo "GitHub release already exists: $TAG"
else
    run gh release create "${release_args[@]}" --verify-tag
fi

upload_args=("$TAG" "${assets[@]}" --repo "$MAIN_REPO")
if [[ "$CLOBBER_ASSETS" = true ]]; then
    upload_args+=(--clobber)
fi
run gh release upload "${upload_args[@]}"

asset_names="$(gh release view "$TAG" --repo "$MAIN_REPO" --json assets --jq '.assets[].name')"
for asset in "${assets[@]}"; do
    name="$(basename "$asset")"
    if ! grep -qxF "$name" <<<"$asset_names"; then
        die "uploaded asset missing from GitHub release: $name"
    fi
done

echo
echo "Published GitHub release: https://github.com/${MAIN_REPO}/releases/tag/${TAG}"
