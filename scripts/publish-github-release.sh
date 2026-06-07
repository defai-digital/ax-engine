#!/usr/bin/env bash
# Publish an AX Engine GitHub release from the local checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MAIN_REPO="${AX_RELEASE_REPO:-defai-digital/ax-engine}"
RELEASE_BINS=(ax-engine-server ax-engine-bench)

TAG=""
DRY_RUN=false
SKIP_CHECKS=false
SKIP_BUILD=false
SKIP_TAG_PUSH=false
ALLOW_DIRTY=false
MINISIGN=true
MINISIGN_SECRET_KEY="${AX_MINISIGN_SECRET_KEY:-$HOME/signkey/ax-engine.minisign.key}"
MINISIGN_PUBLIC_KEY="${AX_MINISIGN_PUBLIC_KEY:-$HOME/signkey/ax-engine.minisign.pub}"
MINISIGN_PUBLIC_KEY_STRING="${AX_MINISIGN_PUBLIC_KEY_STRING:-}"
ARTIFACT_DIR=""
TITLE=""
NOTES_FILE=""
DRAFT=false
PRERELEASE=false
CLOBBER_ASSETS=false

usage() {
    cat <<'EOF'
usage: scripts/publish-github-release.sh <vX.Y.Z> [options]

Publishes the macOS arm64 GitHub release assets for AX Engine:
  1. validate clean checkout and tag/version consistency
  2. run release gates
  3. build release binaries
  4. package tarball, sha256, and release manifest
  5. minisign artifacts by default
  6. push the tag, create/update the GitHub release, upload assets, verify assets

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
  --minisign-key <path>      Secret key path. Default: ~/signkey/ax-engine.minisign.key
  --minisign-pubkey <path>   Public key file path. Default: ~/signkey/ax-engine.minisign.pub
  --minisign-public-key <k>  Public key string for verification.
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

import pathlib
import sys
import tomllib

tag = sys.argv[1]
version = tag.removeprefix("v")
root = pathlib.Path.cwd()
cargo_version = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"]["version"]
pyproject_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]

if len({version, cargo_version, pyproject_version}) != 1:
    print(
        "Version mismatch: "
        f"tag={version}, workspace={cargo_version}, pyproject={pyproject_version}",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(f"Version verified: {version}")
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
    run cargo clippy --all-targets --all-features -- -D warnings
    run bash scripts/check-scripts.sh
    run bash scripts/check-bench-doctor.sh
    run bash scripts/check-metal-kernel-contract.sh
    run bash scripts/check-python-preview.sh
else
    echo "warning: skipping release checks (--skip-checks)" >&2
fi

if [[ "$SKIP_BUILD" = false ]]; then
    run cargo build --release -p ax-engine-server -p ax-engine-bench
else
    echo "warning: skipping build (--skip-build)" >&2
fi

for bin in "${RELEASE_BINS[@]}"; do
    [[ -x "target/release/$bin" ]] || die "missing executable target/release/$bin"
done

mkdir -p "$ARTIFACT_DIR"

ARCHIVE="ax-engine-${TAG}-macos-arm64.tar.gz"
ARCHIVE_PATH="$ARTIFACT_DIR/$ARCHIVE"
SHA256_PATH="$ARTIFACT_DIR/$ARCHIVE.sha256"
MANIFEST_PATH="$ARTIFACT_DIR/ax-engine-${TAG}-macos-arm64.manifest.json"
DOWNLOAD_URL="https://github.com/${MAIN_REPO}/releases/download/${TAG}/${ARCHIVE}"

run tar -czf "$ARCHIVE_PATH" -C target/release "${RELEASE_BINS[@]}"
SHA256="$(shasum -a 256 "$ARCHIVE_PATH" | awk '{print $1}')"
printf '%s  %s\n' "$SHA256" "$ARCHIVE" > "$SHA256_PATH"

python3 - "$TAG" "$VERSION" "$MAIN_REPO" "$head_commit" "$ARCHIVE" "$SHA256" "$DOWNLOAD_URL" "$MANIFEST_PATH" "${RELEASE_BINS[@]}" <<'PY'
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

tag, version, repo, commit, archive, sha256, download_url, manifest_path, *bins = sys.argv[1:]
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
