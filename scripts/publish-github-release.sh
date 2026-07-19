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
    "scripts/prepare_qwen36_mtp_sidecar.py:ax-engine-prepare-qwen36-mtp-sidecar.py"
    "scripts/check_mtp_sidecar_provenance.py:ax-engine-check-mtp-sidecar-provenance.py"
)

TAG=""
DRY_RUN=false
SKIP_CHECKS=false
FULL_LOCAL_CHECKS=false
SKIP_BUILD=false
LOCAL_BUILD=false
SKIP_TAG_PUSH=false
SKIP_BREW_DISPATCH=false
ALLOW_DIRTY=false
MINISIGN=true
MINISIGN_SECRET_KEY="${AX_MINISIGN_SECRET_KEY:-$HOME/signkey/ax.sec}"
MINISIGN_PUBLIC_KEY="${AX_MINISIGN_PUBLIC_KEY:-$HOME/signkey/ax.pub}"
MINISIGN_PUBLIC_KEY_STRING="${AX_MINISIGN_PUBLIC_KEY_STRING:-}"
SIGN_IDENTITY="${AX_CODESIGN_IDENTITY:-}"
NOTARY_PROFILE="${AX_NOTARY_PROFILE:-ax-notary}"
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
RELEASE_BIN_DIR="$ROOT_DIR/target/release"
CANDIDATE_WORKFLOW="release-candidate.yml"

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
  2. require a successful CI run for the exact release commit
  3. reuse a validated macOS release candidate, or build locally for dry-runs
  4. optionally Apple Developer ID sign and notarize binaries
  5. package tarball, sha256, and release manifest
  6. minisign artifacts by default
  7. push the tag, publish and verify assets, then dispatch Homebrew

Options:
  --dry-run                  Run local checks/build/package/sign, but do not push or upload.
  --full-local-checks        Run local release gates in addition to exact-SHA CI verification.
  --skip-checks              Skip local release gates. Published releases still require green CI.
  --local-build              Build release binaries locally instead of using a candidate artifact.
  --skip-build               Reuse existing target/release binaries.
  --skip-tag-push            Do not push the tag before creating/uploading the release.
  --skip-brew-dispatch       Do not dispatch the Homebrew formula update after upload.
  --allow-dirty              Allow a dirty git worktree.
  --artifact-dir <dir>       Output directory. Default: target/release-artifacts/<tag>
  --repo <owner/name>        GitHub repository. Default: defai-digital/ax-engine
  --title <text>             Release title. Default: <tag>
  --notes-file <path>        Release notes file. Default: gh --generate-notes.
  --draft                    Create the release as draft.
  --prerelease               Mark the release as prerelease.
  --clobber-assets           Overwrite existing release assets when uploading.
  --no-minisign              Do not sign release artifacts.
  --minisign-key <path>      Secret key path. Default: ~/signkey/ax.sec
  --minisign-pubkey <path>   Public key file path. Default: ~/signkey/ax.pub
  --minisign-public-key <k>  Public key string for verification.
  --sign-identity <id>       Developer ID Application identity for codesign.
                             Can also be set with AX_CODESIGN_IDENTITY.
  --notary-profile <name>    notarytool Keychain profile. Default: ax-notary.
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

require_green_ci() {
    local ci_info
    local ci_run_id
    local ci_sha
    local ci_url

    ci_info="$(gh api --method GET \
        "repos/${MAIN_REPO}/actions/workflows/ci.yml/runs" \
        -f head_sha="$head_commit" \
        -f status=success \
        -F per_page=1 \
        --jq '.workflow_runs[0] | if . == null then empty else [.id, .head_sha, .html_url] | @tsv end')"
    [[ -n "$ci_info" ]] || {
        die "no successful CI workflow found for release commit $head_commit"
    }
    IFS=$'\t' read -r ci_run_id ci_sha ci_url <<<"$ci_info"
    [[ "$ci_sha" == "$head_commit" ]] || {
        die "CI run $ci_run_id resolved to $ci_sha, expected $head_commit"
    }
    echo "Exact-SHA CI verified: $ci_url"
}

find_candidate_run() {
    local artifact_name="$1"

    # shellcheck disable=SC2016
    gh api --method GET \
        "repos/${MAIN_REPO}/actions/artifacts" \
        -f name="$artifact_name" \
        -F per_page=10 \
        --jq '([.artifacts[] | select(.expired == false)] | sort_by(.created_at) | reverse | .[0]) as $artifact | if $artifact == null then empty else $artifact.workflow_run.id end'
}

validate_candidate_run() {
    local candidate_run_id="$1"
    local run_info
    local workflow_name
    local conclusion

    run_info="$(gh api \
        "repos/${MAIN_REPO}/actions/runs/${candidate_run_id}" \
        --jq '[.name, .conclusion] | @tsv')"
    IFS=$'\t' read -r workflow_name conclusion <<<"$run_info"
    [[ "$workflow_name" == "Build Release Candidate" && "$conclusion" == "success" ]] || {
        die "artifact came from invalid candidate run $candidate_run_id ($workflow_name, $conclusion)"
    }
}

prepare_release_candidate() {
    local artifact_name="ax-engine-release-candidate-${head_commit}"
    local candidate_dir="$ARTIFACT_DIR/release-candidate"
    local candidate_run_id
    local request_id
    local run_info
    local run_title
    local run_url

    candidate_run_id="$(find_candidate_run "$artifact_name")"
    if [[ -z "$candidate_run_id" ]]; then
        request_id="${head_commit:0:12}-$(date -u +%Y%m%d%H%M%S)-$$"
        run_title="Release candidate $TAG [$request_id]"
        run gh workflow run "$CANDIDATE_WORKFLOW" \
            --repo "$MAIN_REPO" \
            --ref main \
            -f tag="$TAG" \
            -f git_commit="$head_commit" \
            -f request_id="$request_id"

        for _ in {1..30}; do
            run_info="$(gh run list \
                --repo "$MAIN_REPO" \
                --workflow "$CANDIDATE_WORKFLOW" \
                --event workflow_dispatch \
                --limit 30 \
                --json databaseId,displayTitle,url \
                --jq "map(select(.displayTitle == \"$run_title\")) | first | if . == null then empty else [.databaseId, .url] | @tsv end")"
            if [[ -n "$run_info" ]]; then
                break
            fi
            sleep 2
        done
        [[ -n "$run_info" ]] || die "could not locate dispatched release-candidate workflow"
        IFS=$'\t' read -r candidate_run_id run_url <<<"$run_info"
        echo "Waiting for release candidate: $run_url"
        run gh run watch "$candidate_run_id" --repo "$MAIN_REPO" --exit-status

        candidate_run_id=""
        for _ in {1..15}; do
            candidate_run_id="$(find_candidate_run "$artifact_name")"
            if [[ -n "$candidate_run_id" ]]; then
                break
            fi
            sleep 2
        done
        [[ -n "$candidate_run_id" ]] || {
            die "release-candidate run succeeded but artifact $artifact_name is missing"
        }
    else
        echo "Reusing release candidate from workflow run $candidate_run_id"
    fi
    validate_candidate_run "$candidate_run_id"

    rm -rf "$candidate_dir"
    mkdir -p "$candidate_dir"
    run gh run download "$candidate_run_id" \
        --repo "$MAIN_REPO" \
        --name "$artifact_name" \
        --dir "$candidate_dir"
    run python3 scripts/release_candidate.py verify \
        --root "$candidate_dir" \
        --tag "$TAG" \
        --commit "$head_commit" \
        --group binaries
    for bin in "${RELEASE_BINS[@]}"; do
        chmod +x "$candidate_dir/bin/$bin"
    done
    RELEASE_BIN_DIR="$candidate_dir/bin"
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
            "$RELEASE_BIN_DIR/$bin"
    done

    echo "Verifying codesignatures"
    for bin in "${RELEASE_BINS[@]}"; do
        run codesign --verify --strict --verbose=2 "$RELEASE_BIN_DIR/$bin"
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
    local notarize_inputs=()
    local bin
    rm -f "$notarize_zip"
    for bin in "${RELEASE_BINS[@]}"; do
        notarize_inputs+=("$RELEASE_BIN_DIR/$bin")
    done
    run zip -j "$notarize_zip" "${notarize_inputs[@]}"
    run xcrun notarytool submit "$notarize_zip" "${NOTARY_ARGS[@]}" --wait
    rm -f "$notarize_zip"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true ;;
        --skip-checks) SKIP_CHECKS=true ;;
        --full-local-checks) FULL_LOCAL_CHECKS=true ;;
        --local-build) LOCAL_BUILD=true ;;
        --skip-build) SKIP_BUILD=true ;;
        --skip-tag-push) SKIP_TAG_PUSH=true ;;
        --skip-brew-dispatch) SKIP_BREW_DISPATCH=true ;;
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
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z][0-9A-Za-z.-]*)?$ ]] || {
    die "release tag must be a version such as v6.9.0"
}
if [[ "$SKIP_CHECKS" = true && "$FULL_LOCAL_CHECKS" = true ]]; then
    die "--skip-checks and --full-local-checks cannot be used together"
fi
if [[ "$SKIP_BUILD" = true && "$LOCAL_BUILD" = true ]]; then
    die "--skip-build and --local-build cannot be used together"
fi

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

python3 scripts/check_version_sync.py --expected "$TAG"

head_commit="$(git rev-parse HEAD)"
if [[ "$DRY_RUN" = false ]]; then
    require_green_ci
fi
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

if [[ "$SKIP_CHECKS" = false && ( "$DRY_RUN" = true || "$FULL_LOCAL_CHECKS" = true ) ]]; then
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
elif [[ "$SKIP_CHECKS" = true ]]; then
    echo "warning: skipping release checks (--skip-checks)" >&2
else
    echo "Reusing exact-SHA GitHub CI; pass --full-local-checks to repeat gates locally."
fi

if [[ "$SKIP_BUILD" = true ]]; then
    echo "warning: skipping build (--skip-build)" >&2
elif [[ "$DRY_RUN" = true || "$LOCAL_BUILD" = true ]]; then
    run cargo build --release -p ax-engine-server -p ax-engine-bench --bins
else
    prepare_release_candidate
fi

for bin in "${RELEASE_BINS[@]}"; do
    [[ -x "$RELEASE_BIN_DIR/$bin" ]] || die "missing executable $RELEASE_BIN_DIR/$bin"
done

codesign_release_binaries

mkdir -p "$ARTIFACT_DIR"
STAGING_DIR="$ARTIFACT_DIR/payload"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

notarize_release_binaries

release_payload=()
for bin in "${RELEASE_BINS[@]}"; do
    cp "$RELEASE_BIN_DIR/$bin" "$STAGING_DIR/$bin"
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

if [[ "$SKIP_BREW_DISPATCH" = true ]]; then
    echo "warning: skipping Homebrew workflow dispatch (--skip-brew-dispatch)" >&2
elif [[ "$DRAFT" = true || "$PRERELEASE" = true ]]; then
    echo "warning: not dispatching Homebrew for a draft or prerelease" >&2
else
    run gh workflow run brew-release.yml \
        --repo "$MAIN_REPO" \
        --ref main \
        -f tag="$TAG"
    echo "Dispatched Homebrew formula update after release asset verification."
fi

echo
echo "Published GitHub release: https://github.com/${MAIN_REPO}/releases/tag/${TAG}"
