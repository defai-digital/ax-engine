#!/usr/bin/env bash
# Stage the pinned pip MLX runtime used by standalone macOS release artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
usage: scripts/prepare-mlx-release-runtime.sh <mlx-lib-dir> <output-dir>

Copies the admitted pip MLX runtime into output-dir:
  libmlx.dylib
  libjaccl.dylib
  mlx.metallib
  MLX-LICENSE.txt

MLX_LICENSE_FILE may point at an explicit MLX license file. Otherwise the
script resolves mlx-<pinned-version>.dist-info/licenses/LICENSE relative to
the standard pip wheel layout containing mlx-lib-dir.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

[[ $# -eq 2 ]] || {
    usage >&2
    exit 2
}

MLX_LIB_DIR="$1"
OUTPUT_DIR="$2"
MLX_PIN="$(tr -d '[:space:]' < "$ROOT_DIR/mlx.version")"

for command in lipo vtool; do
    command -v "$command" >/dev/null 2>&1 || die "'$command' is required"
done
[[ -d "$MLX_LIB_DIR" ]] || die "MLX library directory not found: $MLX_LIB_DIR"
MLX_LIB_DIR="$(cd "$MLX_LIB_DIR" && pwd -P)"
SITE_PACKAGES="$(cd "$MLX_LIB_DIR/../.." && pwd -P)"

case "$MLX_LIB_DIR" in
    */Cellar/mlx/*|*/opt/mlx/*)
        die "release runtime must come from pinned pip MLX, not Homebrew: $MLX_LIB_DIR"
        ;;
esac

RUNTIME_FILES=(libmlx.dylib libjaccl.dylib mlx.metallib)
for name in "${RUNTIME_FILES[@]}"; do
    [[ -s "$MLX_LIB_DIR/$name" ]] || die "pinned MLX runtime is missing $MLX_LIB_DIR/$name"
done

VERSION_HEADER="$MLX_LIB_DIR/../include/mlx/version.h"
[[ -s "$VERSION_HEADER" ]] || {
    die "MLX version header not found beside the pip runtime: $VERSION_HEADER"
}
mlx_version_field() {
    awk -v field="$1" '$1 == "#define" && $2 == field {print $3; exit}' \
        "$VERSION_HEADER"
}
RESOLVED_MLX_VERSION="$(
    printf '%s.%s.%s' \
        "$(mlx_version_field MLX_VERSION_MAJOR)" \
        "$(mlx_version_field MLX_VERSION_MINOR)" \
        "$(mlx_version_field MLX_VERSION_PATCH)"
)"
[[ "$RESOLVED_MLX_VERSION" == "$MLX_PIN" ]] || {
    die "MLX runtime version $RESOLVED_MLX_VERSION does not match repository pin $MLX_PIN"
}

wheel_platform() {
    awk '
        /^Tag: / {
            tag = $2
            if (tag ~ /-macosx_[0-9]+_[0-9]+_arm64$/) {
                sub(/^.*-/, "", tag)
                print tag
            }
        }
    ' "$1" | sort -u
}

MLX_WHEEL_METADATA="$SITE_PACKAGES/mlx-${MLX_PIN}.dist-info/WHEEL"
METAL_WHEEL_METADATA="$SITE_PACKAGES/mlx_metal-${MLX_PIN}.dist-info/WHEEL"
[[ -s "$MLX_WHEEL_METADATA" ]] || die "pinned MLX wheel metadata not found: $MLX_WHEEL_METADATA"
[[ -s "$METAL_WHEEL_METADATA" ]] || {
    die "pinned mlx-metal wheel metadata not found: $METAL_WHEEL_METADATA"
}
MLX_WHEEL_PLATFORM="$(wheel_platform "$MLX_WHEEL_METADATA")"
METAL_WHEEL_PLATFORM="$(wheel_platform "$METAL_WHEEL_METADATA")"
[[ "$MLX_WHEEL_PLATFORM" =~ ^macosx_[0-9]+_[0-9]+_arm64$ ]] || {
    die "MLX wheel must declare exactly one arm64 macOS platform tag"
}
[[ "$METAL_WHEEL_PLATFORM" == "$MLX_WHEEL_PLATFORM" ]] || {
    die "MLX and mlx-metal wheel platforms do not match: $MLX_WHEEL_PLATFORM vs $METAL_WHEEL_PLATFORM"
}
EXPECTED_RUNTIME_MINOS="${MLX_WHEEL_PLATFORM#macosx_}"
EXPECTED_RUNTIME_MINOS="${EXPECTED_RUNTIME_MINOS%_arm64}"
EXPECTED_RUNTIME_MINOS="${EXPECTED_RUNTIME_MINOS/_/.}"

for name in libmlx.dylib libjaccl.dylib; do
    archs="$(lipo -archs "$MLX_LIB_DIR/$name" 2>/dev/null || true)"
    [[ " $archs " == *" arm64 "* ]] || {
        die "$MLX_LIB_DIR/$name does not contain the required arm64 architecture"
    }
    actual_minos="$(
        vtool -show-build "$MLX_LIB_DIR/$name" 2>/dev/null \
            | awk '$1 == "platform" && $2 == "MACOS" { macos = 1; next }
                macos && $1 == "minos" { print $2; exit }'
    )"
    [[ -n "$actual_minos" ]] || die "could not determine macOS minos for $MLX_LIB_DIR/$name"
    [[ "$actual_minos" == "$EXPECTED_RUNTIME_MINOS" ]] || {
        die "$name targets macOS $actual_minos; wheel requires $EXPECTED_RUNTIME_MINOS"
    }
done

MLX_LICENSE_FILE="${MLX_LICENSE_FILE:-}"
if [[ -z "$MLX_LICENSE_FILE" ]]; then
    MLX_LICENSE_FILE="$SITE_PACKAGES/mlx-${MLX_PIN}.dist-info/licenses/LICENSE"
fi
[[ -s "$MLX_LICENSE_FILE" ]] || {
    die "MLX ${MLX_PIN} license not found: $MLX_LICENSE_FILE (set MLX_LICENSE_FILE to override)"
}

mkdir -p "$OUTPUT_DIR"
for name in "${RUNTIME_FILES[@]}"; do
    cp -p "$MLX_LIB_DIR/$name" "$OUTPUT_DIR/$name"
done
cp -p "$MLX_LICENSE_FILE" "$OUTPUT_DIR/MLX-LICENSE.txt"

echo "Staged pinned MLX ${MLX_PIN} (${MLX_WHEEL_PLATFORM}) release runtime from $MLX_LIB_DIR"
for name in "${RUNTIME_FILES[@]}" MLX-LICENSE.txt; do
    echo "  $OUTPUT_DIR/$name"
done
