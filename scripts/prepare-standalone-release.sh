#!/usr/bin/env bash
# Build a relocatable AX Engine payload from immutable binaries + pinned MLX.

set -euo pipefail

usage() {
    cat <<'EOF'
usage: scripts/prepare-standalone-release.sh <binary-dir> <mlx-runtime-dir> <output-dir>

Copies the three AX Engine binaries and pinned MLX runtime into output-dir,
removes builder-host LC_RPATH entries, and installs the two supported relative
run paths:
  @loader_path         direct extraction (runtime colocated with binaries)
  @loader_path/../libexec  Homebrew layout (bin/ next to private libexec/)

The caller must codesign after this script because install_name_tool invalidates
existing Mach-O signatures.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

[[ $# -eq 3 ]] || {
    usage >&2
    exit 2
}

BINARY_DIR="$1"
MLX_RUNTIME_DIR="$2"
OUTPUT_DIR="$3"
BINARY_NAMES=(ax-engine ax-engine-server ax-engine-bench)
RUNTIME_FILES=(libmlx.dylib libjaccl.dylib mlx.metallib MLX-LICENSE.txt)
BUNDLED_DYLIBS=(libmlx.dylib libjaccl.dylib)

for command in install_name_tool otool; do
    command -v "$command" >/dev/null 2>&1 || die "'$command' is required"
done
[[ -d "$BINARY_DIR" ]] || die "binary directory not found: $BINARY_DIR"
[[ -d "$MLX_RUNTIME_DIR" ]] || die "MLX runtime directory not found: $MLX_RUNTIME_DIR"

mkdir -p "$OUTPUT_DIR"
for name in "${BINARY_NAMES[@]}"; do
    [[ -x "$BINARY_DIR/$name" ]] || die "release binary is missing or not executable: $BINARY_DIR/$name"
    cp -p "$BINARY_DIR/$name" "$OUTPUT_DIR/$name"
    chmod +x "$OUTPUT_DIR/$name"
done
for name in "${RUNTIME_FILES[@]}"; do
    [[ -s "$MLX_RUNTIME_DIR/$name" ]] || die "MLX runtime asset is missing: $MLX_RUNTIME_DIR/$name"
    cp -p "$MLX_RUNTIME_DIR/$name" "$OUTPUT_DIR/$name"
done

linked_paths() {
    otool -L "$1" | sed -n '2,$s/^[[:space:]]*\([^[:space:]]*\).*$/\1/p'
}

rpaths() {
    otool -l "$1" | awk '
        $1 == "cmd" && $2 == "LC_RPATH" { want_path = 1; next }
        want_path && $1 == "path" { print $2; want_path = 0 }
    '
}

is_bundled_dylib() {
    local candidate="$1"
    local name
    for name in "${BUNDLED_DYLIBS[@]}"; do
        [[ "$candidate" == "$name" ]] && return 0
    done
    return 1
}

rewrite_dependencies() {
    local image="$1"
    local style="$2"
    local image_id=""
    local dependency
    local basename
    local replacement

    if [[ "$style" == "dylib" ]]; then
        image_id="$(otool -D "$image" | sed -n '2p')"
    fi

    while IFS= read -r dependency; do
        [[ -n "$dependency" && "$dependency" != "$image_id" ]] || continue
        basename="${dependency##*/}"
        is_bundled_dylib "$basename" || continue
        if [[ "$style" == "dylib" ]]; then
            replacement="@loader_path/$basename"
        else
            replacement="@rpath/$basename"
        fi
        if [[ "$dependency" != "$replacement" ]]; then
            install_name_tool -change "$dependency" "$replacement" "$image"
        fi
    done < <(linked_paths "$image")
}

reset_binary_rpaths() {
    local image="$1"
    local existing

    while IFS= read -r existing; do
        [[ -n "$existing" ]] || continue
        install_name_tool -delete_rpath "$existing" "$image"
    done < <(rpaths "$image")
    install_name_tool -add_rpath "@loader_path" "$image"
    install_name_tool -add_rpath "@loader_path/../libexec" "$image"
}

for name in "${BINARY_NAMES[@]}"; do
    image="$OUTPUT_DIR/$name"
    rewrite_dependencies "$image" binary
    reset_binary_rpaths "$image"
done

for name in "${BUNDLED_DYLIBS[@]}"; do
    image="$OUTPUT_DIR/$name"
    install_name_tool -id "@rpath/$name" "$image"
    rewrite_dependencies "$image" dylib
done

echo "Prepared relocatable standalone payload: $OUTPUT_DIR"
