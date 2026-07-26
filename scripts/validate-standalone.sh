#!/usr/bin/env bash
# Fail-closed validation for a relocatable AX Engine macOS release payload.

set -euo pipefail

usage() {
    cat <<'EOF'
usage: scripts/validate-standalone.sh [--skip-smoke] [--doctor] <payload-dir>

Validates the complete standalone contract: required MLX assets, arm64 Mach-O
files, dependency closure, relative rpaths, dylib install names, and clean-env
--help startup for all shipped binaries. --doctor additionally requires
`ax-engine doctor` to report ready on a supported release host.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

SKIP_SMOKE=false
RUN_DOCTOR=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-smoke)
            SKIP_SMOKE=true
            shift
            ;;
        --doctor)
            RUN_DOCTOR=true
            shift
            ;;
        *)
            break
            ;;
    esac
done
if [[ "$SKIP_SMOKE" = true && "$RUN_DOCTOR" = true ]]; then
    die "--skip-smoke and --doctor cannot be used together"
fi
[[ $# -eq 1 ]] || {
    usage >&2
    exit 2
}

PAYLOAD_DIR="$1"
BINARY_NAMES=(ax-engine ax-engine-server ax-engine-bench)
BUNDLED_DYLIBS=(libmlx.dylib libjaccl.dylib)
RUNTIME_FILES=(libmlx.dylib libjaccl.dylib mlx.metallib MLX-LICENSE.txt)

for command in file lipo otool; do
    command -v "$command" >/dev/null 2>&1 || die "'$command' is required"
done
[[ -d "$PAYLOAD_DIR" ]] || die "payload directory not found: $PAYLOAD_DIR"
PAYLOAD_DIR="$(cd "$PAYLOAD_DIR" && pwd -P)"

for name in "${BINARY_NAMES[@]}"; do
    [[ -x "$PAYLOAD_DIR/$name" ]] || die "required executable is missing: $name"
done
for name in "${RUNTIME_FILES[@]}"; do
    [[ -s "$PAYLOAD_DIR/$name" ]] || die "required MLX runtime asset is missing: $name"
done

for name in "${BINARY_NAMES[@]}" "${BUNDLED_DYLIBS[@]}"; do
    archs="$(lipo -archs "$PAYLOAD_DIR/$name" 2>/dev/null || true)"
    [[ " $archs " == *" arm64 "* ]] || die "$name does not contain arm64 Mach-O code"
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

validate_system_or_bundled_dependency() {
    local image_name="$1"
    local dependency="$2"
    local style="$3"
    local basename="${dependency##*/}"

    case "$dependency" in
        /usr/lib/*|/System/Library/*)
            return
            ;;
    esac

    case "$basename" in
        libmlx.dylib|libjaccl.dylib)
            if [[ "$style" == "binary" ]]; then
                [[ "$dependency" == "@rpath/$basename" ]] || {
                    die "$image_name has non-relocatable bundled dependency: $dependency"
                }
            else
                [[ "$dependency" == "@loader_path/$basename" ]] || {
                    die "$image_name has non-colocated bundled dependency: $dependency"
                }
            fi
            [[ -s "$PAYLOAD_DIR/$basename" ]] || {
                die "$image_name references missing bundled dependency: $basename"
            }
            return
            ;;
    esac

    die "$image_name has an unbundled non-system dependency: $dependency"
}

for name in "${BINARY_NAMES[@]}"; do
    image="$PAYLOAD_DIR/$name"
    has_colocated_rpath=false
    has_homebrew_rpath=false
    rpath_count=0
    while IFS= read -r image_rpath; do
        [[ -n "$image_rpath" ]] || continue
        rpath_count=$((rpath_count + 1))
        case "$image_rpath" in
            "@loader_path")
                has_colocated_rpath=true
                ;;
            "@loader_path/../libexec")
                has_homebrew_rpath=true
                ;;
            *)
                die "$name contains an unsupported rpath: $image_rpath"
                ;;
        esac
    done < <(rpaths "$image")
    [[ "$rpath_count" -eq 2 ]] || die "$name must contain exactly two supported rpaths"
    [[ "$has_colocated_rpath" == true ]] || die "$name is missing LC_RPATH @loader_path"
    [[ "$has_homebrew_rpath" == true ]] || {
        die "$name is missing LC_RPATH @loader_path/../libexec"
    }
    while IFS= read -r dependency; do
        [[ -n "$dependency" ]] || continue
        validate_system_or_bundled_dependency "$name" "$dependency" binary
    done < <(linked_paths "$image")
done

for name in ax-engine-server ax-engine-bench; do
    linked_paths "$PAYLOAD_DIR/$name" | grep -Fx "@rpath/libmlx.dylib" >/dev/null || {
        die "$name does not load the bundled @rpath/libmlx.dylib"
    }
done

for name in "${BUNDLED_DYLIBS[@]}"; do
    image="$PAYLOAD_DIR/$name"
    image_id="$(otool -D "$image" | sed -n '2p')"
    [[ "$image_id" == "@rpath/$name" ]] || die "$name has unexpected install id: $image_id"
    while IFS= read -r dependency; do
        [[ -n "$dependency" && "$dependency" != "$image_id" ]] || continue
        validate_system_or_bundled_dependency "$name" "$dependency" dylib
    done < <(linked_paths "$image")
done

if [[ "$SKIP_SMOKE" = false ]]; then
    SMOKE_HOME="$(mktemp -d "${TMPDIR:-/tmp}/ax-engine-standalone-home.XXXXXX")"
    cleanup() {
        rm -rf "$SMOKE_HOME"
    }
    trap cleanup EXIT
    (
        cd "$SMOKE_HOME"
        for name in "${BINARY_NAMES[@]}"; do
            env -i \
                PATH=/usr/bin:/bin:/usr/sbin:/sbin \
                HOME="$SMOKE_HOME" \
                TMPDIR="$SMOKE_HOME" \
                "$PAYLOAD_DIR/$name" --help >/dev/null
        done
        if [[ "$RUN_DOCTOR" = true ]]; then
            env -i \
                PATH=/usr/bin:/bin:/usr/sbin:/sbin \
                HOME="$SMOKE_HOME" \
                TMPDIR="$SMOKE_HOME" \
                "$PAYLOAD_DIR/ax-engine" doctor >/dev/null
        fi
    )
fi

echo "Standalone AX Engine payload validated: $PAYLOAD_DIR"
