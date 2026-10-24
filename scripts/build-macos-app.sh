#!/usr/bin/env bash
# Build a local arm64 app bundle. Publication/signing is a separate release gate.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="$ROOT/packaging/macos/AXEngineApp"
CONFIGURATION="${AX_APP_CONFIGURATION:-debug}"
OUTPUT="$ROOT/build/macos"
RUNTIME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --release) CONFIGURATION=release; shift ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --runtime-dir) RUNTIME="$2"; shift 2 ;;
        --help) echo 'Usage: scripts/build-macos-app.sh [--release] [--output <directory>] [--runtime-dir <staged-runtime>]'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$(uname -m)" == arm64 ]] || { echo 'The native app targets macOS arm64.' >&2; exit 1; }
[[ "$CONFIGURATION" == debug || "$CONFIGURATION" == release ]] || exit 2
swift build --package-path "$PACKAGE" --configuration "$CONFIGURATION"
BIN="$(swift build --package-path "$PACKAGE" --configuration "$CONFIGURATION" --show-bin-path)"
APP="$OUTPUT/AX Engine.app"
[[ ! -e "$APP/Contents/Resources/runtime" ]] || { echo 'Use a fresh output directory when a runtime has already been staged.' >&2; exit 1; }
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN/AXEngineApp" "$APP/Contents/MacOS/AXEngineApp"
# Always regenerate from the untouched source master; never from a masked icon.
"${AX_APP_ICON_PYTHON:-python3}" "$PACKAGE/Scripts/generate_icon.py" --output "$APP/Contents/Resources"
cp "$PACKAGE/Resources/MTPLX-LICENSE.txt" "$PACKAGE/Resources/MTPLX-NOTICE.txt" "$APP/Contents/Resources/"
cp "$PACKAGE/NOTICE" "$APP/Contents/Resources/NOTICE.txt"
VERSION="$(python3 - "$ROOT/Cargo.toml" <<'PY'
import sys, tomllib
with open(sys.argv[1], 'rb') as source:
    print(tomllib.load(source)['workspace']['package']['version'])
PY
)"
python3 - "$APP/Contents/Info.plist" "$VERSION" <<'PY'
import plistlib, sys
with open(sys.argv[1], 'wb') as dest:
    plistlib.dump({
        'CFBundleName': 'AX Engine', 'CFBundleDisplayName': 'AX Engine',
        'CFBundleIdentifier': 'com.automatosx.ax-engine',
        'CFBundleExecutable': 'AXEngineApp', 'CFBundlePackageType': 'APPL',
        'CFBundleShortVersionString': sys.argv[2], 'CFBundleVersion': sys.argv[2],
        'CFBundleIconFile': 'AppIcon', 'LSMinimumSystemVersion': '26.0',
        'NSHighResolutionCapable': True,
        'NSHumanReadableCopyright': 'AX Engine contributors. Selected components: Copyright 2026 Youssof Altoukhi (Apache-2.0).',
        'NSAppTransportSecurity': {'NSAllowsLocalNetworking': True},
    }, dest)
PY
if [[ -n "$RUNTIME" ]]; then
    for file in ax-engine ax-engine-server libmlx.dylib libjaccl.dylib mlx.metallib; do
        [[ -f "$RUNTIME/$file" ]] || { echo "Missing runtime file: $file" >&2; exit 1; }
    done
    [[ ! -e "$APP/Contents/Resources/runtime" ]] || { echo 'Use a fresh output directory when staging a runtime.' >&2; exit 1; }
    ditto "$RUNTIME" "$APP/Contents/Resources/runtime"
fi
# Ad-hoc identity is for this local build only, not Developer ID/notarization.
codesign --force --sign - "$APP"
codesign --verify --strict "$APP"
echo "Local app bundle: $APP"
echo 'This build has not been published or notarized.'
