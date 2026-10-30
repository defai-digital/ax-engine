#!/usr/bin/env bash
# Build a local arm64 app bundle. Publication/signing is a separate release gate.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="$ROOT/packaging/macos/AXEngineApp"
CONFIGURATION="${AX_APP_CONFIGURATION:-debug}"
OUTPUT="$ROOT/build/macos"
RUNTIME=""
DISTRIBUTION=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --release) CONFIGURATION=release; shift ;;
        --output|--runtime-dir)
            [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
            if [[ "$1" == --output ]]; then OUTPUT="$2"; else RUNTIME="$2"; fi
            shift 2 ;;
        --distribution) DISTRIBUTION=true; CONFIGURATION=release; shift ;;
        --help) echo 'Usage: scripts/build-macos-app.sh [--release] [--output <directory>] [--runtime-dir <staged-runtime>] [--distribution]'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$(uname -m)" == arm64 ]] || { echo 'The native app targets macOS arm64.' >&2; exit 1; }
[[ "$CONFIGURATION" == debug || "$CONFIGURATION" == release ]] || exit 2
if [[ "$DISTRIBUTION" == true ]]; then
    [[ -n "$RUNTIME" && -x "$RUNTIME/ax-engine" ]] || { echo 'Distribution requires a staged native runtime.' >&2; exit 1; }
fi
swift build --package-path "$PACKAGE" --configuration "$CONFIGURATION"
BIN="$(swift build --package-path "$PACKAGE" --configuration "$CONFIGURATION" --show-bin-path)"
APP="$OUTPUT/AX Engine.app"
RUNTIME_BUNDLE="$APP/Contents/Helpers/AXEngineRuntime.bundle"
[[ ! -e "$APP/Contents/Helpers" && ! -e "$APP/Contents/Resources/runtime" ]] || { echo 'Use a fresh output directory when a runtime has already been staged.' >&2; exit 1; }
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN/AXEngineApp" "$APP/Contents/MacOS/AXEngineApp"
cp "$BIN/AXEngineModelScanner" "$APP/Contents/MacOS/AXEngineModelScanner"
python3 "$ROOT/scripts/embed-macos-updater.py" "$APP" "$PACKAGE/.build/artifacts/sparkle/Sparkle"
codesign --force --sign - "$APP/Contents/MacOS/AXEngineModelScanner"
# Always regenerate from the untouched source master; never from a masked icon.
"${AX_APP_ICON_PYTHON:-python3}" "$PACKAGE/Scripts/generate_icon.py" --output "$APP/Contents/Resources"
cp "$PACKAGE/Resources/MTPLX-LICENSE.txt" "$PACKAGE/Resources/MTPLX-NOTICE.txt" "$APP/Contents/Resources/"
cp "$PACKAGE/NOTICE" "$APP/Contents/Resources/NOTICE.txt"
cp "$PACKAGE/Resources/PrivacyInfo.xcprivacy" "$APP/Contents/Resources/PrivacyInfo.xcprivacy"
VERSION="$(python3 - "$ROOT/Cargo.toml" <<'PY'
import sys, tomllib
with open(sys.argv[1], 'rb') as source:
    print(tomllib.load(source)['workspace']['package']['version'])
PY
)"
python3 - "$APP/Contents/Info.plist" "$VERSION" "$ROOT" "$DISTRIBUTION" <<'PY'
import json, os, plistlib, sys
from pathlib import Path
config = json.loads((Path(sys.argv[3]) / "packaging/macos/sparkle.json").read_text())
development = {}
if sys.argv[4] != "true":
    root = Path(sys.argv[3])
    native = [root/'target'/profile/'ax-engine' for profile in ['debug', 'release']]
    native = [path for path in native if path.is_file() and os.access(path, os.X_OK)]
    console = root/'.venv/bin/ax-engine'
    executable = max(native, key=lambda p: p.stat().st_mtime) if native else console
    if executable.is_file() and os.access(executable, os.X_OK):
        development['AXEngineDevelopmentRuntime'] = str(executable)
    if console.is_file():
        development['AXEngineDevelopmentPythonBin'] = str(console.parent)
with open(sys.argv[1], 'wb') as dest:
    plistlib.dump({
        'CFBundleName': 'AX Engine', 'CFBundleDisplayName': 'AX Engine',
        'CFBundleIdentifier': 'com.automatosx.ax-engine',
        'CFBundleExecutable': 'AXEngineApp', 'CFBundlePackageType': 'APPL',
        'CFBundleShortVersionString': sys.argv[2], 'CFBundleVersion': sys.argv[2],
        'CFBundleIconFile': 'AppIcon', 'LSMinimumSystemVersion': '26.2',
        'NSHighResolutionCapable': True,
        'AXEngineDistributionBuild': sys.argv[4] == 'true',
        'SUFeedURL': config['feedURL'], 'SUPublicEDKey': config['publicKey'],
        'SUEnableAutomaticChecks': False, 'SUAllowsAutomaticUpdates': False,
        'SUAutomaticallyUpdate': False, 'SUEnableSystemProfiling': False,
        'NSHumanReadableCopyright': 'AX Engine contributors.',
        'NSAppTransportSecurity': {'NSAllowsLocalNetworking': True},
        **development,
    }, dest)
PY
if [[ -n "$RUNTIME" ]]; then
    for file in ax-engine ax-engine-server ax-engine-bench libmlx.dylib libjaccl.dylib mlx.metallib MLX-LICENSE.txt; do
        [[ -f "$RUNTIME/$file" ]] || { echo "Missing runtime file: $file" >&2; exit 1; }
    done
    mkdir -p "$RUNTIME_BUNDLE/Contents/MacOS" "$RUNTIME_BUNDLE/Contents/Resources"
    # Keep native code and data in their standard bundle locations. The internal
    # Metal resource link preserves MLX's colocated lookup and doctor contract
    # without treating the Metal library or license as nested code.
    for file in ax-engine ax-engine-server ax-engine-bench libmlx.dylib libjaccl.dylib; do
        ditto --noextattr "$RUNTIME/$file" "$RUNTIME_BUNDLE/Contents/MacOS/$file"
        codesign --force --sign - "$RUNTIME_BUNDLE/Contents/MacOS/$file"
    done
    for file in mlx.metallib MLX-LICENSE.txt; do
        ditto --noextattr "$RUNTIME/$file" "$RUNTIME_BUNDLE/Contents/Resources/$file"
    done
    ln -s ../Resources/mlx.metallib "$RUNTIME_BUNDLE/Contents/MacOS/mlx.metallib"
    python3 - "$RUNTIME_BUNDLE/Contents/Info.plist" "$VERSION" <<'PY'
import plistlib, sys
with open(sys.argv[1], 'wb') as dest:
    plistlib.dump({
        'CFBundleIdentifier': 'com.automatosx.ax-engine.runtime',
        'CFBundleExecutable': 'ax-engine', 'CFBundlePackageType': 'BNDL',
        'CFBundleVersion': sys.argv[2], 'CFBundleShortVersionString': sys.argv[2],
    }, dest)
PY
    codesign --force --sign - "$RUNTIME_BUNDLE"
fi
if [[ "$DISTRIBUTION" == true ]]; then
    # Relocation invalidates input signatures; execute only the re-signed copy.
    "$RUNTIME_BUNDLE/Contents/MacOS/ax-engine" download --list --json | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("download_backend") == "native-rust-public-v1" else "Distribution requires native public-model downloads; Python-backed runtimes are not supported.")'
fi
# Ad-hoc identity is for this local build only, not Developer ID/notarization.
codesign --force --sign - "$APP"
codesign --verify --deep --strict "$APP"
echo "Local app bundle: $APP"
echo 'This build has not been published or notarized.'
