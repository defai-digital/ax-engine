#!/usr/bin/env python3
"""Embed the pinned Sparkle runtime and sign nested code from the inside out."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import plistlib
import shutil
import subprocess

FRAMEWORK = Path('Contents/Frameworks/Sparkle.framework')
SPARKLE_BINARIES = (
    'Versions/B/Autoupdate',
    'Versions/B/XPCServices/Installer.xpc/Contents/MacOS/Installer',
    'Versions/B/XPCServices/Downloader.xpc/Contents/MacOS/Downloader',
    'Versions/B/Updater.app/Contents/MacOS/Updater',
    'Versions/B/Sparkle',
)
SPARKLE_BUNDLES = (
    'Versions/B/XPCServices/Installer.xpc',
    'Versions/B/XPCServices/Downloader.xpc',
    'Versions/B/Updater.app',
    '.',
)


def embed(app: Path, artifact: Path) -> None:
    source = artifact / 'Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework'
    config = json.loads((Path(__file__).resolve().parents[1] / 'packaging/macos/sparkle.json').read_text())
    metadata = plistlib.loads((source / 'Versions/B/Resources/Info.plist').read_bytes())
    if metadata.get('CFBundleShortVersionString') != config['version']:
        raise ValueError('Sparkle artifact does not match the pinned version')
    destination = app / FRAMEWORK
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ditto', '--noextattr', str(source), str(destination)], check=True)
    for name in SPARKLE_BINARIES:
        binary = destination / name
        subprocess.run(['lipo', str(binary), '-thin', 'arm64', '-output', str(binary)], check=True)
    for name in (*SPARKLE_BINARIES, *SPARKLE_BUNDLES):
        subprocess.run(['codesign', '--force', '--sign', '-', str(destination / name)], check=True)
    (app / 'Contents/Resources').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact / 'LICENSE', app / 'Contents/Resources/SPARKLE-LICENSE.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    parser.add_argument('artifact', type=Path)
    args = parser.parse_args()
    embed(args.app, args.artifact)
