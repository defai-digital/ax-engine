#!/usr/bin/env python3
"""Build and notarize an exact-source macOS GUI candidate; never publish it."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TEAM = "N5ZUZDUJS6"
IDENTITY = "CCA480038EE61E7CB9DF9FAACD842EC06949CD79"
BUNDLE_ID = "com.automatosx.ax-engine"
MIN_MACOS = "26.2"
RUNTIME = Path("Contents/Helpers/AXEngineRuntime.bundle")
_UPDATER_SPEC = importlib.util.spec_from_file_location("embed_updater", ROOT / "scripts/embed-macos-updater.py")
_UPDATER = importlib.util.module_from_spec(_UPDATER_SPEC)
_UPDATER_SPEC.loader.exec_module(_UPDATER)
MACHO = {b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(*args: object, capture: bool = False, env: dict | None = None) -> str:
    result = subprocess.run([str(a) for a in args], cwd=ROOT, check=True,
                            text=True, stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None, env=env)
    return result.stdout if capture else ""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_digest(root: Path) -> str:
    """Bind file paths, content, executable modes and internal symbolic links."""
    rows = []
    require(root.is_dir() and not root.is_symlink(), "Expected a regular app directory")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            require(path.resolve().is_relative_to(root.resolve()), "App symlink escapes its bundle")
            rows.append([relative, "link", os.readlink(path)])
        elif path.is_file():
            rows.append([relative, path.stat().st_mode & 0o777, sha256(path)])
        else:
            require(path.is_dir(), "Special files cannot be shipped")
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def identity(app: Path, version: str, commit: str | None = None) -> dict:
    with (app / "Contents/Info.plist").open("rb") as stream:
        info = plistlib.load(stream)
    require(info.get("CFBundleIdentifier") == BUNDLE_ID, "Unexpected app identity")
    require(info.get("CFBundleShortVersionString") == version and
            info.get("CFBundleVersion") == version, "Unexpected app version")
    require(info.get("LSMinimumSystemVersion") == MIN_MACOS, "Unexpected minimum macOS")
    require(not any(k.startswith("AXEngineDevelopment") for k in info), "Development paths in app")
    if "AXEngineDistributionBuild" in info:
        config = json.loads((ROOT / 'packaging/macos/sparkle.json').read_text())
        require(info['AXEngineDistributionBuild'] is True and
                info.get('SUFeedURL') == config['feedURL'] and
                info.get('SUPublicEDKey') == config['publicKey'] and
                info.get('SUAllowsAutomaticUpdates') is False and
                info.get('SUEnableAutomaticChecks') is False and
                info.get('SUEnableSystemProfiling') is False, 'Invalid updater configuration')
        require((app / _UPDATER.FRAMEWORK).is_dir(), 'Missing updater framework')
    if commit is not None:
        require(info.get("AXEngineSourceCommit") == commit, "App source commit mismatch")
    return info


def signing_targets(app: Path) -> list[Path]:
    expected = [app / "Contents/MacOS" / name for name in ("AXEngineApp", "AXEngineModelScanner")]
    expected += [app / RUNTIME / "Contents/MacOS" / name for name in
                 ("ax-engine", "ax-engine-server", "ax-engine-bench", "libmlx.dylib", "libjaccl.dylib")]
    framework = app / _UPDATER.FRAMEWORK
    # Historical pre-updater candidates remain verifiable. New builds always embed it.
    updater_binaries = [framework / name for name in _UPDATER.SPARKLE_BINARIES] if framework.exists() else []
    updater_bundles = [framework / name for name in _UPDATER.SPARKLE_BUNDLES] if framework.exists() else []
    expected += updater_binaries
    found = []
    for file in app.rglob("*"):
        if file.is_file() and not file.is_symlink():
            with file.open("rb") as stream:
                if stream.read(4) in MACHO:
                    found.append(file)
    require(set(found) == set(expected), "Missing or unexpected native code in app")
    return expected + updater_bundles + [app / RUNTIME, app]


def validate_native_platform(target: Path) -> None:
    require(run("lipo", "-archs", target, capture=True).strip() == "arm64", "Unexpected architecture")
    metadata = run("vtool", "-show-build", target, capture=True)
    floors = re.findall(r"^\s*minos\s+(\d+(?:\.\d+){1,2})\s*$", metadata, re.MULTILINE)
    require(len(floors) == 1 and re.search(r"^\s*platform\s+MACOS\s*$", metadata, re.MULTILINE),
            "Missing or ambiguous native macOS requirement")
    def components(value: str) -> tuple[int, ...]:
        return tuple(int(x) for x in (value.split(".") + ["0", "0"])[:3])
    require(components(floors[0]) <= components(MIN_MACOS), "Native runtime exceeds declared minimum macOS")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_runtime_binaries(output: Path, toolchain: str) -> None:
    """Stage the server from its unwind profile, without reusing an abort binary."""
    target = ROOT / "target"
    cargo = ("rustup", "run", toolchain, "cargo", "build", "--locked", "--target-dir", target)
    run(*cargo, "--release", "-p", "ax-engine-bench", "--bin", "ax-engine",
        "--bin", "ax-engine-bench")
    # Worker catch_unwind containment requires unwind even in the GUI bundle.
    run(*cargo, "--profile", "release-server", "-p", "ax-engine-server",
        "--bin", "ax-engine-server")
    output.mkdir()
    for name in ("ax-engine", "ax-engine-bench", "ax-engine-server"):
        profile = "release-server" if name == "ax-engine-server" else "release"
        shutil.copy2(target / profile / name, output / name)


def build(output: Path, mlx: Path) -> None:
    require(not output.exists(), "Use a new output directory")
    require(not run("git", "status", "--porcelain", capture=True).strip(), "Build requires a clean checkout")
    commit = run("git", "rev-parse", "HEAD", capture=True).strip()
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    require(re.fullmatch(r"\d+\.\d+\.\d+", version) is not None, "Expected a stable source version")
    toolchain = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
    output.mkdir(parents=True)
    build_runtime_binaries(output / "binaries", toolchain)
    run("bash", ROOT / "scripts/prepare-mlx-release-runtime.sh", mlx, output / "mlx")
    run("bash", ROOT / "scripts/prepare-standalone-release.sh", output / "binaries",
        output / "mlx", output / "runtime")
    run("bash", ROOT / "scripts/build-macos-app.sh", "--distribution", "--runtime-dir",
        output / "runtime", "--output", output / "app")
    app = output / "app/AX Engine.app"
    info = identity(app, version)
    info["AXEngineSourceCommit"] = commit
    with (app / "Contents/Info.plist").open("wb") as stream:
        plistlib.dump(info, stream)
    run("codesign", "--force", "--sign", "-", app)
    run("codesign", "--verify", "--deep", "--strict", app)
    require(run("git", "rev-parse", "HEAD", capture=True).strip() == commit and
            not run("git", "status", "--porcelain", capture=True).strip(), "Source changed during build")
    write_json(output / "build.json", {"schema": "ax-engine.gui-build.v1", "version": version,
               "commit": commit, "rust": toolchain, "mlx": (ROOT / "mlx.version").read_text().strip(),
               "appTreeSha256": tree_digest(app)})
    print(f"Built local candidate {version} from {commit}; nothing published.")


def apple_verify(path: Path, *, image: bool = False) -> None:
    run("codesign", "--verify", "--deep", "--strict", path)
    details = subprocess.run(["codesign", "-dv", "--verbose=4", str(path)],
                             capture_output=True, text=True, check=True).stderr
    require(f"TeamIdentifier={TEAM}" in details.splitlines(), "Unexpected Apple signing team")
    if image:
        run("spctl", "--assess", "--type", "open", "--context", "context:primary-signature", path)
    else:
        require("runtime" in details and "Timestamp=" in details, "Missing hardened runtime or timestamp")
        run("spctl", "--assess", "--type", "execute", path)
    run("xcrun", "stapler", "validate", path)


def notarize(path: Path, profile: str, record: Path) -> str:
    # Captured JSON contains request identity/status, never account credentials.
    submitted = json.loads(run("xcrun", "notarytool", "submit", path, "--keychain-profile", profile,
                               "--output-format", "json", capture=True))
    write_json(record, submitted)
    # Persist the request ID before waiting, so timeout/interruption is recoverable
    # with notarytool info/log rather than losing track of an in-flight submission.
    require(isinstance(submitted.get("id"), str), "Notary submission did not return an ID")
    reply = json.loads(run("xcrun", "notarytool", "wait", submitted["id"], "--keychain-profile", profile,
                           "--timeout", "30m", "--output-format", "json", capture=True))
    write_json(record, reply)
    require(reply.get("status") == "Accepted", f"Notarization not accepted; inspect {record.name}")
    return reply["id"]


def package(source: Path, output: Path, signing_identity: str, profile: str) -> None:
    require(not output.exists(), "Use a new package output directory")
    receipt = json.loads((source / "build.json").read_text())
    require(receipt.get("schema") == "ax-engine.gui-build.v1", "Invalid build receipt")
    version, commit = receipt["version"], receipt["commit"]
    require(re.fullmatch(r"\d+\.\d+\.\d+", version) is not None and
            re.fullmatch(r"[a-f0-9]{40}", commit) is not None, "Invalid source identity")
    original = source / "app/AX Engine.app"
    identity(original, version, commit)
    require(tree_digest(original) == receipt["appTreeSha256"], "App changed since build")
    output.mkdir(parents=True)
    app = output / "AX Engine.app"
    run("ditto", original, app)
    for target in signing_targets(app):
        if target.is_file():
            validate_native_platform(target)
        run("codesign", "--force", "--sign", signing_identity, "--options", "runtime", "--timestamp", target)
    run("codesign", "--verify", "--deep", "--strict", app)
    archive = output / "notary-app.zip"
    run("ditto", "-c", "-k", "--keepParent", app, archive)
    app_notary = notarize(archive, profile, output / "app-notary.json")
    run("xcrun", "stapler", "staple", app)
    apple_verify(app)
    image = output / f"AX-Engine-{version}-mac-arm64.dmg"
    with tempfile.TemporaryDirectory(prefix="ax-engine-dmg-") as temporary:
        root = Path(temporary)
        run("ditto", app, root / "AX Engine.app")
        (root / "Applications").symlink_to("/Applications")
        run("hdiutil", "create", "-volname", f"AX Engine {version}", "-srcfolder", root,
            "-format", "UDZO", "-fs", "HFS+", image)
    run("codesign", "--force", "--sign", signing_identity, "--timestamp", image)
    image_notary = notarize(image, profile, output / "dmg-notary.json")
    run("xcrun", "stapler", "staple", image)
    apple_verify(image, image=True)
    write_json(output / "notarization.json", {
        "schema": "ax-engine.gui-notarization.v1", "version": version, "commit": commit,
        "team": TEAM, "bundleId": BUNDLE_ID, "appRequest": app_notary, "dmgRequest": image_notary,
        "filename": image.name, "sha256": sha256(image), "size": image.stat().st_size,
        "appTreeSha256": tree_digest(app),
    })
    print(f"Notarized local candidate: {image}. Nothing published.")


def verify_package(source: Path) -> dict:
    receipt = json.loads((source / "notarization.json").read_text())
    require(receipt.get("schema") == "ax-engine.gui-notarization.v1", "Invalid notarization receipt")
    require(receipt.get("team") == TEAM and receipt.get("bundleId") == BUNDLE_ID, "Invalid signing identity")
    version, commit = receipt["version"], receipt["commit"]
    require(re.fullmatch(r"\d+\.\d+\.\d+", version) is not None and
            re.fullmatch(r"[a-f0-9]{40}", commit) is not None, "Invalid release identity")
    require(receipt["filename"] == f"AX-Engine-{version}-mac-arm64.dmg", "Unexpected artifact name")
    image = source / receipt["filename"]
    require(image.is_file() and not image.is_symlink() and sha256(image) == receipt["sha256"] and
            image.stat().st_size == receipt["size"], "DMG hash or size mismatch")
    apple_verify(image, image=True)
    with tempfile.TemporaryDirectory(prefix="ax-engine-verify-") as temporary:
        mount = Path(temporary) / "mount"
        mount.mkdir()
        run("hdiutil", "attach", image, "-readonly", "-nobrowse", "-mountpoint", mount)
        try:
            app = mount / "AX Engine.app"
            identity(app, version, commit)
            require(tree_digest(app) == receipt["appTreeSha256"], "DMG app differs from notarized app")
            apple_verify(app)
            native = app / RUNTIME / "Contents/MacOS/ax-engine"
            result = json.loads(run(native, "download", "--list", "--json", capture=True))
            require(result.get("download_backend") == "native-rust-public-v1", "Missing native downloads")
        finally:
            run("hdiutil", "detach", mount)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("build")
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--mlx-lib-dir", type=Path, default=ROOT / ".venv/lib/python3.12/site-packages/mlx/lib")
    pack = sub.add_parser("package")
    pack.add_argument("--source", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--identity", default=os.environ.get("AX_CODESIGN_IDENTITY", IDENTITY))
    pack.add_argument("--notary-profile", default=os.environ.get("AX_NOTARY_PROFILE", "ax-notary"))
    verify = sub.add_parser("verify")
    verify.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.output.resolve(), args.mlx_lib_dir.resolve())
    elif args.command == "package":
        package(args.source.resolve(), args.output.resolve(), args.identity, args.notary_profile)
    else:
        result = verify_package(args.source.resolve())
        print(f"Verified notarized GUI {result['version']} from {result['commit']}.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"GUI release failed: {error}", file=sys.stderr)
        sys.exit(1)
