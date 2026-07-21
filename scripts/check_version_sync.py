#!/usr/bin/env python3
"""Verify that published package and installation-guide versions are aligned."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tomllib


class VersionSyncError(ValueError):
    """Report an invalid or inconsistent version surface."""


INSTALL_REQUIREMENT_PATTERN = re.compile(
    r"ax-engine(?:\[[^\]]+\])?>=(\d+\.\d+\.\d+),<\d+"
)


def _required_match(root: pathlib.Path, relative_path: str, pattern: str) -> str:
    text = (root / relative_path).read_text(encoding="utf-8")
    match = re.search(pattern, text)
    if match is None:
        raise VersionSyncError(f"could not parse version from {relative_path}")
    return match.group(1)


def _required_install_version(root: pathlib.Path, relative_path: str) -> str:
    text = (root / relative_path).read_text(encoding="utf-8")
    versions = set(INSTALL_REQUIREMENT_PATTERN.findall(text))
    if not versions:
        raise VersionSyncError(f"could not parse install version from {relative_path}")
    if len(versions) != 1:
        details = ", ".join(sorted(versions))
        raise VersionSyncError(
            f"inconsistent install versions in {relative_path}: {details}"
        )
    return versions.pop()


def load_versions(root: pathlib.Path) -> dict[str, str]:
    cargo = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    javascript = json.loads(
        (root / "sdk/javascript/package.json").read_text(encoding="utf-8")
    )

    return {
        "Cargo.toml": cargo["workspace"]["package"]["version"],
        "pyproject.toml": pyproject["project"]["version"],
        "sdk/javascript/package.json": javascript["version"],
        "sdk/ruby/lib/ax_engine/version.rb": _required_match(
            root,
            "sdk/ruby/lib/ax_engine/version.rb",
            r'\bVERSION\s*=\s*"([^"]+)"',
        ),
        "sdk/go/axengine/client.go": _required_match(
            root,
            "sdk/go/axengine/client.go",
            r'\bconst\s+Version\s*=\s*"([^"]+)"',
        ),
        "sdk/swift/Sources/AxEngine/AxEngineClient.swift": _required_match(
            root,
            "sdk/swift/Sources/AxEngine/AxEngineClient.swift",
            r'\bstatic\s+let\s+version\s*=\s*"([^"]+)"',
        ),
        "README.md": _required_install_version(root, "README.md"),
        "docs/GETTING-STARTED.md": _required_install_version(
            root, "docs/GETTING-STARTED.md"
        ),
        "crates/ax-engine-py/README.md": _required_install_version(
            root, "crates/ax-engine-py/README.md"
        ),
        "docs/sdk/python.md": _required_install_version(root, "docs/sdk/python.md"),
        "docs/sdk/swift.md": _required_match(
            root,
            "docs/sdk/swift.md",
            r"\bcurrent version is `(\d+\.\d+\.\d+)`",
        ),
    }


def verify_versions(root: pathlib.Path, expected: str | None = None) -> str:
    versions = load_versions(root)
    expected_version = (
        expected.removeprefix("v") if expected is not None else versions["Cargo.toml"]
    )
    mismatches = {
        path: version for path, version in versions.items() if version != expected_version
    }
    if mismatches:
        details = ", ".join(
            f"{path}={version}" for path, version in sorted(mismatches.items())
        )
        raise VersionSyncError(
            f"version mismatch: expected={expected_version}, {details}"
        )
    return expected_version


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="repository root (defaults to the parent of this script directory)",
    )
    parser.add_argument(
        "--expected",
        help="expected version or v-prefixed release tag; defaults to Cargo.toml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        version = verify_versions(args.root.resolve(), args.expected)
    except (KeyError, OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        print(f"error: version consistency check failed: {exc}", file=sys.stderr)
        return 1

    print(f"Version verified: {version} (package metadata and install guides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
