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
PYTHON_MIN_VERSION = "3.12"
PYTHON_REQUIRES = f">={PYTHON_MIN_VERSION}"
PYTHON_ABI3_FEATURE = "abi3-py312"

TOML_VERSION_SPECS: dict[str, tuple[str, ...]] = {
    "Cargo.toml": ("workspace", "package", "version"),
    "pyproject.toml": ("project", "version"),
}

REGEX_VERSION_SPECS: tuple[tuple[str, str], ...] = (
    ("sdk/ruby/lib/ax_engine/version.rb", r'\bVERSION\s*=\s*"([^"]+)"'),
    ("sdk/go/axengine/client.go", r'\bconst\s+Version\s*=\s*"([^"]+)"'),
    (
        "sdk/swift/Sources/AxEngine/AxEngineClient.swift",
        r'\bstatic\s+let\s+version\s*=\s*"([^"]+)"',
    ),
    ("docs/sdk/swift.md", r"\bcurrent version is `(\d+\.\d+\.\d+)`"),
)

INSTALL_VERSION_PATHS: tuple[str, ...] = (
    "README.md",
    "docs/GETTING-STARTED.md",
    "crates/ax-engine-py/README.md",
    "docs/sdk/python.md",
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

    toml_documents: dict[str, object] = {
        "Cargo.toml": cargo,
        "pyproject.toml": pyproject,
    }

    versions: dict[str, str] = {"sdk/javascript/package.json": javascript["version"]}

    for relative_path, key_path in TOML_VERSION_SPECS.items():
        value: object = toml_documents[relative_path]
        for key in key_path:
            value = value[key]  # type: ignore[index]
        versions[relative_path] = value  # type: ignore[assignment]

    for relative_path, pattern in REGEX_VERSION_SPECS:
        versions[relative_path] = _required_match(root, relative_path, pattern)

    for relative_path in INSTALL_VERSION_PATHS:
        versions[relative_path] = _required_install_version(root, relative_path)

    return versions


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


def verify_python_policy(root: pathlib.Path) -> str:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    classifiers = set(project.get("classifiers", []))
    expected_classifiers = {
        f"Programming Language :: Python :: {version}"
        for version in ("3.12", "3.13")
    }
    problems: list[str] = []

    if project.get("requires-python") != PYTHON_REQUIRES:
        problems.append(
            f"pyproject.toml requires-python={project.get('requires-python')!r}, "
            f"expected {PYTHON_REQUIRES!r}"
        )
    missing_classifiers = expected_classifiers - classifiers
    if missing_classifiers:
        problems.append(
            "pyproject.toml missing classifiers: " + ", ".join(sorted(missing_classifiers))
        )
    if classifiers & {
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    }:
        problems.append("pyproject.toml still advertises a Python version below 3.12")
    if pyproject.get("tool", {}).get("ruff", {}).get("target-version") != "py312":
        problems.append("pyproject.toml Ruff target-version must be py312")
    if pyproject.get("tool", {}).get("mypy", {}).get("python_version") != PYTHON_MIN_VERSION:
        problems.append("pyproject.toml mypy python_version must be 3.12")

    py_crate = (root / "crates/ax-engine-py/Cargo.toml").read_text(encoding="utf-8")
    if PYTHON_ABI3_FEATURE not in py_crate:
        problems.append(f"Python extension must enable {PYTHON_ABI3_FEATURE}")
    if "abi3-py310" in py_crate or "abi3-py311" in py_crate:
        problems.append("Python extension still enables an ABI floor below 3.12")

    if problems:
        raise VersionSyncError("Python compatibility policy mismatch: " + "; ".join(problems))
    return PYTHON_MIN_VERSION


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
        python_min = verify_python_policy(args.root.resolve())
    except (KeyError, OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        print(f"error: version consistency check failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Version verified: {version}; Python >={python_min} "
        "(package metadata, ABI, and install guides)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
