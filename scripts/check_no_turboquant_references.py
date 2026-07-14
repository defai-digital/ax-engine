#!/usr/bin/env python3
"""Fail closed when retired TurboQuant runtime references reappear.

ADR-002 retired the TurboQuant runtime path in favor of the durable tiered
prefix cache. Historical benchmark artifacts, historical design docs, internal
notes, and release notes may keep the name for provenance; everything else
must not reference it. This checker greps tracked (and untracked,
non-ignored) files and fails when the needle appears outside the allowlist.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

NEEDLE = b"turboquant"

# Repo-relative path prefixes (POSIX separators) that may keep historical
# TurboQuant references.
ALLOWED_PREFIXES = (
    ".git/",
    ".internal/",
    "benchmarks/results/",
    "docs/designs/",
    # Release / migration notes keep the retirement record.
    "CHANGELOG.md",
    "RELEASE-NOTES.md",
    "docs/MIGRATION.md",
    # This checker and its unit test name the needle by design.
    "scripts/check_no_turboquant_references.py",
    "scripts/test_check_no_turboquant_references.py",
)

SKIP_WALK_DIRS = {".git", "target", "node_modules", "__pycache__", ".venv"}

# References to the checker itself (registrations in check-scripts.sh,
# scripts/README.md documentation) are not runtime references.
SELF_REFERENCE = b"check_no_turboquant_references"


class NoTurboquantReferencesError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    path: str
    line_number: int | None

    def render(self) -> str:
        if self.line_number is None:
            return f"{self.path}: path name references the retired TurboQuant runtime"
        return f"{self.path}:{self.line_number}: references the retired TurboQuant runtime"


def is_allowed(relative_path: str) -> bool:
    return any(
        relative_path == prefix.rstrip("/") or relative_path.startswith(prefix)
        for prefix in ALLOWED_PREFIXES
    )


def list_repo_files(root: Path) -> list[str]:
    """List repo-relative candidate files, preferring git's view of the tree."""
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(_walk_files(root))
    return sorted(entry.decode("utf-8") for entry in output.split(b"\0") if entry)


def _walk_files(root: Path) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_WALK_DIRS)
        for filename in sorted(filenames):
            yield (Path(dirpath) / filename).relative_to(root).as_posix()


def find_turboquant_references(root: Path) -> list[Hit]:
    hits: list[Hit] = []
    for relative_path in list_repo_files(root):
        if is_allowed(relative_path):
            continue
        lowered_path = relative_path.lower().encode("utf-8")
        if NEEDLE in lowered_path.replace(SELF_REFERENCE, b""):
            hits.append(Hit(path=relative_path, line_number=None))
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        lowered = content.lower().replace(SELF_REFERENCE, b"")
        if NEEDLE not in lowered:
            continue
        for line_number, line in enumerate(lowered.splitlines(), start=1):
            if NEEDLE in line:
                hits.append(Hit(path=relative_path, line_number=line_number))
    return hits


def check_no_turboquant_references(root: Path) -> None:
    hits = find_turboquant_references(root)
    if hits:
        rendered = "\n".join(f"- {hit.render()}" for hit in hits)
        raise NoTurboquantReferencesError(
            "TurboQuant runtime references are retired (ADR-002) and must not "
            f"appear outside the historical allowlist:\n{rendered}"
        )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root to scan",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        check_no_turboquant_references(args.root)
    except NoTurboquantReferencesError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("ok: no TurboQuant references outside the historical allowlist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
