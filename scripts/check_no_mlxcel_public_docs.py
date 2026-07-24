#!/usr/bin/env python3
"""Keep MLXcel-specific Markdown out of the public docs tree."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

NEEDLE = "mlxcel"


class PublicDocsReferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    path: str
    line_number: int | None

    def render(self) -> str:
        if self.line_number is None:
            return f"{self.path}: filename references MLXcel"
        return f"{self.path}:{self.line_number}: content references MLXcel"


def find_public_docs_references(root: Path) -> list[Hit]:
    docs = root / "docs"
    if not docs.is_dir():
        return []

    hits: list[Hit] = []
    paths = sorted(
        path
        for path in docs.rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".md"
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if NEEDLE in path.name.lower():
            hits.append(Hit(path=relative, line_number=None))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if NEEDLE in line.lower():
                hits.append(Hit(path=relative, line_number=line_number))
    return hits


def check_no_mlxcel_public_docs(root: Path) -> None:
    hits = find_public_docs_references(root)
    if hits:
        rendered = "\n".join(f"- {hit.render()}" for hit in hits)
        raise PublicDocsReferenceError(
            "MLXcel-specific Markdown belongs under .internal/, not docs/:\n"
            f"{rendered}"
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
        check_no_mlxcel_public_docs(args.root)
    except PublicDocsReferenceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("ok: no MLXcel-specific Markdown under docs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
