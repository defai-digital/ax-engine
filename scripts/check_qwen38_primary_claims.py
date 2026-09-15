#!/usr/bin/env python3
"""Keep the Qwen 3.8 27B primary-pack contract honest in public docs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PRIMARY_ALIAS = "qwen3.8-27b:axq"
PRIMARY_REPO = "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP"
PRIMARY_REVISION = "3e290738e96972307c6aeb9934ab170ca0eae1c1"
STATUS_SENTENCE = (
    "Primary optimization target. Checkpoint Tier 1. MTP Tier 2 pending. "
    "AX certification record: Candidate (gates open)."
)
STATUS_FILES = (
    "README.md",
    "docs/SUPPORTED-MODELS.md",
    "docs/GETTING-STARTED.md",
    "docs/model-certifications/qwen3.8-27b-axq.md",
    "docs/TESTING.md",
)
HOST_ALIAS_RE = re.compile(
    r"\b(?:df-macbookpro-m5|df-macbookpro-m3|tn-macstudio-m3|df-macstudio-m2)\b"
)
UNEARNED_TIER2_RE = re.compile(
    r"qwen\s*3\.8[\s\S]{0,160}(?:mtp\s+)?tier\s*2\s+certified"
    r"|(?:mtp\s+)?tier\s*2\s+certified[\s\S]{0,160}qwen\s*3\.8",
    re.IGNORECASE,
)


class PrimaryClaimError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    path: str
    line_number: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line_number}: {self.message}"


def _public_markdown(root: Path) -> list[Path]:
    docs = root / "docs"
    paths = [root / "README.md"]
    if docs.is_dir():
        paths.extend(
            sorted(
                path
                for path in docs.rglob("*.md")
                if path.is_file() and not path.is_symlink()
            )
        )
    return [path for path in paths if path.is_file()]


def _is_historical_benchmark_path(line: str) -> bool:
    return "benchmarks/" in line.replace("\\", "/")


def find_primary_claim_issues(root: Path) -> list[Hit]:
    hits: list[Hit] = []
    for relative in STATUS_FILES:
        path = root / relative
        if not path.is_file():
            hits.append(Hit(path=relative, line_number=1, message="missing required file"))
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if STATUS_SENTENCE not in text:
            hits.append(
                Hit(
                    path=relative,
                    line_number=1,
                    message="missing canonical Qwen 3.8 27B status sentence",
                )
            )
        if PRIMARY_ALIAS not in text:
            hits.append(
                Hit(path=relative, line_number=1, message=f"missing alias {PRIMARY_ALIAS}")
            )

    readme = root / "README.md"
    if readme.is_file():
        readme_text = readme.read_text(encoding="utf-8", errors="replace")
        if f"ax-engine serve {PRIMARY_ALIAS}" not in readme_text:
            hits.append(
                Hit(
                    path="README.md",
                    line_number=1,
                    message=f"missing first-run serve command ax-engine serve {PRIMARY_ALIAS}",
                )
            )
        if PRIMARY_REPO not in readme_text or PRIMARY_REVISION not in readme_text:
            hits.append(
                Hit(
                    path="README.md",
                    line_number=1,
                    message="missing pinned Qwen 3.8 27B repo or revision",
                )
            )

    cert = root / "docs/model-certifications/qwen3.8-27b-axq.md"
    if cert.is_file():
        cert_text = cert.read_text(encoding="utf-8", errors="replace")
        if PRIMARY_REPO not in cert_text or PRIMARY_REVISION not in cert_text:
            hits.append(
                Hit(
                    path="docs/model-certifications/qwen3.8-27b-axq.md",
                    line_number=1,
                    message="certification record missing pinned repo or revision",
                )
            )

    for path in _public_markdown(root):
        relative = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if HOST_ALIAS_RE.search(line) and not _is_historical_benchmark_path(line):
                hits.append(
                    Hit(
                        path=relative,
                        line_number=line_number,
                        message="internal SSH host alias in public docs",
                    )
                )
        folded = " ".join(text.splitlines())
        if UNEARNED_TIER2_RE.search(folded) and "MTP Tier 2 pending" not in text:
            hits.append(
                Hit(
                    path=relative,
                    line_number=1,
                    message="unearned Qwen 3.8 MTP Tier 2 certified claim",
                )
            )
    return hits


def check_qwen38_primary_claims(root: Path) -> None:
    hits = find_primary_claim_issues(root)
    if hits:
        rendered = "\n".join(f"- {hit.render()}" for hit in hits)
        raise PrimaryClaimError(
            "Qwen 3.8 27B primary-pack contract failed:\n" f"{rendered}"
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
        check_qwen38_primary_claims(args.root)
    except PrimaryClaimError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
