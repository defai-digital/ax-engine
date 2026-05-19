#!/usr/bin/env python3
"""Fail closed if no-go direct-MLX FFN probe wrappers enter production code."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


BANNED_SYMBOLS = (
    "gelu_approx_mul_matmul",
    "gelu_approx_quantized_ffn",
)
BANNED_RE = re.compile(r"\b(" + "|".join(re.escape(symbol) for symbol in BANNED_SYMBOLS) + r")\b")
DEFAULT_SCAN_ROOTS = (Path("crates"),)
ALLOWED_PATHS = {
    Path("crates/mlx-sys/src/ops.rs"),
    Path("crates/mlx-sys/src/bin/direct-mlx-hotpath-probe.rs"),
}
ALLOWED_PREFIXES = (
    Path("crates/mlx-sys/native"),
)


class DirectMlxNoProductionRouteError(RuntimeError):
    pass


@dataclass(frozen=True)
class Violation:
    path: Path
    line_number: int
    symbol: str
    line: str

    def render(self) -> str:
        return f"{self.path}:{self.line_number}: banned no-go direct-MLX symbol {self.symbol}: {self.line}"


def _repo_relative(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _is_allowed(path: Path) -> bool:
    if path in ALLOWED_PATHS:
        return True
    return any(path == prefix or prefix in path.parents for prefix in ALLOWED_PREFIXES)


def iter_source_files(root: Path, scan_roots: Sequence[Path]) -> Iterable[Path]:
    for scan_root in scan_roots:
        absolute_root = root / scan_root
        if not absolute_root.exists():
            continue
        for path in absolute_root.rglob("*.rs"):
            relative = _repo_relative(path, root)
            if _is_allowed(relative):
                continue
            yield path


def find_violations(root: Path, scan_roots: Sequence[Path] = DEFAULT_SCAN_ROOTS) -> list[Violation]:
    violations: list[Violation] = []
    for path in iter_source_files(root, scan_roots):
        relative = _repo_relative(path, root)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in BANNED_RE.finditer(line):
                violations.append(
                    Violation(
                        path=relative,
                        line_number=line_number,
                        symbol=match.group(1),
                        line=line.strip(),
                    )
                )
    return violations


def check_no_production_route(root: Path) -> None:
    violations = find_violations(root)
    if violations:
        rendered = "\n".join(violation.render() for violation in violations)
        raise DirectMlxNoProductionRouteError(
            "no-go direct-MLX FFN probe wrappers must not be used outside mlx-sys probe surfaces:\n"
            + rendered
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
        check_no_production_route(args.root)
    except DirectMlxNoProductionRouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("ok: no no-go direct-MLX FFN production route found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
