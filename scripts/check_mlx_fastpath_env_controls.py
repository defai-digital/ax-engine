#!/usr/bin/env python3
"""Fail closed when MLX generation fastpath env controls bypass fastpath.rs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


FASTPATH_OWNER = Path("crates/ax-engine-mlx/src/fastpath.rs")
DEFAULT_SCAN_ROOTS = (Path("crates/ax-engine-mlx/src"),)
FASTPATH_CONTROL_ENVS = frozenset(
    {
        "AX_DISABLE_MLA_PREFIX_RESTORE",
        "AX_DISABLE_TURBOQUANT_FUSED_DECODE",
        "AX_MLX_PACK_DENSE_FFN_GATE_UP",
        "AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS",
        "AX_MLX_PACK_QKV_PROJECTIONS",
        "AX_MLX_PREFILL_FFN_COMPILE_SWIGLU",
    }
)
DIRECT_ENV_CALL_RE = re.compile(
    r"\b(?:std::)?env::var(?:_os)?\(\s*\"(?P<env>AX_[A-Z0-9_]+)\""
)


class MlxFastpathEnvControlError(RuntimeError):
    pass


@dataclass(frozen=True)
class Violation:
    path: Path
    line_number: int
    env_name: str
    line: str

    def render(self) -> str:
        return (
            f"{self.path}:{self.line_number}: direct parse of {self.env_name} "
            f"must use ax-engine-mlx::fastpath: {self.line}"
        )


def _repo_relative(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _is_fastpath_control(env_name: str) -> bool:
    return env_name in FASTPATH_CONTROL_ENVS or env_name.startswith("AX_MLX_PACK_")


def iter_source_files(root: Path, scan_roots: Sequence[Path]) -> Iterable[Path]:
    for scan_root in scan_roots:
        absolute_root = root / scan_root
        if not absolute_root.exists():
            continue
        yield from absolute_root.rglob("*.rs")


def find_direct_parse_violations(
    root: Path,
    scan_roots: Sequence[Path] = DEFAULT_SCAN_ROOTS,
) -> list[Violation]:
    violations: list[Violation] = []
    for path in iter_source_files(root, scan_roots):
        relative = _repo_relative(path, root)
        if relative == FASTPATH_OWNER:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in DIRECT_ENV_CALL_RE.finditer(line):
                env_name = match.group("env")
                if _is_fastpath_control(env_name):
                    violations.append(
                        Violation(
                            path=relative,
                            line_number=line_number,
                            env_name=env_name,
                            line=line.strip(),
                        )
                    )
    return violations


def find_missing_owner_declarations(root: Path) -> list[str]:
    owner = root / FASTPATH_OWNER
    if not owner.exists():
        return sorted(FASTPATH_CONTROL_ENVS)
    text = owner.read_text(encoding="utf-8")
    return sorted(env_name for env_name in FASTPATH_CONTROL_ENVS if env_name not in text)


def check_mlx_fastpath_env_controls(root: Path) -> None:
    missing = find_missing_owner_declarations(root)
    violations = find_direct_parse_violations(root)
    errors: list[str] = []
    if missing:
        errors.append(
            "fastpath.rs must declare the audited MLX fastpath env controls:\n"
            + "\n".join(f"- {env_name}" for env_name in missing)
        )
    if violations:
        rendered = "\n".join(violation.render() for violation in violations)
        errors.append(
            "MLX generation fastpath env controls must not be parsed outside fastpath.rs:\n"
            + rendered
        )
    if errors:
        raise MlxFastpathEnvControlError("\n\n".join(errors))


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
        check_mlx_fastpath_env_controls(args.root)
    except MlxFastpathEnvControlError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("ok: MLX generation fastpath env controls are owned by fastpath.rs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
