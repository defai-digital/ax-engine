#!/usr/bin/env python3
"""Pin the inventory of MLX eval call sites in ax-engine-mlx production code.

Every ``eval`` / ``async_eval`` / ``eval_first_u32`` call in a request path
panics the model worker on MLX failure until the fallible-step boundary
(P0-C, ``.internal/reports/reference-projects-code-audit-2026-07-26.v2.zh-TW.md``)
lands. This guard keeps the inventory reviewed: adding or removing a site
requires regenerating the baseline with ``--update``, which makes the change
visible in review instead of silently growing the panic surface.

Counting rules (kept deliberately simple and deterministic):
- scope: ``crates/ax-engine-mlx/src/**/*.rs`` excluding ``bin/`` and
  ``decode_trace_main.rs`` (probe binaries are not serving paths);
- lines inside ``#[cfg(test)] mod`` blocks are excluded via brace tracking;
- ``//`` comment lines and string-free doc lines are excluded;
- matches ``\\b(async_eval|try_eval|eval_first_u32|eval)\\s*(\\(|::<)``.
  ``try_eval`` is counted separately: converting a bare ``eval`` site to
  ``try_eval`` shows up as an intentional baseline diff, not a silent pass.

Usage:
  python3 scripts/check_mlx_eval_sites.py            # verify against baseline
  python3 scripts/check_mlx_eval_sites.py --update   # regenerate baseline
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCAN_ROOT = ROOT / "crates" / "ax-engine-mlx" / "src"
BASELINE_PATH = ROOT / "scripts" / "mlx_eval_site_baseline.json"

EVAL_RE = re.compile(r"\b(async_eval|try_eval|eval_first_u32|eval)\s*(?:\(|::<)")
CFG_TEST_RE = re.compile(r"^\s*#\[cfg\(test\)\]\s*$")
MOD_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+")


def strip_line_comment(line: str) -> str:
    """Drop a trailing // comment, ignoring // inside string literals."""
    in_string = False
    escaped = False
    for idx, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string and ch == "/" and line[idx : idx + 2] == "//":
            return line[:idx]
    return line


def production_lines(text: str):
    """Yield code lines outside #[cfg(test)] mod blocks."""
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if CFG_TEST_RE.match(line):
            # Find the mod declaration (attributes may stack), then skip
            # the whole brace-balanced block.
            probe = idx + 1
            while probe < len(lines) and lines[probe].lstrip().startswith("#["):
                probe += 1
            if probe < len(lines) and MOD_RE.match(lines[probe]):
                depth = 0
                entered = False
                while probe < len(lines):
                    stripped = strip_line_comment(lines[probe])
                    depth += stripped.count("{") - stripped.count("}")
                    if "{" in stripped:
                        entered = True
                    if entered and depth <= 0:
                        break
                    probe += 1
                idx = probe + 1
                continue
        yield line
        idx += 1


def count_file(path: pathlib.Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in production_lines(path.read_text()):
        code = strip_line_comment(line)
        for match in EVAL_RE.finditer(code):
            kind = match.group(1)
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def scan() -> dict[str, dict[str, int]]:
    inventory: dict[str, dict[str, int]] = {}
    for path in sorted(SCAN_ROOT.rglob("*.rs")):
        rel = path.relative_to(ROOT).as_posix()
        if "/bin/" in rel or rel.endswith("decode_trace_main.rs"):
            continue
        counts = count_file(path)
        if counts:
            inventory[rel] = dict(sorted(counts.items()))
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="regenerate the baseline from the current tree",
    )
    args = parser.parse_args()

    inventory = scan()

    if args.update:
        BASELINE_PATH.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
        total = sum(sum(v.values()) for v in inventory.values())
        print(f"baseline updated: {len(inventory)} files, {total} eval sites")
        return 0

    if not BASELINE_PATH.exists():
        print(
            "missing baseline; run: python3 scripts/check_mlx_eval_sites.py --update",
            file=sys.stderr,
        )
        return 1

    baseline = json.loads(BASELINE_PATH.read_text())
    if inventory == baseline:
        total = sum(sum(v.values()) for v in inventory.values())
        print(f"eval-site inventory matches baseline ({total} sites)")
        return 0

    print("MLX eval-site inventory drifted from baseline:", file=sys.stderr)
    for rel in sorted(set(baseline) | set(inventory)):
        before = baseline.get(rel, {})
        after = inventory.get(rel, {})
        if before != after:
            print(f"  {rel}: {before} -> {after}", file=sys.stderr)
    print(
        "\nEvery bare `eval` in a request path panics the model worker on MLX "
        "failure (see P0-C in the 2026-07-26 v2 reference-projects report). "
        "Review new sites for the fallible-step boundary, then regenerate:\n"
        "  python3 scripts/check_mlx_eval_sites.py --update",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
