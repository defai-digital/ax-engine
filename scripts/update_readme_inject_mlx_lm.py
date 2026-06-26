#!/usr/bin/env python3
"""Refresh the `mlx_lm` cell in the existing Prefill/Decode/TTFT README tables
from a sweep produced by `bench_llama_cpp_metal_sweep.py --include-mlx-lm`.

Only the raw `mlx_lm` median number is replaced. AX cells (raw values and the
parenthesized percentages) are left as-is, so this updater is intentionally
fail-closed by default: use the full row updater when refreshing public README
performance data. Pass --allow-stale-deltas only for an explicit diagnostic
edit where stale AX percentages are acceptable.

Companion to `update_readme_inject_llama_cpp.py`; same sweep doc shape.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SLUG_TO_README = {
    "gemma-4-e2b-it-4bit":     ("Gemma 4 E2B",      "4-bit"),
    "gemma-4-e2b-it-6bit":     ("Gemma 4 E2B",      "6-bit"),
    "gemma-4-e4b-it-4bit":     ("Gemma 4 E4B",      "4-bit"),
    "gemma-4-26b-a4b-it-4bit": ("Gemma 4 26B A4B",  "4-bit"),
    "gemma-4-31b-it-4bit":     ("Gemma 4 31B",      "4-bit"),
    "qwen3_6-27b-4bit":        ("Qwen 3.6 27B",     "4-bit"),
    "qwen3_6-27b-6bit":        ("Qwen 3.6 27B",     "6-bit"),
    "qwen3_6-35b-a3b-4bit":    ("Qwen 3.6 35B A3B", "4-bit"),
    "qwen3_6-35b-a3b-6bit":    ("Qwen 3.6 35B A3B", "6-bit"),
}

TABLE_TARGETS = [
    ("### Prefill throughput", "prefill"),
    ("### Decode throughput", "decode"),
    ("### Time to first token", "ttft"),
]

MLX_HEADER_CELL = "mlx_lm"

_ROW_PT_RE = re.compile(r"^\s*\|.*?\|.*?\|\s*(128|512|2048)\s*\|")
_HEADER_RE = re.compile(r"^\s*\|\s*Model\s*\|")
_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


def fmt_num(val: float | None) -> str:
    if val is None:
        return "n/a"
    if val >= 1000:
        return f"{val:,.1f}"
    return f"{val:.1f}"


def build_mlx_lookup(
    sweep_doc: dict[str, Any],
) -> dict[tuple[str, str, int], dict[str, float | None]]:
    """Return {(model_name, quant, prompt_tokens): {prefill, decode, ttft}} for
    mlx_lm rows only."""
    out: dict[tuple[str, str, int], dict[str, float | None]] = {}
    for row in sweep_doc.get("rows", []):
        slug = row.get("slug")
        if slug not in SLUG_TO_README:
            continue
        if row.get("status") != "ok":
            continue
        model_name, quant = SLUG_TO_README[slug]
        result_doc = row.get("result_doc")
        if not result_doc:
            continue
        for cell in result_doc.get("results", []):
            if cell.get("engine") != "mlx_lm":
                continue
            pt = int(cell["prompt_tokens"])
            prefill = cell["prefill_tok_s"].get("median")
            decode = cell["decode_tok_s"].get("median")
            ttft_raw = cell.get("ttft_ms")
            ttft = ttft_raw.get("median") if isinstance(ttft_raw, dict) else None
            key = (model_name, quant, pt)
            if key in out:
                raise RuntimeError(
                    "duplicate mlx_lm lookup row for "
                    f"model={model_name!r} quant={quant!r} prompt_tokens={pt}"
                )
            out[key] = {"prefill": prefill, "decode": decode, "ttft": ttft}
    return out


def _split_cells(line: str) -> list[str]:
    parts = line.split("|")
    if len(parts) < 3:
        return parts
    return parts[1:-1]


def _join_cells(cells: list[str]) -> str:
    return "|" + "|".join(cells) + "|"


def _mlx_column_index(header_cells: list[str]) -> int:
    """Locate the mlx_lm column by header text; the README layout has put it
    at different positions over time (after llama.cpp Metal* injection it
    sits at index 4; without it, at index 3)."""
    for i, cell in enumerate(header_cells):
        if cell.strip() == MLX_HEADER_CELL:
            return i
    return -1


def update_table(
    lines: list[str],
    section_header_prefix: str,
    metric_key: str,
    lookup: dict[tuple[str, str, int], dict[str, float | None]],
) -> tuple[list[str], int]:
    out: list[str] = []
    cells_changed = 0
    i = 0
    n = len(lines)
    mlx_col: int = -1
    current_model = ""
    current_quant = ""

    while i < n:
        line = lines[i]
        if not line.startswith(section_header_prefix):
            out.append(line)
            i += 1
            continue

        out.append(line)
        i += 1
        mlx_col = -1
        current_model = ""
        current_quant = ""

        while i < n:
            row = lines[i]
            if row.startswith("### ") or row.startswith("## "):
                break

            if _HEADER_RE.match(row):
                mlx_col = _mlx_column_index(_split_cells(row))
                out.append(row)
                i += 1
                continue

            if _SEPARATOR_RE.match(row):
                out.append(row)
                i += 1
                continue

            pt_match = _ROW_PT_RE.match(row)
            if pt_match and mlx_col >= 0:
                pt = int(pt_match.group(1))
                cells = _split_cells(row)
                if cells[0].strip():
                    current_model = cells[0].strip()
                if cells[1].strip():
                    current_quant = cells[1].strip()
                value = lookup.get((current_model, current_quant, pt))
                if value is not None and value.get(metric_key) is not None:
                    new_cell = f" {fmt_num(value[metric_key])} "
                    if mlx_col < len(cells) and cells[mlx_col] != new_cell:
                        cells[mlx_col] = new_cell
                        cells_changed += 1
                out.append(_join_cells(cells))
                i += 1
                continue

            out.append(row)
            i += 1
    return out, cells_changed


def apply(readme_text: str, sweep_doc: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    lookup = build_mlx_lookup(sweep_doc)
    lines = readme_text.splitlines()
    total = 0
    for header_prefix, metric_key in TABLE_TARGETS:
        lines, changed = update_table(lines, header_prefix, metric_key, lookup)
        total += changed
    return "\n".join(lines) + "\n", {
        "rows_in_lookup": len(lookup),
        "cells_changed": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-stale-deltas",
        action="store_true",
        help=(
            "Allow updating only mlx_lm cells while leaving AX percentage "
            "deltas untouched. This is unsafe for public README refreshes."
        ),
    )
    args = parser.parse_args()

    if not args.allow_stale_deltas:
        print(
            "ERROR: refusing to update only mlx_lm cells because AX percentage "
            "deltas would become stale; use update_readme_from_bench.py with "
            "complete per-row artifacts, or rerun with --allow-stale-deltas "
            "for a deliberate diagnostic edit.",
            file=sys.stderr,
        )
        sys.exit(1)

    sweep_doc = json.loads(args.sweep.read_text())
    text = args.readme.read_text()
    new_text, stats = apply(text, sweep_doc)
    print(f"  rows in lookup: {stats['rows_in_lookup']}")
    print(f"  cells changed:  {stats['cells_changed']}")
    if args.dry_run:
        sys.stdout.write(new_text)
        return
    args.readme.write_text(new_text)
    print(f"  {args.readme} updated.")


if __name__ == "__main__":
    main()
