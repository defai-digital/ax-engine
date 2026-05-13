#!/usr/bin/env python3
"""Inject a 'llama.cpp Metal*' column into the existing Prefill/Decode/TTFT
README tables and drop the standalone 'External GGUF baseline' section.

Why in-table:
  Users want side-by-side comparison; a separate section forces them to
  cross-reference. By folding llama.cpp into the same row they can eyeball.

Why the asterisk + footnote:
  llama-bench uses its own internal synthetic prompt tokens, so the column
  is NOT prompt-hash parity with mlx_lm / mlx_swift_lm / ax engine columns.
  The disclaimer paragraph above the tables explains this once, and every
  table header carries an asterisk pointing back to it. No percentages are
  shown in the llama.cpp column because the comparison is shape-only.

Idempotent: rerunning replaces the existing llama.cpp column in place rather
than appending a second one.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SLUG_TO_README = {
    "gemma-4-e2b-it-4bit":         ("Gemma 4 E2B",      "4-bit"),
    "gemma-4-e2b-it-5bit":         ("Gemma 4 E2B",      "5-bit"),
    "gemma-4-e2b-it-6bit":         ("Gemma 4 E2B",      "6-bit"),
    "gemma-4-e2b-it-8bit":         ("Gemma 4 E2B",      "8-bit"),
    "gemma-4-e4b-it-4bit":         ("Gemma 4 E4B",      "4-bit"),
    "gemma-4-26b-a4b-it-4bit":     ("Gemma 4 26B A4B",  "4-bit"),
    "gemma-4-31b-it-4bit":         ("Gemma 4 31B",      "4-bit"),
    "qwen3_5-9b-mlx-4bit":         ("Qwen 3.5 9B",      "4-bit"),
    "qwen3_6-35b-a3b-ud-mlx-4bit": ("Qwen 3.6 35B A3B", "UD-MLX 4-bit"),
    "qwen3_6-35b-a3b-5bit":        ("Qwen 3.6 35B A3B", "MLX 5-bit"),
    "qwen3_6-35b-a3b-6bit":        ("Qwen 3.6 35B A3B", "MLX 6-bit"),
    "qwen3_6-35b-a3b-8bit":        ("Qwen 3.6 35B A3B", "MLX 8-bit"),
    "qwen3-coder-next-4bit":       ("Qwen Coder Next",  "4-bit"),
    "glm-4.7-flash-4bit":          ("GLM 4.7 Flash",    "4-bit"),
}

LLAMA_HEADER_CELL = "llama.cpp Metal*"
LLAMA_SEPARATOR_CELL = "---:"

DISCLAIMER_MARK = "<!-- llama-cpp-column-disclaimer -->"
DISCLAIMER_PARAGRAPH = (
    f"{DISCLAIMER_MARK}\n"
    "**`llama.cpp Metal*` column** — Shape-compatible reference produced by "
    "Metal-enabled `llama-bench`. `llama-bench` generates its own internal "
    "synthetic prompt tokens and does not consume the harness prompt JSON, so "
    "these numbers are NOT prompt-hash parity with the other columns. The "
    "intent is rough side-by-side context against a well-known third-party "
    "Metal runtime, not head-to-head comparison. MLX bit-widths are mapped "
    "to the nearest standard GGUF K-quant "
    "(4→Q4_K_M, 5→Q5_K_M, 6→Q6_K, 8→Q8_0; UD-MLX → unsloth UD-Q4_K_XL). No "
    "percentage delta is shown for this column because the prompt is not "
    "shared. Source: `benchmarks/manifests/llama_cpp_metal/inventory.json`, "
    "`scripts/bench_llama_cpp_metal_sweep.py`."
)

STANDALONE_HEADER = "### External GGUF baseline — llama.cpp Metal"

# Tables to inject into: (section header substring, metric key in cell dict).
TABLE_TARGETS = [
    ("### Prefill throughput", "prefill"),
    ("### Decode throughput", "decode"),
    ("### Time to first token", "ttft"),
]


def fmt_num(val: float | None) -> str:
    if val is None:
        return "n/a"
    if val >= 1000:
        return f"{val:,.1f}"
    return f"{val:.1f}"


def build_llama_lookup(sweep_doc: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, float]]:
    """Return {(model_name, quant, prompt_tokens): {prefill, decode, ttft}}."""
    out: dict[tuple[str, str, int], dict[str, float]] = {}
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
            if cell.get("engine") != "llama_cpp_metal":
                continue
            pt = int(cell["prompt_tokens"])
            prefill = cell["prefill_tok_s"].get("median")
            decode = cell["decode_tok_s"].get("median")
            ttft_raw = cell.get("ttft_ms")
            ttft = ttft_raw.get("median") if isinstance(ttft_raw, dict) else None
            out[(model_name, quant, pt)] = {
                "prefill": prefill,
                "decode": decode,
                "ttft": ttft,
            }
    return out


def remove_standalone_section(lines: list[str]) -> tuple[list[str], bool]:
    start = -1
    end = -1
    for i, line in enumerate(lines):
        if line.startswith(STANDALONE_HEADER):
            start = i
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("### ") or lines[j].startswith("## "):
                    end = j
                    break
            if end < 0:
                end = len(lines)
            break
    if start < 0:
        return lines, False
    # Also strip a blank line before the section if present (so we don't leave
    # two consecutive blank lines after deletion).
    trim_start = start
    if trim_start > 0 and lines[trim_start - 1].strip() == "":
        trim_start -= 1
    return lines[:trim_start] + lines[end:], True


def ensure_disclaimer(lines: list[str]) -> tuple[list[str], bool]:
    # Skip insertion if disclaimer already present (idempotent).
    for line in lines:
        if DISCLAIMER_MARK in line:
            return lines, False

    # Insert directly above the first targeted table section.
    for i, line in enumerate(lines):
        if line.startswith(TABLE_TARGETS[0][0]):
            insertion = DISCLAIMER_PARAGRAPH.splitlines() + [""]
            return lines[:i] + insertion + lines[i:], True
    raise RuntimeError("Could not locate '### Prefill throughput' section to anchor disclaimer.")


_ROW_PT_RE = re.compile(r"^\s*\|.*?\|.*?\|\s*(128|512)\s*\|")
_HEADER_RE = re.compile(r"^\s*\|\s*Model\s*\|")
_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


def _split_cells(line: str) -> list[str]:
    # Markdown rows look like "| a | b | c |"; split keeps leading/trailing
    # empty strings we have to discard.
    parts = line.split("|")
    if len(parts) < 3:
        return parts
    return parts[1:-1]


def _join_cells(cells: list[str]) -> str:
    return "|" + "|".join(cells) + "|"


_LLAMA_COL_INDEX = 3  # after Model | MLX quantization | Prompt tok


def _strip_existing_llama_column(cells: list[str]) -> tuple[list[str], bool]:
    """If the cells already contain a llama.cpp column at the canonical
    position, remove it. Returns (new_cells, was_present)."""
    if len(cells) > _LLAMA_COL_INDEX:
        candidate = cells[_LLAMA_COL_INDEX].strip()
        if candidate == LLAMA_HEADER_CELL or candidate == LLAMA_SEPARATOR_CELL:
            return cells[:_LLAMA_COL_INDEX] + cells[_LLAMA_COL_INDEX + 1:], True
    # Also accept the legacy trailing position so previously-appended columns
    # get migrated to the new canonical slot.
    if cells:
        last = cells[-1].strip()
        if last == LLAMA_HEADER_CELL or last == LLAMA_SEPARATOR_CELL:
            return cells[:-1], True
    return cells, False


def inject_column(
    lines: list[str],
    section_header_prefix: str,
    metric_key: str,
    lookup: dict[tuple[str, str, int], dict[str, float]],
) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.startswith(section_header_prefix):
            out.append(line)
            i += 1
            continue

        # Found the section. Emit the section header line then process the table.
        out.append(line)
        i += 1
        current_model = ""
        current_quant = ""
        # Flag set when we see a pre-existing llama column on the header row;
        # ensures we strip the data-row cell at the matching position on subsequent
        # rows even if their content doesn't match our exact marker strings.
        strip_position: int | None = None

        while i < n:
            row = lines[i]
            if row.startswith("### ") or row.startswith("## "):
                break  # next section, table is done

            if _HEADER_RE.match(row):
                cells = _split_cells(row)
                # Detect both canonical-position and legacy-trailing leftovers.
                if len(cells) > _LLAMA_COL_INDEX and cells[_LLAMA_COL_INDEX].strip() == LLAMA_HEADER_CELL:
                    cells = cells[:_LLAMA_COL_INDEX] + cells[_LLAMA_COL_INDEX + 1:]
                    strip_position = _LLAMA_COL_INDEX
                elif cells and cells[-1].strip() == LLAMA_HEADER_CELL:
                    cells = cells[:-1]
                    strip_position = -1
                cells.insert(_LLAMA_COL_INDEX, f" {LLAMA_HEADER_CELL} ")
                out.append(_join_cells(cells))
                i += 1
                continue

            if _SEPARATOR_RE.match(row):
                cells = _split_cells(row)
                if strip_position is not None and len(cells) > abs(strip_position):
                    cells.pop(strip_position)
                cells.insert(_LLAMA_COL_INDEX, f" {LLAMA_SEPARATOR_CELL} ")
                out.append(_join_cells(cells))
                i += 1
                continue

            pt_match = _ROW_PT_RE.match(row)
            if pt_match:
                pt = int(pt_match.group(1))
                cells = _split_cells(row)
                if strip_position is not None and len(cells) > abs(strip_position):
                    cells.pop(strip_position)
                if cells[0].strip():
                    current_model = cells[0].strip()
                if cells[1].strip():
                    current_quant = cells[1].strip()
                value = lookup.get((current_model, current_quant, pt))
                cell_text = fmt_num(value[metric_key] if value else None)
                cells.insert(_LLAMA_COL_INDEX, f" {cell_text} ")
                out.append(_join_cells(cells))
                i += 1
                continue

            out.append(row)
            i += 1
    return out


def apply(readme_text: str, sweep_doc: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return (new_text, stats)."""
    lookup = build_llama_lookup(sweep_doc)
    lines = readme_text.splitlines()

    lines, removed = remove_standalone_section(lines)
    lines, inserted_disclaimer = ensure_disclaimer(lines)
    for header_prefix, metric_key in TABLE_TARGETS:
        lines = inject_column(lines, header_prefix, metric_key, lookup)

    stats = {
        "rows_in_lookup": len(lookup),
        "removed_standalone_section": removed,
        "inserted_disclaimer": inserted_disclaimer,
    }
    return "\n".join(lines) + "\n", stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sweep_doc = json.loads(args.sweep.read_text())
    original = args.readme.read_text()
    updated, stats = apply(original, sweep_doc)

    print(
        f"  rows in lookup: {stats['rows_in_lookup']}\n"
        f"  removed standalone section: {stats['removed_standalone_section']}\n"
        f"  inserted disclaimer: {stats['inserted_disclaimer']}\n"
        f"  byte delta: {len(updated) - len(original):+d}"
    )

    if args.dry_run:
        sys.stdout.write(updated)
        return

    if updated == original:
        print("  README.md unchanged.")
        return
    args.readme.write_text(updated)
    print("  README.md updated.")


if __name__ == "__main__":
    main()
