#!/usr/bin/env python3
"""Update README.md embedding table rows from benchmark JSON output.

Usage:
    python3 scripts/update_readme_embedding.py \
        --results-dir benchmarks/results/embedding/2026-05-12-readme-refresh-v2
"""
import argparse
import json
import sys
from pathlib import Path

# (model_label in JSON, README display name, use_approx_symbol)
EMBEDDING_MODELS = [
    ("qwen3-embedding-0.6b-8bit", "Qwen3-Embedding 0.6B 8-bit", True),
    ("qwen3-embedding-4b-4bit",   "Qwen3-Embedding 4B 4-bit",   False),
    ("qwen3-embedding-8b-4bit-dwq", "Qwen3-Embedding 8B 4-bit DWQ", False),
]

SECTION_HEADER = "Embedding throughput (tok/s)"
SINGLE_ROW_TAG = "single sentence"
BATCHED_ROW_TAG = "10-sentence batch"


def fmt_num(val: float) -> str:
    if val >= 1000:
        return f"{val:,.1f}"
    return f"{val:.1f}"


def pct_delta(val: float, ref: float) -> float:
    return (val - ref) / ref * 100.0


def fmt_pct(val: float, ref: float, approx: bool = False) -> str:
    pct = pct_delta(val, ref)
    sign = "+" if pct >= 0 else ""
    tilde = "≈" if approx else ""
    return f"({tilde}{sign}{pct:.1f}%)"


def replace_trailing_cells(line: str, new_values: list) -> str:
    parts = line.split("|")
    content = parts[1:-1]
    n = len(new_values)
    for i, val in enumerate(new_values):
        content[-(n - i)] = f" {val} "
    return "|" + "|".join(content) + "|"


def load_model_results(results_dir: Path, model_label: str) -> dict | None:
    """Find the most recent result dir for a model label and load its JSON."""
    candidates = sorted(
        c for c in results_dir.glob(f"*{model_label}*") if c.is_dir()
    )
    if not candidates:
        return None
    json_path = candidates[-1] / "embedding_bench.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def find_embedding_row(lines: list, display_name: str, section_header: str = SECTION_HEADER) -> int:
    """Find the line index of the embedding table row for display_name in named section."""
    in_section = False
    for i, line in enumerate(lines):
        if section_header in line:
            in_section = True
            continue
        if in_section and line.startswith("### ") and section_header not in line:
            in_section = False
            continue
        if not in_section:
            continue
        if display_name in line and "|" in line:
            return i
    return -1


def find_batched_row_after(lines: list, anchor_idx: int) -> int:
    """Given the index of the 'single sentence' row, find the immediately
    following row that has the BATCHED_ROW_TAG. Returns -1 if not found.
    """
    for i in range(anchor_idx + 1, min(anchor_idx + 5, len(lines))):
        if BATCHED_ROW_TAG in lines[i] and "|" in lines[i]:
            return i
    return -1


def update_source_line(lines: list, new_dir: str) -> bool:
    """Update the Source: reference line in the embedding section."""
    in_section = False
    for i, line in enumerate(lines):
        if SECTION_HEADER in line:
            in_section = True
        if not in_section:
            continue
        if "Source:" in line and "benchmarks/results/embedding/" in line:
            # Replace the path inside backticks
            import re
            new_line = re.sub(
                r"`benchmarks/results/embedding/[^`]+`",
                f"`{new_dir}`",
                line,
            )
            if new_line != line:
                lines[i] = new_line
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing timestamped embedding result subdirs")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    with open(args.readme) as f:
        content = f.read()
    lines = content.splitlines()
    orig_lines = lines[:]

    total = 0

    # Update source reference
    rel_dir = str(results_dir).replace(str(Path(args.readme).parent) + "/", "")
    if not rel_dir.endswith("/"):
        rel_dir += "/"
    if update_source_line(lines, rel_dir):
        total += 1

    def update_row(idx: int, ref_tok: float, ax_tok: float, approx: bool) -> None:
        """Replace the last 3 cells (mlx-lm, ax-engine-py, AX vs mlx-lm)."""
        nonlocal total
        mlx_cell = fmt_num(ref_tok)
        ax_cell = fmt_num(ax_tok)
        pct = pct_delta(ax_tok, ref_tok)
        tilde = "≈" if approx else ""
        sign = "+" if pct >= 0 else ""
        delta_cell = f"{tilde}{sign}{pct:.1f}%"
        lines[idx] = replace_trailing_cells(
            lines[idx], [mlx_cell, ax_cell, delta_cell]
        )
        total += 1

    for model_label, display_name, use_approx in EMBEDDING_MODELS:
        data = load_model_results(results_dir, model_label)
        if data is None:
            print(f"  WARN: no result found for {model_label!r} in {results_dir}")
            continue

        single = data.get("results", {})
        batched = data.get("results_batched", {})

        # Single-sentence row
        single_mlx = single.get("mlx_lm", {}).get("median_tokens_per_sec")
        single_ax  = single.get("ax_engine_py", {}).get("median_tokens_per_sec")

        anchor_idx = find_embedding_row(lines, display_name, SECTION_HEADER)
        if anchor_idx < 0:
            print(f"  WARN: row not found for {display_name!r}")
            continue
        if SINGLE_ROW_TAG not in lines[anchor_idx]:
            print(f"  WARN: {display_name!r} anchor row missing {SINGLE_ROW_TAG!r}")
            continue
        if single_mlx is not None and single_ax is not None:
            update_row(anchor_idx, single_mlx, single_ax, use_approx)
            print(f"  {model_label} single:  mlx_lm={single_mlx:.1f}  ax={single_ax:.1f}")

        # Batched row (immediately follows single row)
        batched_idx = find_batched_row_after(lines, anchor_idx)
        if batched_idx < 0:
            print(f"  WARN: batched row not found below {display_name!r}")
            continue
        batched_mlx = batched.get("mlx_lm", {}).get("median_tokens_per_sec")
        batched_ax  = batched.get("ax_engine_py", {}).get("median_tokens_per_sec")
        if batched_mlx is not None and batched_ax is not None:
            update_row(batched_idx, batched_mlx, batched_ax, use_approx)
            print(f"  {model_label} batched: mlx_lm={batched_mlx:.1f}  ax={batched_ax:.1f}")

    print(f"  embedding: {total} cells updated")

    if args.dry_run:
        for i, (orig, new) in enumerate(zip(orig_lines, lines)):
            if orig != new:
                print(f"  line {i+1}:")
                print(f"    OLD: {orig}")
                print(f"    NEW: {new}")
        return

    with open(args.readme, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  README.md updated")


if __name__ == "__main__":
    main()
