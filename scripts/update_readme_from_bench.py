#!/usr/bin/env python3
"""Update README.md performance table rows from a benchmark JSON file.

Usage:
    python3 scripts/update_readme_from_bench.py --slug gemma-4-e2b-it-4bit \
        --json benchmarks/results/mlx-inference/.../gemma-4-e2b-it-4bit.json
"""
import argparse
import json
import sys


class ReadmeBenchUpdateError(RuntimeError):
    pass


INVALID_METRIC_CELL = "—"


# Maps benchmark slug → (README model display name, README quant string)
SLUG_TO_README = {
    "gemma-4-e2b-it-4bit":         ("Gemma 4 E2B",      "4-bit"),
    "gemma-4-e2b-it-6bit":         ("Gemma 4 E2B",      "6-bit"),
    "gemma-4-e4b-it-4bit":         ("Gemma 4 E4B",      "4-bit"),
    "gemma-4-26b-a4b-it-4bit":     ("Gemma 4 26B A4B",  "4-bit"),
    "gemma-4-26b-a4b-it-6bit":     ("Gemma 4 26B A4B",  "6-bit"),
    "gemma-4-31b-it-4bit":         ("Gemma 4 31B",      "4-bit"),
    "gemma-4-31b-it-6bit":         ("Gemma 4 31B",      "6-bit"),
    "qwen3_5-9b-mlx-4bit":         ("Qwen 3.5 9B",      "4-bit"),
    "qwen3_6-27b-4bit":            ("Qwen 3.6 27B",     "4-bit"),
    "qwen3_6-27b-6bit":            ("Qwen 3.6 27B",     "6-bit"),
    "qwen3_6-35b-a3b-4bit":        ("Qwen 3.6 35B A3B", "4-bit"),
    "qwen3_6-35b-a3b-6bit":        ("Qwen 3.6 35B A3B", "6-bit"),
    "qwen3-coder-next-4bit":       ("Qwen Coder Next",  "4-bit"),
    "glm-4.7-flash-4bit":          ("GLM 4.7 Flash",    "4-bit"),
}


def fmt_num(val: float) -> str:
    if val >= 1000:
        return f"{val:,.1f}"
    return f"{val:.1f}"


def pct_delta(val: float, ref: float) -> float:
    return (val - ref) / ref * 100.0


def fmt_pct(val: float, ref: float) -> str:
    pct = pct_delta(val, ref)
    sign = "+" if pct >= 0 else ""
    return f"({sign}{pct:.1f}%)"


def cell_prefill(ax: float, ref: float) -> str:
    """Bold when ax > ref (positive % = better than mlx_lm)."""
    pct = pct_delta(ax, ref)
    s = f"{fmt_num(ax)} {fmt_pct(ax, ref)}"
    return f"**{s}**" if pct > 0 else s


def cell_decode_direct(ax: float, ref: float) -> str:
    """Bold when ax > ref (positive % = better than mlx_lm)."""
    pct = pct_delta(ax, ref)
    s = f"{fmt_num(ax)} {fmt_pct(ax, ref)}"
    return f"**{s}**" if pct > 0 else s


def cell_decode_ngram(ax: float, ref: float) -> str:
    """Bold when n-gram decode is faster than mlx_lm."""
    pct = pct_delta(ax, ref)
    s = f"{fmt_num(ax)} {fmt_pct(ax, ref)}"
    return f"**{s}**" if pct > 0 else s


def cell_ttft(ax: float, ref: float) -> str:
    """Lower is better — bold when ax < ref (negative %)."""
    pct = pct_delta(ax, ref)
    s = f"{fmt_num(ax)} {fmt_pct(ax, ref)}"
    return f"**{s}**" if pct < 0 else s


def replace_trailing_cells(line: str, new_values: list[str]) -> str:
    """Replace the last len(new_values) data cells in a markdown table row.
    Preserves the original content and whitespace in all other cells.
    """
    # Split on | — first and last parts are empty for a well-formed table row
    parts = line.split("|")
    content = parts[1:-1]  # the actual cells (may be " text " or "        ")
    n = len(new_values)
    for i, val in enumerate(new_values):
        content[-(n - i)] = f" {val} "
    return "|" + "|".join(content) + "|"


def _median_or_none(raw):
    """Return the median field from a summary dict, or None if the dict /
    median itself is None. Bench harness emits `null` medians when every
    trial for a metric was invalid (e.g. all-cache-warm prefill rows under
    the recent shared-prefix-cache change). Callers must treat None as
    "no fresh data, leave existing README cell alone."
    """
    if not isinstance(raw, dict):
        return raw
    return raw.get("median")


def extract_bench_values(data: dict) -> dict:
    """Extract metric medians from benchmark JSON, keyed by (engine, prompt_tokens)."""
    out = {}
    for r in data["results"]:
        engine = r["engine"]
        pt = r["prompt_tokens"]
        key = (engine, pt)
        if key in out:
            raise ReadmeBenchUpdateError(
                f"duplicate benchmark row for engine={engine!r} prompt_tokens={pt}"
            )
        out[key] = {
            "prefill": _median_or_none(r.get("prefill_tok_s")),
            "decode":  _median_or_none(r.get("decode_tok_s")),
            "ttft":    _median_or_none(r.get("ttft_ms")),
        }
    return out


def find_anchor_line(lines: list[str], model_name: str, quant: str, section_header: str) -> int:
    """Find the pt=128 table row for (model_name, quant) within the named section."""
    in_target = False
    for i, line in enumerate(lines):
        if line.startswith("#") and section_header in line:
            in_target = True
        elif line.startswith("#") and in_target:
            break
        if not in_target:
            continue
        if model_name in line and quant in line and "| 128 |" in line:
            return i
    return -1


def _split_cells(line: str) -> list[str]:
    parts = line.split("|")
    if len(parts) < 3:
        return []
    return [part.strip() for part in parts[1:-1]]


def _join_cells(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table_column_count(lines: list[str], section_header: str) -> int:
    in_target = False
    for line in lines:
        if line.startswith("#") and section_header in line:
            in_target = True
            continue
        if line.startswith("#") and in_target:
            break
        if in_target and line.startswith("| Model |"):
            return len(_split_cells(line))
    return 0


def _table_header_cells(lines: list[str], section_header: str) -> list[str]:
    in_target = False
    for line in lines:
        if line.startswith("#") and section_header in line:
            in_target = True
            continue
        if line.startswith("#") and in_target:
            break
        if in_target and line.startswith("| Model |"):
            return _split_cells(line)
    return []


def find_insert_line(lines: list[str], section_header: str, model_name: str) -> int:
    """Find where to insert a missing model block in a README performance table."""
    in_target = False
    fallback = -1
    for i, line in enumerate(lines):
        if line.startswith("#") and section_header in line:
            in_target = True
            continue
        if line.startswith("#") and in_target:
            return fallback if fallback >= 0 else i
        if not in_target:
            continue
        if model_name == "Qwen 3.6 27B" and "Qwen 3.6 35B A3B" in line and "| 128 |" in line:
            return i
        if line.startswith("|") and "| 128 |" in line:
            fallback = i + len(collect_prompt_rows(lines, i))
    return fallback if fallback >= 0 else len(lines)


def ensure_model_rows(lines: list[str], model_name: str, quant: str, section_header: str) -> int:
    anchor = find_anchor_line(lines, model_name, quant, section_header)
    if anchor >= 0:
        return anchor

    column_count = _table_column_count(lines, section_header)
    if column_count <= 0:
        return -1

    metric_count = column_count - 3
    insert_at = find_insert_line(lines, section_header, model_name)
    new_rows = []
    for idx, pt in enumerate((128, 512, 2048)):
        if idx == 0:
            cells = [model_name, quant, str(pt), *(["—"] * metric_count)]
        else:
            cells = ["", "", str(pt), *(["—"] * metric_count)]
        new_rows.append(_join_cells(cells))
    lines[insert_at:insert_at] = new_rows
    return insert_at


def find_next_512_line(lines: list[str], after: int) -> int:
    """Find the pt=512 continuation row immediately after index `after`."""
    for i in range(after + 1, min(after + 4, len(lines))):
        if "| 512 |" in lines[i]:
            return i
    return -1


def find_next_2048_line(lines: list[str], after: int) -> int:
    """Find the pt=2048 continuation row immediately after the pt=512 row.
    Returns -1 if no 2048 row exists (older README without the long-context row)."""
    for i in range(after + 1, min(after + 4, len(lines))):
        if "| 2048 |" in lines[i]:
            return i
    return -1


def collect_prompt_rows(lines: list[str], anchor: int) -> list[tuple[int, int]]:
    """Return ordered list of (prompt_tokens, line_index) for this model.
    The pt=128 row is the anchor; pt=512 and optional pt=2048 follow.
    """
    rows: list[tuple[int, int]] = [(128, anchor)]
    idx_512 = find_next_512_line(lines, anchor)
    if idx_512 >= 0:
        rows.append((512, idx_512))
        idx_2048 = find_next_2048_line(lines, idx_512)
        if idx_2048 >= 0:
            rows.append((2048, idx_2048))
    return rows


def update_prefill_rows(lines: list[str], model_name: str, quant: str, vals: dict) -> int:
    changed = 0
    anchor = ensure_model_rows(lines, model_name, quant, "Prefill throughput")
    if anchor < 0:
        print(f"  WARN: prefill pt=128 row not found for {model_name!r} {quant!r}")
        return 0

    for pt, idx in collect_prompt_rows(lines, anchor):
        ref_row = vals.get(("mlx_lm", pt))
        if ref_row is None:
            continue
        ref = ref_row.get("prefill")
        ax  = vals.get(("ax_engine_mlx", pt), {}).get("prefill")
        if ref is None:
            lines[idx] = replace_trailing_cells(
                lines[idx],
                [INVALID_METRIC_CELL, INVALID_METRIC_CELL],
            )
            changed += 1
            continue
        ax_cell = INVALID_METRIC_CELL if ax is None else cell_prefill(ax, ref)
        lines[idx] = replace_trailing_cells(lines[idx], [fmt_num(ref), ax_cell])
        changed += 1

    return changed


def update_decode_rows(lines: list[str], model_name: str, quant: str, vals: dict) -> int:
    changed = 0
    anchor = ensure_model_rows(lines, model_name, quant, "Decode throughput")
    if anchor < 0:
        print(f"  WARN: decode pt=128 row not found for {model_name!r} {quant!r}")
        return 0

    headers = _table_header_cells(lines, "Decode throughput")
    try:
        mlx_col = headers.index("mlx_lm")
        direct_col = headers.index("ax direct baseline")
    except ValueError:
        print("  WARN: decode table does not contain mlx_lm and ax direct baseline columns")
        return 0
    ngram_col = headers.index("ax default n-gram") if "ax default n-gram" in headers else None

    for pt, idx in collect_prompt_rows(lines, anchor):
        ref_row = vals.get(("mlx_lm", pt))
        if ref_row is None:
            continue
        ref       = ref_row.get("decode")
        ax_direct = vals.get(("ax_engine_mlx", pt), {}).get("decode")
        ax_ngram  = vals.get(("ax_engine_mlx_ngram_accel", pt), {}).get("decode")
        cells = _split_cells(lines[idx])
        if len(cells) <= max(mlx_col, direct_col):
            continue
        if ref is None:
            cells[mlx_col] = INVALID_METRIC_CELL
            cells[direct_col] = INVALID_METRIC_CELL
            if ngram_col is not None and len(cells) > ngram_col:
                cells[ngram_col] = INVALID_METRIC_CELL
            lines[idx] = _join_cells(cells)
            changed += 1
            continue
        cells[mlx_col] = fmt_num(ref)
        cells[direct_col] = (
            INVALID_METRIC_CELL
            if ax_direct is None
            else cell_decode_direct(ax_direct, ref)
        )
        if ax_ngram is not None and ngram_col is not None and len(cells) > ngram_col:
            cells[ngram_col] = cell_decode_ngram(ax_ngram, ref)
        lines[idx] = _join_cells(cells)
        changed += 1

    return changed


def update_ttft_rows(lines: list[str], model_name: str, quant: str, vals: dict) -> int:
    changed = 0
    anchor = ensure_model_rows(lines, model_name, quant, "Time to first token")
    if anchor < 0:
        print(f"  WARN: ttft pt=128 row not found for {model_name!r} {quant!r}")
        return 0

    for pt, idx in collect_prompt_rows(lines, anchor):
        ref_row = vals.get(("mlx_lm", pt))
        if ref_row is None:
            continue
        ref_ttft = ref_row.get("ttft")
        ax_ttft  = vals.get(("ax_engine_mlx", pt), {}).get("ttft")
        if ref_ttft is None:
            lines[idx] = replace_trailing_cells(
                lines[idx],
                [INVALID_METRIC_CELL, INVALID_METRIC_CELL],
            )
            changed += 1
            continue
        ax_cell = INVALID_METRIC_CELL if ax_ttft is None else cell_ttft(ax_ttft, ref_ttft)
        lines[idx] = replace_trailing_cells(lines[idx], [fmt_num(ref_ttft), ax_cell])
        changed += 1

    return changed


def main():
    parser = argparse.ArgumentParser(description="Update README bench rows from JSON")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.slug not in SLUG_TO_README:
        print(f"ERROR: unknown slug {args.slug!r}")
        sys.exit(1)

    model_name, quant = SLUG_TO_README[args.slug]

    with open(args.json) as f:
        data = json.load(f)
    vals = extract_bench_values(data)

    with open(args.readme) as f:
        content = f.read()
    lines = content.splitlines()

    orig_lines = lines[:]
    total = 0
    total += update_prefill_rows(lines, model_name, quant, vals)
    total += update_decode_rows(lines, model_name, quant, vals)
    total += update_ttft_rows(lines, model_name, quant, vals)

    print(f"  {args.slug}: {total} row cells updated")
    if total == 0:
        print(
            f"ERROR: {args.json} did not update any README performance rows for {args.slug}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dry_run:
        for i, (orig, new) in enumerate(zip(orig_lines, lines)):
            if orig != new:
                print(f"  line {i+1}:")
                print(f"    OLD: {orig}")
                print(f"    NEW: {new}")
        return

    with open(args.readme, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  README.md updated")


if __name__ == "__main__":
    main()
