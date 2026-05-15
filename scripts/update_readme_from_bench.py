#!/usr/bin/env python3
"""Update README.md performance table rows from a benchmark JSON file.

Usage:
    python3 scripts/update_readme_from_bench.py --slug gemma-4-e2b-it-4bit \
        --json benchmarks/results/mlx-inference/.../gemma-4-e2b-it-4bit.json
"""
import argparse
import json
import sys

# Maps benchmark slug → (README model display name, README quant string)
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
    """Always bold."""
    return f"**{fmt_num(ax)} {fmt_pct(ax, ref)}**"


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


def extract_bench_values(data: dict) -> dict:
    """Extract metric medians from benchmark JSON, keyed by (engine, prompt_tokens)."""
    out = {}
    for r in data["results"]:
        engine = r["engine"]
        pt = r["prompt_tokens"]
        ttft_raw = r.get("ttft_ms")
        ttft = ttft_raw.get("median") if isinstance(ttft_raw, dict) else ttft_raw
        out[(engine, pt)] = {
            "prefill": r["prefill_tok_s"]["median"],
            "decode":  r["decode_tok_s"]["median"],
            "ttft":    ttft,
        }
    return out


def find_anchor_line(lines: list[str], model_name: str, quant: str, section_header: str) -> int:
    """Find the pt=128 table row for (model_name, quant) within the named section."""
    in_target = False
    for i, line in enumerate(lines):
        if line.startswith("### ") and section_header in line:
            in_target = True
        elif line.startswith("### ") and in_target:
            break
        if not in_target:
            continue
        if model_name in line and quant in line and "| 128 |" in line:
            return i
    return -1


def find_next_512_line(lines: list[str], after: int) -> int:
    """Find the pt=512 continuation row immediately after index `after`."""
    for i in range(after + 1, min(after + 4, len(lines))):
        if "| 512 |" in lines[i]:
            return i
    return -1


def find_next_4096_line(lines: list[str], after: int) -> int:
    """Find the pt=4096 continuation row immediately after the pt=512 row.
    Returns -1 if no 4096 row exists (older README without the long-context row)."""
    for i in range(after + 1, min(after + 4, len(lines))):
        if "| 4096 |" in lines[i]:
            return i
    return -1


def collect_prompt_rows(lines: list[str], anchor: int) -> list[tuple[int, int]]:
    """Return ordered list of (prompt_tokens, line_index) for this model.
    The pt=128 row is the anchor; pt=512 and optional pt=4096 follow.
    """
    rows: list[tuple[int, int]] = [(128, anchor)]
    idx_512 = find_next_512_line(lines, anchor)
    if idx_512 >= 0:
        rows.append((512, idx_512))
        idx_4096 = find_next_4096_line(lines, idx_512)
        if idx_4096 >= 0:
            rows.append((4096, idx_4096))
    return rows


def update_prefill_rows(lines: list[str], model_name: str, quant: str, vals: dict) -> int:
    changed = 0
    anchor = find_anchor_line(lines, model_name, quant, "Prefill throughput")
    if anchor < 0:
        print(f"  WARN: prefill pt=128 row not found for {model_name!r} {quant!r}")
        return 0

    for pt, idx in collect_prompt_rows(lines, anchor):
        ref = vals.get(("mlx_lm", pt), {}).get("prefill")
        ax  = vals.get(("ax_engine_mlx", pt), {}).get("prefill")
        if ref is None or ax is None:
            continue
        lines[idx] = replace_trailing_cells(lines[idx], [cell_prefill(ax, ref)])
        changed += 1

    return changed


def update_decode_rows(lines: list[str], model_name: str, quant: str, vals: dict) -> int:
    changed = 0
    anchor = find_anchor_line(lines, model_name, quant, "Decode throughput")
    if anchor < 0:
        print(f"  WARN: decode pt=128 row not found for {model_name!r} {quant!r}")
        return 0

    for pt, idx in collect_prompt_rows(lines, anchor):
        ref       = vals.get(("mlx_lm", pt), {}).get("decode")
        ax_direct = vals.get(("ax_engine_mlx", pt), {}).get("decode")
        ax_ngram  = vals.get(("ax_engine_mlx_ngram_accel", pt), {}).get("decode")
        if ref is None or ax_direct is None:
            continue
        new_cells = [cell_decode_direct(ax_direct, ref)]
        if ax_ngram is not None:
            new_cells.append(cell_decode_ngram(ax_ngram, ref))
        lines[idx] = replace_trailing_cells(lines[idx], new_cells)
        changed += 1

    return changed


def update_ttft_rows(lines: list[str], model_name: str, quant: str, vals: dict) -> int:
    changed = 0
    anchor = find_anchor_line(lines, model_name, quant, "Time to first token")
    if anchor < 0:
        print(f"  WARN: ttft pt=128 row not found for {model_name!r} {quant!r}")
        return 0

    for pt, idx in collect_prompt_rows(lines, anchor):
        ref_ttft = vals.get(("mlx_lm", pt), {}).get("ttft")
        ax_ttft  = vals.get(("ax_engine_mlx", pt), {}).get("ttft")
        if ref_ttft is None or ax_ttft is None:
            continue
        lines[idx] = replace_trailing_cells(lines[idx], [cell_ttft(ax_ttft, ref_ttft)])
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
