#!/usr/bin/env python3
"""
Read completed benchmark JSONs and print the README table rows for
prefill, decode, and TTFT sections. Prints all available models
in README order, skipping any whose JSON is not yet present.

Usage:
    python3 scripts/update_readme_from_results.py --results-dir <dir>
"""
import argparse
import json
import sys
from pathlib import Path


def pct(new, ref):
    if ref == 0:
        return "+∞%"
    v = (new - ref) / ref * 100
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def fmt(v, decimals=1):
    if isinstance(v, float):
        s = f"{v:,.{decimals}f}"
    else:
        s = f"{v}"
    return s


def load_result(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_rows(d):
    """Return dict keyed by (engine, prompt_tokens) → result dict."""
    rows = {}
    for r in d.get("results", []):
        key = (r["engine"], r["prompt_tokens"])
        if key in rows:
            engine, prompt_tokens = key
            raise RuntimeError(
                f"duplicate benchmark row for engine={engine!r} prompt_tokens={prompt_tokens}"
            )
        rows[key] = r
    return rows


def get_decode_median(row):
    return row.get("decode_tok_s", {}).get("median")


def get_prefill_median(row):
    return row.get("prefill_tok_s", {}).get("median")


def get_ttft(row):
    return row.get("ttft_ms", {}).get("median") if isinstance(row.get("ttft_ms"), dict) else row.get("ttft_ms")


README_MODELS = [
    # (readme_label, slug, quant_label, mlx_lm_engine)
    ("Gemma 4 E2B",        "gemma-4-e2b-it-4bit",         "4-bit",        "mlx_lm"),
    ("Gemma 4 E2B",        "gemma-4-e2b-it-6bit",         "6-bit",        "mlx_lm"),
    ("Gemma 4 E4B",        "gemma-4-e4b-it-4bit",         "4-bit",        "mlx_lm"),
    ("Gemma 4 26B A4B",    "gemma-4-26b-a4b-it-4bit",     "4-bit",        "mlx_lm"),
    ("Gemma 4 31B",        "gemma-4-31b-it-4bit",         "4-bit",        "mlx_lm"),
    ("Qwen 3.5 9B",        "qwen3_5-9b-mlx-4bit",         "4-bit",        "mlx_lm"),
    ("Qwen 3.6 27B",       "qwen3_6-27b-4bit",            "4-bit",        "mlx_lm"),
    ("Qwen 3.6 27B",       "qwen3_6-27b-6bit",            "6-bit",        "mlx_lm"),
    ("Qwen 3.6 35B A3B",   "qwen3_6-35b-a3b-4bit",        "4-bit",        "mlx_lm"),
    ("Qwen Coder Next",    "qwen3-coder-next-4bit",        "4-bit",        "mlx_lm"),
    ("GLM 4.7 Flash",      "glm-4.7-flash-4bit",           "4-bit",        "mlx_lm"),
]

PROMPT_TOKENS = [128, 512, 2048]


def bold_if_better(val_str, better):
    return f"**{val_str}**" if better else val_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    rdir = Path(args.results_dir)

    print("=" * 90)
    print("PREFILL TABLE")
    print("=" * 90)
    print("| Model | MLX quantization | Prompt tok | mlx_lm | ax engine |")
    print("|---|---|---:|---:|---:|")

    for label, slug, quant, _ in README_MODELS:
        d = load_result(rdir / f"{slug}.json")
        if d is None:
            print(f"| {label} | {quant} | — | (not yet) | |")
            continue
        rows = extract_rows(d)
        first = True
        for pt in PROMPT_TOKENS:
            lm  = rows.get(("mlx_lm", pt))
            ax  = rows.get(("ax_engine_mlx", pt))
            if lm is None:
                continue
            lm_p  = get_prefill_median(lm)
            ax_p  = get_prefill_median(ax)  if ax  else None

            lm_s  = fmt(lm_p)  if lm_p  else "—"
            ax_s  = f"{fmt(ax_p)} ({pct(ax_p, lm_p)})"  if ax_p  and lm_p else "—"
            if ax_p and lm_p and ax_p > lm_p:
                ax_s = f"**{ax_s}**"

            if first:
                print(f"| {label} | {quant} | {pt} | {lm_s} | {ax_s} |")
                first = False
            else:
                print(f"|        |        | {pt} | {lm_s} | {ax_s} |")

    print()
    print("=" * 90)
    print("DECODE TABLE")
    print("=" * 90)
    print("| Model | MLX quantization | Prompt tok | mlx_lm | ax direct baseline |")
    print("|---|---|---:|---:|---:|")

    for label, slug, quant, _ in README_MODELS:
        d = load_result(rdir / f"{slug}.json")
        if d is None:
            print(f"| {label} | {quant} | — | (not yet) | | |")
            continue
        rows = extract_rows(d)
        first = True
        for pt in PROMPT_TOKENS:
            lm    = rows.get(("mlx_lm", pt))
            ax    = rows.get(("ax_engine_mlx", pt))

            if lm is None:
                continue

            lm_d   = get_decode_median(lm)
            ax_d   = get_decode_median(ax)    if ax    else None

            lm_s   = fmt(lm_d)  if lm_d  else "—"
            ax_s   = f"{fmt(ax_d)} ({pct(ax_d, lm_d)})"    if ax_d   and lm_d else "—"

            if ax_d and lm_d:
                ax_s = bold_if_better(ax_s, ax_d > lm_d)

            if first:
                print(f"| {label} | {quant} | {pt} | {lm_s} | {ax_s} |")
                first = False
            else:
                print(f"|        |        | {pt} | {lm_s} | {ax_s} |")

    print()
    print("=" * 90)
    print("TTFT TABLE")
    print("=" * 90)
    print("| Model | MLX quantization | Prompt tok | mlx_lm | ax engine |")
    print("|---|---|---:|---:|---:|")

    for label, slug, quant, _ in README_MODELS:
        d = load_result(rdir / f"{slug}.json")
        if d is None:
            print(f"| {label} | {quant} | — | (not yet) | |")
            continue
        rows = extract_rows(d)
        first = True
        for pt in PROMPT_TOKENS:
            lm  = rows.get(("mlx_lm", pt))
            ax  = rows.get(("ax_engine_mlx", pt))
            if lm is None:
                continue

            lm_p  = get_prefill_median(lm)
            ax_t  = get_ttft(ax) if ax else None

            lm_ttft  = (pt / lm_p * 1000)  if lm_p  else None

            lm_s  = fmt(lm_ttft, 1)  if lm_ttft  else "—"
            ax_s  = f"{fmt(ax_t, 1)} ({pct(ax_t, lm_ttft)})"         if ax_t     and lm_ttft else "—"
            if ax_t and lm_ttft and ax_t < lm_ttft:
                ax_s = f"**{ax_s}**"

            if first:
                print(f"| {label} | {quant} | {pt} | {lm_s} | {ax_s} |")
                first = False
            else:
                print(f"|        |        | {pt} | {lm_s} | {ax_s} |")


if __name__ == "__main__":
    main()
