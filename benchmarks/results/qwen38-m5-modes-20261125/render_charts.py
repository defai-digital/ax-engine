#!/usr/bin/env python3
"""Render the public M5 greedy mode tables and charts from the serving receipt."""

import argparse
from html import escape
import json
import math
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parent
ROOT = CAMPAIGN.parents[2]
BEGIN = "<!-- qwen38-m5-modes:begin -->"
END = "<!-- qwen38-m5-modes:end -->"
MODEL_NAMES = {"dense": "Qwen 3.8 27B", "flash": "Qwen 3.8 Flash Next"}
METRICS = (
    ("decode_tok_s", "Decode · output tok/s", 10),
    ("prefill_tok_s", "Effective prefill · input tok/s", 100),
)
ENGINE_NAMES = {
    "ax-engine": "AX Engine",
    "splash": "Splash",
    "omlx": "oMLX",
    "mlx-lm": "mlx-lm",
    "mtplx": "MTPLX",
    "tensorfold": "TensorFold",
    "ds4": "ds4",
}
MODE_NAMES = {"direct": "Direct greedy", "dflash2": "DFlash2"}


def label(row, versions):
    key = row["engine"]
    return f"{ENGINE_NAMES[key]} {versions[key]} ({MODE_NAMES[row['mode']]})"


def figure_name(model):
    return f"perf-qwen38-{'27b' if model == 'dense' else 'flash-next'}-m5-modes.svg"


def rate(value):
    return f"{value:.2f}" if value else "0"


def render(model, rows, versions):
    bottom = 159 + 31 * len(rows)
    height = bottom + 55
    width = 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{MODEL_NAMES[model]} AXQ MXFP4 on M5 Max</title>',
        '<desc id="desc">A comparison of single-request decode and effective prompt prefill. '
        'The engine labels identify each version and greedy mode. Zero indicates '
        'the tested configuration did not complete the workload.</desc>',
        f'<rect x="0.5" y="0.5" width="{width-1}" height="{height-1}" rx="16" '
        'fill="#f7f9fc" stroke="#d8e2ef"/>',
        '<g font-family="system-ui, sans-serif" fill="#172840">',
        f'<text x="26" y="37" font-size="23" font-weight="700">{MODEL_NAMES[model]} AXQ MXFP4</text>',
        '<text x="26" y="62" font-size="14" fill="#425674">'
        'MacBook Pro M5 Max 128 GB · One active streaming request · Greedy sampling</text>',
        '<text x="26" y="109" font-size="16" font-weight="700">Engine and decoding mode</text>',
    ]
    for index, row in enumerate(rows):
        y = 160 + 31 * index
        parts.append(f'<text x="26" y="{y + 4}" font-size="13">{escape(label(row, versions))}</text>')
    for index, (key, title, step) in enumerate(METRICS):
        x = 270 + 320 * index
        scale = max(step, math.ceil(max(row[key] for row in rows) / (4 * step)) * step)
        parts.append(f'<text x="{x}" y="109" font-size="16" font-weight="700">{title}</text>')
        for tick in range(5):
            grid = x + 61 * tick
            parts.append(
                f'<line x1="{grid}" x2="{grid}" y1="140" y2="{bottom}" stroke="#d7e0ea"/>'
                f'<text x="{grid}" y="133" text-anchor="middle" font-size="11" '
                f'fill="#425674">{tick * scale}</text>'
            )
        for item, row in enumerate(rows):
            y = 160 + 31 * item
            bar = 61 * row[key] / scale
            color = "#514388" if row["mode"] == "dflash2" else "#176b63"
            if bar:
                parts.append(
                    f'<rect x="{x}" y="{y - 9}" width="{bar:.2f}" height="18" rx="3" fill="{color}"/>'
                )
            parts.append(
                f'<text x="{x + bar + 7:.2f}" y="{y + 4}" font-size="12" font-weight="600">'
                f'{rate(row[key])}</text>'
            )
    parts += [
        f'<text x="26" y="{bottom + 27}" font-size="12" fill="#425674">'
        'Prefill uses the first-text latency difference between two input sizes. Zero means incomplete.</text>',
        '</g></svg>',
    ]
    return "\n".join(parts) + "\n"


def tables(versions, grouped):
    parts = [BEGIN, ""]
    for model, rows in grouped.items():
        name = MODEL_NAMES[model]
        chart = figure_name(model)
        parts += [
            f"### {name} MXFP4",
            "",
            f"![{name} decode and effective prefill by engine version and mode on M5 Max.](docs/assets/{chart})",
            "",
            "| Engine and mode | Decode (output tok/s) | Effective prefill (input tok/s) |",
            "| --- | ---: | ---: |",
        ]
        parts += [
            f"| {label(row, versions)} | {rate(row['decode_tok_s'])} | {rate(row['prefill_tok_s'])} |"
            for row in rows
        ]
        parts.append("")
    return "\n".join(parts + [END])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the figures and README table")
    args = parser.parse_args()
    campaign = json.loads((CAMPAIGN / "provenance.json").read_text())
    versions = campaign["engine_versions"]
    records = [json.loads(line) for line in (CAMPAIGN / "results.jsonl").read_text().splitlines()]
    grouped = {
        model: [row for row in records if row["model"] == model and row["engine"] in ENGINE_NAMES]
        for model in MODEL_NAMES
    }
    for model, rows in grouped.items():
        chart = render(model, rows, versions)
        path = ROOT / "docs/assets" / figure_name(model)
        if args.check:
            if path.read_text() != chart:
                parser.error(f"{path.name} differs from the recorded metrics")
        else:
            path.write_text(chart)
    readme = ROOT / "README.md"
    original = readme.read_text()
    start, end = original.index(BEGIN), original.index(END) + len(END)
    generated = tables(versions, grouped)
    if args.check:
        if original[start:end] != generated:
            parser.error("README benchmark tables differ from the recorded metrics")
    else:
        readme.write_text(original[:start] + generated + original[end:])


if __name__ == "__main__":
    main()
