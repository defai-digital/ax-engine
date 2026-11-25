#!/usr/bin/env python3
"""Render the README throughput chart from the recorded M5 peer results."""

import argparse
from html import escape
import json
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parent
ROOT = CAMPAIGN.parents[2]
OUTPUT = ROOT / "docs/assets/perf-qwen38-mxfp4-m5-peers.svg"
ENGINES = {
    "ax-engine": "AX Engine",
    "splash": "Splash",
    "omlx": "oMLX",
    "mlx-lm": "mlx-lm",
    "mtplx": "MTPLX",
    "tensorfold": "TensorFold",
    "ds4": "ds4",
}
MODELS = {"dense": "Qwen 3.8 27B", "flash": "Qwen 3.8 Flash Next"}


def render_chart(rows):
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="750" height="440" '
        'viewBox="0 0 750 440" role="img" aria-labelledby="title desc">',
        '<title id="title">Qwen 3.8 MXFP4 serving on M5 Max</title>',
        '<desc id="desc">Two horizontal bar charts use the same zero to 100 output tokens '
        'per second scale. AX Engine runs direct decode. Splash uses DFlash2 for 27B. '
        'A zero indicates the engine did not complete the workload.</desc>',
        '<rect x="0.5" y="0.5" width="749" height="439" rx="16" fill="#f7f9fc" stroke="#d8e2ef"/>',
        '<g font-family="system-ui, sans-serif" fill="#18283f">',
        '<text x="25" y="38" font-size="23" font-weight="700">Qwen 3.8 MXFP4 serving</text>',
        '<text x="25" y="63" font-size="14" fill="#425674">'
        'MacBook Pro M5 Max 128 GB · One streaming request · Output tok/s</text>',
    ]
    for column, (model, heading) in enumerate(MODELS.items()):
        x = 25 + column * 367
        bar_x = x + 95
        parts.append(f'<text x="{x}" y="108" font-size="17" font-weight="700">{heading}</text>')
        for mark in (0, 25, 50, 75, 100):
            grid_x = bar_x + 2 * mark
            parts.append(
                f'<line x1="{grid_x}" x2="{grid_x}" y1="140" y2="382" stroke="#d8e2ef"/>'
                f'<text x="{grid_x}" y="133" font-size="11" fill="#425674" '
                f'text-anchor="middle">{mark}</text>'
            )
        for index, (engine, name) in enumerate(ENGINES.items()):
            value = rows[model, engine]
            y = 159 + 34 * index
            name = escape(name)
            color = "#176b63" if engine == "ax-engine" else "#4f71a0"
            parts.append(f'<text x="{x}" y="{y + 4}" font-size="13">{name}</text>')
            if value:
                parts.append(
                    f'<rect x="{bar_x}" y="{y - 9}" width="{2 * value:.2f}" '
                    f'height="18" rx="3" fill="{color}"/>'
                )
            label = f"{value:.2f}" if value else "0"
            parts.append(
                f'<text x="{bar_x + 2 * value + 7:.2f}" y="{y + 4}" '
                f'font-size="12" font-weight="600">{label}</text>'
            )
    parts += [
        '<text x="25" y="409" font-size="12" fill="#425674">'
        'AX uses direct decode. Splash uses DFlash2 for 27B. Zero means the workload did not complete.</text>',
        '</g></svg>',
    ]
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the committed chart")
    args = parser.parse_args()
    results = (json.loads(line) for line in (CAMPAIGN / "results.jsonl").read_text().splitlines())
    rows = {
        (row["model"], row["engine"]): row["tok_s"]
        for row in results
        if row.get("model") in MODELS and row.get("engine") in ENGINES
    }
    chart = render_chart(rows)
    if args.check:
        if OUTPUT.read_text() != chart:
            parser.error(f"{OUTPUT.name} differs from the recorded M5 peer results")
    else:
        OUTPUT.write_text(chart)


if __name__ == "__main__":
    main()
