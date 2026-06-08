#!/usr/bin/env python3
"""Render the Gemma 4 12B memory-bandwidth utilization horizontal bar chart.

Data is static (sourced from the README bandwidth table) and reflects the
M5 Max GPU peak of 577 GB/s measured via MLX reduction probe.

Usage:
  python3 scripts/render_gemma4_12b_bandwidth_chart.py [--assets-dir docs/assets]
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FONT = "Inter,Segoe UI,Arial,sans-serif"

PEAK_GBS = 577.0

# (label, weights_gb, decode_tok_s, effective_bw_gbs, pct_peak, fill, stroke)
ROWS = [
    ("llama.cpp — depth 0 (tg)",       7.38, 60.4, 446, 77, "#f97316", "#c2410c"),
    ("llama.cpp — depth 512",          7.38, 56.6, 418, 72, "#f97316", "#c2410c"),
    ("AX — 8-bit FFN (upstream)",    10.98, 45.0, 494, 86, "#2eaf5f", "#176c37"),
    ("AX — 4-bit FFN (re-quantized)",  6.74, 68.1, 459, 80, "#2eaf5f", "#176c37"),
]

# Chart dimensions
WIDTH = 720
HEIGHT = 280

LABEL_W = 218   # right edge of label area
PLOT_LEFT = LABEL_W + 8
PLOT_RIGHT = 530
PLOT_TOP = 72
PLOT_BOTTOM = 228

PLOT_W = PLOT_RIGHT - PLOT_LEFT
PLOT_H = PLOT_BOTTOM - PLOT_TOP

N = len(ROWS)
BAR_STEP = PLOT_H / N
BAR_H = 22
BAR_PAD = (BAR_STEP - BAR_H) / 2

RED = "#dc2626"
SUBTITLE = "% of M5 Max GPU peak (577 GB/s) · decode is bandwidth-bound · each tok reads weights once"
FOOTNOTE  = "AX Engine v6.0.1 · llama.cpp b9430 · M5 Max · peak measured via MLX reduction probe"


def fx(pct: float) -> float:
    return PLOT_LEFT + (pct / 100.0) * PLOT_W


def bar_y(i: int) -> float:
    return PLOT_TOP + i * BAR_STEP + BAR_PAD


def render() -> str:
    parts: list[str] = []

    def e(s: str) -> str:
        return html.escape(str(s))

    parts += [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}"'
        f' viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title>Gemma 4 12B — Memory Bandwidth Utilization</title>',
        f'<desc>Horizontal bar chart showing the percentage of M5 Max GPU peak memory bandwidth'
        f' ({PEAK_GBS:.0f} GB/s) consumed per decode token for four engine/quantization'
        f' configurations of Gemma 4 12B. AX 8-bit FFN reaches 86%, AX 4-bit FFN 80%,'
        f' llama.cpp depth-0 77%, llama.cpp depth-512 72%.</desc>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>',
        f'<text x="10" y="22" font-family="{FONT}" font-size="15" font-weight="700"'
        f' fill="#111827">Gemma 4 12B — Memory Bandwidth Utilization</text>',
        f'<text x="{WIDTH - 10}" y="22" text-anchor="end" font-family="{FONT}" font-size="10"'
        f' font-weight="700" fill="#374151">Higher is better</text>',
        f'<text x="10" y="40" font-family="{FONT}" font-size="10" fill="#4b5563">'
        f'{e(SUBTITLE)}</text>',
        f'<text x="10" y="54" font-family="{FONT}" font-size="10" fill="#6b7280">'
        f'{e(FOOTNOTE)}</text>',
        f'<rect x="{PLOT_LEFT}" y="{PLOT_TOP}" width="{PLOT_W}" height="{PLOT_H}"'
        f' rx="4" fill="#ffffff" stroke="#dbe3ef"/>',
    ]

    # Vertical grid lines at 0, 25, 50, 75, 100%
    for pct in (0, 25, 50, 75, 100):
        gx = fx(pct)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{PLOT_TOP}" x2="{gx:.1f}" y2="{PLOT_BOTTOM}"'
            f' stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{PLOT_BOTTOM + 14}" text-anchor="middle"'
            f' font-family="{FONT}" font-size="10" fill="#6b7280">{pct}%</text>'
        )

    # Best-value reference line
    best_pct = max(row[4] for row in ROWS)
    best_x = fx(best_pct)
    parts.append(
        f'<line x1="{best_x:.1f}" y1="{PLOT_TOP}" x2="{best_x:.1f}" y2="{PLOT_BOTTOM}"'
        f' stroke="{RED}" stroke-width="1.2" stroke-dasharray="1 4" stroke-linecap="round"/>'
    )
    parts.append(
        f'<text x="{best_x - 4:.1f}" y="66" text-anchor="end"'
        f' font-family="{FONT}" font-size="10" font-weight="700" fill="{RED}">'
        f'highest: {best_pct}%</text>'
    )



    # Bars
    for i, (label, _wgb, tok_s, bw_gbs, pct, fill, stroke) in enumerate(ROWS):
        by = bar_y(i)
        bw = fx(pct) - PLOT_LEFT
        bar_right = PLOT_LEFT + bw
        label_y = by + BAR_H / 2 + 4

        parts.append(
            f'<text x="{LABEL_W}" y="{label_y:.1f}" text-anchor="end"'
            f' font-family="{FONT}" font-size="11" fill="#374151">{e(label)}</text>'
        )
        parts.append(
            f'<rect x="{PLOT_LEFT}" y="{by:.1f}" width="{bw:.1f}" height="{BAR_H}"'
            f' rx="3" fill="{fill}" fill-opacity="0.26" stroke="{stroke}" stroke-width="1.6"/>'
        )
        parts.append(
            f'<line x1="{bar_right:.1f}" y1="{by:.1f}" x2="{bar_right:.1f}" y2="{by + BAR_H:.1f}"'
            f' stroke="{stroke}" stroke-width="2.2"/>'
        )
        parts.append(
            f'<text x="{bar_right + 5:.1f}" y="{label_y:.1f}" text-anchor="start"'
            f' font-family="{FONT}" font-size="11" font-weight="700" fill="#111827">'
            f'{pct}%</text>'
        )
        parts.append(
            f'<text x="{PLOT_RIGHT + 10}" y="{label_y:.1f}" text-anchor="start"'
            f' font-family="{FONT}" font-size="10" fill="#6b7280">'
            f'{bw_gbs} GB/s · {tok_s} tok/s</text>'
        )

    # Legend
    legend_y = PLOT_BOTTOM + 36
    legend_x = PLOT_LEFT
    for lbl, clr, stroke in (
        ("llama.cpp Metal", "#f97316", "#c2410c"),
        ("AX Engine (native MLX)", "#2eaf5f", "#176c37"),
    ):
        parts += [
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" rx="2"'
            f' fill="{clr}" fill-opacity="0.5" stroke="{stroke}" stroke-width="1.4"/>',
            f'<text x="{legend_x + 14}" y="{legend_y}" font-family="{FONT}"'
            f' font-size="10" fill="#374151">{e(lbl)}</text>',
        ]
        legend_x += 160

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=REPO_ROOT / "docs" / "assets")
    args = parser.parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    out = args.assets_dir / "perf-gemma4-12b-bandwidth.svg"
    out.write_text(render())
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
