#!/usr/bin/env python3
"""Render the Gemma 4 12B memory-bandwidth share chart.

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
    ("AX 8-bit FFN", 10.98, 45.0, 494, 86, "#2eaf5f", "#176c37"),
    ("AX 4-bit FFN", 6.74, 68.1, 459, 80, "#2eaf5f", "#176c37"),
    ("llama.cpp depth 0", 7.38, 59.8, 441, 76, "#f97316", "#c2410c"),
    ("llama.cpp depth 512", 7.38, 58.9, 435, 75, "#f97316", "#c2410c"),
]

# Chart dimensions
WIDTH = 800
HEIGHT = 430

LEFT = 64
RIGHT = 640
TOP = 88
BOTTOM = 318
LABEL_W = 158
PLOT_LEFT = LEFT + LABEL_W
PLOT_RIGHT = RIGHT

PLOT_W = PLOT_RIGHT - PLOT_LEFT
PLOT_H = BOTTOM - TOP

N = len(ROWS)
BAR_STEP = PLOT_H / N
BAR_H = 24
BAR_PAD = (BAR_STEP - BAR_H) / 2

HEADROOM_COLOR = "#e5e7eb"
HEADROOM_STROKE = "#cbd5e1"
SUBTITLE = "Used bandwidth vs theoretical headroom · 100% = 577 GB/s M5 Max peak"
TITLE = "Gemma 4 12B - Memory bandwidth share · AX Engine v6.5.1"
FOOTNOTE = "AX Engine v6.5.1 · llama.cpp b9700 · M5 Max · peak measured via MLX reduction probe"


def fx(pct: float) -> float:
    return PLOT_LEFT + (pct / 100.0) * PLOT_W


def bar_y(i: int) -> float:
    return TOP + i * BAR_STEP + BAR_PAD


def render() -> str:
    parts: list[str] = []

    def e(s: str) -> str:
        return html.escape(str(s))

    parts += [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}"'
        f' viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f"<title>{e(TITLE)}</title>",
        f'<desc>100% stacked bars showing effective bandwidth used versus theoretical'
        f' headroom for Gemma 4 12B decode. AX 8-bit FFN uses 86%, AX 4-bit FFN'
        f' 80%, llama.cpp depth 0 76%, and llama.cpp depth 512 75% of the'
        f' {PEAK_GBS:.0f} GB/s M5 Max peak.</desc>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>',
        f'<text id="title" x="{LEFT}" y="24" font-family="{FONT}" font-size="16"'
        f' font-weight="700" fill="#111827">{e(TITLE)}</text>',
        f'<text x="{LEFT}" y="46" font-family="{FONT}" font-size="11" fill="#4b5563">'
        f'{e(SUBTITLE)}</text>',
        f'<text id="desc" x="{LEFT}" y="62" font-family="{FONT}" font-size="10" fill="#6b7280">'
        f'Decode is bandwidth-bound; each generated token reads model weights once.</text>',
        f'<text x="{LEFT}" y="76" font-family="{FONT}" font-size="9" fill="#9ca3af">'
        f'{e(FOOTNOTE)}</text>',
        f'<rect x="{PLOT_LEFT}" y="{TOP}" width="{PLOT_W}" height="{PLOT_H}"'
        f' rx="4" fill="#ffffff" stroke="#dbe3ef"/>',
    ]

    # Vertical grid lines at 0, 25, 50, 75, 100%
    for pct in (0, 25, 50, 75, 100):
        gx = fx(pct)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{TOP}" x2="{gx:.1f}" y2="{BOTTOM}"'
            f' stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{BOTTOM + 18}" text-anchor="middle"'
            f' font-family="{FONT}" font-size="10" fill="#6b7280">{pct}%</text>'
        )

    # Bars
    for i, (label, _wgb, tok_s, bw_gbs, pct, fill, stroke) in enumerate(ROWS):
        by = bar_y(i)
        bw = fx(pct) - PLOT_LEFT
        bar_right = PLOT_LEFT + bw
        label_y = by + BAR_H / 2 + 4
        label_x = max(PLOT_LEFT + 34.0, bar_right - 8.0)

        parts.append(
            f'<text x="{PLOT_LEFT - 12}" y="{label_y:.1f}" text-anchor="end"'
            f' font-family="{FONT}" font-size="11" font-weight="700" fill="#111827">{e(label)}</text>'
        )
        parts.append(
            f'<rect x="{PLOT_LEFT}" y="{by:.1f}" width="{PLOT_W:.1f}" height="{BAR_H}"'
            f' rx="5" fill="{HEADROOM_COLOR}" stroke="{HEADROOM_STROKE}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<rect x="{PLOT_LEFT}" y="{by:.1f}" width="{bw:.1f}" height="{BAR_H}"'
            f' rx="5" fill="{fill}" fill-opacity="0.76"/>'
        )
        parts.append(
            f'<line x1="{bar_right:.1f}" y1="{by:.1f}" x2="{bar_right:.1f}" y2="{by + BAR_H:.1f}"'
            f' stroke="{stroke}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="end"'
            f' font-family="{FONT}" font-size="10" font-weight="700" fill="#ffffff">'
            f'{pct}%</text>'
        )
        parts.append(
            f'<text x="{PLOT_RIGHT + 12}" y="{label_y:.1f}" text-anchor="start"'
            f' font-family="{FONT}" font-size="10" fill="#6b7280">'
            f'{bw_gbs} GB/s · {tok_s} tok/s</text>'
        )

    # Legend
    legend_y = HEIGHT - 22
    legend_x = LEFT
    for lbl, clr, stroke in (
        ("AX Engine used bandwidth", "#2eaf5f", "#176c37"),
        ("llama.cpp used bandwidth", "#f97316", "#c2410c"),
    ):
        parts += [
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" rx="2"'
            f' fill="{clr}" fill-opacity="0.76" stroke="{stroke}" stroke-width="1.2"/>',
            f'<text x="{legend_x + 14}" y="{legend_y}" font-family="{FONT}"'
            f' font-size="10" fill="#374151">{e(lbl)}</text>',
        ]
        legend_x += 170
    parts += [
        f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" rx="2"'
        f' fill="{HEADROOM_COLOR}" stroke="{HEADROOM_STROKE}" stroke-width="1"/>',
        f'<text x="{legend_x + 14}" y="{legend_y}" font-family="{FONT}"'
        f' font-size="10" fill="#374151">Theoretical headroom</text>',
    ]

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
