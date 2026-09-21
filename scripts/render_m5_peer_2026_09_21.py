#!/usr/bin/env python3
"""Render the 2026-09-21 M5 Max peer campaign chart as a deterministic SVG.

Reads the three checked-in summary.json artifacts under
benchmarks/results/mtp-axq-peer/ and emits
docs/assets/perf-m5-peer-2026-09-21.svg. Two linear-scale panels keep the
dense (27-77 tok/s) and MoE (109-240 tok/s) bands readable without a log
axis. Unsupported lanes are omitted from the drawing and footnoted, never
rendered as zero-height bars. No timestamps, no randomness.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "benchmarks" / "results" / "mtp-axq-peer"
OUTPUT_SVG = REPO_ROOT / "docs" / "assets" / "perf-m5-peer-2026-09-21.svg"

LANE_ORDER = ("ax_engine", "mtplx", "omlx", "mlx_lm")
LANE_LABELS = {
    "ax_engine": "AX Engine 7.5.3",
    "mtplx": "MTPLX 2.11.2",
    "omlx": "OMLX 0.6.4",
    "mlx_lm": "mlx-lm 0.31.3",
}
LANE_COLORS = {
    "ax_engine": "#176c37",
    "mtplx": "#9a6a00",
    "omlx": "#1f5f8b",
    "mlx_lm": "#6b7a72",
}


def _load(relative: str) -> dict:
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def _lane_decode(lane: dict) -> float | None:
    value = lane.get("decode_tok_s_median_20")
    return value if isinstance(value, (int, float)) else None


def _group(label: str, sublabel: str, lanes: dict) -> tuple[str, str, list[tuple[str, float]]]:
    bars = []
    for lane_key in LANE_ORDER:
        lane = lanes.get(lane_key)
        if not isinstance(lane, dict):
            continue
        decode = _lane_decode(lane)
        if decode is not None:
            bars.append((lane_key, decode))
    return (label, sublabel, bars)


def collect_groups() -> tuple[list, list, list[str]]:
    """Return (dense_groups, moe_groups, footnote_lines)."""
    base = _load("2026-09-21-apple-m5-max-128gb/summary.json")
    family = _load("2026-09-21-apple-m5-max-128gb-qwen-family/summary.json")
    gemma = _load("2026-09-21-apple-m5-max-128gb-gemma/summary.json")

    flappy = base["measured"]["flappy"]
    long_code = base["measured"]["long_code"]
    q36 = family["models"]["qwen36_35b_a3b_axq_6bit_mtp"]["lanes"]
    q38m = family["models"]["qwen38_27b_axq_mxfp4_mtp"]["lanes"]
    g12 = gemma["models"]["gemma4_12b_it_4bit_unified"]["lanes"]
    g26 = gemma["models"]["gemma4_26b_a4b_it_4bit"]["lanes"]

    def wrap(lanes: dict) -> dict:
        out = {}
        for key in LANE_ORDER:
            lane = lanes.get(key)
            if isinstance(lane, dict) and "decode_tok_s_median_20" in lane:
                out[key] = lane
            elif key in lanes and isinstance(lanes[key], dict):
                out[key] = {"decode_tok_s_median_20": None}
        return out

    dense = [
        _group("Qwen 3.8 27B AXQ 6-bit MTP", "flappy", wrap(flappy)),
        _group("Qwen 3.8 27B AXQ 6-bit MTP", "long_code", wrap(long_code)),
        _group("Qwen 3.8 27B AXQ MXFP4 MTP", "flappy", wrap(q38m)),
        _group("Gemma 4 12B (community 4-bit)", "flappy, direct AR", wrap(g12)),
    ]
    moe = [
        _group("Qwen 3.6 35B-A3B AXQ 6-bit MTP", "flappy", wrap(q36)),
        _group("Gemma 4 26B-A4B (community 4-bit)", "flappy, direct AR", wrap(g26)),
    ]
    footnotes = [
        "Blank lanes mean the runtime could not load that pack (unsupported); never zero.",
        "Gemma lanes are direct AR on community checkpoints (no Assistant-MTP sidecar).",
    ]
    return dense, moe, footnotes


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_panel(
    groups: list, panel_x: float, y0: float, axis_max: float
) -> tuple[list[str], float]:
    lines: list[str] = []
    bar_height = 13
    bar_gap = 2
    group_gap = 16
    plot_left = panel_x + 250
    plot_width = 470
    y = y0
    ticks = [0.0, axis_max * 0.25, axis_max * 0.5, axis_max * 0.75, axis_max]
    for tick in ticks:
        tx = plot_left + plot_width * tick / axis_max
        lines.append(
            f'<line x1="{tx:.1f}" y1="{y0 - 14}" x2="{tx:.1f}" y2="{y0 + 20}" '
            'stroke="#d8e2dc" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{tx:.1f}" y="{y0 - 18}" font-size="10" fill="#7a8a82" '
            f'text-anchor="middle">{tick:.0f}</text>'
        )
    for label, sublabel, bars in groups:
        measured = [b for b in bars]
        block_height = len(measured) * (bar_height + bar_gap)
        lines.append(
            f'<text x="{panel_x + 16}" y="{y + block_height / 2 - 1}" font-size="11.5" '
            f'font-weight="600" fill="#12211a">{_escape(label)}</text>'
        )
        lines.append(
            f'<text x="{panel_x + 16}" y="{y + block_height / 2 + 12}" font-size="10" '
            f'fill="#7a8a82">{_escape(sublabel)}</text>'
        )
        for lane_key, decode in measured:
            width = plot_width * decode / axis_max
            color = LANE_COLORS[lane_key]
            lines.append(
                f'<rect x="{plot_left}" y="{y:.1f}" width="{width:.1f}" '
                f'height="{bar_height}" fill="{color}" rx="2"/>'
            )
            lines.append(
                f'<text x="{plot_left + width + 5:.1f}" y="{y + bar_height - 3:.1f}" '
                f'font-size="10" fill="{color}">{decode:.2f}</text>'
            )
            y += bar_height + bar_gap
        y += group_gap
    return lines, y


def render_svg() -> str:
    dense, moe, footnotes = collect_groups()
    legend_y = 66
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="640" '
        'viewBox="0 0 1180 640" font-family="Helvetica, Arial, sans-serif">',
        '<rect width="1180" height="640" fill="#ffffff"/>',
        '<text x="24" y="34" font-size="18" font-weight="700" fill="#12211a">'
        "M5 Max 128 GB peer campaign 2026-09-21 — decode tok/s (median of 20)</text>",
        '<text x="24" y="54" font-size="12" fill="#4a5a52">'
        "AX Engine 7.5.3 vs MTPLX 2.11.2 vs OMLX 0.6.4 vs mlx-lm 0.31.3 — flappy suite "
        "unless noted; single-host snapshot, not a standing ranking</text>",
    ]
    lx = 24
    for lane_key in LANE_ORDER:
        lines.append(
            f'<rect x="{lx}" y="{legend_y - 10}" width="14" height="14" '
            f'fill="{LANE_COLORS[lane_key]}" rx="2"/>'
        )
        lines.append(
            f'<text x="{lx + 20}" y="{legend_y + 1}" font-size="11.5" '
            f'fill="#12211a">{_escape(LANE_LABELS[lane_key])}</text>'
        )
        lx += 20 + 8 * len(LANE_LABELS[lane_key]) + 40

    dense_lines, after_dense = render_panel(dense, 24, 130, 90.0)
    moe_lines, after_moe = render_panel(moe, 660, 130, 260.0)
    lines.extend(dense_lines)
    lines.extend(moe_lines)
    lines.append(
        f'<text x="274" y="{112}" font-size="12" font-weight="700" fill="#12211a">'
        "Dense band (0–90 tok/s)</text>"
    )
    lines.append(
        f'<text x="{910}" y="{112}" font-size="12" font-weight="700" fill="#12211a">'
        "MoE band (0–260 tok/s)</text>"
    )
    note_y = max(after_dense, after_moe) + 24
    for note in footnotes:
        lines.append(
            f'<text x="24" y="{note_y:.0f}" font-size="10.5" fill="#7a8a82">{_escape(note)}</text>'
        )
        note_y += 16
    lines.append(
        f'<text x="24" y="{note_y + 4:.0f}" font-size="10.5" fill="#7a8a82">'
        "Artifacts: benchmarks/results/mtp-axq-peer/2026-09-21-apple-m5-max-128gb*/"
        "summary.json</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_SVG.write_text(render_svg(), encoding="utf-8")
    print(f"wrote {OUTPUT_SVG.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
