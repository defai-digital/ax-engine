#!/usr/bin/env python3
"""Render Gemma 4 12B direct-decode comparison charts for the README.

Gemma 4 12B is the unified ``gemma4_unified`` architecture. Upstream mlx_lm
0.31.3 has no graph for it (``ValueError: Model type gemma4_unified not
supported``), so the only MLX-side runtime is AX Engine's repo-owned native
graph. The direct comparison is therefore two-way: AX Engine native vs
llama.cpp Metal (a shape-compatible GGUF baseline), with mlx_lm necessarily
absent.

Because llama.cpp Metal rows carry only aggregate medians (llama-bench does
not expose per-trial samples the way the AX HTTP path does), this renders a
grouped *bar* chart of medians per prompt length rather than the family
box-and-whisker used for models mlx_lm can load.

Reads the ``ax.mlx_inference_stack.v2`` artifact written by
bench_mlx_inference_stack.py.

Usage:
  python3 scripts/render_gemma4_12b_direct_charts.py \\
      --artifact benchmarks/results/mlx-inference/2026-06-08-gemma-4-12b-it-4bit-direct/gemma-4-12b-it-4bit.json \\
      --assets-dir docs/assets
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FONT = "Inter,Segoe UI,Arial,sans-serif"
RED = "#dc2626"
PROMPT_TOKENS = (128, 512, 2048)
AX_ENGINE_VERSION = "v6.8.2 (2026-07-11)"
LLAMA_CPP_VERSION = "b9700"

# (engine key in artifact, legend label, fill, stroke) — palette matches the
# existing direct-comparison section in render_readme_performance_charts.py.
# llama.cpp first, AX second — consistent with the README table column order.
SERIES = [
    ("llama_cpp_metal", "llama.cpp Metal", "#f97316", "#c2410c"),
    ("ax_engine_mlx", "AX Engine (native MLX)", "#2eaf5f", "#176c37"),
]

# (metric key, slug, title, unit, lower_is_better)
METRICS = [
    ("decode_tok_s", "decode-tok-s", "Gemma 4 12B 4-bit — Direct decode", "tok/s", False),
    ("prefill_tok_s", "prefill-tok-s", "Gemma 4 12B 4-bit — Prefill", "tok/s", False),
    ("ttft_ms", "ttft-ms", "Gemma 4 12B 4-bit — Time to first token", "ms", True),
]


def metric_median(row: dict[str, Any], key: str) -> float | None:
    if row.get("engine") == "llama_cpp_metal" and key == "decode_tok_s":
        depth_metric = row.get("decode_at_depth_tok_s")
        if isinstance(depth_metric, dict) and isinstance(depth_metric.get("median"), (int, float)):
            return float(depth_metric["median"])
    metric = row.get(key)
    if isinstance(metric, dict) and isinstance(metric.get("median"), (int, float)):
        return float(metric["median"])
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def nice_axis_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for candidate in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def short_number(value: float) -> str:
    if value >= 1000:
        compact = value / 1000
        return f"{compact:.0f}k" if compact.is_integer() else f"{compact:.1f}k"
    if value.is_integer():
        return f"{value:.0f}"
    return f"{value:.1f}"


def collect(rows: list[dict[str, Any]], metric: str) -> dict[str, dict[int, float]]:
    """engine -> prompt_tokens -> median value."""
    out: dict[str, dict[int, float]] = {engine: {} for engine, *_ in SERIES}
    for row in rows:
        engine = row.get("engine")
        pt = row.get("prompt_tokens")
        if engine not in out or pt not in PROMPT_TOKENS:
            continue
        value = metric_median(row, metric)
        if value is not None:
            out[engine][int(pt)] = value
    return out


def render_chart(
    *, title: str, unit: str, lower_is_better: bool, data: dict[str, dict[int, float]]
) -> str:
    width, height = 720, 400
    left, right, top, bottom = 64, 220, 78, 92
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_vals = [v for series in data.values() for v in series.values()]
    if not all_vals:
        raise ValueError("no data to chart")
    axis_max = nice_axis_ceiling(max(all_vals) * 1.12)
    best_by_prompt_tokens: dict[int, float] = {}
    for prompt_tokens in PROMPT_TOKENS:
        prompt_values = [
            series[prompt_tokens]
            for series in data.values()
            if prompt_tokens in series
        ]
        best_by_prompt_tokens[prompt_tokens] = (
            min(prompt_values) if lower_is_better else max(prompt_values)
        )

    def fy(v: float) -> float:
        clamped = max(0.0, min(v, axis_max))
        return top + plot_h - (clamped / axis_max) * plot_h

    n_groups = len(PROMPT_TOKENS)
    n_bars = len(SERIES)
    group_step = plot_w / n_groups
    bar_w = min(46.0, group_step * 0.30)
    gap = bar_w * 0.18
    block = n_bars * bar_w + (n_bars - 1) * gap

    direction = "Lower is better" if lower_is_better else "Higher is better"
    direction_fill = "#dc2626"
    engine_desc = ", ".join(label for _e, label, *_ in SERIES)
    unit_w = max(48, len(unit) * 7 + 24)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>Grouped bar chart of median {html.escape(unit)} comparing "
        f"{html.escape(engine_desc)} for Gemma 4 12B 4-bit at "
        f"{'/'.join(str(p) for p in PROMPT_TOKENS)} prompt tokens. mlx_lm is absent "
        f"because it has no graph for the gemma4_unified architecture. "
        f"AX Engine {html.escape(AX_ENGINE_VERSION)} is compared with llama.cpp Metal "
        f"{html.escape(LLAMA_CPP_VERSION)}.</desc>",
        f'<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{left}" y="26" font-family="{FONT}" font-size="16" font-weight="700" '
        f'fill="#111827">{html.escape(title)}</text>',
        f'<text x="{left}" y="46" font-family="{FONT}" font-size="11" fill="#4b5563">'
        f"median over reps · grouped by prompt tokens · mlx_lm N/A (no gemma4_unified graph)</text>",
        f'<text x="{left}" y="62" font-family="{FONT}" font-size="10" fill="#6b7280">'
        f"AX Engine {html.escape(AX_ENGINE_VERSION)} vs llama.cpp Metal "
        f"{html.escape(LLAMA_CPP_VERSION)} (Unsloth Q4_K_M, shape-compatible)</text>",
        f'<rect x="{width - 34 - unit_w}" y="13" width="{unit_w}" height="22" rx="11" '
        f'fill="#eef2ff" stroke="#c7d2fe"/>',
        f'<text x="{width - 34 - unit_w / 2:.1f}" y="28" text-anchor="middle" '
        f'font-family="{FONT}" font-size="10" font-weight="700" fill="#3730a3">{html.escape(unit)}</text>',
        f'<text x="{width - 34}" y="52" text-anchor="end" font-family="{FONT}" font-size="10" '
        f'font-weight="700" fill="{direction_fill}">{direction}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="6" '
        f'fill="#ffffff" stroke="#dbe3ef"/>',
    ]

    for i in range(5):
        grid_val = axis_max * i / 4
        gy = fy(grid_val)
        parts.append(
            f'<line x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" y2="{gy:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{gy + 3:.1f}" text-anchor="end" font-family="{FONT}" '
            f'font-size="11" fill="#6b7280">{short_number(grid_val)}</text>'
        )

    for gi, pt in enumerate(PROMPT_TOKENS):
        group_center = left + (gi + 0.5) * group_step
        bar0 = group_center - block / 2
        for bi, (engine, _label, fill, stroke) in enumerate(SERIES):
            val = data.get(engine, {}).get(pt)
            cx = bar0 + bi * (bar_w + gap) + bar_w / 2
            bl = cx - bar_w / 2
            if val is None:
                parts.append(
                    f'<text x="{cx:.1f}" y="{top + plot_h - 6:.1f}" text-anchor="middle" '
                    f'font-family="{FONT}" font-size="9" fill="#9ca3af">N/A</text>'
                )
                continue
            y = fy(val)
            bh = top + plot_h - y
            prompt_best = best_by_prompt_tokens[pt]
            label_fill = RED if math.isclose(val, prompt_best) else "#111827"
            parts.extend(
                [
                    f'<rect x="{bl:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="3" '
                    f'fill="{fill}" fill-opacity="0.26" stroke="{stroke}" stroke-width="1.8"/>',
                    f'<line x1="{bl:.1f}" y1="{y:.1f}" x2="{bl + bar_w:.1f}" y2="{y:.1f}" '
                    f'stroke="{stroke}" stroke-width="2.6"/>',
                    f'<text x="{cx:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="{FONT}" '
                    f'font-size="11" font-weight="700" fill="{label_fill}">{short_number(val)}</text>',
                ]
            )
        parts.append(
            f'<text x="{group_center:.1f}" y="{top + plot_h + 20:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="11" font-weight="700" fill="#111827">{pt} tok</text>'
        )

    legend_y = height - 22
    legend_x = left
    for engine, label, fill, stroke in SERIES:
        parts.append(
            f'<rect x="{legend_x:.1f}" y="{legend_y - 9}" width="10" height="10" rx="2" '
            f'fill="{fill}" fill-opacity="0.5" stroke="{stroke}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14:.1f}" y="{legend_y}" font-family="{FONT}" font-size="10" '
            f'fill="#374151">{html.escape(label)}</text>'
        )
        legend_x += max(200.0, len(label) * 7 + 30)

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, default=REPO_ROOT / "docs" / "assets")
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text())
    rows = payload.get("results", [])
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for metric, slug, title, unit, lower_is_better in METRICS:
        data = collect(rows, metric)
        if not any(data.values()):
            continue
        svg = render_chart(
            title=title, unit=unit, lower_is_better=lower_is_better, data=data
        )
        out = args.assets_dir / f"perf-gemma4-12b-direct-{slug}.svg"
        out.write_text(svg)
        written.append(str(out))
    print("Wrote:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
