#!/usr/bin/env python3
"""Render Gemma 4 12B multimodal benchmark charts for the README."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


FONT = "Inter,Segoe UI,Arial,sans-serif"
OUTPUTS = {
    "ttft": "perf-gemma4-12b-multimodal-image-ttft-ms.svg",
    "prefill": "perf-gemma4-12b-multimodal-image-prefill-tok-s.svg",
}


def fmt_value(value: float) -> str:
    if abs(value) >= 1000:
        text = f"{value:,.0f}"
    elif abs(value) >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    return text


def nice_axis_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for candidate in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def metric_summary(artifact: dict[str, Any], key: str) -> dict[str, float]:
    raw = artifact.get("summary", {}).get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"artifact lacks summary.{key}")
    out: dict[str, float] = {}
    for stat in ("median", "min", "max"):
        value = raw.get(stat)
        if not isinstance(value, (int, float)):
            raise ValueError(f"artifact lacks numeric summary.{key}.{stat}")
        out[stat] = float(value)
    return out


def metric_summary_from_row(row: dict[str, Any], key: str) -> dict[str, float]:
    raw = row.get("summary", {}).get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"artifact row lacks summary.{key}")
    out: dict[str, float] = {}
    for stat in ("median", "min", "max"):
        value = raw.get(stat)
        if not isinstance(value, (int, float)):
            raise ValueError(f"artifact row lacks numeric summary.{key}.{stat}")
        out[stat] = float(value)
    return out


def select_native_prefill_row(artifact: dict[str, Any]) -> dict[str, Any]:
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise ValueError("matrix artifact lacks rows")
    measured = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("status") == "measured"
        and row.get("engine") == "ax_engine"
        and row.get("layer") == "native_runtime_prefill"
        and isinstance(row.get("summary"), dict)
    ]
    if not measured:
        raise ValueError("matrix artifact lacks measured AX native_runtime_prefill rows")
    for row in measured:
        if row.get("case_id") == "image_single_256soft":
            return row
    return measured[0]


def chart_inputs(artifact: dict[str, Any]) -> dict[str, Any]:
    schema = artifact.get("schema")
    if schema == "ax.gemma4_multimodal_benchmark.v1":
        row = select_native_prefill_row(artifact)
        prompt = row.get("prompt") or {}
        repetitions = int((artifact.get("benchmark") or {}).get("repetitions") or 0)
        max_output_tokens = int((artifact.get("benchmark") or {}).get("max_output_tokens") or 0)
        image_soft_tokens = sum(int(value) for value in prompt.get("image_soft_tokens") or [])
        audio_soft_tokens = sum(int(value) for value in prompt.get("audio_soft_tokens") or [])
        video_soft_tokens = sum(int(value) for value in prompt.get("video_soft_tokens") or [])
        soft_note = (
            f"{image_soft_tokens} image, {audio_soft_tokens} audio, "
            f"{video_soft_tokens} video soft tokens"
        )
        return {
            "ttft": metric_summary_from_row(row, "runner_prefill_ttft_ms"),
            "prefill": metric_summary_from_row(row, "prefill_tok_s"),
            "case_id": str(row.get("case_id") or "multimodal"),
            "subtitle": "AX Engine native MLX, runner-time multimodal prefill",
            "throughput_subtitle": "AX Engine native MLX, expanded prompt includes soft tokens",
            "footnote": (
                f"{int(prompt.get('expanded_tokens') or 0)} expanded tokens, "
                f"{soft_note}, {repetitions} reps, max_output_tokens={max_output_tokens}"
            ),
        }
    if schema not in (None, "ax.gemma4_image_prefill_ttft.v1"):
        raise ValueError(f"unsupported artifact schema: {schema}")

    prompt = artifact.get("prompt") or {}
    expanded_tokens = int(prompt.get("expanded_tokens") or 0)
    image_soft_tokens = int(prompt.get("image_soft_tokens") or 0)
    repetitions = int(artifact.get("repetitions") or 0)
    max_output_tokens = int(artifact.get("max_output_tokens") or 0)
    return {
        "ttft": metric_summary(artifact, "runner_prefill_ttft_ms"),
        "prefill": metric_summary(artifact, "prefill_tok_s"),
        "case_id": "image_single_256soft",
        "subtitle": "AX Engine native MLX, runner-time image+text prefill",
        "throughput_subtitle": "AX Engine native MLX, expanded prompt includes image soft tokens",
        "footnote": (
            f"{expanded_tokens} expanded tokens, {image_soft_tokens} image soft tokens, "
            f"{repetitions} reps, max_output_tokens={max_output_tokens}"
        ),
    }


def render_single_bar_chart(
    *,
    title: str,
    subtitle: str,
    unit: str,
    lower_is_better: bool,
    median: float,
    min_value: float,
    max_value: float,
    footnote: str,
    bar_label: str = "image + text",
) -> str:
    width, height = 720, 360
    left, right, top, bottom = 82, 170, 82, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    axis_max = nice_axis_ceiling(max(max_value, median) * 1.18)

    def fy(value: float) -> float:
        return top + plot_h - (max(0.0, min(value, axis_max)) / axis_max) * plot_h

    direction = "Lower is better" if lower_is_better else "Higher is better"
    direction_fill = "#dc2626" if lower_is_better else "#374151"
    bar_center = left + plot_w * 0.42
    bar_width = 118
    bar_x = bar_center - bar_width / 2
    y = fy(median)
    bar_h = top + plot_h - y
    err_min_y = fy(min_value)
    err_max_y = fy(max_value)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(subtitle)} Median {html.escape(unit)} is "
        f"{fmt_value(median)} with min {fmt_value(min_value)} and max {fmt_value(max_value)}.</desc>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text id="title" x="{left}" y="28" font-family="{FONT}" font-size="16" '
        f'font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="{left}" y="48" font-family="{FONT}" font-size="11" fill="#4b5563">'
        f"{html.escape(subtitle)}</text>",
        f'<text x="{width - 32}" y="31" text-anchor="end" font-family="{FONT}" '
        f'font-size="10" font-weight="700" fill="{direction_fill}">{direction}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="6" '
        f'fill="#ffffff" stroke="#dbe3ef"/>',
    ]

    for i in range(5):
        grid_value = axis_max * i / 4
        grid_y = fy(grid_value)
        parts.append(
            f'<line x1="{left}" y1="{grid_y:.1f}" x2="{left + plot_w}" y2="{grid_y:.1f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{grid_y + 4:.1f}" text-anchor="end" '
            f'font-family="{FONT}" font-size="11" fill="#6b7280">{fmt_value(grid_value)}</text>'
        )

    parts.extend(
        [
            f'<rect x="{bar_x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_h:.1f}" '
            'rx="5" fill="#2eaf5f" fill-opacity="0.28" stroke="#176c37" stroke-width="2"/>',
            f'<line x1="{bar_x:.1f}" y1="{y:.1f}" x2="{bar_x + bar_width:.1f}" y2="{y:.1f}" '
            'stroke="#176c37" stroke-width="3"/>',
            f'<line x1="{bar_center:.1f}" y1="{err_max_y:.1f}" x2="{bar_center:.1f}" '
            f'y2="{err_min_y:.1f}" stroke="#111827" stroke-width="1.4"/>',
            f'<line x1="{bar_center - 24:.1f}" y1="{err_max_y:.1f}" x2="{bar_center + 24:.1f}" '
            f'y2="{err_max_y:.1f}" stroke="#111827" stroke-width="1.4"/>',
            f'<line x1="{bar_center - 24:.1f}" y1="{err_min_y:.1f}" x2="{bar_center + 24:.1f}" '
            f'y2="{err_min_y:.1f}" stroke="#111827" stroke-width="1.4"/>',
            f'<text x="{bar_center:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="13" font-weight="700" fill="#111827">'
            f"{fmt_value(median)}</text>",
            f'<text x="{bar_center:.1f}" y="{top + plot_h + 24:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="11" font-weight="700" fill="#111827">'
            f"{html.escape(bar_label)}</text>",
            f'<text x="{left + plot_w + 16}" y="{top + 28}" font-family="{FONT}" '
            f'font-size="11" fill="#374151">median: <tspan font-weight="700">'
            f"{fmt_value(median)} {html.escape(unit)}</tspan></text>",
            f'<text x="{left + plot_w + 16}" y="{top + 48}" font-family="{FONT}" '
            f'font-size="11" fill="#6b7280">min: {fmt_value(min_value)}</text>',
            f'<text x="{left + plot_w + 16}" y="{top + 66}" font-family="{FONT}" '
            f'font-size="11" fill="#6b7280">max: {fmt_value(max_value)}</text>',
            f'<text x="{left}" y="{height - 20}" font-family="{FONT}" font-size="10" '
            f'fill="#6b7280">{html.escape(footnote)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def render(artifact_path: Path, assets_dir: Path) -> list[Path]:
    artifact = json.loads(artifact_path.read_text())
    inputs = chart_inputs(artifact)
    ttft = inputs["ttft"]
    prefill = inputs["prefill"]
    case_id = inputs["case_id"]

    assets_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        (
            OUTPUTS["ttft"],
            render_single_bar_chart(
                title="Gemma 4 12B multimodal image prefill TTFT",
                subtitle=inputs["subtitle"],
                unit="ms",
                lower_is_better=True,
                median=ttft["median"],
                min_value=ttft["min"],
                max_value=ttft["max"],
                footnote=inputs["footnote"],
                bar_label=case_id,
            ),
        ),
        (
            OUTPUTS["prefill"],
            render_single_bar_chart(
                title="Gemma 4 12B multimodal image prefill throughput",
                subtitle=inputs["throughput_subtitle"],
                unit="tok/s",
                lower_is_better=False,
                median=prefill["median"],
                min_value=prefill["min"],
                max_value=prefill["max"],
                footnote=inputs["footnote"],
                bar_label=case_id,
            ),
        ),
    ]

    written: list[Path] = []
    for filename, svg in outputs:
        path = assets_dir / filename
        path.write_text(svg)
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "benchmarks/results/gemma4-multimodal/"
            "2026-06-09-gemma4-12b-image-prefill-ttft.json"
        ),
    )
    parser.add_argument("--assets-dir", type=Path, default=Path("docs/assets"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for path in render(args.artifact, args.assets_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
