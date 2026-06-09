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
LEGACY_OUTPUTS = {
    "ttft": "perf-gemma4-12b-multimodal-image-ttft-ms.svg",
    "prefill": "perf-gemma4-12b-multimodal-image-prefill-tok-s.svg",
}
MATRIX_OUTPUTS = {
    "ttft": "perf-gemma4-12b-multimodal-ttft-ms.svg",
    "prefill": "perf-gemma4-12b-multimodal-prefill-tok-s.svg",
    "peer": "perf-gemma4-12b-multimodal-peer-chat-ms.svg",
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


def metric_median_from_row(row: dict[str, Any], key: str) -> float | None:
    raw = row.get("summary", {}).get(key)
    if not isinstance(raw, dict):
        return None
    value = raw.get("median")
    return float(value) if isinstance(value, (int, float)) else None


def native_prefill_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise ValueError("matrix artifact lacks rows")
    measured = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("status") == "measured"
        and row.get("engine") in {"ax_engine", "ax_engine_mlx"}
        and row.get("layer") == "native_runtime_prefill"
        and isinstance(row.get("summary"), dict)
    ]
    if not measured:
        raise ValueError("matrix artifact lacks measured AX native_runtime_prefill rows")
    return measured


def peer_comparison_series(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        return []
    ax_by_case = {
        row.get("case_id"): row
        for row in rows
        if isinstance(row, dict)
        and row.get("status") == "measured"
        and row.get("engine") in {"ax_engine", "ax_engine_mlx"}
        and row.get("layer") == "openai_chat_e2e"
    }
    series = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("status") != "measured"
            or row.get("engine") != "llama_cpp_metal"
            or row.get("layer") != "peer_comparison"
        ):
            continue
        case_id = row.get("case_id")
        ax_row = ax_by_case.get(case_id)
        if not isinstance(case_id, str) or not isinstance(ax_row, dict):
            continue
        capability = row.get("capability")
        if not isinstance(capability, dict) or capability.get("cache_policy") != "prompt_cache_disabled":
            continue
        peer_prompt_tokens = metric_median_from_row(row, "prompt_tokens_reported")
        peer_server_prompt_tokens = metric_median_from_row(row, "server_prompt_tokens")
        if (
            peer_prompt_tokens is not None
            and peer_server_prompt_tokens is not None
            and peer_prompt_tokens > 0
            and peer_server_prompt_tokens < peer_prompt_tokens * 0.9
        ):
            continue
        ax_output_tokens = metric_median_from_row(ax_row, "output_tokens")
        peer_output_tokens = metric_median_from_row(row, "output_tokens")
        if ax_output_tokens != peer_output_tokens:
            continue
        ax_stats = metric_summary_from_row(ax_row, "client_wall_ms")
        peer_stats = metric_summary_from_row(row, "client_wall_ms")
        series.append(
            {
                "label": case_id,
                "modalities": "+".join(row.get("modalities") or []),
                "ax_prompt_tokens": metric_median_from_row(ax_row, "prompt_tokens_reported"),
                "peer_prompt_tokens": metric_median_from_row(row, "prompt_tokens_reported"),
                "peer_cached_tokens": metric_median_from_row(
                    row, "prompt_cached_tokens_reported"
                ),
                "output_tokens": ax_output_tokens,
                "ax": ax_stats,
                "peer": peer_stats,
            }
        )
    return series


def peer_exclusions(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        return []
    ax_by_case = {
        row.get("case_id"): row
        for row in rows
        if isinstance(row, dict)
        and row.get("status") == "measured"
        and row.get("engine") in {"ax_engine", "ax_engine_mlx"}
        and row.get("layer") == "openai_chat_e2e"
    }
    exclusions: list[dict[str, Any]] = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("status") != "measured"
            or row.get("engine") != "llama_cpp_metal"
            or row.get("layer") != "peer_comparison"
        ):
            continue
        case_id = row.get("case_id")
        ax_row = ax_by_case.get(case_id)
        if not isinstance(case_id, str) or not isinstance(ax_row, dict):
            continue
        capability = row.get("capability")
        if not isinstance(capability, dict) or capability.get("cache_policy") != "prompt_cache_disabled":
            exclusions.append(
                {
                    "case_id": case_id,
                    "reason": "peer_cache_policy_not_disabled",
                    "cache_policy": (
                        capability.get("cache_policy") if isinstance(capability, dict) else None
                    ),
                }
            )
            continue
        peer_prompt_tokens = metric_median_from_row(row, "prompt_tokens_reported")
        peer_server_prompt_tokens = metric_median_from_row(row, "server_prompt_tokens")
        if (
            peer_prompt_tokens is not None
            and peer_server_prompt_tokens is not None
            and peer_prompt_tokens > 0
            and peer_server_prompt_tokens < peer_prompt_tokens * 0.9
        ):
            exclusions.append(
                {
                    "case_id": case_id,
                    "reason": "peer_server_prompt_tokens_too_low",
                    "peer_prompt_tokens": peer_prompt_tokens,
                    "peer_server_prompt_tokens": peer_server_prompt_tokens,
                }
            )
            continue
        ax_output_tokens = metric_median_from_row(ax_row, "output_tokens")
        peer_output_tokens = metric_median_from_row(row, "output_tokens")
        if ax_output_tokens != peer_output_tokens:
            exclusions.append(
                {
                    "case_id": case_id,
                    "reason": "output_token_mismatch",
                    "ax_output_tokens": ax_output_tokens,
                    "peer_output_tokens": peer_output_tokens,
                }
            )
    return exclusions


def chart_inputs(artifact: dict[str, Any]) -> dict[str, Any]:
    schema = artifact.get("schema")
    if schema == "ax.gemma4_multimodal_benchmark.v1":
        repetitions = int((artifact.get("benchmark") or {}).get("repetitions") or 0)
        max_output_tokens = int((artifact.get("benchmark") or {}).get("max_output_tokens") or 0)
        rows = native_prefill_rows(artifact)

        def series_for(metric: str) -> list[dict[str, Any]]:
            series = []
            for row in rows:
                prompt = row.get("prompt") or {}
                soft_tokens = prompt.get("soft_tokens") or {}
                series.append(
                    {
                        "label": str(row.get("case_id") or "multimodal"),
                        "modalities": "+".join(row.get("modalities") or []),
                        "expanded_tokens": int(prompt.get("expanded_tokens") or 0),
                        "soft_tokens": sum(
                            int(soft_tokens.get(modality) or 0)
                            for modality in ("image", "audio", "video")
                        ),
                        **metric_summary_from_row(row, metric),
                    }
                )
            return series

        peer_series = peer_comparison_series(artifact)
        excluded_peer = peer_exclusions(artifact)
        peer_note = (
            "Cold endpoint latency only; llama.cpp prompt cache disabled; "
            "prompt token accounting differs by engine; "
            f"{len(excluded_peer)} measured peer case(s) excluded"
        )
        return {
            "ttft_series": series_for("runner_prefill_ttft_ms"),
            "prefill_series": series_for("prefill_tok_s"),
            "peer_series": peer_series,
            "peer_exclusions": excluded_peer,
            "case_id": "matrix",
            "subtitle": "AX Engine native MLX, runner-time multimodal prefill",
            "throughput_subtitle": "AX Engine native MLX, expanded prompt includes soft tokens",
            "peer_subtitle": "OpenAI chat endpoint latency, no prompt-cache peer rows only",
            "footnote": f"{len(rows)} case(s), {repetitions} reps, max_output_tokens={max_output_tokens}",
            "peer_footnote": f"{peer_note}; {repetitions} reps, max_output_tokens={max_output_tokens}",
        }
    if schema not in (None, "ax.gemma4_image_prefill_ttft.v1"):
        raise ValueError(f"unsupported artifact schema: {schema}")

    prompt = artifact.get("prompt") or {}
    expanded_tokens = int(prompt.get("expanded_tokens") or 0)
    image_soft_tokens = int(prompt.get("image_soft_tokens") or 0)
    repetitions = int(artifact.get("repetitions") or 0)
    max_output_tokens = int(artifact.get("max_output_tokens") or 0)
    return {
        "ttft_series": [
            {
                "label": "image_single_256soft",
                "modalities": "image",
                "expanded_tokens": expanded_tokens,
                "soft_tokens": image_soft_tokens,
                **metric_summary(artifact, "runner_prefill_ttft_ms"),
            }
        ],
        "prefill_series": [
            {
                "label": "image_single_256soft",
                "modalities": "image",
                "expanded_tokens": expanded_tokens,
                "soft_tokens": image_soft_tokens,
                **metric_summary(artifact, "prefill_tok_s"),
            }
        ],
        "peer_series": [],
        "peer_exclusions": [],
        "case_id": "image_single_256soft",
        "subtitle": "AX Engine native MLX, runner-time image+text prefill",
        "throughput_subtitle": "AX Engine native MLX, expanded prompt includes image soft tokens",
        "peer_subtitle": "",
        "footnote": (
            f"{expanded_tokens} expanded tokens, {image_soft_tokens} image soft tokens, "
            f"{repetitions} reps, max_output_tokens={max_output_tokens}"
        ),
        "peer_footnote": "",
    }


def output_filenames(artifact: dict[str, Any]) -> dict[str, str]:
    if artifact.get("schema") == "ax.gemma4_multimodal_benchmark.v1":
        return MATRIX_OUTPUTS
    return LEGACY_OUTPUTS


def render_bar_chart(
    *,
    title: str,
    subtitle: str,
    unit: str,
    lower_is_better: bool,
    series: list[dict[str, Any]],
    footnote: str,
) -> str:
    width = 920 if len(series) > 6 else 720
    height = 420 if len(series) > 6 else 360
    left, right, top, bottom = 82, 170, 82, 112
    plot_w = width - left - right
    plot_h = height - top - bottom
    axis_max = nice_axis_ceiling(max(item["max"] for item in series) * 1.18)

    def fy(value: float) -> float:
        return top + plot_h - (max(0.0, min(value, axis_max)) / axis_max) * plot_h

    direction = "Lower is better" if lower_is_better else "Higher is better"
    direction_fill = "#dc2626" if lower_is_better else "#374151"
    bar_gap = 12
    bar_width = min(56, max(18, (plot_w - bar_gap * (len(series) + 1)) / max(1, len(series))))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(subtitle)} {len(series)} measured case(s).</desc>",
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

    for index, item in enumerate(series):
        median = float(item["median"])
        min_value = float(item["min"])
        max_value = float(item["max"])
        bar_x = left + bar_gap + index * (bar_width + bar_gap)
        bar_center = bar_x + bar_width / 2
        y = fy(median)
        bar_h = top + plot_h - y
        err_min_y = fy(min_value)
        err_max_y = fy(max_value)
        parts.extend(
            [
                f'<rect x="{bar_x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" '
                'rx="4" fill="#2eaf5f" fill-opacity="0.28" stroke="#176c37" stroke-width="1.6"/>',
                f'<line x1="{bar_x:.1f}" y1="{y:.1f}" x2="{bar_x + bar_width:.1f}" y2="{y:.1f}" '
                'stroke="#176c37" stroke-width="2"/>',
                f'<line x1="{bar_center:.1f}" y1="{err_max_y:.1f}" x2="{bar_center:.1f}" '
                f'y2="{err_min_y:.1f}" stroke="#111827" stroke-width="1.2"/>',
                f'<line x1="{bar_center - 8:.1f}" y1="{err_max_y:.1f}" x2="{bar_center + 8:.1f}" '
                f'y2="{err_max_y:.1f}" stroke="#111827" stroke-width="1.2"/>',
                f'<line x1="{bar_center - 8:.1f}" y1="{err_min_y:.1f}" x2="{bar_center + 8:.1f}" '
                f'y2="{err_min_y:.1f}" stroke="#111827" stroke-width="1.2"/>',
                f'<text x="{bar_center:.1f}" y="{y - 7:.1f}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="10" font-weight="700" fill="#111827">'
                f"{fmt_value(median)}</text>",
                f'<text x="{bar_center:.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="9" fill="#111827" '
                f'transform="rotate(45 {bar_center:.1f} {top + plot_h + 18:.1f})">'
                f"{html.escape(str(item['label']))}</text>",
            ]
        )

    best = min(series, key=lambda item: item["median"]) if lower_is_better else max(series, key=lambda item: item["median"])
    parts.extend(
        [
            f'<text x="{left + plot_w + 16}" y="{top + 28}" font-family="{FONT}" '
            f'font-size="11" fill="#374151">best: <tspan font-weight="700">'
            f"{html.escape(str(best['label']))}</tspan></text>",
            f'<text x="{left + plot_w + 16}" y="{top + 48}" font-family="{FONT}" '
            f'font-size="11" fill="#6b7280">{fmt_value(float(best["median"]))} {html.escape(unit)}</text>',
            f'<text x="{left + plot_w + 16}" y="{top + 70}" font-family="{FONT}" '
            f'font-size="10" fill="#6b7280">error bars: min-max</text>',
            f'<text x="{left}" y="{height - 20}" font-family="{FONT}" font-size="10" '
            f'fill="#6b7280">{html.escape(footnote)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def render_peer_comparison_chart(
    *,
    title: str,
    subtitle: str,
    series: list[dict[str, Any]],
    footnote: str,
) -> str:
    width = 920
    height = 430 if len(series) > 6 else 360
    left, right, top, bottom = 82, 180, 82, 118
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_max = [
        float(item[engine]["max"])
        for item in series
        for engine in ("ax", "peer")
    ]
    axis_max = nice_axis_ceiling(max(all_max) * 1.18)

    def fy(value: float) -> float:
        return top + plot_h - (max(0.0, min(value, axis_max)) / axis_max) * plot_h

    group_gap = 14
    group_w = max(34, (plot_w - group_gap * (len(series) + 1)) / max(1, len(series)))
    bar_gap = 4
    bar_w = max(10, min(24, (group_w - bar_gap) / 2))
    engine_order = ("peer", "ax")
    colors = {
        "peer": ("#f59e0b", "#b45309", "llama.cpp"),
        "ax": ("#2eaf5f", "#176c37", "AX Engine"),
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(subtitle)} {len(series)} measured peer case(s).</desc>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text id="title" x="{left}" y="28" font-family="{FONT}" font-size="16" '
        f'font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="{left}" y="48" font-family="{FONT}" font-size="11" fill="#4b5563">'
        f"{html.escape(subtitle)}</text>",
        f'<text x="{width - 32}" y="31" text-anchor="end" font-family="{FONT}" '
        'font-size="10" font-weight="700" fill="#dc2626">Lower is better</text>',
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

    for index, item in enumerate(series):
        group_x = left + group_gap + index * (group_w + group_gap)
        for offset, engine in enumerate(engine_order):
            median = float(item[engine]["median"])
            bar_x = group_x + offset * (bar_w + bar_gap)
            bar_center = bar_x + bar_w / 2
            y = fy(median)
            fill, stroke, _ = colors[engine]
            parts.extend(
                [
                    f'<rect x="{bar_x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                    f'height="{top + plot_h - y:.1f}" rx="3" fill="{fill}" '
                    f'fill-opacity="0.30" stroke="{stroke}" stroke-width="1.5"/>',
                    f'<line x1="{bar_x:.1f}" y1="{y:.1f}" x2="{bar_x + bar_w:.1f}" '
                    f'y2="{y:.1f}" stroke="{stroke}" stroke-width="2"/>',
                    f'<text x="{bar_center:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
                    f'font-family="{FONT}" font-size="8" font-weight="700" fill="#111827">'
                    f"{fmt_value(median)}</text>",
                ]
            )
        label_x = group_x + group_w / 2
        parts.append(
            f'<text x="{label_x:.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="9" fill="#111827" '
            f'transform="rotate(45 {label_x:.1f} {top + plot_h + 18:.1f})">'
            f"{html.escape(str(item['label']))}</text>"
        )

    legend_y = top + 30
    for index, engine in enumerate(engine_order):
        fill, stroke, label = colors[engine]
        y = legend_y + index * 22
        parts.extend(
            [
                f'<rect x="{left + plot_w + 16}" y="{y - 10}" width="12" height="12" '
                f'rx="2" fill="{fill}" fill-opacity="0.30" stroke="{stroke}"/>',
                f'<text x="{left + plot_w + 36}" y="{y}" font-family="{FONT}" '
                f'font-size="11" fill="#374151">{html.escape(label)}</text>',
            ]
        )

    parts.extend(
        [
            f'<text x="{left + plot_w + 16}" y="{legend_y + 60}" font-family="{FONT}" '
            f'font-size="10" fill="#6b7280">unit: endpoint ms</text>',
            f'<text x="{left + plot_w + 16}" y="{legend_y + 78}" font-family="{FONT}" '
            f'font-size="10" fill="#6b7280">not throughput</text>',
            f'<text x="{left}" y="{height - 20}" font-family="{FONT}" font-size="10" '
            f'fill="#6b7280">{html.escape(footnote)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def render(artifact_path: Path, assets_dir: Path) -> list[Path]:
    artifact = json.loads(artifact_path.read_text())
    inputs = chart_inputs(artifact)
    output_names = output_filenames(artifact)
    ttft = inputs["ttft_series"]
    prefill = inputs["prefill_series"]
    peer = inputs["peer_series"]

    assets_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        (
            output_names["ttft"],
            render_bar_chart(
                title="Gemma 4 12B multimodal prefill TTFT",
                subtitle=inputs["subtitle"],
                unit="ms",
                lower_is_better=True,
                series=ttft,
                footnote=inputs["footnote"],
            ),
        ),
        (
            output_names["prefill"],
            render_bar_chart(
                title="Gemma 4 12B multimodal prefill throughput",
                subtitle=inputs["throughput_subtitle"],
                unit="tok/s",
                lower_is_better=False,
                series=prefill,
                footnote=inputs["footnote"],
            ),
        ),
    ]
    if peer and "peer" in output_names:
        outputs.append(
            (
                output_names["peer"],
                render_peer_comparison_chart(
                    title="Gemma 4 12B multimodal chat latency",
                    subtitle=inputs["peer_subtitle"],
                    series=peer,
                    footnote=inputs["peer_footnote"],
                ),
            )
        )

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
            "2026-06-09-gemma4-12b-multimodal-matrix.json"
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
