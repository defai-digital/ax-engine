#!/usr/bin/env python3
"""Render Gemma 4 assistant-MTP box-and-whisker SVG charts for the README.

Companion to bench_gemma4_assistant_mtp.py. Reads the per-suite artifacts a run
writes (``<model>/<suite>/{direct,mtp,mtp-ngram}.json``) and renders grouped
box-and-whisker SVGs in the same visual style as the Qwen fair-MTP charts, one
chart per model x metric. Decode charts also include a same-prompt AX direct
baseline; other metrics stay focused on AX assistant-MTP and AX assistant
MTP+n-gram (Gemma has no MTPLX reference). Groups are the prompt suites.

Per-run distributions come from each prompt case's ``trials`` array
(decode / prefill / ttft); accept rate is per prompt case (accepted/drafted from
the ``ax_mlx_gemma4_assistant_mtp`` telemetry).

Usage:
  python3 scripts/render_gemma4_assistant_mtp_charts.py \\
      --results-dir benchmarks/results/gemma4-assistant-mtp/2026-06-06-gemma4-26b-31b-assistant-mtp \\
      --assets-dir docs/assets
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Engine keys written by bench_gemma4_assistant_mtp.py.
ENGINE_DIRECT = "ax_engine_mlx"
ENGINE_MTP = "ax_engine_gemma4_assistant_mtp"
ENGINE_NGRAM = "ax_engine_gemma4_assistant_mtp_ngram"
DEFAULT_ENGINES = [ENGINE_MTP, ENGINE_NGRAM]
DECODE_ENGINES = [ENGINE_DIRECT, ENGINE_MTP, ENGINE_NGRAM]
ENGINE_LABELS = {
    ENGINE_DIRECT: "AX direct (same prompts)",
    ENGINE_MTP: "AX assistant MTP",
    ENGINE_NGRAM: "AX assistant MTP+n-gram",
}
ENGINE_COLORS = {ENGINE_DIRECT: "#d97706", ENGINE_MTP: "#2eaf5f", ENGINE_NGRAM: "#137a3d"}

MODELS = [
    ("12b-4bit-ffn4", "Gemma 4 12B 4-bit-FFN"),
    ("12b-4bit", "Gemma 4 12B 4-bit"),
    ("26b-a4b-4bit", "Gemma 4 26B A4B 4-bit"),
    ("31b-4bit", "Gemma 4 31B 4-bit"),
]
# Maps a model key to the compact slug used in chart filenames.
MODEL_SHORT = {
    "12b-4bit-ffn4": "12b",
    "12b-4bit": "12b",
    "26b-a4b-4bit": "26b",
    "31b-4bit": "31b",
}
SUITES = [
    ("flappy", "flappy"),
    ("long_code", "long_code"),
    ("python_modules_long", "python_modules_long"),
]
FOOTNOTE = "Apple M5 Max · gate 0.90 first / 0.999 deep · GPU exact · T=0.6/top_p=0.95/top_k=20"
FONT = "Inter,Segoe UI,Arial,sans-serif"


@dataclass(frozen=True)
class BoxStats:
    values: tuple[float, ...]
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float


def percentile(sorted_values: list[float], p: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def box_stats(values: list[float]) -> BoxStats:
    ordered = sorted(values)
    return BoxStats(
        values=tuple(values),
        minimum=ordered[0],
        q1=percentile(ordered, 0.25),
        median=percentile(ordered, 0.50),
        q3=percentile(ordered, 0.75),
        maximum=ordered[-1],
    )


def nice_axis_ceiling(max_value: float) -> float:
    if not math.isfinite(max_value) or max_value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(max_value))
    scaled = max_value / magnitude
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        if scaled <= step:
            return step * magnitude
    return 10.0 * magnitude


def axis_label(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.0f}%"
    if value >= 1000:
        return f"{value / 1000:g}k"
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def point_label(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "ms":
        return f"{value:.0f}"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def write_box_whisker_svg(
    path: Path,
    *,
    title: str,
    subtitle: str,
    engines: list[str],
    unit: str,
    direction_label: str,
    groups: list[dict[str, Any]],
    width: int = 760,
    axis_min: float = 0.0,
    axis_max: float | None = None,
    lower_is_better: bool = False,
    footnote: str = "",
) -> None:
    height = 460
    left = 64
    right = 200
    header_right = 34
    top = 86
    bottom = 106
    plot_right = width - right
    plot_w = plot_right - left
    plot_h = height - top - bottom
    group_w = plot_w / max(len(groups), 1)
    gap = 8.0
    box_w = min(34.0, max(12.0, (group_w - 56 - gap * (len(engines) - 1)) / max(len(engines), 1)))

    stats_by_key: dict[tuple[int, str], BoxStats] = {}
    all_values: list[float] = []
    all_medians: list[float] = []
    for gi, group in enumerate(groups):
        for engine in engines:
            values = [float(v) for v in group["values"].get(engine, []) if v is not None]
            if not values:
                continue
            stats = box_stats(values)
            stats_by_key[(gi, engine)] = stats
            all_values.extend(values)
            all_medians.append(stats.median)

    max_value = axis_max if axis_max is not None else nice_axis_ceiling(max(all_values or [1.0]) * 1.08)
    if max_value <= axis_min:
        max_value = axis_min + 1.0
    best_value = min(all_medians) if lower_is_better and all_medians else max(all_medians or [0.0])

    def fy(value: float) -> float:
        clamped = max(axis_min, min(value, max_value))
        return top + plot_h - ((clamped - axis_min) / (max_value - axis_min)) * plot_h

    def engine_centers(group_x: float, count: int) -> list[float]:
        if count <= 1:
            return [group_x + group_w / 2]
        side_pad = min(76.0, max(44.0, group_w * 0.12))
        preferred_span = max(0.0, group_w - side_pad * 2)
        minimum_span = (box_w + gap) * (count - 1)
        max_span = max(0.0, group_w - box_w)
        span = min(max(preferred_span, minimum_span), max_span)
        start_x = group_x + (group_w - span) / 2
        step = span / (count - 1) if count > 1 else 0.0
        return [start_x + step * i for i in range(count)]

    direction_fill = "#dc2626"
    best_line_label = "lowest median" if lower_is_better else "highest median"
    best_side_label = "lowest" if lower_is_better else "highest"
    best_label = f"{best_side_label}: {point_label(best_value, unit)}"
    engine_desc = ", ".join(ENGINE_LABELS[e] for e in engines)
    group_desc = ", ".join(group["label"] for group in groups)
    unit_w = max(48, len(unit) * 7 + 24)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>Grouped box-and-whisker plot comparing {html.escape(engine_desc)} "
        f"across {html.escape(group_desc)} prompt suites.</desc>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="24" y="24" font-family="{FONT}" font-size="16" font-weight="700" '
        f'fill="#111827">{html.escape(title)}</text>',
        f'<text x="24" y="46" font-family="{FONT}" font-size="11" fill="#4b5563">'
        f"{html.escape(subtitle)}</text>",
        *(
            [
                f'<text x="24" y="62" font-family="{FONT}" font-size="10" '
                f'fill="#6b7280">{html.escape(footnote)}</text>'
            ]
            if footnote
            else []
        ),
        f'<rect x="{width - header_right - unit_w}" y="13" width="{unit_w}" height="22" '
        f'rx="11" fill="#eef2ff" stroke="#c7d2fe"/>',
        f'<text x="{width - header_right - unit_w / 2:.1f}" y="28" text-anchor="middle" '
        f'font-family="{FONT}" font-size="10" font-weight="700" fill="#3730a3">'
        f"{html.escape(unit)}</text>",
        f'<text x="{width - header_right}" y="52" text-anchor="end" font-family="{FONT}" '
        f'font-size="10" font-weight="700" fill="{direction_fill}">'
        f"{html.escape(direction_label)}</text>",
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="6" '
        f'fill="#ffffff" stroke="#dbe3ef"/>',
    ]

    for i in range(5):
        value = axis_min + (max_value - axis_min) * i / 4
        y = fy(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 3:.1f}" text-anchor="end" font-family="{FONT}" '
            f'font-size="11" fill="#6b7280">{axis_label(value, unit)}</text>'
        )

    if best_value > 0:
        best_y = fy(best_value)
        parts.append(
            f'<line x1="{left}" y1="{best_y:.1f}" x2="{plot_right}" y2="{best_y:.1f}" '
            f'stroke="#dc2626" stroke-width="1.2" stroke-dasharray="1 4" stroke-linecap="round"/>'
        )
        parts.append(
            f'<text x="{plot_right + 8}" y="{max(top + 11, best_y - 5):.1f}" text-anchor="start" '
            f'font-family="{FONT}" font-size="11" font-weight="700" fill="#dc2626" '
            f'data-label="{html.escape(best_line_label)}">{html.escape(best_label)}</text>'
        )

    dot_slots = 9
    dot_jitter = [(-0.36 + 0.72 * i / (dot_slots - 1)) * box_w for i in range(dot_slots)]
    for gi, group in enumerate(groups):
        group_x = left + group_w * gi
        centers = engine_centers(group_x, len(engines))
        parts.append(
            f'<text x="{group_x + group_w / 2:.1f}" y="{height - 62}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="11" font-weight="700" fill="#111827">'
            f'{html.escape(group["label"])}</text>'
        )
        for ei, engine in enumerate(engines):
            stats = stats_by_key.get((gi, engine))
            if stats is None:
                continue
            x = centers[ei]
            color = ENGINE_COLORS[engine]
            y_min = fy(stats.minimum)
            y_q1 = fy(stats.q1)
            y_med = fy(stats.median)
            y_q3 = fy(stats.q3)
            y_max = fy(stats.maximum)
            box_top = min(y_q1, y_q3)
            box_h = max(abs(y_q3 - y_q1), 1.0)
            cap_left = x - box_w * 0.36
            cap_right = x + box_w * 0.36
            box_left = x - box_w / 2
            for vi, value in enumerate(stats.values):
                parts.append(
                    f'<circle cx="{x + dot_jitter[vi % len(dot_jitter)]:.1f}" '
                    f'cy="{fy(value):.1f}" r="1.45" fill="{color}" fill-opacity="0.46"/>'
                )
            parts.extend(
                [
                    f'<line x1="{x:.1f}" y1="{y_max:.1f}" x2="{x:.1f}" y2="{y_min:.1f}" '
                    f'stroke="{color}" stroke-opacity="0.86" stroke-width="1.8"/>',
                    f'<line x1="{cap_left:.1f}" y1="{y_max:.1f}" x2="{cap_right:.1f}" y2="{y_max:.1f}" '
                    f'stroke="{color}" stroke-opacity="0.86" stroke-width="1.8"/>',
                    f'<line x1="{cap_left:.1f}" y1="{y_min:.1f}" x2="{cap_right:.1f}" y2="{y_min:.1f}" '
                    f'stroke="{color}" stroke-opacity="0.86" stroke-width="1.8"/>',
                    f'<rect x="{box_left:.1f}" y="{box_top:.1f}" width="{box_w:.1f}" '
                    f'height="{box_h:.1f}" rx="2" fill="{color}" fill-opacity="0.18" '
                    f'stroke="{color}" stroke-opacity="0.9" stroke-width="1.8"/>',
                    f'<line x1="{box_left:.1f}" y1="{y_med:.1f}" x2="{box_left + box_w:.1f}" '
                    f'y2="{y_med:.1f}" stroke="{color}" stroke-opacity="0.96" stroke-width="2.6"/>',
                    f'<text x="{box_left + box_w + 6:.1f}" y="{y_med + 4:.1f}" text-anchor="start" '
                    f'font-family="{FONT}" font-size="11" font-weight="700" fill="#111827" '
                    f'stroke="#ffffff" stroke-width="3" paint-order="stroke">'
                    f"{html.escape(point_label(stats.median, unit))}</text>",
                ]
            )

    legend_y = height - 18
    legend_step = max(150.0, (width - left - right) / max(len(engines), 1))
    legend_x = left
    for engine in engines:
        color = ENGINE_COLORS[engine]
        parts.append(
            f'<rect x="{legend_x:.1f}" y="{legend_y - 9}" width="10" height="10" rx="2" '
            f'fill="{color}" fill-opacity="0.72"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14:.1f}" y="{legend_y}" font-family="{FONT}" '
            f'font-size="10" fill="#374151">{html.escape(ENGINE_LABELS[engine])}</text>'
        )
        legend_x += legend_step

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


# Candidate per-suite artifact basenames per mode, newest naming first.
# bench_gemma4_assistant_mtp.py writes profile-keyed names
# (assistant_mtp_default / assistant_mtp_ngram_default); older runs wrote the
# bare mode name (mtp / mtp-ngram). Try both so this renders new and archived
# result trees alike.
MODE_FILE_CANDIDATES = {
    "direct": ("direct",),
    "mtp": ("assistant_mtp_default", "mtp"),
    "mtp-ngram": ("assistant_mtp_ngram_default", "mtp-ngram"),
}


def mode_artifact_path(results_dir: Path, model_key: str, suite_key: str, mode_file: str) -> Path | None:
    suite_dir = results_dir / model_key / suite_key
    candidates = MODE_FILE_CANDIDATES.get(mode_file, (mode_file,))
    return next(
        (suite_dir / f"{name}.json" for name in candidates if (suite_dir / f"{name}.json").exists()),
        None,
    )


def engine_rows(results_dir: Path, model_key: str, suite_key: str, mode_file: str, engine: str) -> list[dict[str, Any]]:
    suite_dir = results_dir / model_key / suite_key
    candidates = MODE_FILE_CANDIDATES.get(mode_file, (mode_file,))
    path = mode_artifact_path(results_dir, model_key, suite_key, mode_file)
    if path is None:
        raise FileNotFoundError(suite_dir / f"{candidates[0]}.json")
    payload = json.loads(path.read_text())
    return [r for r in payload.get("results", []) if r.get("engine") == engine]


def trial_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        for trial in row.get("trials", []) or []:
            v = trial.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
    return out


def accept_values(rows: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for row in rows:
        tel = row.get("ax_mlx_gemma4_assistant_mtp") or {}
        drafted = int(tel.get("ax_mlx_gemma4_assistant_mtp_draft_tokens", 0) or 0)
        accepted = int(tel.get("ax_mlx_gemma4_assistant_mtp_accepted_tokens", 0) or 0)
        if drafted > 0:
            out.append(accepted / drafted * 100.0)
    return out


def build_groups(results_dir: Path, model_key: str, metric: str, engines: list[str]) -> list[dict[str, Any]]:
    mode_for_engine = {ENGINE_DIRECT: "direct", ENGINE_MTP: "mtp", ENGINE_NGRAM: "mtp-ngram"}
    trial_key = {"decode": "decode_tok_s", "prefill": "prefill_tok_s", "ttft": "ttft_ms"}.get(metric)
    groups: list[dict[str, Any]] = []
    for suite_key, suite_label in SUITES:
        values: dict[str, list[float]] = {}
        for engine in engines:
            rows = engine_rows(results_dir, model_key, suite_key, mode_for_engine[engine], engine)
            values[engine] = accept_values(rows) if metric == "accept" else trial_values(rows, trial_key)
        groups.append({"label": suite_label, "values": values})
    return groups


def engines_for_metric(results_dir: Path, model_key: str, metric: str) -> list[str]:
    if metric != "decode":
        return DEFAULT_ENGINES
    direct_present = [
        mode_artifact_path(results_dir, model_key, suite_key, "direct") is not None
        for suite_key, _suite_label in SUITES
    ]
    if all(direct_present):
        return DECODE_ENGINES
    if any(direct_present):
        missing = [
            suite_key
            for (suite_key, _suite_label), present in zip(SUITES, direct_present, strict=True)
            if not present
        ]
        raise FileNotFoundError(
            f"{model_key}: partial direct baseline; missing direct artifact(s) for "
            + ", ".join(missing)
        )
    return DEFAULT_ENGINES


METRICS = [
    ("decode", "decode-tok-s", "Decode throughput", "tok/s", "Higher is better", False, 0.0, None),
    ("accept", "accept-rate", "Assistant accept rate", "%", "Higher is better", False, 90.0, 100.0),
    ("prefill", "prefill-tok-s", "Prefill throughput", "tok/s", "Higher is better", False, 0.0, None),
    ("ttft", "ttft-ms", "Time to first token", "ms", "Lower is better", True, 0.0, None),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, default=REPO_ROOT / "docs" / "assets")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated subset of model keys to render. Defaults to every "
        f"model with artifacts present. Choices: {', '.join(k for k, _ in MODELS)}",
    )
    args = parser.parse_args()

    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    models = [(k, label) for k, label in MODELS if not selected or k in selected]

    args.assets_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for model_key, model_label in models:
        for metric, slug, title_metric, unit, direction, lower_is_better, axis_min, axis_max in METRICS:
            engines = engines_for_metric(args.results_dir, model_key, metric)
            groups = build_groups(args.results_dir, model_key, metric, engines)
            short = MODEL_SHORT[model_key]
            out = args.assets_dir / f"perf-gemma4-assistant-mtp-{short}-{slug}.svg"
            write_box_whisker_svg(
                out,
                title=f"{model_label} — {title_metric}",
                subtitle="All suites | box=IQR | whiskers=min/max | dots=runs",
                engines=engines,
                unit=unit,
                direction_label=direction,
                groups=groups,
                axis_min=axis_min,
                axis_max=axis_max,
                lower_is_better=lower_is_better,
                footnote=FOOTNOTE,
            )
            written.append(str(out))
    print("Wrote:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
