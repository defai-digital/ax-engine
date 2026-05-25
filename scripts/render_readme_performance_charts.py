#!/usr/bin/env python3
"""Render README performance SVG charts from benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import check_readme_performance_artifacts as readme_artifacts

LABEL_TO_SLUG = {
    label: slug for slug, label in readme_artifacts.ARTIFACT_LABELS.items()
}

PROMPT_TOKENS = (128, 512, 2048)

SERIES = [
    ("llama_cpp_metal", "llama.cpp", "#f97316", "#c2410c"),
    ("mlx_lm", "mlx_lm", "#f2b705", "#9a6a00"),
    ("ax_engine_mlx", "ax_engine", "#2eaf5f", "#176c37"),
    ("ax_engine_mlx_ngram_accel", "ax_engine + n-gram", "#137a3d", "#0b4f28"),
]

WIDTH = 460
HEIGHT = 276
LEFT = 52
RIGHT = 442
TOP = 46
BOTTOM = 206
FONT = "Inter,Segoe UI,Arial,sans-serif"
RED = "#dc2626"


@dataclass(frozen=True)
class SeriesStats:
    engine: str
    label: str
    color: str
    dot_color: str
    values: tuple[float, ...]
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float


@dataclass(frozen=True)
class PromptGroupStats:
    prompt_tokens: int
    series: tuple[SeriesStats, ...]


@dataclass(frozen=True)
class ChartSpec:
    title: str
    subtitle: str | None
    unit: str
    direction_label: str
    output_slug: str
    metric: str
    series_engines: tuple[str, ...]


@dataclass(frozen=True)
class MtpBenchmarkRow:
    model_bundle: str
    suite: str
    ax_depth_cap: int
    ax_mtp_tok_s: float
    ax_accept_rate: float
    mtplx_tok_s: float
    mtplx_depth: int
    mtplx_accept_rate: float


CHARTS = [
    ChartSpec(
        title="Prefill rate",
        subtitle=None,
        unit="tok/s",
        direction_label="Higher is better",
        output_slug="prefill",
        metric="prefill",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
    ),
    ChartSpec(
        title="Decode rate",
        subtitle="AX n-gram is the default policy",
        unit="tok/s",
        direction_label="Higher is better",
        output_slug="decode",
        metric="decode",
        series_engines=(
            "llama_cpp_metal",
            "mlx_lm",
            "ax_engine_mlx",
            "ax_engine_mlx_ngram_accel",
        ),
    ),
    ChartSpec(
        title="TTFT",
        subtitle=None,
        unit="ms",
        direction_label="Lower is better",
        output_slug="ttft",
        metric="ttft",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
    ),
]


MTP_WIDTH = 380
MTP_HEIGHT = 238
MTP_LEFT = 52
MTP_RIGHT = 354
MTP_TOP = 50
MTP_BOTTOM = 168
MTP_AX_COLOR = "#2eaf5f"
MTP_AX_DOT = "#176c37"
MTP_MTPLX_COLOR = "#f2b705"
MTP_MTPLX_DOT = "#9a6a00"

MTP_CHART_OUTPUTS = {
    ("speed", "tok_s"): "perf-mtp-speed-tok-s.svg",
    ("speed", "accept_rate"): "perf-mtp-speed-accept-rate.svg",
    ("quality", "tok_s"): "perf-mtp-quality-tok-s.svg",
    ("quality", "accept_rate"): "perf-mtp-quality-accept-rate.svg",
}


class ChartError(RuntimeError):
    pass


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ChartError("cannot calculate percentile of an empty series")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def summarize(
    values: list[float],
    engine: str,
    label: str,
    color: str,
    dot_color: str,
) -> SeriesStats:
    ordered = sorted(values)
    return SeriesStats(
        engine=engine,
        label=label,
        color=color,
        dot_color=dot_color,
        values=tuple(values),
        minimum=ordered[0],
        q1=percentile(ordered, 0.25),
        median=percentile(ordered, 0.50),
        q3=percentile(ordered, 0.75),
        maximum=ordered[-1],
    )


def metric_median(row: dict[str, Any], key: str) -> float:
    metric = row.get(key)
    if isinstance(metric, dict) and isinstance(metric.get("median"), (int, float)):
        return float(metric["median"])
    if isinstance(metric, (int, float)):
        return float(metric)
    raise ChartError(f"missing numeric median for {key} in {row.get('engine', '<unknown>')}")


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def strip_markdown_cell(cell: str) -> str:
    without_links = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cell)
    return without_links.replace("**", "").replace("`", "").strip()


def parse_numeric_cell(cell: str) -> float:
    cleaned = strip_markdown_cell(cell).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ChartError(f"metric cell has no numeric value: {cell!r}")
    return float(match.group(0))


def parse_percent_cell(cell: str) -> float:
    cleaned = strip_markdown_cell(cell)
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ChartError(f"percent cell has no numeric value: {cell!r}")
    value = float(match.group(0))
    return value / 100.0 if "%" in cleaned else value


def parse_int_cell(cell: str) -> int:
    value = parse_numeric_cell(cell)
    if not float(value).is_integer():
        raise ChartError(f"integer cell has a non-integer value: {cell!r}")
    return int(value)


def readme_model_slugs(readme: Path) -> list[str]:
    slugs: list[str] = []
    seen: set[tuple[str, str]] = set()
    for metric in readme_artifacts.parse_readme_metrics(readme.resolve()):
        label = (metric.model, metric.quantization)
        if label in seen:
            continue
        slug = LABEL_TO_SLUG.get(label)
        if slug is None:
            raise ChartError(f"README model has no artifact slug mapping: {label}")
        slugs.append(slug)
        seen.add(label)
    if not slugs:
        raise ChartError("README performance tables contain no chartable models")
    return slugs


def load_rows(results_dir: Path, slugs: list[str], *, required: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for slug in slugs:
        path = results_dir / f"{slug}.json"
        if not path.exists():
            if required:
                missing.append(str(path))
            continue
        payload = json.loads(path.read_text())
        for row in payload.get("results", []):
            if row.get("prompt_tokens") in PROMPT_TOKENS:
                row_copy = dict(row)
                row_copy["_slug"] = slug
                rows.append(row_copy)
    if missing:
        raise ChartError("missing README benchmark artifacts:\n" + "\n".join(missing))
    return rows


def load_composite_rows(readme: Path, metric: str, slugs: list[str]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    sources = readme_artifacts.default_artifact_sources(readme.resolve())
    series_engines = {engine for engine, _label, _color, _dot_color in SERIES}

    for source in sources:
        if source.include_tables is not None and metric not in source.include_tables:
            continue
        source_rows = load_rows(source.artifact_dir, slugs, required=False)
        for row in source_rows:
            engine = row.get("engine")
            if engine not in series_engines:
                continue
            if source.include_engines is not None and engine not in source.include_engines:
                continue
            prompt_tokens = row.get("prompt_tokens")
            if prompt_tokens not in PROMPT_TOKENS:
                continue
            if (
                source.include_prompt_tokens is not None
                and prompt_tokens not in source.include_prompt_tokens
            ):
                continue
            generation_tokens = row.get("generation_tokens")
            if not isinstance(generation_tokens, int):
                raise ChartError(f"missing generation_tokens for {engine}")
            merged[(str(row["_slug"]), str(engine), int(prompt_tokens), generation_tokens)] = row

    return list(merged.values())


def series_for_chart(spec: ChartSpec) -> list[tuple[str, str, str, str]]:
    series_by_engine = {
        engine: (engine, label, color, dot_color)
        for engine, label, color, dot_color in SERIES
    }
    missing = [engine for engine in spec.series_engines if engine not in series_by_engine]
    if missing:
        raise ChartError(
            f"{spec.metric} chart references unknown series: {', '.join(missing)}"
        )
    return [series_by_engine[engine] for engine in spec.series_engines]


def infer_llama_results_dir(repo_root: Path) -> Path:
    root = repo_root / "benchmarks" / "results" / "llama-cpp-metal"
    if not root.exists():
        raise ChartError(
            f"could not infer llama.cpp Metal artifact directory; {root} does not exist"
        )
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "sweep_results.json").exists()
    ]
    if not candidates:
        raise ChartError(
            f"could not infer llama.cpp Metal artifact directory; no sweep_results.json under {root}"
        )
    complete = [
        path
        for path in candidates
        if all(
            (path / f"{slug}.json").exists()
            for slug in readme_model_slugs(repo_root / "README.md")
        )
    ]
    if complete:
        return max(complete, key=lambda path: path.name).resolve()
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def collect_values(
    rows: list[dict[str, Any]],
    spec: ChartSpec,
    expected_model_count: int,
) -> list[PromptGroupStats]:
    groups: list[PromptGroupStats] = []
    for prompt_tokens in PROMPT_TOKENS:
        series_stats: list[SeriesStats] = []
        for engine, label, color, dot_color in series_for_chart(spec):
            values: list[float] = []
            for row in rows:
                if row.get("engine") != engine or row.get("prompt_tokens") != prompt_tokens:
                    continue
                if spec.metric == "prefill":
                    values.append(metric_median(row, "prefill_tok_s"))
                elif spec.metric == "decode":
                    values.append(metric_median(row, "decode_tok_s"))
                elif spec.metric == "ttft":
                    if engine in {"llama_cpp_metal", "mlx_lm"}:
                        values.append(prompt_tokens / metric_median(row, "prefill_tok_s") * 1000)
                    else:
                        values.append(metric_median(row, "ttft_ms"))
                else:
                    raise ChartError(f"unknown metric: {spec.metric}")
            if len(values) != expected_model_count:
                raise ChartError(
                    f"{spec.metric} chart expected {expected_model_count} values "
                    f"for {engine} at {prompt_tokens} prompt tokens, found {len(values)}"
                )
            series_stats.append(summarize(values, engine, label, color, dot_color))
        groups.append(PromptGroupStats(prompt_tokens, tuple(series_stats)))
    return groups


def y_scale(value: float, axis_max: float) -> float:
    clamped = max(0.0, min(value, axis_max))
    return BOTTOM - (clamped / axis_max) * (BOTTOM - TOP)


def nice_axis_ceiling(value: float) -> float:
    if value <= 0:
        raise ChartError("chart axis requires a positive maximum value")
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for candidate in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def chart_axis_max(group: PromptGroupStats) -> float:
    largest_value = max(stat.maximum for stat in group.series)
    return nice_axis_ceiling(largest_value * 1.05)


def short_number(value: float) -> str:
    if value >= 1000:
        compact = value / 1000
        if compact.is_integer():
            return f"{compact:.0f}k"
        return f"{compact:.1f}k"
    if value.is_integer():
        return f"{value:.0f}"
    return f"{value:.1f}"


def series_x_positions(series_count: int) -> list[float]:
    if series_count < 1:
        raise ChartError("chart requires at least one series")
    plot_width = RIGHT - LEFT
    step = plot_width / series_count
    return [LEFT + step * (index + 0.5) for index in range(series_count)]


def box_width(series_count: int) -> float:
    plot_width = RIGHT - LEFT
    return min(34.0, plot_width / max(series_count, 1) * 0.38)


def best_median(group: PromptGroupStats, spec: ChartSpec) -> float:
    medians = [stat.median for stat in group.series]
    if spec.metric == "ttft":
        return min(medians)
    return max(medians)


def chart_output_name(spec: ChartSpec, prompt_tokens: int) -> str:
    return f"perf-{spec.output_slug}-{prompt_tokens}-box-whisker.svg"


def render_chart(spec: ChartSpec, group: PromptGroupStats) -> str:
    axis_max = chart_axis_max(group)
    series_description = ", ".join(stat.label for stat in group.series)
    best_median_description = "lowest median" if spec.metric == "ttft" else "highest median"
    title = f"{spec.title} - {group.prompt_tokens} tok"
    x_positions = series_x_positions(len(group.series))
    current_box_width = box_width(len(group.series))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f"<title>{escape(title)}</title>",
        (
            f"<desc>Box-and-Whisker Plot comparing {escape(series_description)} "
            f"at {group.prompt_tokens} prompt tokens across README benchmark rows "
            f"on a linear y-axis. A red dotted horizontal line marks the "
            f"{best_median_description} in this chart.</desc>"
        ),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        f'<text x="52" y="22" font-family="{FONT}" font-size="16" font-weight="700" fill="#111827">{escape(title)}</text>',
    ]
    if spec.subtitle:
        lines.append(
            f'<text x="52" y="40" font-family="{FONT}" font-size="10" fill="#6b7280">{escape(spec.subtitle)}</text>'
        )
    lines.extend(
        [
            f'<text x="442" y="22" text-anchor="end" font-family="{FONT}" font-size="10" fill="#6b7280">{escape(spec.unit)}</text>',
            f'<text x="442" y="40" text-anchor="end" font-family="{FONT}" font-size="10" font-weight="700" fill="#374151">{escape(spec.direction_label)}</text>',
        ]
    )

    for value in (0.0, axis_max / 2, axis_max):
        y = y_scale(value, axis_max)
        lines.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{RIGHT}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(
            f'<text x="44" y="{y + 3:.1f}" text-anchor="end" font-family="{FONT}" font-size="10" fill="#6b7280">{short_number(value)}</text>'
        )

    best_y = y_scale(best_median(group, spec), axis_max)
    lines.append(
        f'<line x1="{LEFT}" y1="{best_y:.1f}" x2="{RIGHT}" y2="{best_y:.1f}" stroke="{RED}" stroke-width="1.2" stroke-dasharray="1 4" stroke-linecap="round"/>'
    )

    dot_offsets = (
        -current_box_width * 0.24,
        -current_box_width * 0.12,
        0,
        current_box_width * 0.12,
        current_box_width * 0.24,
    )
    for stat, x in zip(group.series, x_positions):
        y_min = y_scale(stat.minimum, axis_max)
        y_q1 = y_scale(stat.q1, axis_max)
        y_med = y_scale(stat.median, axis_max)
        y_q3 = y_scale(stat.q3, axis_max)
        y_max = y_scale(stat.maximum, axis_max)
        box_y = min(y_q1, y_q3)
        box_h = max(abs(y_q3 - y_q1), 1.0)
        cap_left = x - current_box_width * 0.36
        cap_right = x + current_box_width * 0.36
        box_left = x - current_box_width / 2
        lines.extend(
            [
                f'<line x1="{x:g}" y1="{y_max:.1f}" x2="{x:g}" y2="{y_min:.1f}" stroke="{stat.color}" stroke-width="1.7"/>',
                f'<line x1="{cap_left:g}" y1="{y_max:.1f}" x2="{cap_right:g}" y2="{y_max:.1f}" stroke="{stat.color}" stroke-width="1.7"/>',
                f'<line x1="{cap_left:g}" y1="{y_min:.1f}" x2="{cap_right:g}" y2="{y_min:.1f}" stroke="{stat.color}" stroke-width="1.7"/>',
                f'<rect x="{box_left:g}" y="{box_y:.1f}" width="{current_box_width:g}" height="{box_h:.1f}" rx="3" fill="{stat.color}" fill-opacity="0.18" stroke="{stat.color}" stroke-width="1.7"/>',
                f'<line x1="{box_left:g}" y1="{y_med:.1f}" x2="{box_left + current_box_width:g}" y2="{y_med:.1f}" stroke="{stat.color}" stroke-width="2.4"/>',
                f'<text x="{x:g}" y="226" text-anchor="middle" font-family="{FONT}" font-size="9" fill="#111827">{escape(stat.label)}</text>',
                f'<text x="{x:g}" y="242" text-anchor="middle" font-family="{FONT}" font-size="9" fill="#6b7280">med {short_number(stat.median)}</text>',
            ]
        )
        for value_index, value in enumerate(stat.values):
            dot_x = x + dot_offsets[value_index % len(dot_offsets)]
            dot_y = y_scale(value, axis_max)
            lines.append(
                f'<circle cx="{dot_x:g}" cy="{dot_y:.1f}" r="1.5" fill="{stat.dot_color}" fill-opacity="0.9"/>'
            )

    lines.append("</svg>")
    return "".join(lines) + "\n"


def load_mtp_rows(performance_doc: Path) -> list[MtpBenchmarkRow]:
    lines = performance_doc.read_text().splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("| Model bundle | Suite | AX depth cap | AX MTP"):
            start = index
            break
    if start is None:
        raise ChartError(f"could not find MTP results table in {performance_doc}")

    table_lines: list[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    if len(table_lines) < 3:
        raise ChartError(f"MTP results table is empty in {performance_doc}")

    rows: list[MtpBenchmarkRow] = []
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        if len(cells) != 9:
            raise ChartError(f"unexpected MTP table row shape: {line}")
        raw_bundle = strip_markdown_cell(cells[0])
        model_bundle = raw_bundle.split("(", 1)[0].strip()
        rows.append(
            MtpBenchmarkRow(
                model_bundle=model_bundle,
                suite=strip_markdown_cell(cells[1]),
                ax_depth_cap=parse_int_cell(cells[2]),
                ax_mtp_tok_s=parse_numeric_cell(cells[3]),
                ax_accept_rate=parse_percent_cell(cells[4]),
                mtplx_tok_s=parse_numeric_cell(cells[5]),
                mtplx_depth=parse_int_cell(cells[6]),
                mtplx_accept_rate=parse_percent_cell(cells[7]),
            )
        )
    if not rows:
        raise ChartError(f"MTP results table has no data rows in {performance_doc}")
    return rows


def mtp_y_scale(value: float, axis_max: float) -> float:
    clamped = max(0.0, min(value, axis_max))
    return MTP_BOTTOM - (clamped / axis_max) * (MTP_BOTTOM - MTP_TOP)


def mtp_row_key(row: MtpBenchmarkRow) -> str:
    return row.model_bundle.lower().strip()


def mtp_depth_key(row: MtpBenchmarkRow) -> int:
    if row.ax_depth_cap != row.mtplx_depth:
        raise ChartError(
            f"MTP row mixes AX depth={row.ax_depth_cap} and MTPLX depth={row.mtplx_depth}"
        )
    return row.ax_depth_cap


def mtp_metric_value(row: MtpBenchmarkRow, engine: str, metric: str) -> float:
    if metric == "tok_s":
        return row.mtplx_tok_s if engine == "mtplx" else row.ax_mtp_tok_s
    if metric == "accept_rate":
        value = row.mtplx_accept_rate if engine == "mtplx" else row.ax_accept_rate
        return value * 100.0
    raise ChartError(f"unknown MTP chart metric: {metric}")


def mtp_metric_label(value: float, metric: str) -> str:
    if metric == "accept_rate":
        return f"{value:.1f}%"
    return f"{value:.1f}"


def render_mtp_metric_chart(rows: list[MtpBenchmarkRow], metric: str) -> str:
    rows = sorted(rows, key=mtp_depth_key)
    if not rows:
        raise ChartError("MTP metric chart requires at least one row")
    model_bundle = rows[0].model_bundle
    suite = rows[0].suite
    if any(row.model_bundle != model_bundle or row.suite != suite for row in rows):
        raise ChartError("MTP metric chart rows must share model bundle and suite")
    title_metric = "tok/s" if metric == "tok_s" else "accept rate"
    unit = "tok/s" if metric == "tok_s" else "%"
    title = f"{model_bundle} {title_metric}"
    axis_max = (
        100.0
        if metric == "accept_rate"
        else nice_axis_ceiling(
            max(max(row.mtplx_tok_s, row.ax_mtp_tok_s) for row in rows) * 1.15
        )
    )
    best_value = max(
        max(
            mtp_metric_value(row, "mtplx", metric),
            mtp_metric_value(row, "ax", metric),
        )
        for row in rows
    )
    group_step = (MTP_RIGHT - MTP_LEFT) / len(rows)
    bar_width = min(42.0, group_step * 0.26)
    pair_gap = bar_width * 0.22
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{MTP_WIDTH}" height="{MTP_HEIGHT}" viewBox="0 0 {MTP_WIDTH} {MTP_HEIGHT}" role="img" aria-labelledby="title desc">',
        f"<title>{escape(title)}</title>",
        (
            f"<desc>Bar chart comparing MTPLX 0.3.7 on the left and AX native "
            f"MTP on the right at depth 2 and depth 3 for "
            f"{escape(model_bundle)} {escape(title_metric)} on the "
            f"{escape(suite)} suite.</desc>"
        ),
        f'<rect width="{MTP_WIDTH}" height="{MTP_HEIGHT}" fill="#ffffff"/>',
        f'<text x="{MTP_LEFT}" y="22" font-family="{FONT}" font-size="16" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="{MTP_LEFT}" y="40" font-family="{FONT}" font-size="10" fill="#6b7280">{escape(suite)} suite</text>',
        f'<text x="{MTP_RIGHT}" y="22" text-anchor="end" font-family="{FONT}" font-size="10" fill="#6b7280">{escape(unit)}</text>',
        f'<text x="{MTP_RIGHT}" y="40" text-anchor="end" font-family="{FONT}" font-size="10" font-weight="700" fill="#374151">Higher is better</text>',
    ]

    for value in (0.0, axis_max / 2, axis_max):
        y = mtp_y_scale(value, axis_max)
        lines.append(
            f'<line x1="{MTP_LEFT}" y1="{y:.1f}" x2="{MTP_RIGHT}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{MTP_LEFT - 8}" y="{y + 3:.1f}" text-anchor="end" font-family="{FONT}" font-size="10" fill="#6b7280">{short_number(value)}</text>'
        )

    best_y = mtp_y_scale(best_value, axis_max)
    lines.append(
        f'<line x1="{MTP_LEFT}" y1="{best_y:.1f}" x2="{MTP_RIGHT}" y2="{best_y:.1f}" stroke="{RED}" stroke-width="1.2" stroke-dasharray="1 4" stroke-linecap="round"/>'
    )

    for index, row in enumerate(rows):
        group_x = MTP_LEFT + group_step * (index + 0.5)
        bars = (
            (
                "MTPLX",
                "mtplx",
                MTP_MTPLX_COLOR,
                MTP_MTPLX_DOT,
                group_x - bar_width / 2 - pair_gap,
            ),
            (
                "AX MTP",
                "ax",
                MTP_AX_COLOR,
                MTP_AX_DOT,
                group_x + bar_width / 2 + pair_gap,
            ),
        )
        for label, engine, color, dot_color, center_x in bars:
            value = mtp_metric_value(row, engine, metric)
            y = mtp_y_scale(value, axis_max)
            height = MTP_BOTTOM - y
            left = center_x - bar_width / 2
            lines.extend(
                [
                    f'<rect x="{left:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" rx="3" fill="{color}" fill-opacity="0.22" stroke="{color}" stroke-width="1.8"/>',
                    f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + bar_width:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2.4"/>',
                    f'<circle cx="{center_x:.1f}" cy="{y:.1f}" r="2.2" fill="{dot_color}"/>',
                    f'<text x="{center_x:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="{FONT}" font-size="10" font-weight="700" fill="#111827">{mtp_metric_label(value, metric)}</text>',
                    f'<text x="{center_x:.1f}" y="190" text-anchor="middle" font-family="{FONT}" font-size="9" fill="#111827">{escape(label)}</text>',
                ]
            )
        lines.append(
            f'<text x="{group_x:.1f}" y="207" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#6b7280">d={row.ax_depth_cap}</text>'
        )

    lines.extend(
        [
            f'<text x="{MTP_WIDTH / 2:.1f}" y="226" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#6b7280">MTPLX left · AX MTP right</text>',
            "</svg>",
        ]
    )
    return "".join(lines) + "\n"


def infer_results_dir_from_readme(readme: Path) -> Path:
    text = readme.read_text()
    patterns = [
        r"readme-performance-artifacts:[^\n]*base=([^;\s]+)",
        r"readme-performance-artifacts:[^\n]*reference=([^;\s]+)",
        r"Artifact directory: `([^`]+)`",
        r"artifacts are in\s+`([^`]+)`",
        r"Source:\s+`([^`]+)`\s+for all rows",
    ]
    match = next((match for pattern in patterns if (match := re.search(pattern, text))), None)
    if not match:
        raise ChartError(f"could not infer artifact directory from {readme}")
    return (readme.parent / match.group(1)).resolve()


def write_chart(path: Path, content: str, check: bool) -> bool:
    if check:
        return path.exists() and path.read_text() == content
    path.write_text(content)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Benchmark artifact directory. Defaults to the README artifact directory.",
    )
    parser.add_argument(
        "--llama-results-dir",
        type=Path,
        help=(
            "llama.cpp Metal artifact directory. Defaults to the newest "
            "benchmarks/results/llama-cpp-metal/*/sweep_results.json directory."
        ),
    )
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument(
        "--performance-doc",
        type=Path,
        default=Path("docs/PERFORMANCE.md"),
        help="Performance doc containing the MTP Mode benchmark table.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated SVGs match files on disk without writing.",
    )
    args = parser.parse_args()

    readme_slugs = readme_model_slugs(args.readme)
    results_dir = args.results_dir or infer_results_dir_from_readme(args.readme)
    llama_results_dir = args.llama_results_dir or infer_llama_results_dir(args.readme.parent)
    llama_rows = load_rows(llama_results_dir, readme_slugs, required=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mismatches: list[Path] = []
    mtp_rows = {
        (mtp_row_key(row), mtp_depth_key(row)): row
        for row in load_mtp_rows(args.performance_doc)
    }
    for (row_key, metric), output_name in MTP_CHART_OUTPUTS.items():
        rows = [
            row
            for (candidate_key, _depth), row in mtp_rows.items()
            if candidate_key == row_key
        ]
        if not rows:
            raise ChartError(f"MTP performance table has no {row_key!r} rows")
        depths = {mtp_depth_key(row) for row in rows}
        if depths != {2, 3}:
            raise ChartError(
                f"MTP performance table for {row_key!r} requires depths 2 and 3, found {sorted(depths)}"
            )
        mtp_output_path = args.output_dir / output_name
        mtp_content = render_mtp_metric_chart(rows, metric)
        if not write_chart(mtp_output_path, mtp_content, args.check):
            mismatches.append(mtp_output_path)

    for spec in CHARTS:
        if args.results_dir:
            benchmark_rows = load_rows(results_dir, readme_slugs, required=True)
        else:
            benchmark_rows = load_composite_rows(args.readme, spec.metric, readme_slugs)
        rows = benchmark_rows + llama_rows
        groups = collect_values(rows, spec, expected_model_count=len(readme_slugs))
        for group in groups:
            output_path = args.output_dir / chart_output_name(spec, group.prompt_tokens)
            content = render_chart(spec, group)
            if not write_chart(output_path, content, args.check):
                mismatches.append(output_path)

    if mismatches:
        print("README performance charts are stale:", file=sys.stderr)
        for path in mismatches:
            print(f"  {path}", file=sys.stderr)
        return 1

    if args.check:
        print("README performance charts are up to date")
    else:
        benchmark_source = (
            str(results_dir)
            if args.results_dir
            else f"README composite sources in {args.readme}"
        )
        print(
            f"Rendered README performance charts from {benchmark_source} "
            f"and {llama_results_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
