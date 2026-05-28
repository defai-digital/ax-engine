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
    ("ax_engine_mlx_ngram_accel", "ax+n-gram", "#137a3d", "#0b4f28"),
]

FAMILY_SLUGS: dict[str, list[str]] = {
    "gemma4": [
        "gemma-4-e2b-it-4bit",
        "gemma-4-e2b-it-5bit",
        "gemma-4-e2b-it-6bit",
        "gemma-4-e2b-it-8bit",
        "gemma-4-e4b-it-4bit",
        "gemma-4-26b-a4b-it-4bit",
        "gemma-4-31b-it-4bit",
    ],
    "qwen": [
        "qwen3_6-27b-4bit",
        "qwen3_6-27b-5bit",
        "qwen3_6-27b-6bit",
        "qwen3_6-27b-8bit",
        "qwen3_6-35b-a3b-4bit",
    ],
}

FAMILY_LABELS: dict[str, str] = {
    "gemma4": "Gemma 4",
    "qwen": "Qwen 3.6",
}

FAMILY_CHART_WIDTH = 700
FAMILY_CHART_HEIGHT = 340
FAMILY_LEFT = 52
FAMILY_RIGHT = 684
FAMILY_TOP = 46
FAMILY_BOTTOM = 276

# fill-opacity and stroke-opacity per context length (lighter = shorter prompt)
CTX_FILL_OPACITY = {128: 0.08, 512: 0.17, 2048: 0.30}
CTX_STROKE_OPACITY = {128: 0.55, 512: 0.78, 2048: 1.0}

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
class EngineContextStats:
    prompt_tokens: int
    stats: SeriesStats


@dataclass(frozen=True)
class EngineGroupStats:
    engine: str
    label: str
    color: str
    dot_color: str
    context_stats: tuple[EngineContextStats, ...]


@dataclass(frozen=True)
class ChartSpec:
    title: str
    subtitle: str | None
    unit: str
    direction_label: str
    output_slug: str
    metric: str
    series_engines: tuple[str, ...]
    family: str


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
        title="Gemma 4 — Prefill rate",
        subtitle=None,
        unit="tok/s",
        direction_label="Higher is better",
        output_slug="gemma4-prefill",
        metric="prefill",
        family="gemma4",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
    ),
    ChartSpec(
        title="Gemma 4 — Decode rate",
        subtitle="AX n-gram is the default policy",
        unit="tok/s",
        direction_label="Higher is better",
        output_slug="gemma4-decode",
        metric="decode",
        family="gemma4",
        series_engines=(
            "llama_cpp_metal",
            "mlx_lm",
            "ax_engine_mlx",
            "ax_engine_mlx_ngram_accel",
        ),
    ),
    ChartSpec(
        title="Gemma 4 — TTFT",
        subtitle=None,
        unit="ms",
        direction_label="Lower is better",
        output_slug="gemma4-ttft",
        metric="ttft",
        family="gemma4",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
    ),
    ChartSpec(
        title="Qwen 3.6 — Prefill rate",
        subtitle=None,
        unit="tok/s",
        direction_label="Higher is better",
        output_slug="qwen-prefill",
        metric="prefill",
        family="qwen",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
    ),
    ChartSpec(
        title="Qwen 3.6 — Decode rate",
        subtitle="AX n-gram is the default policy",
        unit="tok/s",
        direction_label="Higher is better",
        output_slug="qwen-decode",
        metric="decode",
        family="qwen",
        series_engines=(
            "llama_cpp_metal",
            "mlx_lm",
            "ax_engine_mlx",
            "ax_engine_mlx_ngram_accel",
        ),
    ),
    ChartSpec(
        title="Qwen 3.6 — TTFT",
        subtitle=None,
        unit="ms",
        direction_label="Lower is better",
        output_slug="qwen-ttft",
        metric="ttft",
        family="qwen",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
    ),
]


MTP_WIDTH = 380
MTP_HEIGHT = 216
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

# ---------------------------------------------------------------------------
# N-gram chart constants (shared by all three ngram charts)
# ---------------------------------------------------------------------------

NGRAM_CHART_WIDTH = 540
NGRAM_CHART_HEIGHT = 292
NGRAM_LEFT = 52
NGRAM_RIGHT = 512
NGRAM_TOP = 56
NGRAM_BOTTOM = 228

NGRAM_OPPORTUNITY_SERIES: list[tuple[str, str, str, str]] = [
    ("ax_direct",  "ax direct",          "#86efac", "#16a34a"),
    ("ax_ngram",   "ax + n-gram",        "#2eaf5f", "#176c37"),
    ("lightning",  "lightning (temp=0.6)", "#f2b705", "#9a6a00"),
    ("oracle",     "oracle (bound)",     "#d1d5db", "#6b7280"),
]

NGRAM_TOKS_SERIES: list[tuple[str, str, str, str]] = [
    ("ax_direct",  "ax direct",           "#86efac", "#16a34a"),
    ("ax_ngram",   "ax + n-gram",         "#2eaf5f", "#176c37"),
    ("lightning",  "lightning (temp=0.6)", "#f2b705", "#9a6a00"),
]

NGRAM_ACCEPT_SERIES: list[tuple[str, str, str, str]] = [
    ("ax_ngram",   "ax n-gram",           "#2eaf5f", "#176c37"),
    ("lightning",  "lightning (temp=0.6)", "#f2b705", "#9a6a00"),
]

NGRAM_OPPORTUNITY_CATEGORIES: list[tuple[str, str]] = [
    ("high_repeat", "high repeat"),
    ("med_repeat",  "med repeat"),
    ("low_repeat",  "low repeat"),
]

NGRAM_ARTIFACT_GLOB = "benchmarks/results/ngram-compare/*/artifact.json"

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


def nice_axis_ceiling(value: float) -> float:
    if value <= 0:
        raise ChartError("chart axis requires a positive maximum value")
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for candidate in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def short_number(value: float) -> str:
    if value >= 1000:
        compact = value / 1000
        if compact.is_integer():
            return f"{compact:.0f}k"
        return f"{compact:.1f}k"
    if value.is_integer():
        return f"{value:.0f}"
    return f"{value:.1f}"


def chart_output_name(spec: ChartSpec) -> str:
    return f"perf-{spec.output_slug}-box-whisker.svg"


def collect_family_values(
    rows: list[dict[str, Any]],
    spec: ChartSpec,
) -> list[EngineGroupStats]:
    family_slugs = set(FAMILY_SLUGS[spec.family])
    family_rows = [r for r in rows if r.get("_slug") in family_slugs]
    expected_count = len(FAMILY_SLUGS[spec.family])

    engine_groups: list[EngineGroupStats] = []
    for engine, label, color, dot_color in series_for_chart(spec):
        context_stats_list: list[EngineContextStats] = []
        for prompt_tokens in PROMPT_TOKENS:
            values: list[float] = []
            for row in family_rows:
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
            if len(values) != expected_count:
                raise ChartError(
                    f"{spec.metric} chart expected {expected_count} values "
                    f"for {engine} at {prompt_tokens} tok in family {spec.family!r}, "
                    f"found {len(values)}"
                )
            context_stats_list.append(
                EngineContextStats(
                    prompt_tokens=prompt_tokens,
                    stats=summarize(values, engine, label, color, dot_color),
                )
            )
        engine_groups.append(
            EngineGroupStats(
                engine=engine,
                label=label,
                color=color,
                dot_color=dot_color,
                context_stats=tuple(context_stats_list),
            )
        )
    return engine_groups


def render_family_chart(spec: ChartSpec, engine_groups: list[EngineGroupStats]) -> str:
    all_maxima = [cs.stats.maximum for eg in engine_groups for cs in eg.context_stats]
    axis_max = nice_axis_ceiling(max(all_maxima) * 1.05)

    all_medians = [cs.stats.median for eg in engine_groups for cs in eg.context_stats]
    best_med = min(all_medians) if spec.metric == "ttft" else max(all_medians)
    best_label = "lowest median" if spec.metric == "ttft" else "highest median"

    n_engines = len(engine_groups)
    plot_width = FAMILY_RIGHT - FAMILY_LEFT
    group_step = plot_width / n_engines
    sub_spacing = 30.0
    sub_bar_w = 16.0
    sub_offsets = [-sub_spacing, 0.0, sub_spacing]

    def fy(v: float) -> float:
        clamped = max(0.0, min(v, axis_max))
        return FAMILY_BOTTOM - (clamped / axis_max) * (FAMILY_BOTTOM - FAMILY_TOP)

    engine_desc = ", ".join(eg.label for eg in engine_groups)
    ctx_desc = "/".join(str(pt) for pt in PROMPT_TOKENS)
    family_label = FAMILY_LABELS.get(spec.family, spec.family)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{FAMILY_CHART_WIDTH}" height="{FAMILY_CHART_HEIGHT}"'
        f' viewBox="0 0 {FAMILY_CHART_WIDTH} {FAMILY_CHART_HEIGHT}"'
        f' role="img" aria-labelledby="title desc">',
        f"<title>{escape(spec.title)}</title>",
        f"<desc>Grouped box-and-whisker plot comparing {escape(engine_desc)}"
        f" at {escape(ctx_desc)} prompt tokens for {escape(family_label)} models."
        f" A red dotted line marks the {best_label}.</desc>",
        f'<rect width="{FAMILY_CHART_WIDTH}" height="{FAMILY_CHART_HEIGHT}" fill="#ffffff"/>',
        f'<text x="{FAMILY_LEFT}" y="22" font-family="{FONT}"'
        f' font-size="16" font-weight="700" fill="#111827">{escape(spec.title)}</text>',
    ]
    if spec.subtitle:
        lines.append(
            f'<text x="{FAMILY_LEFT}" y="40" font-family="{FONT}"'
            f' font-size="10" fill="#6b7280">{escape(spec.subtitle)}</text>'
        )
    lines.extend(
        [
            f'<text x="{FAMILY_RIGHT}" y="22" text-anchor="end" font-family="{FONT}"'
            f' font-size="10" fill="#6b7280">{escape(spec.unit)}</text>',
            f'<text x="{FAMILY_RIGHT}" y="40" text-anchor="end" font-family="{FONT}"'
            f' font-size="10" font-weight="700" fill="#374151">{escape(spec.direction_label)}</text>',
        ]
    )

    for grid_val in (0.0, axis_max / 2, axis_max):
        gy = fy(grid_val)
        lines.append(
            f'<line x1="{FAMILY_LEFT}" y1="{gy:.1f}"'
            f' x2="{FAMILY_RIGHT}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="44" y="{gy + 3:.1f}" text-anchor="end" font-family="{FONT}"'
            f' font-size="10" fill="#6b7280">{short_number(grid_val)}</text>'
        )

    best_y = fy(best_med)
    lines.append(
        f'<line x1="{FAMILY_LEFT}" y1="{best_y:.1f}"'
        f' x2="{FAMILY_RIGHT}" y2="{best_y:.1f}"'
        f' stroke="{RED}" stroke-width="1.2" stroke-dasharray="1 4" stroke-linecap="round"/>'
    )

    dot_jitter = (-3.0, -1.5, 0.0, 1.5, 3.0)
    y_ctx = FAMILY_BOTTOM + 15
    y_eng = FAMILY_BOTTOM + 32

    for i, eg in enumerate(engine_groups):
        group_center = FAMILY_LEFT + (i + 0.5) * group_step

        for j, cs in enumerate(eg.context_stats):
            sub_x = group_center + sub_offsets[j]
            fill_op = CTX_FILL_OPACITY[cs.prompt_tokens]
            stroke_op = CTX_STROKE_OPACITY[cs.prompt_tokens]
            s = cs.stats

            y_min = fy(s.minimum)
            y_q1 = fy(s.q1)
            y_med = fy(s.median)
            y_q3 = fy(s.q3)
            y_max_v = fy(s.maximum)
            box_y = min(y_q1, y_q3)
            box_h = max(abs(y_q3 - y_q1), 1.0)
            cap_left = sub_x - sub_bar_w * 0.36
            cap_right = sub_x + sub_bar_w * 0.36
            box_left = sub_x - sub_bar_w / 2

            sa = f'stroke="{eg.color}" stroke-opacity="{stroke_op}"'
            lines.extend(
                [
                    f'<line x1="{sub_x:g}" y1="{y_max_v:.1f}"'
                    f' x2="{sub_x:g}" y2="{y_min:.1f}" {sa} stroke-width="1.7"/>',
                    f'<line x1="{cap_left:g}" y1="{y_max_v:.1f}"'
                    f' x2="{cap_right:g}" y2="{y_max_v:.1f}" {sa} stroke-width="1.7"/>',
                    f'<line x1="{cap_left:g}" y1="{y_min:.1f}"'
                    f' x2="{cap_right:g}" y2="{y_min:.1f}" {sa} stroke-width="1.7"/>',
                    f'<rect x="{box_left:g}" y="{box_y:.1f}"'
                    f' width="{sub_bar_w:g}" height="{box_h:.1f}" rx="2"'
                    f' fill="{eg.color}" fill-opacity="{fill_op}" {sa} stroke-width="1.7"/>',
                    f'<line x1="{box_left:g}" y1="{y_med:.1f}"'
                    f' x2="{box_left + sub_bar_w:g}" y2="{y_med:.1f}" {sa} stroke-width="2.4"/>',
                    f'<text x="{sub_x:g}" y="{y_ctx}"'
                    f' text-anchor="middle" font-family="{FONT}"'
                    f' font-size="8" fill="#6b7280">{cs.prompt_tokens}</text>',
                ]
            )

            for vi, val in enumerate(s.values):
                dx = dot_jitter[vi % len(dot_jitter)]
                dy = fy(val)
                dot_op = round(stroke_op * 0.9, 2)
                lines.append(
                    f'<circle cx="{sub_x + dx:g}" cy="{dy:.1f}" r="1.4"'
                    f' fill="{eg.dot_color}" fill-opacity="{dot_op}"/>'
                )

        lines.append(
            f'<text x="{group_center:g}" y="{y_eng}"'
            f' text-anchor="middle" font-family="{FONT}"'
            f' font-size="9" font-weight="700" fill="#111827">{escape(eg.label)}</text>'
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


def mtp_bundle_order(row: MtpBenchmarkRow) -> int:
    order = {"speed": 0, "quality": 1}
    return order.get(mtp_row_key(row), 99)


def mtp_bundle_chart_label(model_bundle: str) -> str:
    normalized = model_bundle.lower().strip()
    if normalized == "speed":
        return "Speed to Speed"
    if normalized == "quality":
        return "Quality"
    return model_bundle


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
    rows = sorted(rows, key=lambda row: (row.suite, mtp_depth_key(row)))
    if not rows:
        raise ChartError("MTP metric chart requires at least one row")
    model_bundle = rows[0].model_bundle
    if any(row.model_bundle != model_bundle for row in rows):
        raise ChartError("MTP metric chart rows must share model bundle")
    depths = {mtp_depth_key(row) for row in rows}
    depth_label = f"d={next(iter(depths))}" if len(depths) == 1 else "mixed depth"
    title_metric = "tok/s" if metric == "tok_s" else "accept %"
    chart_label = mtp_bundle_chart_label(model_bundle)
    title = f"{chart_label} {depth_label} {title_metric}"
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
            f"<desc>Bar chart comparing artifact-backed MTPLX 0.3.7 on the "
            f"left and AX native MTP on the right for "
            f"{escape(chart_label)} {escape(title_metric)} on flappy and "
            f"long_code prompt suites.</desc>"
        ),
        f'<rect width="{MTP_WIDTH}" height="{MTP_HEIGHT}" fill="#ffffff"/>',
        f'<text x="{MTP_LEFT}" y="22" font-family="{FONT}" font-size="16" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="{MTP_RIGHT}" y="22" text-anchor="end" font-family="{FONT}" font-size="10" font-weight="700" fill="#374151">Higher is better</text>',
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
            f'<text x="{group_x:.1f}" y="207" text-anchor="middle" font-family="{FONT}" font-size="10" font-weight="700" fill="#111827">{escape(row.suite)}</text>'
        )

    lines.append("</svg>")
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


def find_latest_ngram_artifact(repo_root: Path) -> Path | None:
    """Return the most recent ngram-compare artifact that has usable data.

    Prefers artifacts with non-empty ax_direct results, falling back to the
    alphabetically latest if none qualify.
    """
    candidates = sorted(
        (
            p
            for p in (repo_root / "benchmarks" / "results" / "ngram-compare").glob(
                "*/artifact.json"
            )
            if p.is_file()
        ),
        key=lambda p: p.parent.name,
    )

    # Return the alphabetically latest artifact that has both ax_direct and lightning data.
    # Naming convention: suffixes sort naturally so the most recent/complete run wins.
    for path in reversed(candidates):
        try:
            art = json.loads(path.read_text())
            if art.get("ax_direct") and art.get("lightning"):
                return path
        except Exception:
            continue
    return candidates[-1] if candidates else None


def _ngram_category_median(results: list[dict], category: str, key: str) -> float | None:
    import statistics as _stats

    vals = [
        r[key]
        for r in results
        if r.get("category") == category and isinstance(r.get(key), (int, float))
    ]
    return _stats.median(vals) if vals else None


def render_ngram_opportunity_chart(artifact: dict) -> str:
    """Grouped bar chart comparing ax direct / ax n-gram / lightning / oracle by category."""
    categories = NGRAM_OPPORTUNITY_CATEGORIES
    series = NGRAM_OPPORTUNITY_SERIES

    # Build data matrix: series_key → category_id → median tok/s
    data: dict[str, dict[str, float]] = {}
    for s_key, _label, _color, _dot in series:
        data[s_key] = {}
        for cat_id, _cat_label in categories:
            val = _ngram_category_median(artifact.get(s_key, []), cat_id, "tok_s_median")
            data[s_key][cat_id] = val if val is not None else 0.0

    all_vals = [v for cat_vals in data.values() for v in cat_vals.values() if v > 0]
    if not all_vals:
        raise ChartError("ngram opportunity artifact has no usable tok/s data")
    axis_max = nice_axis_ceiling(max(all_vals) * 1.05)

    plot_w = NGRAM_RIGHT - NGRAM_LEFT
    plot_h = NGRAM_BOTTOM - NGRAM_TOP
    n_groups = len(categories)
    n_bars = len(series)
    bar_w = 20.0
    bar_gap = 4.0
    group_block = n_bars * bar_w + (n_bars - 1) * bar_gap  # 92px

    def fy(v: float) -> float:
        return NGRAM_BOTTOM - (max(0.0, min(v, axis_max)) / axis_max) * plot_h

    title = "N-gram opportunity — Qwen3-4B 4-bit"
    subtitle = "all paths: temp=0.6/top_p=0.95/top_k=20 · oracle: theoretical upper bound"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{NGRAM_CHART_WIDTH}" height="{NGRAM_CHART_HEIGHT}"'
        f' viewBox="0 0 {NGRAM_CHART_WIDTH} {NGRAM_CHART_HEIGHT}"'
        f' role="img" aria-labelledby="title desc">',
        f"<title>{escape(title)}</title>",
        f"<desc>Grouped bar chart comparing ax direct, ax n-gram, lightning (temp=0.6), and oracle"
        f" upper-bound decode throughput (tok/s) across high, med, and low repeat prompt"
        f" categories for Qwen3-4B 4-bit.</desc>",
        f'<rect width="{NGRAM_CHART_WIDTH}" height="{NGRAM_CHART_HEIGHT}" fill="#ffffff"/>',
        f'<text x="{NGRAM_LEFT}" y="22" font-family="{FONT}"'
        f' font-size="15" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="{NGRAM_LEFT}" y="38" font-family="{FONT}"'
        f' font-size="9" fill="#6b7280">{escape(subtitle)}</text>',
        f'<text x="{NGRAM_RIGHT}" y="22" text-anchor="end" font-family="{FONT}"'
        f' font-size="10" fill="#6b7280">tok/s</text>',
        f'<text x="{NGRAM_RIGHT}" y="38" text-anchor="end" font-family="{FONT}"'
        f' font-size="10" font-weight="700" fill="#374151">Higher is better</text>',
    ]

    # Grid lines and Y axis labels
    for grid_val in (0.0, axis_max * 0.5, axis_max):
        gy = fy(grid_val)
        lines.append(
            f'<line x1="{NGRAM_LEFT}" y1="{gy:.1f}"'
            f' x2="{NGRAM_RIGHT}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{NGRAM_LEFT - 6}" y="{gy + 3:.1f}" text-anchor="end"'
            f' font-family="{FONT}" font-size="10" fill="#6b7280">{short_number(grid_val)}</text>'
        )

    # Best median reference line
    best_y = fy(max(all_vals))
    lines.append(
        f'<line x1="{NGRAM_LEFT}" y1="{best_y:.1f}"'
        f' x2="{NGRAM_RIGHT}" y2="{best_y:.1f}"'
        f' stroke="{RED}" stroke-width="1.2" stroke-dasharray="1 4" stroke-linecap="round"/>'
    )

    group_step = plot_w / n_groups
    y_cat_label = NGRAM_BOTTOM + 16

    for gi, (cat_id, cat_label) in enumerate(categories):
        group_center = NGRAM_LEFT + (gi + 0.5) * group_step
        bar0_left = group_center - group_block / 2

        for bi, (s_key, s_label, color, dot_color) in enumerate(series):
            bar_center = bar0_left + bi * (bar_w + bar_gap) + bar_w / 2
            bar_left = bar_center - bar_w / 2
            val = data[s_key][cat_id]
            y_top = fy(val)
            bar_h = NGRAM_BOTTOM - y_top

            lines.extend([
                f'<rect x="{bar_left:.1f}" y="{y_top:.1f}"'
                f' width="{bar_w:.0f}" height="{bar_h:.1f}" rx="2"'
                f' fill="{color}" fill-opacity="0.30" stroke="{dot_color}" stroke-width="1.6"/>',
                f'<line x1="{bar_left:.1f}" y1="{y_top:.1f}"'
                f' x2="{bar_left + bar_w:.1f}" y2="{y_top:.1f}"'
                f' stroke="{dot_color}" stroke-width="2.4"/>',
            ])
            if val > 0:
                lines.append(
                    f'<text x="{bar_center:.1f}" y="{y_top - 4:.1f}"'
                    f' text-anchor="middle" font-family="{FONT}"'
                    f' font-size="8" fill="{dot_color}">{val:.0f}</text>'
                )

        lines.append(
            f'<text x="{group_center:.1f}" y="{y_cat_label}"'
            f' text-anchor="middle" font-family="{FONT}"'
            f' font-size="10" font-weight="700" fill="#111827">{escape(cat_label)}</text>'
        )

    # Legend
    legend_x = NGRAM_LEFT
    legend_y = NGRAM_BOTTOM + 32
    leg_box = 10
    leg_gap = 6
    leg_item_w = 115
    for li, (_s_key, s_label, color, dot_color) in enumerate(series):
        lx = legend_x + li * leg_item_w
        lines.extend([
            f'<rect x="{lx}" y="{legend_y}" width="{leg_box}" height="{leg_box}"'
            f' rx="2" fill="{color}" fill-opacity="0.40" stroke="{dot_color}" stroke-width="1.4"/>',
            f'<text x="{lx + leg_box + leg_gap}" y="{legend_y + 9}"'
            f' font-family="{FONT}" font-size="9" fill="#374151">{escape(s_label)}</text>',
        ])

    lines.append("</svg>")
    return "".join(lines) + "\n"


def _render_ngram_grouped_bars(
    series: list[tuple[str, str, str, str]],
    categories: list[tuple[str, str]],
    data: dict[str, dict[str, float]],
    title: str,
    subtitle: str,
    y_label: str,
    axis_max: float,
    y_fmt: str = "{:.0f}",
    reference_line: bool = True,
) -> str:
    """Generic grouped-bar SVG renderer for ngram charts."""
    plot_w = NGRAM_RIGHT - NGRAM_LEFT
    plot_h = NGRAM_BOTTOM - NGRAM_TOP
    n_groups = len(categories)
    n_bars = len(series)
    bar_w = 26.0
    bar_gap = 6.0
    group_block = n_bars * bar_w + (n_bars - 1) * bar_gap

    def fy(v: float) -> float:
        return NGRAM_BOTTOM - (max(0.0, min(v, axis_max)) / axis_max) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{NGRAM_CHART_WIDTH}" height="{NGRAM_CHART_HEIGHT}"'
        f' viewBox="0 0 {NGRAM_CHART_WIDTH} {NGRAM_CHART_HEIGHT}"'
        f' role="img" aria-labelledby="title desc">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(title)}</desc>",
        f'<rect width="{NGRAM_CHART_WIDTH}" height="{NGRAM_CHART_HEIGHT}" fill="#ffffff"/>',
        f'<text x="{NGRAM_LEFT}" y="22" font-family="{FONT}"'
        f' font-size="15" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="{NGRAM_LEFT}" y="38" font-family="{FONT}"'
        f' font-size="9" fill="#6b7280">{escape(subtitle)}</text>',
        f'<text x="{NGRAM_RIGHT}" y="22" text-anchor="end" font-family="{FONT}"'
        f' font-size="10" fill="#6b7280">{escape(y_label)}</text>',
        f'<text x="{NGRAM_RIGHT}" y="38" text-anchor="end" font-family="{FONT}"'
        f' font-size="10" font-weight="700" fill="#374151">Higher is better</text>',
    ]

    for grid_val in (0.0, axis_max * 0.5, axis_max):
        gy = fy(grid_val)
        lines.append(
            f'<line x1="{NGRAM_LEFT}" y1="{gy:.1f}"'
            f' x2="{NGRAM_RIGHT}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{NGRAM_LEFT - 6}" y="{gy + 3:.1f}" text-anchor="end"'
            f' font-family="{FONT}" font-size="10" fill="#6b7280">'
            f'{y_fmt.format(grid_val)}</text>'
        )

    if reference_line:
        all_vals = [v for cat_vals in data.values() for v in cat_vals.values() if v > 0]
        if all_vals:
            best_y = fy(max(all_vals))
            lines.append(
                f'<line x1="{NGRAM_LEFT}" y1="{best_y:.1f}"'
                f' x2="{NGRAM_RIGHT}" y2="{best_y:.1f}"'
                f' stroke="{RED}" stroke-width="1.2" stroke-dasharray="1 4" stroke-linecap="round"/>'
            )

    group_step = plot_w / n_groups
    y_cat_label = NGRAM_BOTTOM + 16

    for gi, (cat_id, cat_label) in enumerate(categories):
        group_center = NGRAM_LEFT + (gi + 0.5) * group_step
        bar0_left = group_center - group_block / 2

        for bi, (s_key, _s_label, color, dot_color) in enumerate(series):
            bar_center = bar0_left + bi * (bar_w + bar_gap) + bar_w / 2
            bar_left = bar_center - bar_w / 2
            val = data[s_key][cat_id]
            y_top = fy(val)
            bar_h = NGRAM_BOTTOM - y_top

            lines.extend([
                f'<rect x="{bar_left:.1f}" y="{y_top:.1f}"'
                f' width="{bar_w:.0f}" height="{bar_h:.1f}" rx="2"'
                f' fill="{color}" fill-opacity="0.30" stroke="{dot_color}" stroke-width="1.6"/>',
                f'<line x1="{bar_left:.1f}" y1="{y_top:.1f}"'
                f' x2="{bar_left + bar_w:.1f}" y2="{y_top:.1f}"'
                f' stroke="{dot_color}" stroke-width="2.4"/>',
            ])
            if val > 0:
                label_str = y_fmt.format(val)
                lines.append(
                    f'<text x="{bar_center:.1f}" y="{y_top - 4:.1f}"'
                    f' text-anchor="middle" font-family="{FONT}"'
                    f' font-size="8" fill="{dot_color}">{label_str}</text>'
                )

        lines.append(
            f'<text x="{group_center:.1f}" y="{y_cat_label}"'
            f' text-anchor="middle" font-family="{FONT}"'
            f' font-size="10" font-weight="700" fill="#111827">{escape(cat_label)}</text>'
        )

    legend_x = NGRAM_LEFT
    legend_y = NGRAM_BOTTOM + 32
    leg_box = 10
    leg_gap = 6
    leg_item_w = 140
    for li, (_s_key, s_label, color, dot_color) in enumerate(series):
        lx = legend_x + li * leg_item_w
        lines.extend([
            f'<rect x="{lx}" y="{legend_y}" width="{leg_box}" height="{leg_box}"'
            f' rx="2" fill="{color}" fill-opacity="0.40" stroke="{dot_color}" stroke-width="1.4"/>',
            f'<text x="{lx + leg_box + leg_gap}" y="{legend_y + 9}"'
            f' font-family="{FONT}" font-size="9" fill="#374151">{escape(s_label)}</text>',
        ])

    lines.append("</svg>")
    return "".join(lines) + "\n"


def render_ngram_toks_chart(artifact: dict) -> str:
    """Throughput comparison: ax direct / ax n-gram / lightning n-gram."""
    series = NGRAM_TOKS_SERIES
    categories = NGRAM_OPPORTUNITY_CATEGORIES

    data: dict[str, dict[str, float]] = {}
    for s_key, _label, _color, _dot in series:
        data[s_key] = {}
        for cat_id, _cat_label in categories:
            val = _ngram_category_median(artifact.get(s_key, []), cat_id, "tok_s_median")
            data[s_key][cat_id] = val if val is not None else 0.0

    all_vals = [v for cat_vals in data.values() for v in cat_vals.values() if v > 0]
    if not all_vals:
        raise ChartError("ngram toks artifact has no usable tok/s data")
    axis_max = nice_axis_ceiling(max(all_vals) * 1.05)

    return _render_ngram_grouped_bars(
        series=series,
        categories=categories,
        data=data,
        title="N-gram throughput — ax-engine vs lightning",
        subtitle="all paths: temp=0.6/top_p=0.95/top_k=20",
        y_label="tok/s",
        axis_max=axis_max,
        y_fmt="{:.0f}",
        reference_line=True,
    )


def render_ngram_accept_chart(artifact: dict) -> str:
    """Accept rate comparison: ax n-gram vs lightning n-gram, dynamic y-axis scale."""
    series = NGRAM_ACCEPT_SERIES
    categories = NGRAM_OPPORTUNITY_CATEGORIES

    data: dict[str, dict[str, float]] = {}
    accept_keys = {"ax_ngram": "ngram_accept_rate", "lightning": "lightning_accept_rate"}
    for s_key, _label, _color, _dot in series:
        data[s_key] = {}
        rate_key = accept_keys[s_key]
        for cat_id, _cat_label in categories:
            val = _ngram_category_median(artifact.get(s_key, []), cat_id, rate_key)
            data[s_key][cat_id] = (val * 100.0) if val is not None else 0.0

    all_vals = [v for cat_vals in data.values() for v in cat_vals.values() if v > 0]
    # Scale to actual data with 25% headroom, capped at 100%
    axis_max = min(100.0, nice_axis_ceiling(max(all_vals) * 1.25)) if all_vals else 100.0

    return _render_ngram_grouped_bars(
        series=series,
        categories=categories,
        data=data,
        title="N-gram accept rate — ax-engine vs lightning",
        subtitle="draft token acceptance at temp=0.6 · higher = better speculation quality",
        y_label="accept %",
        axis_max=axis_max,
        y_fmt="{:.0f}%",
        reference_line=True,
    )


# Canonical display names for model slugs extracted from artifact directory names.
NGRAM_MODEL_DISPLAY: dict[str, str] = {
    "qwen3-4b-4bit":            "Qwen3-4B",
    "glm-4-7-flash-4bit":       "GLM-4.7F",
    "gemma-4-e2b-4bit":         "E2B 4bit",
    "gemma-4-e2b-5bit":         "E2B 5bit",
    "gemma-4-e2b-6bit":         "E2B 6bit",
    "gemma-4-e2b-8bit":         "E2B 8bit",
    "gemma-4-e4b-4bit":         "E4B 4bit",
    "gemma-4-26b-a4b-4bit":     "G26B-A4B",
    "gemma-4-31b-4bit":         "G31B 4bit",
    "qwen3-6-27b-4bit":         "Q27B 4bit",
    "qwen3-6-27b-5bit":         "Q27B 5bit",
    "qwen3-6-27b-6bit":         "Q27B 6bit",
    "qwen3-6-27b-8bit":         "Q27B 8bit",
    "qwen3-6-35b-a3b-4bit":     "Q35B-A3B",
}

# Preferred order for model display in multi-model charts (small → large).
NGRAM_MODEL_ORDER = [
    "qwen3-4b-4bit",
    "glm-4-7-flash-4bit",
    "gemma-4-e2b-4bit",
    "gemma-4-e2b-5bit",
    "gemma-4-e2b-6bit",
    "gemma-4-e2b-8bit",
    "gemma-4-e4b-4bit",
    "gemma-4-26b-a4b-4bit",
    "gemma-4-31b-4bit",
    "qwen3-6-27b-4bit",
    "qwen3-6-27b-5bit",
    "qwen3-6-27b-6bit",
    "qwen3-6-27b-8bit",
    "qwen3-6-35b-a3b-4bit",
]


def find_ngram_artifacts_by_model(repo_root: Path) -> dict[str, dict]:
    """Return {model_slug: artifact_dict} using the latest `-ngram` run per slug."""
    base = repo_root / "benchmarks" / "results" / "ngram-compare"
    result: dict[str, dict] = {}
    for slug in NGRAM_MODEL_ORDER:
        # Prefer the `-ngram` suffixed run, fall back to any run containing the slug.
        candidates = sorted(base.glob(f"*{slug}*/artifact.json"), key=lambda p: p.parent.name)
        for path in reversed(candidates):
            try:
                art = json.loads(path.read_text())
                if art.get("ax_direct"):
                    result[slug] = art
                    break
            except Exception:
                continue
    return result


def render_ngram_models_speedup_chart(artifacts: dict[str, dict]) -> str:
    """Grouped bar: ax_direct vs ax_ngram tok/s for high-repeat, one group per model."""
    models = [s for s in NGRAM_MODEL_ORDER if s in artifacts]
    if not models:
        raise ChartError("no ngram model artifacts found")

    direct_vals: list[float] = []
    ngram_vals: list[float] = []
    for slug in models:
        art = artifacts[slug]
        d = _ngram_category_median(art.get("ax_direct", []), "high_repeat", "tok_s_median") or 0.0
        n = _ngram_category_median(art.get("ax_ngram", []), "high_repeat", "tok_s_median") or 0.0
        direct_vals.append(d)
        ngram_vals.append(n)

    all_vals = [v for v in direct_vals + ngram_vals if v > 0]
    if not all_vals:
        raise ChartError("no usable tok/s data for models speedup chart")
    axis_max = nice_axis_ceiling(max(all_vals) * 1.05)

    n_models = len(models)
    W = max(700, 55 + n_models * 75 + 30)
    H = 320
    LEFT, TOP, BOTTOM = 55, 56, 248
    RIGHT = W - 25
    plot_w = RIGHT - LEFT
    plot_h = BOTTOM - TOP
    bar_w = 14.0
    bar_gap = 3.0
    group_block = 2 * bar_w + bar_gap

    def fy(v: float) -> float:
        return BOTTOM - (max(0.0, min(v, axis_max)) / axis_max) * plot_h

    title = "N-gram throughput — ax direct vs ax n-gram (high-repeat, all models)"
    subtitle = "temp=0.6/top_p=0.95/top_k=20 · bars: ax direct (light) / ax+n-gram (dark)"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"'
        f' viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(title)}</desc>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{LEFT}" y="22" font-family="{FONT}" font-size="13" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="{LEFT}" y="36" font-family="{FONT}" font-size="9" fill="#6b7280">{escape(subtitle)}</text>',
        f'<text x="{RIGHT}" y="22" text-anchor="end" font-family="{FONT}" font-size="10" fill="#6b7280">tok/s</text>',
        f'<text x="{RIGHT}" y="36" text-anchor="end" font-family="{FONT}" font-size="10" font-weight="700" fill="#374151">Higher is better</text>',
    ]
    for grid_val in (0.0, axis_max * 0.5, axis_max):
        gy = fy(grid_val)
        lines.append(f'<line x1="{LEFT}" y1="{gy:.1f}" x2="{RIGHT}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{LEFT-6}" y="{gy+3:.1f}" text-anchor="end" font-family="{FONT}" font-size="10" fill="#6b7280">{short_number(grid_val)}</text>')

    group_step = plot_w / n_models
    for gi, slug in enumerate(models):
        group_center = LEFT + (gi + 0.5) * group_step
        bar0 = group_center - group_block / 2
        d_val, n_val = direct_vals[gi], ngram_vals[gi]
        label = NGRAM_MODEL_DISPLAY.get(slug, slug)

        for bi, (val, color, dot_color) in enumerate([
            (d_val, "#86efac", "#16a34a"),
            (n_val, "#2eaf5f", "#176c37"),
        ]):
            bx = bar0 + bi * (bar_w + bar_gap)
            bc = bx + bar_w / 2
            y_top = fy(val)
            bh = BOTTOM - y_top
            lines.extend([
                f'<rect x="{bx:.1f}" y="{y_top:.1f}" width="{bar_w:.0f}" height="{bh:.1f}" rx="2"'
                f' fill="{color}" fill-opacity="0.35" stroke="{dot_color}" stroke-width="1.4"/>',
                f'<line x1="{bx:.1f}" y1="{y_top:.1f}" x2="{bx+bar_w:.1f}" y2="{y_top:.1f}" stroke="{dot_color}" stroke-width="2.2"/>',
            ])
            if val > 0:
                lines.append(f'<text x="{bc:.1f}" y="{y_top-3:.1f}" text-anchor="middle" font-family="{FONT}" font-size="7.5" fill="{dot_color}">{val:.0f}</text>')

        lines.append(f'<text x="{group_center:.1f}" y="{BOTTOM+14}" text-anchor="middle" font-family="{FONT}" font-size="8.5" font-weight="700" fill="#111827">{escape(label)}</text>')

        if d_val > 0 and n_val > 0:
            speedup = n_val / d_val
            sy = fy(n_val) - 13
            lines.append(f'<text x="{group_center:.1f}" y="{sy:.1f}" text-anchor="middle" font-family="{FONT}" font-size="7.5" font-weight="700" fill="#176c37">{speedup:.2f}×</text>')

    ly = BOTTOM + 32
    for li, (label, color, dot_color) in enumerate([
        ("ax direct", "#86efac", "#16a34a"),
        ("ax + n-gram", "#2eaf5f", "#176c37"),
    ]):
        lx = LEFT + li * 120
        lines.extend([
            f'<rect x="{lx}" y="{ly}" width="10" height="10" rx="2" fill="{color}" fill-opacity="0.4" stroke="{dot_color}" stroke-width="1.4"/>',
            f'<text x="{lx+14}" y="{ly+9}" font-family="{FONT}" font-size="9" fill="#374151">{escape(label)}</text>',
        ])

    lines.append("</svg>")
    return "".join(lines) + "\n"


def render_ngram_models_accept_chart(artifacts: dict[str, dict]) -> str:
    """Grouped bar: ax n-gram accept rate vs lightning accept rate, one group per model."""
    models = [s for s in NGRAM_MODEL_ORDER if s in artifacts]
    if not models:
        raise ChartError("no ngram model artifacts found")

    ax_vals: list[float] = []
    lt_vals: list[float] = []
    has_lightning = False
    for slug in models:
        art = artifacts[slug]
        a = _ngram_category_median(art.get("ax_ngram", []), "high_repeat", "ngram_accept_rate")
        l = _ngram_category_median(art.get("lightning", []), "high_repeat", "lightning_accept_rate")
        ax_vals.append((a * 100.0) if a is not None else 0.0)
        lt_val = (l * 100.0) if l is not None else 0.0
        lt_vals.append(lt_val)
        if lt_val > 0:
            has_lightning = True

    all_vals = [v for v in ax_vals + lt_vals if v > 0]
    axis_max = min(100.0, nice_axis_ceiling(max(all_vals) * 1.25)) if all_vals else 100.0

    n_models = len(models)
    n_bars = 2 if has_lightning else 1
    W = max(700, 55 + n_models * 75 + 30)
    H = 320
    LEFT, TOP, BOTTOM = 55, 56, 248
    RIGHT = W - 25
    plot_w = RIGHT - LEFT
    plot_h = BOTTOM - TOP
    bar_w = 14.0
    bar_gap = 3.0
    group_block = n_bars * bar_w + (n_bars - 1) * bar_gap

    def fy(v: float) -> float:
        return BOTTOM - (max(0.0, min(v, axis_max)) / axis_max) * plot_h

    title = "N-gram accept rate — ax-engine vs lightning (high-repeat)"
    subtitle = "temp=0.6 · draft token acceptance per decode step · higher = better"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"'
        f' viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(title)}</desc>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{LEFT}" y="22" font-family="{FONT}" font-size="13" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="{LEFT}" y="36" font-family="{FONT}" font-size="9" fill="#6b7280">{escape(subtitle)}</text>',
        f'<text x="{RIGHT}" y="22" text-anchor="end" font-family="{FONT}" font-size="10" fill="#6b7280">accept %</text>',
        f'<text x="{RIGHT}" y="36" text-anchor="end" font-family="{FONT}" font-size="10" font-weight="700" fill="#374151">Higher is better</text>',
    ]
    for grid_val in (0.0, axis_max * 0.5, axis_max):
        gy = fy(grid_val)
        lines.append(f'<line x1="{LEFT}" y1="{gy:.1f}" x2="{RIGHT}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{LEFT-6}" y="{gy+3:.1f}" text-anchor="end" font-family="{FONT}" font-size="10" fill="#6b7280">{grid_val:.0f}%</text>')

    group_step = plot_w / n_models
    bar_series = [
        (ax_vals,  "#2eaf5f", "#176c37", "ax n-gram"),
        (lt_vals,  "#f2b705", "#9a6a00", "lightning"),
    ][:n_bars]

    for gi, slug in enumerate(models):
        group_center = LEFT + (gi + 0.5) * group_step
        bar0 = group_center - group_block / 2
        label = NGRAM_MODEL_DISPLAY.get(slug, slug)

        for bi, (vals, color, dot_color, _) in enumerate(bar_series):
            val = vals[gi]
            bx = bar0 + bi * (bar_w + bar_gap)
            bc = bx + bar_w / 2
            y_top = fy(val)
            bh = BOTTOM - y_top
            lines.extend([
                f'<rect x="{bx:.1f}" y="{y_top:.1f}" width="{bar_w:.0f}" height="{bh:.1f}" rx="2"'
                f' fill="{color}" fill-opacity="0.35" stroke="{dot_color}" stroke-width="1.4"/>',
                f'<line x1="{bx:.1f}" y1="{y_top:.1f}" x2="{bx+bar_w:.1f}" y2="{y_top:.1f}" stroke="{dot_color}" stroke-width="2.2"/>',
            ])
            if val > 0:
                lines.append(f'<text x="{bc:.1f}" y="{y_top-3:.1f}" text-anchor="middle" font-family="{FONT}" font-size="7.5" fill="{dot_color}">{val:.0f}%</text>')

        lines.append(f'<text x="{group_center:.1f}" y="{BOTTOM+14}" text-anchor="middle" font-family="{FONT}" font-size="8.5" font-weight="700" fill="#111827">{escape(label)}</text>')

    ly = BOTTOM + 32
    for li, (_, color, dot_color, label) in enumerate(bar_series):
        lx = LEFT + li * 120
        lines.extend([
            f'<rect x="{lx}" y="{ly}" width="10" height="10" rx="2" fill="{color}" fill-opacity="0.4" stroke="{dot_color}" stroke-width="1.4"/>',
            f'<text x="{lx+14}" y="{ly+9}" font-family="{FONT}" font-size="9" fill="#374151">{escape(label)}</text>',
        ])

    lines.append("</svg>")
    return "".join(lines) + "\n"


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
    mtp_rows = load_mtp_rows(args.performance_doc)
    for (row_key, metric), output_name in MTP_CHART_OUTPUTS.items():
        rows = [row for row in mtp_rows if mtp_row_key(row) == row_key]
        if not rows:
            raise ChartError(f"MTP performance table has no {row_key!r} rows")
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
        engine_groups = collect_family_values(rows, spec)
        output_path = args.output_dir / chart_output_name(spec)
        content = render_family_chart(spec, engine_groups)
        if not write_chart(output_path, content, args.check):
            mismatches.append(output_path)

    # N-gram charts (Qwen3-4B): reads from the latest ngram-compare artifact.
    ngram_artifact_path = find_latest_ngram_artifact(args.readme.parent)
    if ngram_artifact_path is not None:
        ngram_artifact = json.loads(ngram_artifact_path.read_text())
        for out_name, renderer in [
            ("perf-ngram-opportunity.svg", render_ngram_opportunity_chart),
            ("perf-ngram-toks.svg",        render_ngram_toks_chart),
            ("perf-ngram-accept.svg",      render_ngram_accept_chart),
        ]:
            content = renderer(ngram_artifact)
            out_path = args.output_dir / out_name
            if not write_chart(out_path, content, args.check):
                mismatches.append(out_path)

    # N-gram multi-model charts: one group per model slug.
    ngram_artifacts = find_ngram_artifacts_by_model(args.readme.parent)
    if ngram_artifacts:
        for out_name, renderer in [
            ("perf-ngram-models-toks.svg",   render_ngram_models_speedup_chart),
            ("perf-ngram-models-accept.svg", render_ngram_models_accept_chart),
        ]:
            content = renderer(ngram_artifacts)
            out_path = args.output_dir / out_name
            if not write_chart(out_path, content, args.check):
                mismatches.append(out_path)

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
