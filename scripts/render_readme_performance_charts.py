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
    ("llama_cpp_metal", "llama.cpp b9430", "#f97316", "#c2410c"),
    ("mlx_lm", "mlx-lm 0.31.3", "#f2b705", "#9a6a00"),
    ("ax_engine_mlx", "AX Engine v5.1.8", "#2eaf5f", "#176c37"),
    ("ax_engine_mlx_ngram_accel", "AX+ngram v5.1.8", "#137a3d", "#0b4f28"),
]
DIRECT_VERSIONS_FOOTNOTE = "llama.cpp b9430 · mlx-lm 0.31.3 · AX Engine v5.1.8"

FAMILY_SLUGS: dict[str, list[str]] = {
    "gemma4": [
        "gemma-4-e2b-it-4bit",
        "gemma-4-e2b-it-6bit",
        "gemma-4-e4b-it-4bit",
        "gemma-4-26b-a4b-it-4bit",
        "gemma-4-31b-it-4bit",
    ],
    "qwen": [
        "qwen3_6-27b-4bit",
        "qwen3_6-27b-6bit",
        "qwen3_6-35b-a3b-4bit",
        "qwen3_6-35b-a3b-6bit",
    ],
}

FAMILY_LABELS: dict[str, str] = {
    "gemma4": "Gemma 4",
    "qwen": "Qwen 3.6",
}

FAMILY_CHART_WIDTH = 800
FAMILY_CHART_HEIGHT = 430
FAMILY_LEFT = 64
FAMILY_RIGHT = 640
FAMILY_TOP = 86
FAMILY_BOTTOM = 316

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


@dataclass(frozen=True)
class EmbeddingDeltaRow:
    label: str
    detail: str
    reference_label: str
    reference_tok_s: float
    ax_tok_s: float
    delta_pct: float


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
        subtitle="AX direct baseline, no speculative drafting",
        unit="tok/s",
        direction_label="Higher is better",
        output_slug="gemma4-decode",
        metric="decode",
        family="gemma4",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
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
        subtitle="AX direct baseline, no speculative drafting",
        unit="tok/s",
        direction_label="Higher is better",
        output_slug="qwen-decode",
        metric="decode",
        family="qwen",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
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

MTP_6BIT_CHART_OUTPUT = "perf-mtp-6bit-ax-acceleration.svg"
MTP_PEER_CHART_OUTPUTS = {
    "decode": "perf-mtp-peer-comparison-apples-to-apples.svg",
    "prefill": "perf-mtp-peer-comparison-prefill-apples-to-apples.svg",
    "ttft": "perf-mtp-peer-comparison-ttft-apples-to-apples.svg",
    "accept": "perf-mtp-peer-comparison-accept-rate-apples-to-apples.svg",
}
MTP_6BIT_WIDTH = 1080
MTP_6BIT_HEIGHT = 844
MTP_6BIT_LEFT = 210.0
MTP_6BIT_RIGHT = 1008.0
MTP_6BIT_TOP = 68.0
MTP_6BIT_BOTTOM = 768.0
MTP_6BIT_LABEL_X = 72
MTP_6BIT_DIRECT_COLOR = "#2eaf5f"
MTP_6BIT_DIRECT_TEXT = "#176c37"
MTP_6BIT_MTP_COLOR = RED
MTP_6BIT_MTP_TEXT = "#991b1b"
MTP_6BIT_ROW_GAP = 32.0
MTP_6BIT_GROUP_GAP = 18.0
MTP_6BIT_GROUP_SIZE = 3

EMBEDDING_FAIR_ARTIFACTS = (
    Path(
        "benchmarks/results/embedding-fair/2026-06-28-qwen-after-batch-fix/"
        "2026-06-28-051508/embedding_fair.json"
    ),
    Path(
        "benchmarks/results/embedding-fair/"
        "2026-06-28-embeddinggemma-after-batch-fix/"
        "2026-06-28-051549/embedding_fair.json"
    ),
)
EMBEDDING_AX_REFRESH_ARTIFACTS = (
    Path(
        "benchmarks/results/embedding-fair/2026-06-29-qwen-refresh/"
        "2026-06-29-003717/embedding_fair.json"
    ),
    Path(
        "benchmarks/results/embedding-fair/"
        "2026-06-29-embeddinggemma-refresh/"
        "2026-06-29-003743/embedding_fair.json"
    ),
)
# Same-session paired ingest artifacts: the reference (mlx-lm / mlx-embeddings)
# and AX series are measured interleaved in one process. These are the only
# artifacts the ingest-scale charts draw from — an `--ax-only` run divided by a
# *different* run's frozen reference is rejected (see
# load_embedding_paired_scale_delta_rows), because AX throughput alone drifts
# run-to-run by more than the reported delta.
EMBEDDING_SCALE_ARTIFACT = Path(
    "benchmarks/results/embedding-scale/2026-06-29-qwen-paired-refresh/"
    "2026-06-29-003753/embedding_ingest_scale.json"
)
EMBEDDINGGEMMA_SCALE_ARTIFACT = Path(
    "benchmarks/results/embedding-scale/2026-06-29-embeddinggemma-paired-refresh/"
    "2026-06-29-004503/embedding_ingest_scale.json"
)
EMBEDDING_FAIR_CHART_OUTPUT = "perf-embedding-fair-ax-vs-reference.svg"
EMBEDDING_SCALE_CHART_OUTPUT = "perf-embedding-ingest-scale-ax-vs-mlx-lm.svg"
EMBEDDINGGEMMA_SCALE_CHART_OUTPUT = (
    "perf-embeddinggemma-ingest-scale-ax-vs-mlx-embeddings.svg"
)
EMBEDDING_CHART_WIDTH = 1080
EMBEDDING_CHART_LEFT = 360.0
EMBEDDING_CHART_RIGHT = 1012.0
EMBEDDING_CHART_TOP = 82.0
EMBEDDING_CHART_ROW_GAP = 30.0
EMBEDDING_CHART_POSITIVE = "#2eaf5f"
EMBEDDING_CHART_POSITIVE_TEXT = "#176c37"
EMBEDDING_CHART_NEGATIVE = "#dc2626"
EMBEDDING_CHART_NEGATIVE_TEXT = "#991b1b"

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
    ("ax_direct", "ax direct", "#86efac", "#16a34a"),
    ("ax_ngram", "ax + n-gram", "#2eaf5f", "#176c37"),
    ("lightning", "lightning (temp=0.6)", "#f2b705", "#9a6a00"),
    ("oracle", "oracle (bound)", "#d1d5db", "#6b7280"),
]

NGRAM_TOKS_SERIES: list[tuple[str, str, str, str]] = [
    ("ax_direct", "ax direct", "#86efac", "#16a34a"),
    ("ax_ngram", "ax + n-gram", "#2eaf5f", "#176c37"),
    ("lightning", "lightning (temp=0.6)", "#f2b705", "#9a6a00"),
]

NGRAM_ACCEPT_SERIES: list[tuple[str, str, str, str]] = [
    ("ax_ngram", "ax n-gram", "#2eaf5f", "#176c37"),
    ("lightning", "lightning (temp=0.6)", "#f2b705", "#9a6a00"),
]

NGRAM_OPPORTUNITY_CATEGORIES: list[tuple[str, str]] = [
    ("high_repeat", "high repeat"),
    ("med_repeat", "med repeat"),
    ("low_repeat", "low repeat"),
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
    raise ChartError(
        f"missing numeric median for {key} in {row.get('engine', '<unknown>')}"
    )


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


def load_rows(
    results_dir: Path, slugs: list[str], *, required: bool
) -> list[dict[str, Any]]:
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


def load_composite_rows(
    readme: Path, metric: str, slugs: list[str]
) -> list[dict[str, Any]]:
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
            if (
                source.include_engines is not None
                and engine not in source.include_engines
            ):
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
            merged[
                (str(row["_slug"]), str(engine), int(prompt_tokens), generation_tokens)
            ] = row

    return list(merged.values())


def load_llama_rows_from_readme(readme: Path) -> list[dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    table_specs = [
        ("#### Prefill throughput", "prefill", "prefill_tok_s"),
        ("#### Decode throughput", "decode", "decode_tok_s"),
        ("#### Time to first token", "ttft", "ttft_ms"),
    ]
    text = readme.read_text()
    for heading_prefix, table_name, metric_key in table_specs:
        for metric in readme_artifacts.parse_readme_table(
            text,
            heading_prefix=heading_prefix,
            table_name=table_name,
            column_map={"llama.cpp metal*": "llama_cpp_metal"},
        ):
            slug = LABEL_TO_SLUG.get((metric.model, metric.quantization))
            if slug is None:
                continue
            key = (slug, metric.prompt_tokens)
            row = rows.setdefault(
                key,
                {
                    "_slug": slug,
                    "engine": "llama_cpp_metal",
                    "prompt_tokens": metric.prompt_tokens,
                    "generation_tokens": metric.generation_tokens or 128,
                },
            )
            row[metric_key] = {"median": metric.displayed_value}
    return list(rows.values())


def series_for_chart(spec: ChartSpec) -> list[tuple[str, str, str, str]]:
    series_by_engine = {
        engine: (engine, label, color, dot_color)
        for engine, label, color, dot_color in SERIES
    }
    missing = [
        engine for engine in spec.series_engines if engine not in series_by_engine
    ]
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
                if (
                    row.get("engine") != engine
                    or row.get("prompt_tokens") != prompt_tokens
                ):
                    continue
                if spec.metric == "prefill":
                    values.append(metric_median(row, "prefill_tok_s"))
                elif spec.metric == "decode":
                    values.append(metric_median(row, "decode_tok_s"))
                elif spec.metric == "ttft":
                    # Use the pre-computed ttft_ms median for all engines.
                    # bench_mlx_inference_stack.py already derives correct
                    # per-trial TTFTs for mlx_lm/llama_cpp via
                    # attach_derived_ttft_ms, so no special-case inversion
                    # is needed here.
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
    lower_is_better = spec.metric == "ttft"
    best_med = min(all_medians) if lower_is_better else max(all_medians)
    best_line_label = "lowest median" if lower_is_better else "highest median"
    best_side = "lowest" if lower_is_better else "highest"

    n_engines = len(engine_groups)
    plot_width = FAMILY_RIGHT - FAMILY_LEFT
    plot_height = FAMILY_BOTTOM - FAMILY_TOP
    group_step = plot_width / n_engines
    sub_spacing = 30.0
    sub_bar_w = 16.0
    sub_offsets = [-sub_spacing, 0.0, sub_spacing]

    def fy(v: float) -> float:
        clamped = max(0.0, min(v, axis_max))
        return FAMILY_BOTTOM - (clamped / axis_max) * plot_height

    engine_desc = ", ".join(eg.label for eg in engine_groups)
    ctx_desc = "/".join(str(pt) for pt in PROMPT_TOKENS)
    family_label = FAMILY_LABELS.get(spec.family, spec.family)
    direction_fill = RED if lower_is_better else "#374151"

    header_right = FAMILY_CHART_WIDTH - 34
    unit_w = max(48, len(spec.unit) * 7 + 24)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{FAMILY_CHART_WIDTH}" height="{FAMILY_CHART_HEIGHT}"'
        f' viewBox="0 0 {FAMILY_CHART_WIDTH} {FAMILY_CHART_HEIGHT}"'
        f' role="img" aria-labelledby="title desc">',
        f"<title>{escape(spec.title)}</title>",
        f"<desc>Grouped box-and-whisker plot comparing {escape(engine_desc)}"
        f" at {escape(ctx_desc)} prompt tokens for {escape(family_label)} models."
        f" A red dotted line marks the {best_line_label}.</desc>",
        # Background
        f'<rect width="{FAMILY_CHART_WIDTH}" height="{FAMILY_CHART_HEIGHT}" fill="#f8fafc"/>',
        # Title
        f'<text x="{FAMILY_LEFT}" y="24" font-family="{FONT}"'
        f' font-size="16" font-weight="700" fill="#111827">{escape(spec.title)}</text>',
        # Subtitle
        f'<text x="{FAMILY_LEFT}" y="46" font-family="{FONT}"'
        f' font-size="11" fill="#4b5563">'
        f'{escape(spec.subtitle) if spec.subtitle else "box=IQR | whiskers=min/max | dots=runs | opacity: 128/512/2048 tok"}'
        f"</text>",
        # Footnote (versions)
        f'<text x="{FAMILY_LEFT}" y="62" font-family="{FONT}"'
        f' font-size="10" fill="#6b7280">{escape(DIRECT_VERSIONS_FOOTNOTE)}</text>',
        # Unit pill badge (top-right)
        f'<rect x="{header_right - unit_w}" y="13" width="{unit_w}" height="22"'
        f' rx="11" fill="#eef2ff" stroke="#c7d2fe"/>',
        f'<text x="{header_right - unit_w / 2:.1f}" y="28" text-anchor="middle"'
        f' font-family="{FONT}" font-size="10" font-weight="700"'
        f' fill="#3730a3">{escape(spec.unit)}</text>',
        # Direction label
        f'<text x="{header_right}" y="52" text-anchor="end" font-family="{FONT}"'
        f' font-size="10" font-weight="700" fill="{direction_fill}">'
        f"{escape(spec.direction_label)}</text>",
        # Plot area (white rect with border)
        f'<rect x="{FAMILY_LEFT}" y="{FAMILY_TOP}" width="{plot_width}" height="{plot_height}"'
        f' rx="6" fill="#ffffff" stroke="#dbe3ef"/>',
    ]

    # 5 grid lines (0%, 25%, 50%, 75%, 100%)
    for i in range(5):
        grid_val = axis_max * i / 4
        gy = fy(grid_val)
        lines.append(
            f'<line x1="{FAMILY_LEFT}" y1="{gy:.1f}"'
            f' x2="{FAMILY_RIGHT}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{FAMILY_LEFT - 8}" y="{gy + 3:.1f}" text-anchor="end"'
            f' font-family="{FONT}" font-size="11" fill="#6b7280">'
            f"{short_number(grid_val)}</text>"
        )

    # Best median reference line + right-side label
    best_y = fy(best_med)
    best_label_str = f"{best_side}: {short_number(best_med)}"
    lines.append(
        f'<line x1="{FAMILY_LEFT}" y1="{best_y:.1f}"'
        f' x2="{FAMILY_RIGHT}" y2="{best_y:.1f}"'
        f' stroke="{RED}" stroke-width="1.2" stroke-dasharray="1 4" stroke-linecap="round"/>'
    )
    lines.append(
        f'<text x="{FAMILY_RIGHT + 8}" y="{max(FAMILY_TOP + 11, best_y - 5):.1f}"'
        f' text-anchor="start" font-family="{FONT}"'
        f' font-size="11" font-weight="700" fill="{RED}"'
        f' data-label="{escape(best_line_label)}">'
        f"{escape(best_label_str)}</text>"
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
            label_x = box_left + sub_bar_w + 4

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
                    f'<text x="{label_x:g}" y="{y_med + 3.5:.1f}" text-anchor="start"'
                    f' font-family="{FONT}" font-size="9" font-weight="700"'
                    f' fill="#111827" stroke="#ffffff" stroke-width="3"'
                    f' paint-order="stroke">{escape(short_number(s.median))}</text>',
                    f'<text x="{sub_x:g}" y="{y_ctx}"'
                    f' text-anchor="middle" font-family="{FONT}"'
                    f' font-size="9" fill="#6b7280">{cs.prompt_tokens}</text>',
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
            f' font-size="10" font-weight="700" fill="#111827">{escape(eg.label)}</text>'
        )

    # Context-length legend (opacity shading guide)
    legend_y = FAMILY_BOTTOM + 56
    legend_x = FAMILY_LEFT
    legend_step = 120
    for pt, label in ((128, "128 tok"), (512, "512 tok"), (2048, "2048 tok")):
        fill_op = CTX_FILL_OPACITY[pt]
        stroke_op = CTX_STROKE_OPACITY[pt]
        lines.extend(
            [
                f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" rx="2"'
                f' fill="#6b7280" fill-opacity="{fill_op}"'
                f' stroke="#6b7280" stroke-opacity="{stroke_op}" stroke-width="1.4"/>',
                f'<text x="{legend_x + 14}" y="{legend_y}"'
                f' font-family="{FONT}" font-size="10" fill="#374151">{escape(label)}</text>',
            ]
        )
        legend_x += legend_step

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
        return []

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


def find_mtp_6bit_summary(readme: Path) -> Path | None:
    text = readme.read_text()
    match = re.search(
        r"\]\((benchmarks/results/mtp-6bit/[^)]+/summary\.json)\)", text
    )
    if match is None:
        return None
    summary_path = readme.parent / match.group(1)
    if not summary_path.exists():
        raise ChartError(f"README references missing MTP summary: {summary_path}")
    return summary_path


def load_mtp_6bit_rows(summary_path: Path) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ChartError(f"MTP 6-bit summary has no rows: {summary_path}")
    for row in rows:
        if not isinstance(row, dict):
            raise ChartError(f"MTP 6-bit summary row is not an object: {summary_path}")
        for key in (
            "model",
            "suite_id",
            "ax_direct_decode_tok_s",
            "ax_mtp_decode_tok_s",
            "ax_mtp_speedup_x",
        ):
            if key not in row:
                raise ChartError(f"MTP 6-bit summary row missing {key}: {summary_path}")
    return rows


def mtp_6bit_suite_label(suite_id: str) -> str:
    if suite_id == "python_modules_long":
        return "python_modules"
    return suite_id


def mtp_6bit_axis_max(rows: list[dict[str, Any]]) -> float:
    max_value = max(
        max(
            float(row["ax_direct_decode_tok_s"]),
            float(row["ax_mtp_decode_tok_s"]),
        )
        for row in rows
    )
    return max(40.0, math.ceil(max_value * 1.1 / 40.0) * 40.0)


def mtp_6bit_x_scale(value: float, axis_max: float) -> float:
    width = MTP_6BIT_RIGHT - MTP_6BIT_LEFT
    return (max(0.0, value) / axis_max) * width


def render_mtp_6bit_ax_acceleration_chart(
    rows: list[dict[str, Any]], summary_path: Path
) -> str:
    axis_max = mtp_6bit_axis_max(rows)
    tick_step = axis_max / 4.0
    model_order = tuple(dict.fromkeys(str(row["model"]) for row in rows))
    axis_bottom = (
        106.0
        + len(rows) * MTP_6BIT_ROW_GAP
        + max(0, len(model_order) - 1) * MTP_6BIT_GROUP_GAP
        + 16.0
    )
    height = int(axis_bottom + 88.0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{MTP_6BIT_WIDTH}" height="{height}" viewBox="0 0 {MTP_6BIT_WIDTH} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">AX MTP decode throughput with and without MTP</title>',
        (
            '<desc id="desc">Horizontal grouped bar chart comparing AX direct '
            "decode throughput with MTP off against AX MTP decode throughput "
            "with MTP on for each supported 6-bit AX MTP package and prompt suite. "
            "Labels show the resulting same-package speedup.</desc>"
        ),
        f'<rect width="{MTP_6BIT_WIDTH}" height="{height}" fill="#ffffff"/>',
        f'<text x="{MTP_6BIT_LABEL_X}" y="32" font-family="{FONT}" font-size="20" font-weight="700" fill="#111827">AX MTP decode: MTP off vs MTP on</text>',
        f'<text x="{MTP_6BIT_LABEL_X}" y="54" font-family="{FONT}" font-size="12" fill="#4b5563">Each row compares the same prepared 6-bit package and prompt suite: AX direct has MTP off; AX MTP has MTP on.</text>',
        f'<text x="{MTP_6BIT_RIGHT:.0f}" y="54" text-anchor="end" font-family="{FONT}" font-size="11" font-weight="700" fill="#374151">Decode throughput, tok/s</text>',
    ]

    for tick_index in range(5):
        value = tick_step * tick_index
        x = MTP_6BIT_LEFT + mtp_6bit_x_scale(value, axis_max)
        stroke = "#9ca3af" if tick_index == 0 else "#d1d5db"
        width = "1.4" if tick_index == 0 else "1.2"
        lines.extend(
            [
                f'<line x1="{x:.1f}" y1="{MTP_6BIT_TOP:.0f}" x2="{x:.1f}" y2="{axis_bottom:.1f}" stroke="{stroke}" stroke-width="{width}"/>',
                f'<text x="{x:.1f}" y="{axis_bottom + 18.0:.1f}" text-anchor="middle" font-family="{FONT}" font-size="11" fill="#4b5563">{short_number(value)}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{(MTP_6BIT_LEFT + MTP_6BIT_RIGHT) / 2:.1f}" y="{axis_bottom + 40.0:.1f}" text-anchor="middle" font-family="{FONT}" font-size="11" fill="#6b7280">Higher is better</text>',
            f'<rect x="{MTP_6BIT_LABEL_X}" y="70" width="12" height="12" rx="2" fill="{MTP_6BIT_DIRECT_COLOR}"/>',
            f'<text x="90" y="80" font-family="{FONT}" font-size="12" fill="#374151">MTP off / AX direct</text>',
            f'<rect x="232" y="70" width="12" height="12" rx="2" fill="{MTP_6BIT_MTP_COLOR}"/>',
            f'<text x="250" y="80" font-family="{FONT}" font-size="12" fill="#374151">MTP on / AX MTP</text>',
            f'<text x="414" y="80" font-family="{FONT}" font-size="12" fill="#6b7280">Speedup label = MTP on / MTP off</text>',
        ]
    )

    previous_model: str | None = None
    label_y = 106.0
    for row in rows:
        model = str(row["model"])
        if model != previous_model:
            if previous_model is not None:
                separator_y = label_y - MTP_6BIT_GROUP_GAP / 2.0
                lines.append(
                    f'<line x1="{MTP_6BIT_LABEL_X}" y1="{separator_y:.1f}" x2="{MTP_6BIT_RIGHT:.0f}" y2="{separator_y:.1f}" stroke="#eef2f7" stroke-width="1"/>'
                )
                label_y += MTP_6BIT_GROUP_GAP
            previous_model = model

        suite = mtp_6bit_suite_label(str(row["suite_id"]))
        direct = float(row["ax_direct_decode_tok_s"])
        mtp = float(row["ax_mtp_decode_tok_s"])
        speedup = float(row["ax_mtp_speedup_x"])
        direct_width = mtp_6bit_x_scale(direct, axis_max)
        mtp_width = mtp_6bit_x_scale(mtp, axis_max)
        direct_end = MTP_6BIT_LEFT + direct_width
        mtp_end = MTP_6BIT_LEFT + mtp_width

        lines.extend(
            [
                (
                    f'<text x="{MTP_6BIT_LABEL_X}" y="{label_y - 6.0:.1f}"'
                    f' font-family="{FONT}" font-size="11" fill="#111827">'
                    f'<tspan x="{MTP_6BIT_LABEL_X}" y="{label_y - 6.0:.1f}"'
                    f' font-weight="700">{escape(model)}</tspan>'
                    f'<tspan x="{MTP_6BIT_LABEL_X}" y="{label_y + 8.0:.1f}"'
                    f' font-size="10" fill="#4b5563">{escape(suite)}</tspan></text>'
                ),
                f'<rect x="{MTP_6BIT_LEFT:.0f}" y="{label_y - 15.0:.1f}" width="{direct_width:.1f}" height="10" rx="2" fill="{MTP_6BIT_DIRECT_COLOR}"/>',
                f'<rect x="{MTP_6BIT_LEFT:.0f}" y="{label_y - 2.0:.1f}" width="{mtp_width:.1f}" height="10" rx="2" fill="{MTP_6BIT_MTP_COLOR}"/>',
                f'<text x="{direct_end + 5.0:.1f}" y="{label_y - 7.0:.1f}" font-family="{FONT}" font-size="10" font-weight="700" fill="{MTP_6BIT_DIRECT_TEXT}">{direct:.1f}</text>',
                f'<text x="{mtp_end + 5.0:.1f}" y="{label_y + 6.0:.1f}" font-family="{FONT}" font-size="10" font-weight="700" fill="{MTP_6BIT_MTP_TEXT}">{mtp:.1f}</text>',
                f'<text x="{mtp_end + 52.0:.1f}" y="{label_y:.1f}" font-family="{FONT}" font-size="11" font-weight="700" fill="#111827">{speedup:.2f}x</text>',
            ]
        )
        label_y += MTP_6BIT_ROW_GAP

    source_label = (
        f"Source: {summary_path.parent.as_posix()} / summary.json. "
        "Pure MTP; no MTP+n-gram stacking."
    )
    lines.append(
        f'<text x="{MTP_6BIT_LABEL_X}" y="{axis_bottom + 64.0:.1f}" font-family="{FONT}" font-size="10" fill="#6b7280">{escape(source_label)}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


MTP_PEER_WIDTH = 1120
MTP_PEER_LEFT = 248.0
MTP_PEER_RIGHT = 1052.0
MTP_PEER_TOP = 112.0
MTP_PEER_ROW_GAP = 118.0
MTP_PEER_BAR_H = 20.0
MTP_PEER_COLORS = {
    "ax_engine": "#2eaf5f",
    "mtplx": "#f2b705",
    "lightning_mlx": "#2563eb",
}
MTP_PEER_LABELS = {
    "ax_engine": "AX Engine",
    "mtplx": "MTPLX",
    "lightning_mlx": "lightning-mlx",
}
# Engine versions behind the peer-comparison artifacts, surfaced on each chart so the
# run is reproducible. Update alongside any re-benchmark. Provenance:
#   AX Engine     = [workspace.package] version in Cargo.toml
#   MTPLX         = /opt/homebrew/var/mtplx/venv-1.0.4 (pip: mtplx 1.0.4)
#   lightning-mlx = .internal/reference/lightning-mlx v0.7.0 (git rev ec19b3d, incl. post-tag streaming fix #3)
MTP_PEER_VERSIONS = {
    "ax_engine": "6.6.1",
    "mtplx": "1.0.4",
    "lightning_mlx": "0.7.0",
}
MTP_PEER_METRICS = {
    "decode": {
        "field": "decode_tok_s",
        "title": "Qwen3.6 MTP peer comparison decode",
        "desc": "Decode throughput, tokens per second.",
        "unit": "tok/s",
        "suffix": "",
        "higher_is_better": True,
    },
    "prefill": {
        "field": "prefill_tok_s",
        "title": "Qwen3.6 MTP peer comparison prefill",
        "desc": "Prefill throughput, tokens per second.",
        "unit": "tok/s",
        "suffix": "",
        "higher_is_better": True,
    },
    "ttft": {
        "field": "ttft_ms",
        "title": "Qwen3.6 MTP peer comparison TTFT",
        "desc": "Time to first token, milliseconds.",
        "unit": "ms",
        "suffix": "",
        "higher_is_better": False,
    },
    "accept": {
        "field": "accept_rate",
        "title": "Qwen3.6 MTP peer comparison accept rate",
        "desc": "MTP draft-token acceptance rate.",
        "unit": "%",
        "suffix": "%",
        "higher_is_better": True,
    },
}


def find_mtp_peer_summary(readme: Path) -> Path | None:
    text = readme.read_text()
    match = re.search(
        r"\]\((benchmarks/results/mtp-qwen36-matrix/[^)]+peer-comparison-apples-to-apples(?:-refresh)?/summary\.json)\)",
        text,
    )
    if match is None:
        return None
    summary_path = readme.parent / match.group(1)
    if not summary_path.exists():
        raise ChartError(f"README references missing MTP peer summary: {summary_path}")
    return summary_path


def load_mtp_peer_rows(summary_path: Path) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    rows = []
    for row in summary.get("rows", []):
        if not isinstance(row, dict):
            continue
        engine = row.get("engine")
        metrics = row.get("metrics")
        if (
            engine not in MTP_PEER_LABELS
            or not isinstance(metrics, dict)
            or metrics.get("status") != "ok"
        ):
            continue
        decode = metrics.get("decode_tok_s")
        if not isinstance(decode, int | float):
            continue
        rows.append(row)
    if not rows:
        raise ChartError(f"MTP peer summary has no chartable rows: {summary_path}")
    return rows


def render_mtp_peer_comparison_chart(
    rows: list[dict[str, Any]], summary_path: Path, metric_key: str
) -> str:
    summary = json.loads(summary_path.read_text())
    contract_label = str(
        summary.get("contract", {}).get("benchmark_contract", "production-configuration")
    ).replace("-", " ")
    metric_config = MTP_PEER_METRICS[metric_key]
    metric_field = str(metric_config["field"])
    targets = []
    for row in rows:
        label = str(row["model_label"])
        if label not in targets:
            targets.append(label)
    height = int(MTP_PEER_TOP + len(targets) * MTP_PEER_ROW_GAP + 88)
    values: list[float] = []
    for row in rows:
        metric_value = row["metrics"].get(metric_field)
        if isinstance(metric_value, int | float):
            value = float(metric_value)
            values.append(value * 100.0 if metric_key == "accept" else value)
    if not values:
        raise ChartError(f"MTP peer summary has no {metric_key} values")
    axis_max = 100.0 if metric_key == "accept" else nice_axis_ceiling(max(values) * 1.15)

    def x_scale(value: float) -> float:
        return (max(0.0, value) / axis_max) * (MTP_PEER_RIGHT - MTP_PEER_LEFT)

    by_target_engine = {}
    degenerate_by_target_engine = {}
    for row in rows:
        metric_value = row["metrics"].get(metric_field)
        if isinstance(metric_value, int | float):
            value = float(metric_value)
            key = (str(row["model_label"]), str(row["engine"]))
            by_target_engine[key] = (
                value * 100.0 if metric_key == "accept" else value
            )
            degenerate_by_target_engine[key] = bool(
                row["metrics"].get("degeneracy_gate", {}).get("degenerate")
            )
    higher_is_better = bool(metric_config["higher_is_better"])
    direction = "Higher is better" if higher_is_better else "Lower is better"
    # Mark the winning engine per target for every metric, respecting direction:
    # the max for higher-is-better (decode/prefill/accept), the min for ttft.
    best_by_target: dict[str, float] = {}
    for target in targets:
        target_values = [
            value
            for (row_target, _engine), value in by_target_engine.items()
            if row_target == target
            and not degenerate_by_target_engine.get((row_target, _engine), False)
        ]
        if len(target_values) > 1:
            best_by_target[target] = (
                max(target_values) if higher_is_better else min(target_values)
            )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{MTP_PEER_WIDTH}" height="{height}" viewBox="0 0 {MTP_PEER_WIDTH} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(str(metric_config["title"]))} {escape(contract_label)}</title>',
        (
            '<desc id="desc">Grouped horizontal bar chart comparing AX Engine, '
            f'MTPLX, and lightning-mlx {escape(str(metric_config["unit"]))} on '
            "the flappy prompt suite. Degenerate rows are shown muted and are "
            "not eligible for best-row highlighting.</desc>"
        ),
        f'<rect width="{MTP_PEER_WIDTH}" height="{height}" fill="#f8fafc"/>',
        f'<text x="32" y="34" font-family="{FONT}" font-size="22" font-weight="700" fill="#111827">{escape(str(metric_config["title"]))}</text>',
        f'<text x="32" y="60" font-family="{FONT}" font-size="14" fill="#374151">flappy suite · 1000 generated tokens · 5 measured reps · 2 warmups · 15s cooldown · {escape(contract_label)}</text>',
        f'<text x="32" y="80" font-family="{FONT}" font-size="13" fill="#6b7280">{escape(str(metric_config["desc"]))}</text>',
    ]
    lines.append(
        f'<text x="{MTP_PEER_RIGHT:.1f}" y="80" text-anchor="end" font-family="{FONT}" font-size="13" font-weight="700" fill="#dc2626">{escape(direction)}</text>'
    )
    for tick_index in range(5):
        tick = axis_max * tick_index / 4.0
        x = MTP_PEER_LEFT + x_scale(tick)
        stroke = "#cbd5e1" if tick_index == 0 else "#e5e7eb"
        lines.extend(
            [
                f'<line x1="{x:.1f}" y1="{MTP_PEER_TOP - 26:.1f}" x2="{x:.1f}" y2="{height - 66}" stroke="{stroke}" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{height - 43}" text-anchor="middle" font-family="{FONT}" font-size="12" fill="#4b5563">{short_number(tick)}</text>',
            ]
        )
    # No standalone axis unit label: it collided with the last (rightmost) tick,
    # and the unit is already stated in the description line above the plot.
    legend_x = 648.0
    for engine, label in MTP_PEER_LABELS.items():
        lines.extend(
            [
                f'<rect x="{legend_x:.1f}" y="25" width="13" height="13" rx="2" fill="{MTP_PEER_COLORS[engine]}"/>',
                f'<text x="{legend_x + 19:.1f}" y="37" font-family="{FONT}" font-size="13" fill="#374151">{escape(label)}</text>',
            ]
        )
        legend_x += 136.0
    for index, target in enumerate(targets):
        base_y = MTP_PEER_TOP + index * MTP_PEER_ROW_GAP
        short_target = target.replace("Qwen3.6 ", "")
        lines.append(
            f'<text x="32" y="{base_y + 18:.1f}" font-family="{FONT}" font-size="15" font-weight="700" fill="#111827">{escape(short_target)}</text>'
        )
        for engine_index, engine in enumerate(MTP_PEER_LABELS):
            value = by_target_engine.get((target, engine))
            y = base_y + 32.0 + engine_index * (MTP_PEER_BAR_H + 10.0)
            lines.append(
                f'<text x="{MTP_PEER_LEFT - 14:.1f}" y="{y + 15:.1f}" text-anchor="end" font-family="{FONT}" font-size="12" fill="#374151">{escape(MTP_PEER_LABELS[engine])}</text>'
            )
            if value is None:
                lines.append(
                    f'<text x="{MTP_PEER_LEFT:.1f}" y="{y + 15:.1f}" font-family="{FONT}" font-size="12" fill="#9ca3af">unsupported</text>'
                )
                continue
            width = x_scale(value)
            # Best value per target is highlighted in red; no text suffix.
            is_degenerate = degenerate_by_target_engine.get((target, engine), False)
            is_best = (
                target in best_by_target
                and value == best_by_target[target]
                and not is_degenerate
            )
            value_fill = "#dc2626" if is_best else "#111827"
            opacity = "0.38" if is_degenerate else "1"
            label_suffix = " degenerate" if is_degenerate else ""
            lines.extend(
                [
                    f'<rect x="{MTP_PEER_LEFT:.1f}" y="{y:.1f}" width="{width:.1f}" height="{MTP_PEER_BAR_H:.1f}" rx="3" fill="{MTP_PEER_COLORS[engine]}" opacity="{opacity}"/>',
                    f'<text x="{MTP_PEER_LEFT + width + 8:.1f}" y="{y + 15:.1f}" font-family="{FONT}" font-size="12" font-weight="700" fill="{value_fill}">{value:.1f}{escape(str(metric_config["suffix"]))}{escape(label_suffix)}</text>',
                ]
            )
    versions_text = " · ".join(
        f"{MTP_PEER_LABELS[engine]} {version}"
        for engine, version in MTP_PEER_VERSIONS.items()
    )
    source_label = (
        f"Source: {summary_path.parent.name}/summary.json"
        f"   ·   Versions: {versions_text}"
    )
    lines.append(
        f'<text x="32" y="{height - 18}" font-family="{FONT}" font-size="11" fill="#6b7280">{escape(source_label)}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def infer_results_dir_from_readme(readme: Path) -> Path:
    text = readme.read_text()
    patterns = [
        r"readme-performance-artifacts:[^\n]*base=([^;\s]+)",
        r"readme-performance-artifacts:[^\n]*reference=([^;\s]+)",
        r"Artifact directory: `([^`]+)`",
        r"artifacts are in\s+`([^`]+)`",
        r"Source:\s+`([^`]+)`\s+for all rows",
    ]
    match = next(
        (match for pattern in patterns if (match := re.search(pattern, text))), None
    )
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


def _ngram_category_median(
    results: list[dict], category: str, key: str
) -> float | None:
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
            val = _ngram_category_median(
                artifact.get(s_key, []), cat_id, "tok_s_median"
            )
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
    subtitle = (
        "all paths: temp=0.6/top_p=0.95/top_k=20 · oracle: theoretical upper bound"
    )

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

            lines.extend(
                [
                    f'<rect x="{bar_left:.1f}" y="{y_top:.1f}"'
                    f' width="{bar_w:.0f}" height="{bar_h:.1f}" rx="2"'
                    f' fill="{color}" fill-opacity="0.30" stroke="{dot_color}" stroke-width="1.6"/>',
                    f'<line x1="{bar_left:.1f}" y1="{y_top:.1f}"'
                    f' x2="{bar_left + bar_w:.1f}" y2="{y_top:.1f}"'
                    f' stroke="{dot_color}" stroke-width="2.4"/>',
                ]
            )
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
        lines.extend(
            [
                f'<rect x="{lx}" y="{legend_y}" width="{leg_box}" height="{leg_box}"'
                f' rx="2" fill="{color}" fill-opacity="0.40" stroke="{dot_color}" stroke-width="1.4"/>',
                f'<text x="{lx + leg_box + leg_gap}" y="{legend_y + 9}"'
                f' font-family="{FONT}" font-size="9" fill="#374151">{escape(s_label)}</text>',
            ]
        )

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
            f"{y_fmt.format(grid_val)}</text>"
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

            lines.extend(
                [
                    f'<rect x="{bar_left:.1f}" y="{y_top:.1f}"'
                    f' width="{bar_w:.0f}" height="{bar_h:.1f}" rx="2"'
                    f' fill="{color}" fill-opacity="0.30" stroke="{dot_color}" stroke-width="1.6"/>',
                    f'<line x1="{bar_left:.1f}" y1="{y_top:.1f}"'
                    f' x2="{bar_left + bar_w:.1f}" y2="{y_top:.1f}"'
                    f' stroke="{dot_color}" stroke-width="2.4"/>',
                ]
            )
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
        lines.extend(
            [
                f'<rect x="{lx}" y="{legend_y}" width="{leg_box}" height="{leg_box}"'
                f' rx="2" fill="{color}" fill-opacity="0.40" stroke="{dot_color}" stroke-width="1.4"/>',
                f'<text x="{lx + leg_box + leg_gap}" y="{legend_y + 9}"'
                f' font-family="{FONT}" font-size="9" fill="#374151">{escape(s_label)}</text>',
            ]
        )

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
            val = _ngram_category_median(
                artifact.get(s_key, []), cat_id, "tok_s_median"
            )
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
    accept_keys = {
        "ax_ngram": "ngram_accept_rate",
        "lightning": "lightning_accept_rate",
    }
    for s_key, _label, _color, _dot in series:
        data[s_key] = {}
        rate_key = accept_keys[s_key]
        for cat_id, _cat_label in categories:
            val = _ngram_category_median(artifact.get(s_key, []), cat_id, rate_key)
            data[s_key][cat_id] = (val * 100.0) if val is not None else 0.0

    all_vals = [v for cat_vals in data.values() for v in cat_vals.values() if v > 0]
    # Scale to actual data with 25% headroom, capped at 100%
    axis_max = (
        min(100.0, nice_axis_ceiling(max(all_vals) * 1.25)) if all_vals else 100.0
    )

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
    "qwen3-4b-4bit": "Qwen3-4B",
    "glm-4-7-flash-4bit": "GLM-4.7F",
    "gemma-4-e2b-4bit": "E2B 4bit",
    "gemma-4-e2b-6bit": "E2B 6bit",
    "gemma-4-e4b-4bit": "E4B 4bit",
    "gemma-4-26b-a4b-4bit": "G26B-A4B",
    "gemma-4-31b-4bit": "G31B 4bit",
    "qwen3-6-27b-4bit": "Q27B 4bit",
    "qwen3-6-27b-6bit": "Q27B 6bit",
    "qwen3-6-35b-a3b-4bit": "Q35B-A3B",
}

# Preferred order for model display in multi-model charts (small → large).
NGRAM_MODEL_ORDER = [
    "qwen3-4b-4bit",
    "glm-4-7-flash-4bit",
    "gemma-4-e2b-4bit",
    "gemma-4-e2b-6bit",
    "gemma-4-e4b-4bit",
    "gemma-4-26b-a4b-4bit",
    "gemma-4-31b-4bit",
    "qwen3-6-27b-4bit",
    "qwen3-6-27b-6bit",
    "qwen3-6-35b-a3b-4bit",
]


def find_ngram_artifacts_by_model(repo_root: Path) -> dict[str, dict]:
    """Return {model_slug: artifact_dict} using the latest `-ngram` run per slug."""
    base = repo_root / "benchmarks" / "results" / "ngram-compare"
    result: dict[str, dict] = {}
    for slug in NGRAM_MODEL_ORDER:
        # Prefer the `-ngram` suffixed run, fall back to any run containing the slug.
        candidates = sorted(
            base.glob(f"*{slug}*/artifact.json"), key=lambda p: p.parent.name
        )
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
        d = (
            _ngram_category_median(
                art.get("ax_direct", []), "high_repeat", "tok_s_median"
            )
            or 0.0
        )
        n = (
            _ngram_category_median(
                art.get("ax_ngram", []), "high_repeat", "tok_s_median"
            )
            or 0.0
        )
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
    subtitle = (
        "temp=0.6/top_p=0.95/top_k=20 · bars: ax direct (light) / ax+n-gram (dark)"
    )

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
        lines.append(
            f'<line x1="{LEFT}" y1="{gy:.1f}" x2="{RIGHT}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{LEFT - 6}" y="{gy + 3:.1f}" text-anchor="end" font-family="{FONT}" font-size="10" fill="#6b7280">{short_number(grid_val)}</text>'
        )

    group_step = plot_w / n_models
    for gi, slug in enumerate(models):
        group_center = LEFT + (gi + 0.5) * group_step
        bar0 = group_center - group_block / 2
        d_val, n_val = direct_vals[gi], ngram_vals[gi]
        label = NGRAM_MODEL_DISPLAY.get(slug, slug)

        for bi, (val, color, dot_color) in enumerate(
            [
                (d_val, "#86efac", "#16a34a"),
                (n_val, "#2eaf5f", "#176c37"),
            ]
        ):
            bx = bar0 + bi * (bar_w + bar_gap)
            bc = bx + bar_w / 2
            y_top = fy(val)
            bh = BOTTOM - y_top
            lines.extend(
                [
                    f'<rect x="{bx:.1f}" y="{y_top:.1f}" width="{bar_w:.0f}" height="{bh:.1f}" rx="2"'
                    f' fill="{color}" fill-opacity="0.35" stroke="{dot_color}" stroke-width="1.4"/>',
                    f'<line x1="{bx:.1f}" y1="{y_top:.1f}" x2="{bx + bar_w:.1f}" y2="{y_top:.1f}" stroke="{dot_color}" stroke-width="2.2"/>',
                ]
            )
            if val > 0:
                lines.append(
                    f'<text x="{bc:.1f}" y="{y_top - 3:.1f}" text-anchor="middle" font-family="{FONT}" font-size="7.5" fill="{dot_color}">{val:.0f}</text>'
                )

        lines.append(
            f'<text x="{group_center:.1f}" y="{BOTTOM + 14}" text-anchor="middle" font-family="{FONT}" font-size="8.5" font-weight="700" fill="#111827">{escape(label)}</text>'
        )

        if d_val > 0 and n_val > 0:
            speedup = n_val / d_val
            sy = fy(n_val) - 13
            lines.append(
                f'<text x="{group_center:.1f}" y="{sy:.1f}" text-anchor="middle" font-family="{FONT}" font-size="7.5" font-weight="700" fill="#176c37">{speedup:.2f}×</text>'
            )

    ly = BOTTOM + 32
    for li, (label, color, dot_color) in enumerate(
        [
            ("ax direct", "#86efac", "#16a34a"),
            ("ax + n-gram", "#2eaf5f", "#176c37"),
        ]
    ):
        lx = LEFT + li * 120
        lines.extend(
            [
                f'<rect x="{lx}" y="{ly}" width="10" height="10" rx="2" fill="{color}" fill-opacity="0.4" stroke="{dot_color}" stroke-width="1.4"/>',
                f'<text x="{lx + 14}" y="{ly + 9}" font-family="{FONT}" font-size="9" fill="#374151">{escape(label)}</text>',
            ]
        )

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
        a = _ngram_category_median(
            art.get("ax_ngram", []), "high_repeat", "ngram_accept_rate"
        )
        l = _ngram_category_median(
            art.get("lightning", []), "high_repeat", "lightning_accept_rate"
        )
        ax_vals.append((a * 100.0) if a is not None else 0.0)
        lt_val = (l * 100.0) if l is not None else 0.0
        lt_vals.append(lt_val)
        if lt_val > 0:
            has_lightning = True

    all_vals = [v for v in ax_vals + lt_vals if v > 0]
    axis_max = (
        min(100.0, nice_axis_ceiling(max(all_vals) * 1.25)) if all_vals else 100.0
    )

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
        lines.append(
            f'<line x1="{LEFT}" y1="{gy:.1f}" x2="{RIGHT}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{LEFT - 6}" y="{gy + 3:.1f}" text-anchor="end" font-family="{FONT}" font-size="10" fill="#6b7280">{grid_val:.0f}%</text>'
        )

    group_step = plot_w / n_models
    bar_series = [
        (ax_vals, "#2eaf5f", "#176c37", "ax n-gram"),
        (lt_vals, "#f2b705", "#9a6a00", "lightning"),
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
            lines.extend(
                [
                    f'<rect x="{bx:.1f}" y="{y_top:.1f}" width="{bar_w:.0f}" height="{bh:.1f}" rx="2"'
                    f' fill="{color}" fill-opacity="0.35" stroke="{dot_color}" stroke-width="1.4"/>',
                    f'<line x1="{bx:.1f}" y1="{y_top:.1f}" x2="{bx + bar_w:.1f}" y2="{y_top:.1f}" stroke="{dot_color}" stroke-width="2.2"/>',
                ]
            )
            if val > 0:
                lines.append(
                    f'<text x="{bc:.1f}" y="{y_top - 3:.1f}" text-anchor="middle" font-family="{FONT}" font-size="7.5" fill="{dot_color}">{val:.0f}%</text>'
                )

        lines.append(
            f'<text x="{group_center:.1f}" y="{BOTTOM + 14}" text-anchor="middle" font-family="{FONT}" font-size="8.5" font-weight="700" fill="#111827">{escape(label)}</text>'
        )

    ly = BOTTOM + 32
    for li, (_, color, dot_color, label) in enumerate(bar_series):
        lx = LEFT + li * 120
        lines.extend(
            [
                f'<rect x="{lx}" y="{ly}" width="10" height="10" rx="2" fill="{color}" fill-opacity="0.4" stroke="{dot_color}" stroke-width="1.4"/>',
                f'<text x="{lx + 14}" y="{ly + 9}" font-family="{FONT}" font-size="9" fill="#374151">{escape(label)}</text>',
            ]
        )

    lines.append("</svg>")
    return "".join(lines) + "\n"


def embedding_reference_key(artifact: dict[str, Any]) -> tuple[str, str]:
    reference = artifact.get("reference", "mlx_lm")
    if reference == "mlx_embeddings":
        return "mlx_embeddings", "mlx-embeddings"
    return "mlx_lm", "mlx-lm"


def embedding_model_label(label: str) -> str:
    return {
        "qwen3-embedding-0.6b-8bit": "Qwen3 0.6B 8-bit",
        "qwen3-embedding-4b-4bit-dwq": "Qwen3 4B 4-bit DWQ",
        "qwen3-embedding-8b-4bit-dwq": "Qwen3 8B 4-bit DWQ",
        "embeddinggemma-300m-8bit": "EmbeddingGemma 300M 8-bit",
    }.get(label, label)


def embedding_workload_label(workload: str) -> str:
    return {
        "short_query_b8": "short query",
        "fixed_64_b8": "64-token chunks",
        "fixed_256_b8": "256-token chunks",
    }.get(workload, workload)


def load_embedding_fair_delta_rows(repo_root: Path) -> list[EmbeddingDeltaRow]:
    rows: list[EmbeddingDeltaRow] = []
    wanted_workloads = ("short_query_b8", "fixed_64_b8", "fixed_256_b8")
    for ref_relative_path, ax_relative_path in zip(
        EMBEDDING_FAIR_ARTIFACTS, EMBEDDING_AX_REFRESH_ARTIFACTS, strict=True
    ):
        ref_path = repo_root / ref_relative_path
        ax_path = repo_root / ax_relative_path
        if not ref_path.exists():
            raise ChartError(f"missing embedding fair artifact: {ref_path}")
        if not ax_path.exists():
            raise ChartError(f"missing embedding AX refresh artifact: {ax_path}")
        ref_artifact = json.loads(ref_path.read_text())
        ax_artifact = json.loads(ax_path.read_text())
        ref_key, ref_label = embedding_reference_key(ref_artifact)
        ax_by_model = {
            str(model.get("model_label")): {
                str(row.get("workload")): row for row in model.get("rows", [])
            }
            for model in ax_artifact.get("models", [])
        }
        for model in ref_artifact.get("models", []):
            raw_model_label = str(model.get("model_label", ""))
            model_label = embedding_model_label(raw_model_label)
            ref_by_workload = {
                str(row.get("workload")): row for row in model.get("rows", [])
            }
            ax_model_rows = ax_by_model.get(raw_model_label)
            if ax_model_rows is None:
                raise ChartError(f"{ax_path} missing model {raw_model_label}")
            for workload in wanted_workloads:
                ref_row = ref_by_workload.get(workload)
                ax_row = ax_model_rows.get(workload)
                if ref_row is None:
                    raise ChartError(f"{ref_path} missing workload {workload}")
                if ax_row is None:
                    raise ChartError(f"{ax_path} missing workload {workload}")
                ref_results = ref_row.get("results", {})
                ax_results = ax_row.get("results", {})
                ref = ref_results.get(ref_key)
                ax = ax_results.get("ax_engine_py")
                if not isinstance(ref, dict) or not isinstance(ax, dict):
                    raise ChartError(
                        f"missing embedding results for {raw_model_label} {workload}"
                    )
                ref_tps = float(ref["median_tokens_per_sec"])
                ax_tps = float(ax["median_tokens_per_sec"])
                delta = ((ax_tps - ref_tps) / ref_tps * 100.0) if ref_tps else 0.0
                rows.append(
                    EmbeddingDeltaRow(
                        label=model_label,
                        detail=embedding_workload_label(workload),
                        reference_label=ref_label,
                        reference_tok_s=ref_tps,
                        ax_tok_s=ax_tps,
                        delta_pct=delta,
                    )
                )
    return rows


def load_embedding_paired_scale_delta_rows(
    repo_root: Path, artifact_relative_path: Path
) -> list[EmbeddingDeltaRow]:
    """Load AX-vs-reference deltas from one *same-session* paired ingest artifact.

    Both the reference (mlx-lm / mlx-embeddings) and the AX series are read from
    the same artifact, so they were measured interleaved in a single process —
    the only methodology that controls for thermal and run-to-run drift.

    Artifacts produced with ``--ax-only`` carry no reference series and are
    rejected: dividing a fresh AX run by a *different* run's frozen reference is
    a cross-run comparison, and the same default binary drifts in AX throughput
    alone by more than the deltas these charts report.
    """
    artifact_path = repo_root / artifact_relative_path
    if not artifact_path.exists():
        raise ChartError(f"missing embedding scale artifact: {artifact_path}")
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("ax_only"):
        raise ChartError(
            f"{artifact_path} is an ax_only artifact; ingest-scale charts require a "
            "same-session paired run (omit --ax-only) so the reference is measured "
            "in the same process as AX"
        )
    ref_key, ref_label = embedding_reference_key(artifact)
    rows: list[EmbeddingDeltaRow] = []
    for model in artifact.get("models", []):
        model_label = embedding_model_label(str(model.get("model_label", "")))
        for row in model.get("rows", []):
            results = row.get("results", {})
            ref = results.get(ref_key)
            ax = results.get("ax_engine_py")
            if not isinstance(ref, dict) or not isinstance(ax, dict):
                raise ChartError(
                    f"{artifact_path} {row.get('workload')} lacks paired "
                    f"{ref_key}/ax_engine_py results"
                )
            ref_tps = float(ref["median_tokens_per_sec"])
            ax_tps = float(ax["median_tokens_per_sec"])
            delta = ((ax_tps - ref_tps) / ref_tps * 100.0) if ref_tps else 0.0
            rows.append(
                EmbeddingDeltaRow(
                    label=model_label,
                    detail=(
                        f"{row['total_chunks']} x {row['chunk_tokens']} tok, "
                        f"batch {row['batch_size']}"
                    ),
                    reference_label=ref_label,
                    reference_tok_s=ref_tps,
                    ax_tok_s=ax_tps,
                    delta_pct=delta,
                )
            )
    return rows


def load_embedding_scale_delta_rows(repo_root: Path) -> list[EmbeddingDeltaRow]:
    return load_embedding_paired_scale_delta_rows(repo_root, EMBEDDING_SCALE_ARTIFACT)


def render_embedding_delta_chart(
    rows: list[EmbeddingDeltaRow],
    *,
    title: str,
    subtitle: str,
    source_label: str,
) -> str:
    if not rows:
        raise ChartError(f"{title} has no rows")
    max_abs = max(abs(row.delta_pct) for row in rows)
    axis_abs = max(5.0, math.ceil((max_abs + 1.0) / 5.0) * 5.0)
    chart_height = int(EMBEDDING_CHART_TOP + len(rows) * EMBEDDING_CHART_ROW_GAP + 76)
    plot_width = EMBEDDING_CHART_RIGHT - EMBEDDING_CHART_LEFT
    zero_x = EMBEDDING_CHART_LEFT + plot_width / 2.0

    def fx(delta_pct: float) -> float:
        clamped = max(-axis_abs, min(delta_pct, axis_abs))
        return EMBEDDING_CHART_LEFT + ((clamped + axis_abs) / (2.0 * axis_abs)) * plot_width

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{EMBEDDING_CHART_WIDTH}"'
        f' height="{chart_height}" viewBox="0 0 {EMBEDDING_CHART_WIDTH} {chart_height}"'
        f' role="img" aria-labelledby="title desc">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(subtitle)} Bars show AX Engine throughput percentage difference versus the reference backend; zero means parity.</desc>",
        f'<rect width="{EMBEDDING_CHART_WIDTH}" height="{chart_height}" fill="#f8fafc"/>',
        f'<text x="44" y="30" font-family="{FONT}" font-size="18" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="44" y="52" font-family="{FONT}" font-size="12" fill="#4b5563">{escape(subtitle)}</text>',
        f'<text x="{EMBEDDING_CHART_RIGHT:.0f}" y="52" text-anchor="end" font-family="{FONT}" font-size="11" font-weight="700" fill="#374151">AX vs reference, tok/s delta</text>',
    ]

    for tick in (-axis_abs, -axis_abs / 2.0, 0.0, axis_abs / 2.0, axis_abs):
        x = fx(tick)
        stroke = "#cbd5e1" if tick == 0 else "#e5e7eb"
        width = "1.6" if tick == 0 else "1"
        lines.append(
            f'<line x1="{x:.1f}" y1="68" x2="{x:.1f}" y2="{chart_height - 50}" stroke="{stroke}" stroke-width="{width}"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{chart_height - 30}" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#6b7280">{tick:+.0f}%</text>'
        )

    lines.append(
        f'<text x="{zero_x:.1f}" y="{chart_height - 14}" text-anchor="middle" font-family="{FONT}" font-size="10" font-weight="700" fill="#6b7280">0% = reference backend</text>'
    )

    for idx, row in enumerate(rows):
        y = EMBEDDING_CHART_TOP + idx * EMBEDDING_CHART_ROW_GAP
        end_x = fx(row.delta_pct)
        bar_x = min(zero_x, end_x)
        bar_w = max(abs(end_x - zero_x), 1.0)
        positive = row.delta_pct >= 0
        color = EMBEDDING_CHART_POSITIVE if positive else EMBEDDING_CHART_NEGATIVE
        text_color = (
            EMBEDDING_CHART_POSITIVE_TEXT if positive else EMBEDDING_CHART_NEGATIVE_TEXT
        )
        label_x = end_x + 6 if positive else end_x - 6
        anchor = "start" if positive else "end"
        label_stroke = "#ffffff"
        if positive and end_x > EMBEDDING_CHART_RIGHT - 92:
            label_x = end_x - 6
            anchor = "end"
            text_color = "#ffffff"
            label_stroke = EMBEDDING_CHART_POSITIVE_TEXT
        elif not positive and end_x < EMBEDDING_CHART_LEFT + 92:
            label_x = end_x + 6
            anchor = "start"
            text_color = "#ffffff"
            label_stroke = EMBEDDING_CHART_NEGATIVE_TEXT
        lines.extend(
            [
                f'<text x="44" y="{y + 4:.1f}" font-family="{FONT}" font-size="11" font-weight="700" fill="#111827">{escape(row.label)}</text>',
                f'<text x="44" y="{y + 18:.1f}" font-family="{FONT}" font-size="10" fill="#6b7280">{escape(row.detail)} · ref {escape(row.reference_label)}</text>',
                f'<rect x="{bar_x:.1f}" y="{y - 8:.1f}" width="{bar_w:.1f}" height="14" rx="3" fill="{color}"/>',
                f'<text x="{label_x:.1f}" y="{y + 3.7:.1f}" text-anchor="{anchor}" font-family="{FONT}" font-size="10" font-weight="700" fill="{text_color}" stroke="{label_stroke}" stroke-width="3" paint-order="stroke">{row.delta_pct:+.1f}%</text>',
                f'<text x="{EMBEDDING_CHART_RIGHT + 12:.1f}" y="{y + 3.7:.1f}" font-family="{FONT}" font-size="10" fill="#374151">{short_number(row.ax_tok_s)} tok/s</text>',
            ]
        )

    lines.append(
        f'<text x="44" y="{chart_height - 14}" font-family="{FONT}" font-size="10" fill="#6b7280">{escape(source_label)}</text>'
    )
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
    llama_rows = (
        load_rows(args.llama_results_dir, readme_slugs, required=True)
        if args.llama_results_dir
        else load_llama_rows_from_readme(args.readme)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mismatches: list[Path] = []
    mtp_rows = load_mtp_rows(args.performance_doc)
    if mtp_rows:
        for (row_key, metric), output_name in MTP_CHART_OUTPUTS.items():
            rows = [row for row in mtp_rows if mtp_row_key(row) == row_key]
            if not rows:
                raise ChartError(f"MTP performance table has no {row_key!r} rows")
            mtp_output_path = args.output_dir / output_name
            mtp_content = render_mtp_metric_chart(rows, metric)
            if not write_chart(mtp_output_path, mtp_content, args.check):
                mismatches.append(mtp_output_path)

    mtp_6bit_summary_path = find_mtp_6bit_summary(args.readme)
    if mtp_6bit_summary_path is not None:
        mtp_6bit_output_path = args.output_dir / MTP_6BIT_CHART_OUTPUT
        mtp_6bit_content = render_mtp_6bit_ax_acceleration_chart(
            load_mtp_6bit_rows(mtp_6bit_summary_path), mtp_6bit_summary_path
        )
        if not write_chart(mtp_6bit_output_path, mtp_6bit_content, args.check):
            mismatches.append(mtp_6bit_output_path)

    mtp_peer_summary_path = find_mtp_peer_summary(args.readme)
    if mtp_peer_summary_path is not None:
        mtp_peer_rows = load_mtp_peer_rows(mtp_peer_summary_path)
        for metric_key, output_name in MTP_PEER_CHART_OUTPUTS.items():
            mtp_peer_output_path = args.output_dir / output_name
            mtp_peer_content = render_mtp_peer_comparison_chart(
                mtp_peer_rows, mtp_peer_summary_path, metric_key
            )
            if not write_chart(mtp_peer_output_path, mtp_peer_content, args.check):
                mismatches.append(mtp_peer_output_path)

    embedding_fair_output_path = args.output_dir / EMBEDDING_FAIR_CHART_OUTPUT
    embedding_fair_content = render_embedding_delta_chart(
        load_embedding_fair_delta_rows(args.readme.parent),
        title="Embedding throughput: AX vs reference",
        subtitle="Batch=8, contiguous CPU float32 [B,H] output; higher delta means AX is faster.",
        source_label="Sources: embedding-fair reference baselines from 2026-06-28 and AX refresh artifacts from 2026-06-29",
    )
    if not write_chart(embedding_fair_output_path, embedding_fair_content, args.check):
        mismatches.append(embedding_fair_output_path)

    embedding_scale_output_path = args.output_dir / EMBEDDING_SCALE_CHART_OUTPUT
    embedding_scale_content = render_embedding_delta_chart(
        load_embedding_scale_delta_rows(args.readme.parent),
        title="Embedding ingest scale: AX vs mlx-lm",
        subtitle="512 chunks per trial, repeated batches, contiguous CPU float32 [B,H] output.",
        source_label="Sources: embedding-scale Qwen3-Embedding 0.6B same-session paired artifact from 2026-06-29",
    )
    if not write_chart(embedding_scale_output_path, embedding_scale_content, args.check):
        mismatches.append(embedding_scale_output_path)

    embeddinggemma_scale_output_path = args.output_dir / EMBEDDINGGEMMA_SCALE_CHART_OUTPUT
    embeddinggemma_scale_content = render_embedding_delta_chart(
        load_embedding_paired_scale_delta_rows(
            args.readme.parent, EMBEDDINGGEMMA_SCALE_ARTIFACT
        ),
        title="EmbeddingGemma ingest scale: AX vs mlx-embeddings",
        subtitle="512 chunks per trial, repeated batches, contiguous CPU float32 [B,H] output.",
        source_label="Sources: embedding-scale EmbeddingGemma 300M 8-bit same-session paired artifact from 2026-06-29",
    )
    if not write_chart(
        embeddinggemma_scale_output_path, embeddinggemma_scale_content, args.check
    ):
        mismatches.append(embeddinggemma_scale_output_path)

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
            ("perf-ngram-toks.svg", render_ngram_toks_chart),
            ("perf-ngram-accept.svg", render_ngram_accept_chart),
        ]:
            content = renderer(ngram_artifact)
            out_path = args.output_dir / out_name
            if not write_chart(out_path, content, args.check):
                mismatches.append(out_path)

    # N-gram multi-model charts: one group per model slug.
    ngram_artifacts = find_ngram_artifacts_by_model(args.readme.parent)
    if ngram_artifacts:
        for out_name, renderer in [
            ("perf-ngram-models-toks.svg", render_ngram_models_speedup_chart),
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
            f"and {'README llama.cpp cells' if args.llama_results_dir is None else args.llama_results_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
