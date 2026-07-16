#!/usr/bin/env python3
"""Render README performance SVG charts from benchmark artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import check_readme_performance_artifacts as readme_artifacts


def _load_embedding_publish_gate():
    """Load the embedding publish gate without requiring package install."""
    path = Path(__file__).resolve().parent / "check_embedding_publish_gate.py"
    spec = importlib.util.spec_from_file_location(
        "check_embedding_publish_gate_runtime", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPO_ROOT = Path(__file__).resolve().parents[1]


def display_source_path(path: Path) -> str:
    """Return a portable path for a chart source label."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def workspace_version() -> str:
    cargo_toml = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(
        r"\[workspace\.package\][\s\S]*?^version\s*=\s*\"([^\"]+)\"",
        cargo_toml,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("could not determine workspace package version")
    return match.group(1)


AX_ENGINE_VERSION = workspace_version()

LABEL_TO_SLUG = {
    label: slug for slug, label in readme_artifacts.ARTIFACT_LABELS.items()
}

PROMPT_TOKENS = (128, 512, 2048)
AX_ENGINE_CHART_LABEL = f"AX Engine v{AX_ENGINE_VERSION}"

SERIES = [
    ("llama_cpp_metal", "llama.cpp b9910", "#f97316", "#c2410c"),
    ("mlx_lm", "mlx-lm 0.31.3", "#f2b705", "#9a6a00"),
    ("ax_engine_mlx", AX_ENGINE_CHART_LABEL, "#2eaf5f", "#176c37"),
    ("ax_engine_mlx_ngram_accel", f"AX+ngram v{AX_ENGINE_VERSION}", "#137a3d", "#0b4f28"),
]
DIRECT_VERSIONS_FOOTNOTE = (
    f"AX Engine v{AX_ENGINE_VERSION} snapshot (2026-07-14) · retained "
    "mlx-lm 0.31.3 · retained llama.cpp b9910 · cross-run distribution"
)

FAMILY_SLUGS: dict[str, list[str]] = {
    "gemma4": [
        "gemma-4-e2b-it-4bit",
        "gemma-4-e2b-it-6bit",
        "gemma-4-e4b-it-4bit",
        "gemma-4-26b-a4b-it-4bit",
        "gemma-4-26b-a4b-it-6bit",
        "gemma-4-31b-it-4bit",
        "gemma-4-31b-it-6bit",
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

BOX_FILL_OPACITY = 0.26
BOX_STROKE_OPACITY = 0.9
DOT_FILL_OPACITY = 0.82

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


@dataclass(frozen=True)
class EmbeddingEngineBoxRow:
    label: str
    stats: SeriesStats


@dataclass(frozen=True)
class EmbeddingModelBoxGroup:
    label: str
    engine_rows: tuple[EmbeddingEngineBoxRow, ...]


BOX_CHART_SUBTITLE = (
    "AX v6.9.0 snapshot vs retained peer rows | cross-run distribution; "
    "use the exact table for per-model values."
)
CHARTS: tuple[ChartSpec, ...] = (
    ChartSpec(
        title="Gemma 4 — Prefill rate",
        subtitle=BOX_CHART_SUBTITLE,
        unit="tok/s",
        direction_label="Higher is better",
        output_slug="gemma4-prefill",
        metric="prefill",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
        family="gemma4",
    ),
    ChartSpec(
        title="Gemma 4 — Decode rate",
        subtitle=BOX_CHART_SUBTITLE,
        unit="tok/s",
        direction_label="Higher is better",
        output_slug="gemma4-decode",
        metric="decode",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
        family="gemma4",
    ),
    ChartSpec(
        title="Gemma 4 — TTFT",
        subtitle=BOX_CHART_SUBTITLE,
        unit="ms",
        direction_label="Lower is better",
        output_slug="gemma4-ttft",
        metric="ttft",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
        family="gemma4",
    ),
    ChartSpec(
        title="Qwen 3.6 — Prefill rate",
        subtitle=BOX_CHART_SUBTITLE,
        unit="tok/s",
        direction_label="Higher is better",
        output_slug="qwen-prefill",
        metric="prefill",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
        family="qwen",
    ),
    ChartSpec(
        title="Qwen 3.6 — Decode rate",
        subtitle=BOX_CHART_SUBTITLE,
        unit="tok/s",
        direction_label="Higher is better",
        output_slug="qwen-decode",
        metric="decode",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
        family="qwen",
    ),
    ChartSpec(
        title="Qwen 3.6 — TTFT",
        subtitle=BOX_CHART_SUBTITLE,
        unit="ms",
        direction_label="Lower is better",
        output_slug="qwen-ttft",
        metric="ttft",
        series_engines=("llama_cpp_metal", "mlx_lm", "ax_engine_mlx"),
        family="qwen",
    ),
)


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

MTP_6BIT_APPROXIMATE_CHART_OUTPUT = "perf-mtp-6bit-ax-approximate-diagnostic.svg"
MTP_6BIT_EXACT_CHART_OUTPUT = "perf-mtp-6bit-ax-acceleration.svg"
MTP_6BIT_APPROXIMATE_SCHEMA = "ax.mtp_6bit_approximate_diagnostic_summary.v2"
MTP_6BIT_EXACT_SCHEMA = "ax.mtp_6bit_ax_acceleration_summary.v3"
MTP_6BIT_EXACT_TARGET_IDS = (
    "qwen3.6-27b-6bit",
    "qwen3.6-35b-a3b",
    "gemma-4-12b",
    "gemma-4-26b",
    "gemma-4-31b",
)
MTP_6BIT_EXACT_SUITES = ("flappy", "long_code", "python_modules_long")
MTP_6BIT_NGRAM_ZERO_KEYS = (
    "ax_ngram_accepted_tokens",
    "ax_ngram_draft_tokens",
    "ax_ngram_rejected_tokens",
    "ax_mtp_ngram_accepted_tokens",
    "ax_mtp_ngram_proposed_tokens",
    "ax_mtp_ngram_submitted_tokens",
    "ax_mtp_ngram_submitted_accepted_tokens",
    "ax_mtp_ngram_hit_steps",
    "ax_mtp_ngram_attempt_steps",
)
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
MTP_6BIT_MTP_COLOR = "#c084fc"
MTP_6BIT_MTP_TEXT = "#7e22ce"
MTP_6BIT_ROW_GAP = 32.0
MTP_6BIT_GROUP_GAP = 18.0
MTP_6BIT_GROUP_SIZE = 3

# Retained mlx-lm reference plus a fresh AX-only refresh for all Qwen3 sizes.
EMBEDDING_SCALE_PAIRED_ARTIFACT = Path(
    "benchmarks/results/embedding/embedding-scale/2026-07-12-qwen-paired-v2/"
    "2026-07-12-145710/embedding_ingest_scale.json"
)
# Historical overlay path retained for optional diagnostic reloads / tests.
EMBEDDING_SCALE_REFERENCE_ARTIFACT = Path(
    "benchmarks/results/embedding/embedding-scale/2026-07-03-qwen-paired-refresh/"
    "2026-07-02-215823/embedding_ingest_scale.json"
)
EMBEDDING_SCALE_AX_ARTIFACT = Path(
    "benchmarks/results/embedding/embedding-scale/"
    "2026-07-16-ax-only-full-refresh-qwen/"
    "2026-07-16-135740/embedding_ingest_scale.json"
)
# Back-compat aliases used by older call sites and tests.
EMBEDDING_SCALE_PAIRED_06_ARTIFACT = EMBEDDING_SCALE_PAIRED_ARTIFACT
EMBEDDING_SCALE_AX_OVERLAY_ARTIFACT = EMBEDDING_SCALE_AX_ARTIFACT
EMBEDDINGGEMMA_SCALE_REFERENCE_ARTIFACT = Path(
    "benchmarks/results/embedding/embedding-scale/"
    "2026-07-02-embeddinggemma-paired-cooldown15-refresh/"
    "2026-07-02-175206/embedding_ingest_scale.json"
)
EMBEDDINGGEMMA_SCALE_AX_ARTIFACT = Path(
    "benchmarks/results/embedding/embedding-scale/"
    "2026-07-16-ax-only-full-refresh-embeddinggemma/"
    "2026-07-16-153523/embedding_ingest_scale.json"
)
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
EMBEDDING_BOX_REFERENCE_COLOR = "#f2b705"
EMBEDDING_BOX_REFERENCE_DOT_COLOR = "#9a6a00"
EMBEDDING_BOX_AX_COLOR = "#2eaf5f"
EMBEDDING_BOX_AX_DOT_COLOR = "#176c37"

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

AX_DIRECT_SNAPSHOT_METRICS = ("decode_tok_s", "prefill_tok_s", "ttft_ms")


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
            merge_chart_row(
                merged,
                (str(row["_slug"]), str(engine), int(prompt_tokens), generation_tokens),
                row,
                metric,
            )

    return list(merged.values())


def load_retained_mlx_lm_rows(readme: Path, slugs: list[str]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for source in readme_artifacts.default_artifact_sources(readme.resolve()):
        if (
            source.include_engines is not None
            and "mlx_lm" not in source.include_engines
        ):
            continue
        for row in load_rows(source.artifact_dir, slugs, required=False):
            if row.get("engine") != "mlx_lm":
                continue
            prompt_tokens = row.get("prompt_tokens")
            generation_tokens = row.get("generation_tokens")
            if prompt_tokens not in PROMPT_TOKENS or not isinstance(
                generation_tokens, int
            ):
                continue
            rows[(str(row["_slug"]), int(prompt_tokens), generation_tokens)] = row
    return list(rows.values())


def merge_chart_row(
    rows: dict[tuple[str, str, int, int], dict[str, Any]],
    key: tuple[str, str, int, int],
    row: dict[str, Any],
    metric: str,
) -> None:
    existing = rows.get(key)
    if existing is None:
        rows[key] = row
        return

    _slug, engine, _prompt_tokens, _generation_tokens = key
    if engine not in readme_artifacts.AX_ENGINE_ROWS:
        rows[key] = row
        return

    candidate = readme_artifacts.metric_median(row, metric)
    previous = readme_artifacts.metric_median(existing, metric)
    if not readme_artifacts.metric_is_regressed(metric, candidate, previous):
        rows[key] = row


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


def find_ax_direct_snapshot(readme: Path) -> Path | None:
    match = re.search(
        r"readme-ax-direct-snapshot:\s*([^\s]+)",
        readme.read_text(encoding="utf-8"),
    )
    if match is None:
        return None
    path_value = match.group(1)
    relative = (readme.parent / path_value).resolve()
    if relative.exists():
        return relative
    if path_value.startswith("benchmarks/"):
        return (REPO_ROOT / path_value).resolve()
    return relative


def load_ax_direct_snapshot(snapshot_path: Path) -> dict[str, Any]:
    if not snapshot_path.exists():
        raise ChartError(f"AX-only direct snapshot is missing: {snapshot_path}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("publication_candidate") is not True:
        raise ChartError("AX-only direct snapshot is not publication eligible")
    if snapshot.get("ax_direct_only") is not True:
        raise ChartError("AX-only direct snapshot is not AX-only")

    rows = snapshot.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ChartError("AX-only direct snapshot must contain all 12 model rows")

    normalized_rows: list[dict[str, Any]] = []
    versions: set[str] = set()
    commits: set[str] = set()
    mlx_versions: set[str] = set()
    for row in rows:
        if row.get("status") != "ok":
            raise ChartError("AX-only direct snapshot has a non-successful model row")
        result_doc = row.get("result_doc")
        if not isinstance(result_doc, dict):
            raise ChartError("AX-only direct snapshot row has no benchmark result")
        build = result_doc.get("build")
        host = result_doc.get("host")
        if not isinstance(build, dict) or not isinstance(host, dict):
            raise ChartError("AX-only direct snapshot row has incomplete provenance")
        version = build.get("engine_version")
        commit = build.get("commit")
        toolchain = host.get("toolchain")
        if not isinstance(version, str) or not isinstance(commit, str):
            raise ChartError("AX-only direct snapshot row has incomplete build identity")
        if not isinstance(toolchain, dict) or not isinstance(
            toolchain.get("homebrew_mlx"), str
        ):
            raise ChartError("AX-only direct snapshot row has incomplete MLX identity")
        versions.add(version)
        commits.add(commit)
        mlx_versions.add(toolchain["homebrew_mlx"])

        model = row.get("readme_model")
        quant = row.get("readme_quant")
        if not isinstance(model, str) or not isinstance(quant, str):
            raise ChartError("AX-only direct snapshot row has no display label")
        if model.startswith("Gemma 4 "):
            family = "gemma"
            short_model = model.removeprefix("Gemma 4 ")
        elif model.startswith("Qwen 3.6 "):
            family = "qwen"
            short_model = model.removeprefix("Qwen 3.6 ")
        else:
            raise ChartError(f"AX-only direct snapshot has unsupported model: {model}")
        slug = LABEL_TO_SLUG.get((model, quant))
        if slug is None:
            raise ChartError(f"AX-only direct snapshot model has no slug: {model}")

        metrics: dict[str, dict[int, float]] = {
            metric: {} for metric in AX_DIRECT_SNAPSHOT_METRICS
        }
        results = result_doc.get("results")
        if not isinstance(results, list):
            raise ChartError("AX-only direct snapshot row has no prompt results")
        for result in results:
            prompt_tokens = result.get("prompt_tokens")
            if prompt_tokens not in PROMPT_TOKENS:
                continue
            for metric in AX_DIRECT_SNAPSHOT_METRICS:
                value = result.get(metric)
                if not isinstance(value, dict) or not isinstance(value.get("median"), (int, float)):
                    raise ChartError(
                        f"AX-only direct snapshot is missing {metric} median"
                    )
                metrics[metric][prompt_tokens] = float(value["median"])
        if any(set(values) != set(PROMPT_TOKENS) for values in metrics.values()):
            raise ChartError("AX-only direct snapshot row has incomplete prompt coverage")
        normalized_rows.append(
            {
                "family": family,
                "slug": slug,
                "label": f"{short_model} {quant}",
                "metrics": metrics,
            }
        )

    if len(versions) != 1 or len(commits) != 1 or len(mlx_versions) != 1:
        raise ChartError("AX-only direct snapshot has mixed benchmark provenance")
    if versions != {AX_ENGINE_VERSION}:
        raise ChartError(
            f"AX-only direct snapshot version {sorted(versions)} does not match {AX_ENGINE_VERSION}"
        )
    return {
        "rows": normalized_rows,
        "engine_version": versions.pop(),
        "commit": commits.pop(),
        "mlx_version": mlx_versions.pop(),
        "date": str(snapshot.get("started_at", "unknown"))[:10],
    }


def ax_direct_snapshot_chart_rows(
    snapshot: dict[str, Any], metric: str
) -> list[dict[str, Any]]:
    metric_key = {
        "decode": "decode_tok_s",
        "prefill": "prefill_tok_s",
        "ttft": "ttft_ms",
    }.get(metric)
    if metric_key is None:
        raise ChartError(f"unknown AX-only direct snapshot metric: {metric}")
    rows: list[dict[str, Any]] = []
    for snapshot_row in snapshot["rows"]:
        for prompt_tokens in PROMPT_TOKENS:
            rows.append(
                {
                    "_slug": snapshot_row["slug"],
                    "engine": "ax_engine_mlx",
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": 128,
                    metric_key: {
                        "median": snapshot_row["metrics"][metric_key][prompt_tokens]
                    },
                }
            )
    return rows


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

    lower_is_better = spec.metric == "ttft"
    best_by_prompt_tokens: dict[int, float] = {}
    for prompt_tokens in PROMPT_TOKENS:
        prompt_medians = [
            cs.stats.median
            for eg in engine_groups
            for cs in eg.context_stats
            if cs.prompt_tokens == prompt_tokens
        ]
        best_by_prompt_tokens[prompt_tokens] = (
            min(prompt_medians) if lower_is_better else max(prompt_medians)
        )

    n_engines = len(engine_groups)
    n_prompt_groups = len(PROMPT_TOKENS)
    plot_width = FAMILY_RIGHT - FAMILY_LEFT
    plot_height = FAMILY_BOTTOM - FAMILY_TOP
    group_step = plot_width / n_prompt_groups
    sub_spacing = 54.0
    sub_bar_w = 18.0
    sub_offsets = [
        (idx - (n_engines - 1) / 2) * sub_spacing for idx in range(n_engines)
    ]
    stats_by_engine_prompt = {
        eg.engine: {cs.prompt_tokens: cs for cs in eg.context_stats}
        for eg in engine_groups
    }

    def fy(v: float) -> float:
        clamped = max(0.0, min(v, axis_max))
        return FAMILY_BOTTOM - (clamped / axis_max) * plot_height

    engine_desc = ", ".join(eg.label for eg in engine_groups)
    ctx_desc = "/".join(str(pt) for pt in PROMPT_TOKENS)
    family_label = FAMILY_LABELS.get(spec.family, spec.family)
    direction_fill = RED

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
        f" The x-axis is grouped by prompt-token count, with engines nested inside each group."
        f" Within each prompt-token group, the best median value label is red.</desc>",
        # Background
        f'<rect width="{FAMILY_CHART_WIDTH}" height="{FAMILY_CHART_HEIGHT}" fill="#f8fafc"/>',
        # Title
        f'<text x="{FAMILY_LEFT}" y="24" font-family="{FONT}"'
        f' font-size="16" font-weight="700" fill="#111827">{escape(spec.title)}</text>',
        # Subtitle
        f'<text x="{FAMILY_LEFT}" y="46" font-family="{FONT}"'
        f' font-size="11" fill="#4b5563">'
        f'{escape(spec.subtitle) if spec.subtitle else "grouped by prompt tokens | box=IQR | whiskers=min/max | dots=runs"}'
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

    dot_jitter = (-3.0, -1.5, 0.0, 1.5, 3.0)
    y_prompt = FAMILY_BOTTOM + 20

    for i, prompt_tokens in enumerate(PROMPT_TOKENS):
        group_center = FAMILY_LEFT + (i + 0.5) * group_step

        for j, eg in enumerate(engine_groups):
            try:
                cs = stats_by_engine_prompt[eg.engine][prompt_tokens]
            except KeyError as exc:
                raise ChartError(
                    f"missing {eg.engine} stats for {prompt_tokens} tok in {spec.title}"
                ) from exc
            sub_x = group_center + sub_offsets[j]
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
            prompt_best = best_by_prompt_tokens[prompt_tokens]
            label_fill = RED if math.isclose(s.median, prompt_best) else "#111827"

            sa = f'stroke="{eg.color}" stroke-opacity="{BOX_STROKE_OPACITY}"'
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
                    f' fill="{eg.color}" fill-opacity="{BOX_FILL_OPACITY}" {sa} stroke-width="1.7"/>',
                    f'<line x1="{box_left:g}" y1="{y_med:.1f}"'
                    f' x2="{box_left + sub_bar_w:g}" y2="{y_med:.1f}" {sa} stroke-width="2.4"/>',
                    f'<text x="{label_x:g}" y="{y_med + 3.5:.1f}" text-anchor="start"'
                    f' font-family="{FONT}" font-size="9" font-weight="700"'
                    f' fill="{label_fill}" stroke="#ffffff" stroke-width="3"'
                    f' paint-order="stroke">{escape(short_number(s.median))}</text>',
                ]
            )

            for vi, val in enumerate(s.values):
                dx = dot_jitter[vi % len(dot_jitter)]
                dy = fy(val)
                lines.append(
                    f'<circle cx="{sub_x + dx:g}" cy="{dy:.1f}" r="1.4"'
                    f' fill="{eg.dot_color}" fill-opacity="{DOT_FILL_OPACITY}"/>'
                )

        lines.append(
            f'<text x="{group_center:g}" y="{y_prompt}"'
            f' text-anchor="middle" font-family="{FONT}"'
            f' font-size="11" font-weight="700" fill="#111827">{prompt_tokens} tok</text>'
        )

    # Engine legend.
    legend_y = FAMILY_BOTTOM + 56
    legend_x = FAMILY_LEFT
    for eg in engine_groups:
        lines.extend(
            [
                f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" rx="2"'
                f' fill="{eg.color}" fill-opacity="0.5"'
                f' stroke="{eg.color}" stroke-width="1.4"/>',
                f'<text x="{legend_x + 14}" y="{legend_y}"'
                f' font-family="{FONT}" font-size="10" fill="#374151">{escape(eg.label)}</text>',
            ]
        )
        legend_x += max(170, len(eg.label) * 7 + 30)

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
        f'<text x="{MTP_RIGHT}" y="22" text-anchor="end" font-family="{FONT}" font-size="10" font-weight="700" fill="{RED}">Higher is better</text>',
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
        r"\]\((benchmarks/results/(?:speculative/)?mtp-6bit/[^)]+/summary\.json)\)",
        text,
    )
    if match is None:
        return None
    summary_path = readme_artifacts.resolve_results_doc_path(
        readme, match.group(1)
    )
    if not summary_path.exists():
        raise ChartError(f"README references missing MTP summary: {summary_path}")
    return summary_path


def load_mtp_6bit_summary(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    schema = summary.get("schema")
    if schema not in {MTP_6BIT_APPROXIMATE_SCHEMA, MTP_6BIT_EXACT_SCHEMA}:
        raise ChartError(f"unsupported MTP 6-bit summary schema {schema!r}: {summary_path}")
    approximate = schema == MTP_6BIT_APPROXIMATE_SCHEMA
    expected_publication_candidate = not approximate
    if summary.get("publication_candidate") is not expected_publication_candidate:
        raise ChartError(
            "MTP 6-bit summary publication_candidate does not match its schema: "
            f"{summary_path}"
        )
    expected_claim_type = (
        "approximate_optimistic_diagnostic" if approximate else "exact_mtp_acceleration"
    )
    if summary.get("claim_type") != expected_claim_type:
        raise ChartError(
            f"MTP 6-bit summary has invalid claim_type for {schema}: {summary_path}"
        )
    engine_version = summary.get("engine_version")
    if not isinstance(engine_version, str) or not re.fullmatch(
        r"\d+\.\d+\.\d+", engine_version
    ):
        raise ChartError(f"MTP 6-bit summary has no semantic engine_version: {summary_path}")
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ChartError(f"MTP 6-bit summary has no rows: {summary_path}")
    exact_row_keys: set[tuple[str, str]] = set()
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
        if row.get("publication_candidate") is not expected_publication_candidate:
            raise ChartError(
                "MTP 6-bit summary row publication_candidate does not match its schema: "
                f"{summary_path}"
            )
        if approximate:
            for key in (
                "ax_mtp_draft_quality_pct",
                "ax_mtp_draft_quality_kind",
                "ax_mtp_step_coverage_pct",
                "ax_mtp_fallback_prompt_count",
                "prompt_count",
            ):
                if key not in row:
                    raise ChartError(
                        f"MTP approximate diagnostic row missing {key}: {summary_path}"
                    )
        else:
            for key in (
                "model_id",
                "ax_mtp_step_coverage_pct",
                "ax_mtp_fallback_prompt_count",
                "ax_mtp_direct_fallback_steps",
                "publication_reasons",
                "ax_mtp_ngram_telemetry",
            ):
                if key not in row:
                    raise ChartError(
                        f"MTP exact acceleration row missing {key}: {summary_path}"
                    )
            row_key = (str(row["model_id"]), str(row["suite_id"]))
            if row_key in exact_row_keys:
                raise ChartError(f"MTP exact acceleration has duplicate row {row_key!r}")
            exact_row_keys.add(row_key)
            try:
                direct = float(row["ax_direct_decode_tok_s"])
                mtp = float(row["ax_mtp_decode_tok_s"])
                speedup = float(row["ax_mtp_speedup_x"])
                coverage = float(row["ax_mtp_step_coverage_pct"])
            except (TypeError, ValueError) as error:
                raise ChartError(
                    f"MTP exact acceleration row has invalid metrics: {row_key!r}"
                ) from error
            if direct <= 0.0 or mtp <= 0.0 or speedup <= 1.0:
                raise ChartError(
                    f"MTP exact acceleration row does not win decode: {row_key!r}"
                )
            if abs(speedup - mtp / direct) > 0.001:
                raise ChartError(
                    f"MTP exact acceleration row speedup is inconsistent: {row_key!r}"
                )
            if coverage != 100.0:
                raise ChartError(
                    f"MTP exact acceleration row lacks full coverage: {row_key!r}"
                )
            if row["ax_mtp_fallback_prompt_count"] != 0:
                raise ChartError(
                    f"MTP exact acceleration row has fallback prompts: {row_key!r}"
                )
            if row["ax_mtp_direct_fallback_steps"] != 0:
                raise ChartError(
                    f"MTP exact acceleration row has fallback steps: {row_key!r}"
                )
            if row["publication_reasons"] != []:
                raise ChartError(
                    f"MTP exact acceleration row has publication reasons: {row_key!r}"
                )
            ngram = row["ax_mtp_ngram_telemetry"]
            if not isinstance(ngram, dict) or any(
                ngram.get(key) != 0 for key in MTP_6BIT_NGRAM_ZERO_KEYS
            ):
                raise ChartError(
                    f"MTP exact acceleration row has nonzero n-gram telemetry: {row_key!r}"
                )
    if not approximate:
        expected_row_keys = {
            (model_id, suite_id)
            for model_id in MTP_6BIT_EXACT_TARGET_IDS
            for suite_id in MTP_6BIT_EXACT_SUITES
        }
        if exact_row_keys != expected_row_keys:
            raise ChartError(
                "MTP exact acceleration summary is not the complete supported matrix: "
                f"{summary_path}"
            )
    return summary


def load_mtp_6bit_rows(summary_path: Path) -> list[dict[str, Any]]:
    return load_mtp_6bit_summary(summary_path)["rows"]


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
    rows: list[dict[str, Any]],
    summary_path: Path,
    *,
    approximate_diagnostic: bool = False,
    engine_version: str | None = None,
) -> str:
    run_date_match = re.match(r"(\d{4}-\d{2}-\d{2})", summary_path.parent.name)
    run_date = run_date_match.group(1) if run_date_match else summary_path.parent.name
    chart_title = (
        f"AX approximate MTP diagnostic ({run_date})"
        if approximate_diagnostic
        else f"AX MTP decode ({run_date}): MTP off vs MTP on"
    )
    axis_max = mtp_6bit_axis_max(rows)
    tick_step = axis_max / 4.0
    model_order = tuple(dict.fromkeys(str(row["model"]) for row in rows))
    axis_bottom = (
        106.0
        + len(rows) * MTP_6BIT_ROW_GAP
        + max(0, len(model_order) - 1) * MTP_6BIT_GROUP_GAP
        + 16.0
    )
    height = int(axis_bottom + 102.0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{MTP_6BIT_WIDTH}" height="{height}" viewBox="0 0 {MTP_6BIT_WIDTH} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(chart_title)}</title>',
        (
            '<desc id="desc">Horizontal grouped bar chart comparing AX direct '
            "decode throughput with MTP off against AX MTP decode throughput "
            "with MTP on for each supported 6-bit AX MTP package and prompt suite. "
            + (
                "The MTP rows are non-publishable optimistic diagnostics, not exact-distribution evidence.</desc>"
                if approximate_diagnostic
                else "The winning throughput label in each row is red.</desc>"
            )
        ),
        f'<rect width="{MTP_6BIT_WIDTH}" height="{height}" fill="#ffffff"/>',
        f'<text x="{MTP_6BIT_LABEL_X}" y="32" font-family="{FONT}" font-size="20" font-weight="700" fill="#111827">{escape(chart_title)}</text>',
        f'<text x="{MTP_6BIT_LABEL_X}" y="54" font-family="{FONT}" font-size="12" fill="#4b5563">{escape("Non-publishable greedy optimistic diagnostic; MTP can fall back per prompt." if approximate_diagnostic else "Each row compares the same prepared 6-bit package and prompt suite: AX direct has MTP off; AX MTP has MTP on.")}</text>',
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
            f'<text x="{(MTP_6BIT_LEFT + MTP_6BIT_RIGHT) / 2:.1f}" y="{axis_bottom + 40.0:.1f}" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="{RED}">{"Throughput diagnostic only" if approximate_diagnostic else "Higher is better"}</text>',
            f'<rect x="{MTP_6BIT_LABEL_X}" y="70" width="12" height="12" rx="2" fill="{MTP_6BIT_DIRECT_COLOR}"/>',
            f'<text x="90" y="80" font-family="{FONT}" font-size="12" fill="#374151">MTP off / AX direct</text>',
            f'<rect x="232" y="70" width="12" height="12" rx="2" fill="{MTP_6BIT_MTP_COLOR}"/>',
            f'<text x="250" y="80" font-family="{FONT}" font-size="12" fill="#374151">{"Approx. MTP diagnostic" if approximate_diagnostic else "MTP on / AX MTP"}</text>',
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
        direct_width = mtp_6bit_x_scale(direct, axis_max)
        mtp_width = mtp_6bit_x_scale(mtp, axis_max)
        direct_end = MTP_6BIT_LEFT + direct_width
        mtp_end = MTP_6BIT_LEFT + mtp_width
        row_best = max(direct, mtp)
        direct_text = (
            MTP_6BIT_DIRECT_TEXT
            if approximate_diagnostic
            else RED if math.isclose(direct, row_best) else MTP_6BIT_DIRECT_TEXT
        )
        mtp_text = (
            MTP_6BIT_MTP_TEXT
            if approximate_diagnostic
            else RED if math.isclose(mtp, row_best) else MTP_6BIT_MTP_TEXT
        )

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
                f'<text x="{direct_end + 5.0:.1f}" y="{label_y - 7.0:.1f}" font-family="{FONT}" font-size="10" font-weight="700" fill="{direct_text}">{direct:.1f}</text>',
                f'<text x="{mtp_end + 5.0:.1f}" y="{label_y + 6.0:.1f}" font-family="{FONT}" font-size="10" font-weight="700" fill="{mtp_text}">{mtp:.1f}</text>',
            ]
        )
        label_y += MTP_6BIT_ROW_GAP

    version = engine_version or "unrecorded"
    version_label = f"Runtime: AX Engine v{version} ({run_date})."
    source_label = f"Source: {display_source_path(summary_path.parent)} / summary.json. " + (
        "No MTP+n-gram stacking; approximate and not publication eligible."
        if approximate_diagnostic
        else "Pure MTP; no MTP+n-gram stacking."
    )
    lines.append(
        f'<text x="{MTP_6BIT_LABEL_X}" y="{axis_bottom + 60.0:.1f}" font-family="{FONT}" font-size="10" fill="#6b7280">{escape(version_label)}</text>'
    )
    lines.append(
        f'<text x="{MTP_6BIT_LABEL_X}" y="{axis_bottom + 76.0:.1f}" font-family="{FONT}" font-size="10" fill="#6b7280">{escape(source_label)}</text>'
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
    "ax_engine": AX_ENGINE_CHART_LABEL,
    "mtplx": "MTPLX",
    "lightning_mlx": "lightning-mlx",
}
# Engine versions behind the peer-comparison artifacts, surfaced on each chart so the
# run is reproducible. Update alongside any re-benchmark. Provenance:
#   AX Engine     = [workspace.package] version in Cargo.toml
#   MTPLX         = /opt/homebrew/var/mtplx/venv-2.0.1 (pip: mtplx 2.0.1)
#   lightning-mlx = .internal/reference/lightning-mlx v0.7.0 (git rev ec19b3d, incl. post-tag streaming fix #3)
MTP_PEER_VERSIONS = {
    "ax_engine": AX_ENGINE_VERSION,
    "mtplx": "2.0.1",
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
    summary_path = readme_artifacts.resolve_results_doc_path(
        readme, match.group(1)
    )
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
    run_date_match = re.match(r"(\d{4}-\d{2}-\d{2})", summary_path.parent.name)
    run_date = run_date_match.group(1) if run_date_match else summary_path.parent.name
    metric_config = MTP_PEER_METRICS[metric_key]
    chart_title = f'{metric_config["title"]} ({run_date})'
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
        f'<title id="title">{escape(str(chart_title))} {escape(contract_label)}</title>',
        (
            '<desc id="desc">Grouped horizontal bar chart comparing AX Engine, '
            f'MTPLX, and lightning-mlx {escape(str(metric_config["unit"]))} on '
            "the flappy prompt suite. Degenerate rows are shown muted and are "
            "not eligible for best-row highlighting.</desc>"
        ),
        f'<rect width="{MTP_PEER_WIDTH}" height="{height}" fill="#f8fafc"/>',
        f'<text x="32" y="34" font-family="{FONT}" font-size="22" font-weight="700" fill="#111827">{escape(str(chart_title))}</text>',
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
        f' font-size="10" font-weight="700" fill="{RED}">Higher is better</text>',
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
        f' font-size="10" font-weight="700" fill="{RED}">Higher is better</text>',
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
        f'<text x="{RIGHT}" y="36" text-anchor="end" font-family="{FONT}" font-size="10" font-weight="700" fill="{RED}">Higher is better</text>',
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
        f'<text x="{RIGHT}" y="36" text-anchor="end" font-family="{FONT}" font-size="10" font-weight="700" fill="{RED}">Higher is better</text>',
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


def _assert_embedding_publish_gate(
    artifact_path: Path, *, claim: str, allow_legacy: bool = True
) -> None:
    """Reject chart inputs that fail the embedding publication gate.

    Historical v1 retained rows pass only with allow_legacy=True (default for
    chart rendering of frozen artifacts). Fresh v2 paired artifacts are fully
    validated including runtime_identity / libmlx fingerprints.
    """
    gate = _load_embedding_publish_gate()
    try:
        gate.validate_artifact(
            artifact_path, claim=claim, allow_legacy=allow_legacy
        )
    except gate.PublishGateError as exc:
        raise ChartError(str(exc)) from exc


def load_embedding_overlay_scale_delta_rows(
    repo_root: Path, reference_relative_path: Path, ax_relative_path: Path
) -> list[EmbeddingDeltaRow]:
    reference_path = repo_root / reference_relative_path
    ax_path = repo_root / ax_relative_path
    if not reference_path.exists():
        raise ChartError(f"missing embedding scale reference artifact: {reference_path}")
    if not ax_path.exists():
        raise ChartError(f"missing embedding scale AX artifact: {ax_path}")
    # Overlay charts pair a historical reference snapshot with a later AX-only
    # refresh. Gate each side under its allowed claim; legacy v1 is retained.
    _assert_embedding_publish_gate(
        reference_path, claim="paired_delta", allow_legacy=True
    )
    _assert_embedding_publish_gate(
        ax_path, claim="ax_absolute_trend", allow_legacy=True
    )
    reference_artifact = json.loads(reference_path.read_text())
    ax_artifact = json.loads(ax_path.read_text())
    if reference_artifact.get("ax_only"):
        raise ChartError(f"{reference_path} is AX-only; expected reference artifact")
    if not ax_artifact.get("ax_only"):
        raise ChartError(f"{ax_path} is not AX-only")
    ref_key, ref_label = embedding_reference_key(reference_artifact)
    reference_models = {
        str(model.get("model_label", "")): model
        for model in reference_artifact.get("models", [])
    }
    rows: list[EmbeddingDeltaRow] = []
    for model in ax_artifact.get("models", []):
        raw_model_label = str(model.get("model_label", ""))
        reference_model = reference_models.get(raw_model_label)
        if reference_model is None:
            raise ChartError(f"{reference_path} missing model {raw_model_label}")
        model_label = embedding_model_label(raw_model_label)
        reference_rows = {
            str(row.get("workload")): row for row in reference_model.get("rows", [])
        }
        for row in model.get("rows", []):
            workload = str(row.get("workload", ""))
            reference_row = reference_rows.get(workload)
            if reference_row is None:
                raise ChartError(f"{reference_path} missing workload {workload}")
            ref = reference_row.get("results", {}).get(ref_key)
            ax = row.get("results", {}).get("ax_engine_py")
            if not isinstance(ref, dict) or not isinstance(ax, dict):
                raise ChartError(
                    f"{workload} lacks {ref_key}/ax_engine_py results"
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


def load_embedding_paired_scale_delta_rows(
    repo_root: Path, artifact_relative_path: Path
) -> list[EmbeddingDeltaRow]:
    artifact_path = repo_root / artifact_relative_path
    if not artifact_path.exists():
        raise ChartError(f"missing embedding scale paired artifact: {artifact_path}")
    _assert_embedding_publish_gate(
        artifact_path, claim="paired_delta", allow_legacy=True
    )
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("ax_only"):
        raise ChartError(f"{artifact_path} is AX-only; expected paired artifact")
    ref_key, ref_label = embedding_reference_key(artifact)
    rows: list[EmbeddingDeltaRow] = []
    for model in artifact.get("models", []):
        raw_model_label = str(model.get("model_label", ""))
        model_label = embedding_model_label(raw_model_label)
        for row in model.get("rows", []):
            ref = row.get("results", {}).get(ref_key)
            ax = row.get("results", {}).get("ax_engine_py")
            if not isinstance(ref, dict) or not isinstance(ax, dict):
                raise ChartError(
                    f"{artifact_path} lacks {ref_key}/ax_engine_py results for "
                    f"{raw_model_label} {row.get('workload')}"
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
    # Publication chart: retained mlx-lm reference plus AX-only refresh.
    return load_embedding_overlay_scale_delta_rows(
        repo_root, EMBEDDING_SCALE_PAIRED_ARTIFACT, EMBEDDING_SCALE_AX_ARTIFACT
    )


def format_embedding_delta_pct(delta_pct: float) -> str:
    if abs(delta_pct) < 0.05:
        return "0.0%"
    return f"{delta_pct:+.1f}%"


# Left-to-right group order for Qwen3 embedding ingest charts (small → large).
EMBEDDING_MODEL_CHART_ORDER = (
    "Qwen3 0.6B 8-bit",
    "Qwen3 4B 4-bit DWQ",
    "Qwen3 8B 4-bit DWQ",
    "EmbeddingGemma 300M 8-bit",
)


def embedding_box_groups(rows: list[EmbeddingDeltaRow]) -> list[EmbeddingModelBoxGroup]:
    grouped: dict[str, list[EmbeddingDeltaRow]] = {}
    order: list[str] = []
    for row in rows:
        if row.label not in grouped:
            grouped[row.label] = []
            order.append(row.label)
        grouped[row.label].append(row)

    preferred = {label: index for index, label in enumerate(EMBEDDING_MODEL_CHART_ORDER)}
    order = sorted(
        order,
        key=lambda label: (preferred.get(label, len(preferred)), label),
    )

    box_groups: list[EmbeddingModelBoxGroup] = []
    for label in order:
        model_rows = grouped[label]
        references = {row.reference_label for row in model_rows}
        if len(references) != 1:
            raise ChartError(f"{label} mixes embedding reference backends")
        reference_label = next(iter(references))
        box_groups.append(
            EmbeddingModelBoxGroup(
                label=label,
                engine_rows=(
                    EmbeddingEngineBoxRow(
                        label=reference_label,
                        stats=summarize(
                            [row.reference_tok_s for row in model_rows],
                            "embedding_reference",
                            reference_label,
                            EMBEDDING_BOX_REFERENCE_COLOR,
                            EMBEDDING_BOX_REFERENCE_DOT_COLOR,
                        ),
                    ),
                    EmbeddingEngineBoxRow(
                        label="AX Engine",
                        stats=summarize(
                            [row.ax_tok_s for row in model_rows],
                            "ax_engine_py",
                            "AX Engine",
                            EMBEDDING_BOX_AX_COLOR,
                            EMBEDDING_BOX_AX_DOT_COLOR,
                        ),
                    ),
                ),
            )
        )
    return box_groups


def embedding_box_label_lines(label: str) -> tuple[str, str]:
    return {
        "Qwen3 0.6B 8-bit": ("Qwen3 0.6B", "8-bit"),
        "Qwen3 4B 4-bit DWQ": ("Qwen3 4B", "4-bit DWQ"),
        "Qwen3 8B 4-bit DWQ": ("Qwen3 8B", "4-bit DWQ"),
        "EmbeddingGemma 300M 8-bit": ("EmbeddingGemma 300M", "8-bit"),
    }.get(label, (label, ""))


def render_embedding_box_chart(
    rows: list[EmbeddingDeltaRow],
    *,
    title: str,
    subtitle: str,
    source_label: str,
    ax_label: str | None = None,
) -> str:
    box_groups = embedding_box_groups(rows)
    if not box_groups:
        raise ChartError(f"{title} has no rows")
    ax_chart_label = ax_label or AX_ENGINE_CHART_LABEL
    all_maxima = [
        engine_row.stats.maximum
        for group in box_groups
        for engine_row in group.engine_rows
    ]
    max_value = max(all_maxima)
    axis_max = max(1.0, math.ceil(max_value * 1.08 / 10000.0) * 10000.0)
    plot_width = FAMILY_RIGHT - FAMILY_LEFT
    plot_height = FAMILY_BOTTOM - FAMILY_TOP
    group_step = plot_width / len(box_groups)
    sub_spacing = 44.0
    box_w = 20.0
    dot_jitter = (-3.0, -1.8, -0.6, 0.6, 1.8, 3.0)
    reference_label = box_groups[0].engine_rows[0].label

    def fy(value: float) -> float:
        clamped = max(0.0, min(value, axis_max))
        return FAMILY_BOTTOM - (clamped / axis_max) * plot_height

    unit_w = 59
    header_right = FAMILY_CHART_WIDTH - 34

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{FAMILY_CHART_WIDTH}" height="{FAMILY_CHART_HEIGHT}"'
        f' viewBox="0 0 {FAMILY_CHART_WIDTH} {FAMILY_CHART_HEIGHT}"'
        f' role="img" aria-labelledby="title desc">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(subtitle)} Grouped box-and-whisker plot comparing "
        f"{escape(reference_label)} and AX Engine throughput across the "
        f"displayed embedding model group(s). Within each group, the best "
        f"median label is red.</desc>",
        f'<rect width="{FAMILY_CHART_WIDTH}" height="{FAMILY_CHART_HEIGHT}" fill="#f8fafc"/>',
        f'<text x="{FAMILY_LEFT}" y="24" font-family="{FONT}"'
        f' font-size="16" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="{FAMILY_LEFT}" y="46" font-family="{FONT}"'
        f' font-size="11" fill="#4b5563">{escape(subtitle)}</text>',
        f'<text x="{FAMILY_LEFT}" y="62" font-family="{FONT}"'
        f' font-size="10" fill="#6b7280">{escape(source_label)}</text>',
        f'<rect x="{header_right - unit_w}" y="13" width="{unit_w}" height="22"'
        f' rx="11" fill="#eef2ff" stroke="#c7d2fe"/>',
        f'<text x="{header_right - unit_w / 2:.1f}" y="28" text-anchor="middle"'
        f' font-family="{FONT}" font-size="10" font-weight="700"'
        f' fill="#3730a3">tok/s</text>',
        f'<text x="{header_right}" y="52" text-anchor="end" font-family="{FONT}"'
        f' font-size="10" font-weight="700" fill="{RED}">'
        f"Higher is better</text>",
        f'<rect x="{FAMILY_LEFT}" y="{FAMILY_TOP}" width="{plot_width}"'
        f' height="{plot_height}" rx="6" fill="#ffffff" stroke="#dbe3ef"/>',
    ]

    for i in range(5):
        tick = axis_max * i / 4.0
        gy = fy(tick)
        lines.append(
            f'<line x1="{FAMILY_LEFT}" y1="{gy:.1f}"'
            f' x2="{FAMILY_RIGHT}" y2="{gy:.1f}"'
            f' stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{FAMILY_LEFT - 8}" y="{gy + 3:.1f}" text-anchor="end"'
            f' font-family="{FONT}" font-size="11" fill="#6b7280">'
            f"{short_number(tick)}</text>"
        )

    for idx, group in enumerate(box_groups):
        group_center = FAMILY_LEFT + (idx + 0.5) * group_step
        group_best_median = max(
            engine_row.stats.median for engine_row in group.engine_rows
        )
        for engine_idx, engine_row in enumerate(group.engine_rows):
            sub_x = group_center + (engine_idx - 0.5) * sub_spacing
            s = engine_row.stats
            y_min = fy(s.minimum)
            y_q1 = fy(s.q1)
            y_med = fy(s.median)
            y_q3 = fy(s.q3)
            y_max = fy(s.maximum)
            box_y = min(y_q1, y_q3)
            box_h = max(abs(y_q3 - y_q1), 1.0)
            cap_left = sub_x - box_w * 0.36
            cap_right = sub_x + box_w * 0.36
            box_left = sub_x - box_w / 2.0
            label_anchor = "end" if engine_idx == 0 else "start"
            label_x = box_left - 5 if engine_idx == 0 else box_left + box_w + 5
            label = short_number(s.median)
            label_fill = (
                RED if math.isclose(s.median, group_best_median) else s.dot_color
            )
            sa = f'stroke="{s.color}" stroke-opacity="{BOX_STROKE_OPACITY}"'
            lines.extend(
                [
                    f'<line x1="{sub_x:g}" y1="{y_max:.1f}"'
                    f' x2="{sub_x:g}" y2="{y_min:.1f}" {sa} stroke-width="1.7"/>',
                    f'<line x1="{cap_left:g}" y1="{y_max:.1f}"'
                    f' x2="{cap_right:g}" y2="{y_max:.1f}" {sa} stroke-width="1.7"/>',
                    f'<line x1="{cap_left:g}" y1="{y_min:.1f}"'
                    f' x2="{cap_right:g}" y2="{y_min:.1f}" {sa} stroke-width="1.7"/>',
                    f'<rect x="{box_left:g}" y="{box_y:.1f}" width="{box_w:g}"'
                    f' height="{box_h:.1f}" rx="2" fill="{s.color}"'
                    f' fill-opacity="{BOX_FILL_OPACITY}" {sa} stroke-width="1.7"/>',
                    f'<line x1="{box_left:g}" y1="{y_med:.1f}"'
                    f' x2="{box_left + box_w:g}" y2="{y_med:.1f}"'
                    f' {sa} stroke-width="2.4"/>',
                    f'<text x="{label_x:g}" y="{y_med + 3.5:.1f}"'
                    f' text-anchor="{label_anchor}" font-family="{FONT}"'
                    f' font-size="9" font-weight="700" fill="{label_fill}"'
                    f' stroke="#ffffff" stroke-width="3" paint-order="stroke">'
                    f"{escape(label)}</text>",
                ]
            )
            for vi, value in enumerate(s.values):
                dx = dot_jitter[vi % len(dot_jitter)]
                lines.append(
                    f'<circle cx="{sub_x + dx:g}" cy="{fy(value):.1f}" r="1.4"'
                    f' fill="{s.dot_color}" fill-opacity="{DOT_FILL_OPACITY}"/>'
                )

        label_line_1, label_line_2 = embedding_box_label_lines(group.label)
        lines.append(
            f'<text x="{group_center:g}" y="{FAMILY_BOTTOM + 23}"'
            f' text-anchor="middle" font-family="{FONT}" font-size="11"'
            f' font-weight="700" fill="#111827">{escape(label_line_1)}</text>'
        )
        if label_line_2:
            lines.append(
                f'<text x="{group_center:g}" y="{FAMILY_BOTTOM + 37}"'
                f' text-anchor="middle" font-family="{FONT}" font-size="10"'
                f' fill="#6b7280">{escape(label_line_2)}</text>'
            )

    legend_y = FAMILY_BOTTOM + 62
    legend_items = [
        (
            reference_label,
            EMBEDDING_BOX_REFERENCE_COLOR,
            EMBEDDING_BOX_REFERENCE_DOT_COLOR,
        ),
        (ax_chart_label, EMBEDDING_BOX_AX_COLOR, EMBEDDING_BOX_AX_DOT_COLOR),
    ]
    legend_x = FAMILY_LEFT
    for label, color, stroke in legend_items:
        lines.append(
            f'<rect x="{legend_x}" y="{legend_y}" width="10" height="10" rx="2"'
            f' fill="{color}" fill-opacity="0.5" stroke="{stroke}" stroke-width="1.4"/>'
        )
        lines.append(
            f'<text x="{legend_x + 14}" y="{legend_y + 9}" font-family="{FONT}"'
            f' font-size="10" fill="#374151">{escape(label)}</text>'
        )
        legend_x += 128

    lines.append(
        f'<text x="{FAMILY_LEFT}" y="{FAMILY_BOTTOM + 90}" font-family="{FONT}"'
        f' font-size="10" fill="#6b7280">'
        f"box=IQR | whiskers=min/max | dots=six chunk/batch shapes | "
        f"best median label is red</text>"
    )
    lines.append("</svg>")
    return "".join(lines) + "\n"


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
                f'<text x="{label_x:.1f}" y="{y + 3.7:.1f}" text-anchor="{anchor}" font-family="{FONT}" font-size="10" font-weight="700" fill="{text_color}" stroke="{label_stroke}" stroke-width="3" paint-order="stroke">{format_embedding_delta_pct(row.delta_pct)}</text>',
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
    parser.add_argument(
        "--readme", type=Path, default=Path("docs/PERFORMANCE-RESULTS.md")
    )
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
    parser.add_argument(
        "--only-ax-direct-snapshot",
        action="store_true",
        help="Render or verify only the README direct peer boxplot charts.",
    )
    args = parser.parse_args()
    repo_root = readme_artifacts.repo_root_for(args.readme)

    if args.only_ax_direct_snapshot:
        snapshot_path = find_ax_direct_snapshot(args.readme)
        if snapshot_path is None:
            raise ChartError("README does not declare an AX-only direct snapshot")
        snapshot = load_ax_direct_snapshot(snapshot_path)
        readme_slugs = readme_model_slugs(args.readme)
        llama_rows = load_llama_rows_from_readme(args.readme)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        mismatches = []
        for spec in CHARTS:
            rows = (
                load_retained_mlx_lm_rows(args.readme, readme_slugs)
                + ax_direct_snapshot_chart_rows(snapshot, spec.metric)
                + llama_rows
            )
            output_path = args.output_dir / chart_output_name(spec)
            content = render_family_chart(spec, collect_family_values(rows, spec))
            if not write_chart(output_path, content, args.check):
                mismatches.append(output_path)
        if mismatches:
            print("AX-only direct snapshot charts are stale:", file=sys.stderr)
            for path in mismatches:
                print(f"  {path}", file=sys.stderr)
            return 1
        print(
            "AX-only direct snapshot charts are "
            f"{'up to date' if args.check else 'rendered'}"
        )
        return 0

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
        mtp_6bit_summary = load_mtp_6bit_summary(mtp_6bit_summary_path)
        mtp_6bit_approximate = (
            mtp_6bit_summary["schema"] == MTP_6BIT_APPROXIMATE_SCHEMA
        )
        mtp_6bit_output_path = args.output_dir / (
            MTP_6BIT_APPROXIMATE_CHART_OUTPUT
            if mtp_6bit_approximate
            else MTP_6BIT_EXACT_CHART_OUTPUT
        )
        mtp_6bit_content = render_mtp_6bit_ax_acceleration_chart(
            mtp_6bit_summary["rows"],
            mtp_6bit_summary_path,
            approximate_diagnostic=mtp_6bit_approximate,
            engine_version=str(mtp_6bit_summary["engine_version"]),
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

    embedding_scale_output_path = args.output_dir / EMBEDDING_SCALE_CHART_OUTPUT
    embedding_scale_content = render_embedding_box_chart(
        load_embedding_scale_delta_rows(repo_root),
        title="Qwen3 embedding ingest scale (batched)",
        subtitle=(
            "Both series are matrix-batched encode (B=8/32/64) | "
            "box=IQR | whiskers=min/max | dots=six chunk×batch shapes."
        ),
        source_label=(
            "Sources: retained 2026-07-12 mlx-lm reference + 2026-07-16 "
            "AX-only refresh (0.6B/4B/8B); cross-run directional view, not B=1"
        ),
        ax_label=AX_ENGINE_CHART_LABEL,
    )
    if not write_chart(embedding_scale_output_path, embedding_scale_content, args.check):
        mismatches.append(embedding_scale_output_path)

    embeddinggemma_scale_output_path = args.output_dir / EMBEDDINGGEMMA_SCALE_CHART_OUTPUT
    embeddinggemma_scale_content = render_embedding_box_chart(
        load_embedding_overlay_scale_delta_rows(
            repo_root,
            EMBEDDINGGEMMA_SCALE_REFERENCE_ARTIFACT,
            EMBEDDINGGEMMA_SCALE_AX_ARTIFACT,
        ),
        title="EmbeddingGemma ingest scale",
        subtitle=(
            "One model group | box=IQR | whiskers=min/max | dots=six "
            "chunk/batch shapes."
        ),
        source_label=(
            "Sources: 2026-07-02 EmbeddingGemma paired reference + "
            "2026-07-16 AX-only refresh; cross-run directional view"
        ),
    )
    if not write_chart(
        embeddinggemma_scale_output_path, embeddinggemma_scale_content, args.check
    ):
        mismatches.append(embeddinggemma_scale_output_path)

    ax_direct_snapshot = None
    ax_direct_snapshot_path = find_ax_direct_snapshot(args.readme)
    if ax_direct_snapshot_path is not None:
        ax_direct_snapshot = load_ax_direct_snapshot(ax_direct_snapshot_path)

    for spec in CHARTS:
        if args.results_dir:
            benchmark_rows = load_rows(results_dir, readme_slugs, required=True)
        else:
            if ax_direct_snapshot is None:
                benchmark_rows = load_composite_rows(
                    args.readme, spec.metric, readme_slugs
                )
            else:
                benchmark_rows = load_retained_mlx_lm_rows(
                    args.readme, readme_slugs
                ) + ax_direct_snapshot_chart_rows(ax_direct_snapshot, spec.metric)
        rows = benchmark_rows + llama_rows
        engine_groups = collect_family_values(rows, spec)
        output_path = args.output_dir / chart_output_name(spec)
        content = render_family_chart(spec, engine_groups)
        if not write_chart(output_path, content, args.check):
            mismatches.append(output_path)

    # N-gram charts (Qwen3-4B): reads from the latest ngram-compare artifact.
    ngram_artifact_path = find_latest_ngram_artifact(repo_root)
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
    ngram_artifacts = find_ngram_artifacts_by_model(repo_root)
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
