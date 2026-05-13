#!/usr/bin/env python3
"""Render README performance box-and-whisker SVG charts from benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


README_MODELS = [
    "gemma-4-e2b-it-4bit",
    "gemma-4-e2b-it-5bit",
    "gemma-4-e2b-it-6bit",
    "gemma-4-e2b-it-8bit",
    "gemma-4-e4b-it-4bit",
    "gemma-4-26b-a4b-it-4bit",
    "gemma-4-31b-it-4bit",
    "qwen3_5-9b-mlx-4bit",
    "qwen3_6-35b-a3b-ud-mlx-4bit",
    "qwen3_6-35b-a3b-5bit",
    "qwen3_6-35b-a3b-6bit",
    "qwen3_6-35b-a3b-8bit",
    "qwen3-coder-next-4bit",
    "glm-4.7-flash-4bit",
]

PROMPT_TOKENS = (128, 512)

SERIES = [
    ("llama_cpp_metal", "llama.cpp", "#f97316", "#c2410c"),
    ("mlx_lm", "mlx_lm", "#f2b705", "#9a6a00"),
    ("mlx_swift_lm", "mlx_swift_lm", "#4062bb", "#243b87"),
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
class ChartSpec:
    title: str
    subtitle: str | None
    unit: str
    direction_label: str
    output_name: str
    metric: str
    axis_max: float


CHARTS = [
    ChartSpec(
        title="Prefill rate",
        subtitle=None,
        unit="tok/s",
        direction_label="Higher is better",
        output_name="perf-prefill-box-whisker.svg",
        metric="prefill",
        axis_max=9000.0,
    ),
    ChartSpec(
        title="Decode rate",
        subtitle="AX n-gram is the default policy",
        unit="tok/s",
        direction_label="Higher is better",
        output_name="perf-decode-box-whisker.svg",
        metric="decode",
        axis_max=700.0,
    ),
    ChartSpec(
        title="TTFT",
        subtitle=None,
        unit="ms",
        direction_label="Lower is better",
        output_name="perf-ttft-box-whisker.svg",
        metric="ttft",
        axis_max=900.0,
    ),
]


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


def load_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for slug in README_MODELS:
        path = results_dir / f"{slug}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text())
        for row in payload.get("results", []):
            if row.get("prompt_tokens") in PROMPT_TOKENS:
                rows.append(row)
    if missing:
        raise ChartError("missing README benchmark artifacts:\n" + "\n".join(missing))
    return rows


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
        if all((path / f"{slug}.json").exists() for slug in README_MODELS)
    ]
    if complete:
        return max(complete, key=lambda path: path.name).resolve()
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def collect_values(rows: list[dict[str, Any]], metric: str) -> list[SeriesStats]:
    stats: list[SeriesStats] = []
    for engine, label, color, dot_color in SERIES:
        values: list[float] = []
        for row in rows:
            if row.get("engine") != engine:
                continue
            if metric == "prefill":
                values.append(metric_median(row, "prefill_tok_s"))
            elif metric == "decode":
                values.append(metric_median(row, "decode_tok_s"))
            elif metric == "ttft":
                if engine in {"llama_cpp_metal", "mlx_lm", "mlx_swift_lm"}:
                    prompt_tokens = row.get("prompt_tokens")
                    if not isinstance(prompt_tokens, int):
                        raise ChartError(f"missing prompt_tokens for {engine}")
                    values.append(prompt_tokens / metric_median(row, "prefill_tok_s") * 1000)
                else:
                    values.append(metric_median(row, "ttft_ms"))
            else:
                raise ChartError(f"unknown metric: {metric}")
        if len(values) != len(README_MODELS) * len(PROMPT_TOKENS):
            raise ChartError(
                f"{metric} chart expected {len(README_MODELS) * len(PROMPT_TOKENS)} "
                f"values for {engine}, found {len(values)}"
            )
        stats.append(summarize(values, engine, label, color, dot_color))
    return stats


def y_scale(value: float, axis_max: float) -> float:
    clamped = max(0.0, min(value, axis_max))
    return BOTTOM - (clamped / axis_max) * (BOTTOM - TOP)


def short_number(value: float) -> str:
    if value >= 1000:
        compact = value / 1000
        if compact.is_integer():
            return f"{compact:.0f}k"
        return f"{compact:.1f}k"
    if value.is_integer():
        return f"{value:.0f}"
    return f"{value:.1f}"


def median_label(value: float) -> str:
    if value >= 1000:
        return f"med {value / 1000:.1f}k"
    return f"med {value:.0f}"


def render_chart(spec: ChartSpec, stats: list[SeriesStats]) -> str:
    x_positions = [78.0, 159.25, 240.5, 321.75, 403.0]
    dot_offsets = (-8.5, -5.0, -2.0, 1.5, 5.0, 8.5, -6.5, 3.5)
    ngram = next(item for item in stats if item.engine == "ax_engine_mlx_ngram_accel")
    ngram_median_y = y_scale(ngram.median, spec.axis_max)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f"<title>{escape(spec.title)}</title>",
        (
            f"<desc>Box-and-Whisker Plot comparing llama.cpp Metal, mlx_lm, "
            f"mlx_swift_lm, ax_engine, and ax_engine plus n-gram across 28 README benchmark rows. "
            f"A red dotted horizontal line marks the ax_engine plus n-gram median.</desc>"
        ),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        f'<text x="52" y="22" font-family="{FONT}" font-size="16" font-weight="700" fill="#111827">{escape(spec.title)}</text>',
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

    for value in (0.0, spec.axis_max / 2, spec.axis_max):
        y = y_scale(value, spec.axis_max)
        lines.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{RIGHT}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(
            f'<text x="44" y="{y + 3:.1f}" text-anchor="end" font-family="{FONT}" font-size="10" fill="#6b7280">{short_number(value)}</text>'
        )

    lines.append(
        f'<line x1="{LEFT}" y1="{ngram_median_y:.1f}" x2="{RIGHT}" y2="{ngram_median_y:.1f}" stroke="{RED}" stroke-width="1.2" stroke-dasharray="2 4"/>'
    )
    lines.append(f'<rect x="300" y="50" width="135" height="160" fill="none" stroke="{RED}" stroke-width="1.4"/>')

    box_width = 34
    for stat, x in zip(stats, x_positions):
        y_min = y_scale(stat.minimum, spec.axis_max)
        y_q1 = y_scale(stat.q1, spec.axis_max)
        y_med = y_scale(stat.median, spec.axis_max)
        y_q3 = y_scale(stat.q3, spec.axis_max)
        y_max = y_scale(stat.maximum, spec.axis_max)
        box_y = min(y_q1, y_q3)
        box_h = max(abs(y_q3 - y_q1), 1.0)
        cap_left = x - 10
        cap_right = x + 10
        box_left = x - box_width / 2
        lines.extend(
            [
                f'<line x1="{x:g}" y1="{y_max:.1f}" x2="{x:g}" y2="{y_min:.1f}" stroke="{stat.color}" stroke-width="2"/>',
                f'<line x1="{cap_left:g}" y1="{y_max:.1f}" x2="{cap_right:g}" y2="{y_max:.1f}" stroke="{stat.color}" stroke-width="2"/>',
                f'<line x1="{cap_left:g}" y1="{y_min:.1f}" x2="{cap_right:g}" y2="{y_min:.1f}" stroke="{stat.color}" stroke-width="2"/>',
                f'<rect x="{box_left:g}" y="{box_y:.1f}" width="{box_width}" height="{box_h:.1f}" rx="4" fill="{stat.color}" fill-opacity="0.18" stroke="{stat.color}" stroke-width="2"/>',
                f'<line x1="{box_left:g}" y1="{y_med:.1f}" x2="{box_left + box_width:g}" y2="{y_med:.1f}" stroke="{stat.color}" stroke-width="3"/>',
                f'<text x="{x:g}" y="230" text-anchor="middle" font-family="{FONT}" font-size="9" fill="#111827">{escape(stat.label)}</text>',
                f'<text x="{x:g}" y="246" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#6b7280">{median_label(stat.median)}</text>',
            ]
        )
        for index, value in enumerate(stat.values):
            dot_x = x + dot_offsets[index % len(dot_offsets)]
            dot_y = y_scale(value, spec.axis_max)
            lines.append(f'<circle cx="{dot_x:g}" cy="{dot_y:.1f}" r="2" fill="{stat.dot_color}" fill-opacity="0.9"/>')

    lines.append("</svg>")
    return "".join(lines) + "\n"


def infer_results_dir_from_readme(readme: Path) -> Path:
    text = readme.read_text()
    patterns = [
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
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated SVGs match files on disk without writing.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir or infer_results_dir_from_readme(args.readme)
    llama_results_dir = args.llama_results_dir or infer_llama_results_dir(args.readme.parent)
    rows = load_rows(results_dir) + load_rows(llama_results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mismatches: list[Path] = []
    for spec in CHARTS:
        stats = collect_values(rows, spec.metric)
        output_path = args.output_dir / spec.output_name
        content = render_chart(spec, stats)
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
        print(f"Rendered README performance charts from {results_dir} and {llama_results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
