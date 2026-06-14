#!/usr/bin/env python3
"""Render Qwen3-Coder-Next README direct-mode grouped bar charts.

Qwen3-Coder-Next is a single-model direct-only case, so this script renders
one chart per metric with prompt length on the x-axis and one bar per engine
inside each prompt group.

Usage:
  python3 scripts/render_qwen_coder_next_charts.py \
    --artifact benchmarks/results/mlx-inference/2026-06-13-qwen3-coder-next-prefill-probe/qwen3-coder-next-4bit-p128-p2048-step4096.json \
    --artifact benchmarks/results/mlx-inference/2026-06-13-qwen3-coder-next-prefill-probe/qwen3-coder-next-4bit-p512-step4096.json \
    --llama-artifact benchmarks/results/llama-cpp-metal/2026-05-13-full-sweep/qwen3-coder-next-4bit.json \
    --llama-artifact benchmarks/results/llama-cpp-metal/2026-06-13-qwen3-coder-next-2048/qwen3-coder-next-4bit-p2048.json \
    --assets-dir docs/assets
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FONT = "Inter,Segoe UI,Arial,sans-serif"
RED = "#dc2626"

PROMPT_TOKENS = (128, 512, 2048)

# Fallbacks used only when a version cannot be recovered from the artifacts.
MLX_LM_VERSION_FALLBACK = "0.31.3"

WIDTH = 800
HEIGHT = 430
LEFT = 64
RIGHT = 640
TOP = 86
BOTTOM = 316

SERIES = [
    ("llama_cpp_metal", "llama.cpp Metal", "#f97316", "#c2410c"),
    ("mlx_lm", "mlx-lm 0.31.3", "#f2b705", "#9a6a00"),
    ("ax_engine_mlx", "AX Engine native MLX", "#2eaf5f", "#176c37"),
]

METRICS = [
    ("decode_tok_s", "decode-tok-s", "Qwen3-Coder-Next 4-bit - Direct decode", "tok/s", False),
    ("prefill_tok_s", "prefill-tok-s", "Qwen3-Coder-Next 4-bit - Prefill", "tok/s", False),
    ("ttft_ms", "ttft-ms", "Qwen3-Coder-Next 4-bit - Time to first token", "ms", True),
]


@dataclass(frozen=True)
class Stats:
    values: tuple[float, ...]
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def metric_median(row: dict[str, Any], key: str) -> float | None:
    metric = row.get(key)
    if isinstance(metric, dict) and isinstance(metric.get("median"), (int, float)):
        return float(metric["median"])
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def trial_values(row: dict[str, Any], key: str) -> list[float]:
    values: list[float] = []
    trial_sources = {
        "decode_tok_s": ("decode_trials",),
        "prefill_tok_s": ("prefill_trials",),
        "ttft_ms": ("trials", "prefill_trials"),
    }.get(key, ("trials",))
    for source in trial_sources:
        for trial in row.get(source) or []:
            if isinstance(trial, dict) and isinstance(trial.get(key), (int, float)):
                values.append(float(trial[key]))
        if values:
            return values
    for trial in row.get("trials") or []:
        if isinstance(trial, dict) and isinstance(trial.get(key), (int, float)):
            values.append(float(trial[key]))
    if values:
        return values
    median = metric_median(row, key)
    return [median] if median is not None else []


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate percentile of an empty series")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def summarize(values: list[float]) -> Stats:
    ordered = sorted(values)
    return Stats(
        values=tuple(values),
        minimum=ordered[0],
        q1=percentile(ordered, 0.25),
        median=percentile(ordered, 0.50),
        q3=percentile(ordered, 0.75),
        maximum=ordered[-1],
    )


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


def collect(rows: list[dict[str, Any]], metric: str) -> dict[str, dict[int, Stats]]:
    data: dict[str, dict[int, Stats]] = {engine: {} for engine, *_ in SERIES}
    for row in rows:
        engine = row.get("engine")
        prompt_tokens = row.get("prompt_tokens")
        if engine not in data or prompt_tokens not in PROMPT_TOKENS:
            continue
        values = trial_values(row, metric)
        if values:
            data[str(engine)][int(prompt_tokens)] = summarize(values)
    return data


def render_chart(
    *,
    title: str,
    unit: str,
    lower_is_better: bool,
    data: dict[str, dict[int, Stats]],
    labels: dict[str, str],
    provenance: str,
) -> str:
    all_stats = [stats for engine_data in data.values() for stats in engine_data.values()]
    if not all_stats:
        raise ValueError("no data to chart")
    axis_max = nice_axis_ceiling(max(stats.maximum for stats in all_stats) * 1.05)
    best = (
        min(stats.median for stats in all_stats)
        if lower_is_better
        else max(stats.median for stats in all_stats)
    )

    plot_width = RIGHT - LEFT
    plot_height = BOTTOM - TOP
    group_step = plot_width / len(PROMPT_TOKENS)
    bar_gap = 8.0
    bar_w = 34.0
    series_offsets = [
        -(bar_w + bar_gap),
        0.0,
        bar_w + bar_gap,
    ]
    header_right = WIDTH - 34
    unit_w = max(48, len(unit) * 7 + 24)
    direction = "Lower is better" if lower_is_better else "Higher is better"
    direction_fill = RED if lower_is_better else "#374151"

    def fy(value: float) -> float:
        clamped = max(0.0, min(value, axis_max))
        return BOTTOM - (clamped / axis_max) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f"<title>{escape(title)}</title>",
        f"<desc>Grouped bar chart of median values comparing llama.cpp Metal, mlx-lm, "
        f"and AX Engine native MLX at 128/512/2048 prompt tokens for "
        f"Qwen3-Coder-Next 4-bit.</desc>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>',
        f'<text x="{LEFT}" y="24" font-family="{FONT}" font-size="16" '
        f'font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="{LEFT}" y="46" font-family="{FONT}" font-size="11" fill="#4b5563">'
        f"median over reps | grouped by prompt tokens | generation=128 tokens</text>",
        f'<text x="{LEFT}" y="62" font-family="{FONT}" font-size="10" fill="#6b7280">'
        f"direct-only Qwen3-Coder-Next comparison; llama.cpp rows are shape-compatible GGUF</text>",
        f'<text x="{LEFT}" y="76" font-family="{FONT}" font-size="9" fill="#9ca3af">'
        f"{escape(provenance)}</text>",
        f'<rect x="{header_right - unit_w}" y="13" width="{unit_w}" height="22" '
        f'rx="11" fill="#eef2ff" stroke="#c7d2fe"/>',
        f'<text x="{header_right - unit_w / 2:.1f}" y="28" text-anchor="middle" '
        f'font-family="{FONT}" font-size="10" font-weight="700" fill="#3730a3">{escape(unit)}</text>',
        f'<text x="{header_right}" y="52" text-anchor="end" font-family="{FONT}" '
        f'font-size="10" font-weight="700" fill="{direction_fill}">{direction}</text>',
        f'<rect x="{LEFT}" y="{TOP}" width="{plot_width}" height="{plot_height}" '
        f'rx="6" fill="#ffffff" stroke="#dbe3ef"/>',
    ]

    for i in range(5):
        grid_value = axis_max * i / 4
        gy = fy(grid_value)
        parts.append(
            f'<line x1="{LEFT}" y1="{gy:.1f}" x2="{RIGHT}" y2="{gy:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{LEFT - 8}" y="{gy + 3:.1f}" text-anchor="end" '
            f'font-family="{FONT}" font-size="11" fill="#6b7280">{short_number(grid_value)}</text>'
        )

    best_y = fy(best)
    parts.append(
        f'<line x1="{LEFT}" y1="{best_y:.1f}" x2="{RIGHT}" y2="{best_y:.1f}" '
        f'stroke="{RED}" stroke-width="1.2" stroke-dasharray="1 4" stroke-linecap="round"/>'
    )
    parts.append(
        f'<text x="{RIGHT + 8}" y="{max(TOP + 11, best_y - 5):.1f}" '
        f'text-anchor="start" font-family="{FONT}" font-size="11" font-weight="700" '
        f'fill="{RED}">{"lowest" if lower_is_better else "highest"}: {short_number(best)}</text>'
    )

    for group_index, prompt_tokens in enumerate(PROMPT_TOKENS):
        center = LEFT + (group_index + 0.5) * group_step
        for series_index, (engine, _label, fill, stroke) in enumerate(SERIES):
            stats = data.get(engine, {}).get(prompt_tokens)
            bar_x = center + series_offsets[series_index] - bar_w / 2
            bar_center = bar_x + bar_w / 2
            if stats is None:
                parts.append(
                    f'<text x="{bar_center:.1f}" y="{BOTTOM - 8:.1f}" text-anchor="middle" '
                    f'font-family="{FONT}" font-size="9" fill="#9ca3af">N/A</text>'
                )
                continue
            y_med = fy(stats.median)
            bar_h = max(1.0, BOTTOM - y_med)
            parts.extend(
                [
                    f'<rect x="{bar_x:.1f}" y="{y_med:.1f}" width="{bar_w:.1f}" '
                    f'height="{bar_h:.1f}" rx="3" fill="{fill}" fill-opacity="0.40" '
                    f'stroke="{stroke}" stroke-width="1.8"/>',
                    f'<line x1="{bar_x:.1f}" y1="{y_med:.1f}" x2="{bar_x + bar_w:.1f}" '
                    f'y2="{y_med:.1f}" stroke="{stroke}" stroke-width="2.6"/>',
                    f'<text x="{bar_center:.1f}" y="{max(TOP + 11, y_med - 6):.1f}" '
                    f'text-anchor="middle" font-family="{FONT}" font-size="10" '
                    f'font-weight="700" fill="#111827">{short_number(stats.median)}</text>',
                ]
            )

        parts.append(
            f'<text x="{center:.1f}" y="{BOTTOM + 30}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="11" font-weight="700" fill="#111827">{prompt_tokens} tok</text>'
        )

    legend_y = HEIGHT - 22
    legend_x = LEFT
    for engine, default_label, fill, stroke in SERIES:
        label = labels.get(engine, default_label)
        parts.append(
            f'<rect x="{legend_x:.1f}" y="{legend_y - 9}" width="12" height="10" rx="2" '
            f'fill="{fill}" fill-opacity="0.40" stroke="{stroke}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<text x="{legend_x + 16:.1f}" y="{legend_y}" font-family="{FONT}" '
            f'font-size="10" fill="#374151">{escape(label)}</text>'
        )
        legend_x += 18 + len(label) * 6.0

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def load_payloads(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in paths]


def rows_of(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        rows.extend(payload.get("results", []))
    return rows


def cargo_workspace_version() -> str | None:
    cargo = REPO_ROOT / "Cargo.toml"
    try:
        text = cargo.read_text()
    except OSError:
        return None
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    return match.group(1) if match else None


@dataclass(frozen=True)
class Versions:
    ax_version: str | None
    ax_commit: str | None
    mlx_lm: str | None
    llama_build: str | None
    llama_ggml: str | None
    llama_flash_attn: bool


def _llama_ggml_from_evidence(evidence: str) -> str | None:
    match = re.search(r"Cellar/ggml/([0-9][0-9.]*)/", evidence or "")
    return match.group(1) if match else None


def extract_versions(
    mlx_payloads: list[dict[str, Any]], llama_payloads: list[dict[str, Any]]
) -> Versions:
    # AX Engine: release version from the workspace, build commit from the MLX artifact.
    ax_commit = None
    for payload in mlx_payloads:
        commit = (payload.get("build") or {}).get("commit")
        if commit:
            ax_commit = commit[:8]
            break

    # mlx-lm: prefer a version recorded in the artifact, else fall back to the constant.
    mlx_lm = None
    for row in rows_of(mlx_payloads):
        if row.get("engine") == "mlx_lm":
            mlx_lm = row.get("mlx_lm_version") or row.get("engine_version")
            if mlx_lm:
                break
    if mlx_lm is None:
        mlx_lm = MLX_LM_VERSION_FALLBACK

    # llama.cpp: build number, ggml version, and flash-attn flag from the llama rows.
    llama_build = llama_ggml = None
    llama_fa = False
    for row in rows_of(llama_payloads):
        info = row.get("llama_cpp")
        if not isinstance(info, dict):
            continue
        build_number = info.get("build_number")
        if build_number is not None and llama_build is None:
            llama_build = f"b{build_number}"
        if llama_ggml is None:
            llama_ggml = _llama_ggml_from_evidence(row.get("llama_cpp_device_evidence", ""))
        if info.get("flash_attn") in (True, 1):
            llama_fa = True

    return Versions(
        ax_version=cargo_workspace_version(),
        ax_commit=ax_commit,
        mlx_lm=mlx_lm,
        llama_build=llama_build,
        llama_ggml=llama_ggml,
        llama_flash_attn=llama_fa,
    )


def series_labels(versions: Versions) -> dict[str, str]:
    ax = "AX Engine"
    if versions.ax_version:
        ax = f"AX Engine {versions.ax_version}"
    llama = "llama.cpp Metal"
    if versions.llama_build:
        suffix = " FA" if versions.llama_flash_attn else ""
        llama = f"llama.cpp Metal {versions.llama_build}{suffix}"
    mlx_lm = f"mlx-lm {versions.mlx_lm}" if versions.mlx_lm else "mlx-lm"
    return {
        "llama_cpp_metal": llama,
        "mlx_lm": mlx_lm,
        "ax_engine_mlx": ax,
    }


def provenance_line(versions: Versions) -> str:
    bits: list[str] = []
    if versions.ax_version:
        commit = f" @{versions.ax_commit}" if versions.ax_commit else ""
        bits.append(f"AX {versions.ax_version}{commit}")
    if versions.mlx_lm:
        bits.append(f"mlx-lm {versions.mlx_lm}")
    if versions.llama_build:
        ggml = f", ggml {versions.llama_ggml}" if versions.llama_ggml else ""
        fa = ", FA on" if versions.llama_flash_attn else ""
        bits.append(f"llama.cpp {versions.llama_build}{ggml}{fa}")
    return "versions: " + " | ".join(bits) if bits else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=Path, required=True)
    parser.add_argument("--llama-artifact", action="append", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, default=REPO_ROOT / "docs" / "assets")
    args = parser.parse_args()

    mlx_payloads = load_payloads(args.artifact)
    llama_payloads = load_payloads(args.llama_artifact)
    rows = rows_of([*mlx_payloads, *llama_payloads])
    versions = extract_versions(mlx_payloads, llama_payloads)
    labels = series_labels(versions)
    provenance = provenance_line(versions)
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for metric, slug, title, unit, lower_is_better in METRICS:
        data = collect(rows, metric)
        svg = render_chart(
            title=title,
            unit=unit,
            lower_is_better=lower_is_better,
            data=data,
            labels=labels,
            provenance=provenance,
        )
        out = args.assets_dir / f"perf-qwen-coder-next-{slug}.svg"
        out.write_text(svg)
        written.append(str(out))
    print("Wrote:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
