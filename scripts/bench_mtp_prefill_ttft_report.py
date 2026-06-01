#!/usr/bin/env python3
"""Generate prefill rate and TTFT comparison tables and SVG charts from MTP fair bench artifacts.

Reads an existing bench_qwen36_mtp_fair output directory (which contains summary.json and
per-engine artifact JSON files) and emits:
  - prefill-tok-s.md / prefill-tok-s-<model>.svg
  - ttft-ms.md / ttft-ms-<model>.svg
  - prefill-ttft-report.md  (combined tables)

Sources per engine:
  ax_engine / ax_engine_ngram  — row["prefill_tok_s"]["median"], row["ttft_ms"]["median"]
  mtplx                        — case["prompt_tokens"] / run["prompt_eval_time_s"] per measured run
  lightning_mlx / _ngram       — run["ttft_s"] per measured run; approx prefill = prompt_tokens/ttft_s
                                 (Lightning runs via HTTP server; ttft_s includes socket overhead)

Usage:
  python3 scripts/bench_mtp_prefill_ttft_report.py \\
      --result-dir benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-ngram3
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENGINE_LABELS = {
    "mtplx": "MTPLX 0.3.7",
    "lightning_mlx": "Light. MTP",
    "lightning_mtp_ngram": "Light. ngram+MTP",
    "ax_engine": "AX MTP",
    "ax_engine_ngram": "AX MTP+n-gram",
}
ENGINE_COLORS = {
    "mtplx": "#14532d",
    "lightning_mlx": "#7c3aed",
    "lightning_mtp_ngram": "#1e3a8a",
    "ax_engine": "#f97316",
    "ax_engine_ngram": "#eab308",
}
ENGINE_ORDER = [
    "mtplx",
    "lightning_mlx",
    "lightning_mtp_ngram",
    "ax_engine",
    "ax_engine_ngram",
]
LIGHTNING_ENGINES = {"lightning_mlx", "lightning_mtp_ngram"}
AX_ENGINES = {"ax_engine", "ax_engine_ngram"}


@dataclass(frozen=True)
class BoxStats:
    values: tuple[float, ...]
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate percentile for empty values")
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


def numeric_samples(value: Any) -> list[float]:
    if isinstance(value, dict):
        values = value.get("values")
        if isinstance(values, list):
            return [float(v) for v in values if isinstance(v, int | float)]
        median = value.get("median")
        return [float(median)] if isinstance(median, int | float) else []
    if isinstance(value, int | float):
        return [float(value)]
    return []


# ---------------------------------------------------------------------------
# Per-case extraction
# ---------------------------------------------------------------------------


def _ax_case_prefill_ttft_samples(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    return numeric_samples(row.get("prefill_tok_s")), numeric_samples(row.get("ttft_ms"))


def _ax_case_prefill_ttft(row: dict[str, Any]) -> tuple[float | None, float | None]:
    prefill_values, ttft_values = _ax_case_prefill_ttft_samples(row)
    return median_or_none(prefill_values), median_or_none(ttft_values)


def _mtplx_case_prefill_ttft_samples(
    case: dict[str, Any],
) -> tuple[list[float], list[float]]:
    prompt_tokens = case.get("prompt_tokens")
    if not prompt_tokens:
        return [], []
    measured_runs = [r for r in case.get("runs", []) if r.get("measured")]
    prefill_times = [
        float(r["prompt_eval_time_s"])
        for r in measured_runs
        if r.get("prompt_eval_time_s") and float(r["prompt_eval_time_s"]) > 0
    ]
    if not prefill_times:
        return [], []
    return [float(prompt_tokens) / t for t in prefill_times], [t * 1000 for t in prefill_times]


def _mtplx_case_prefill_ttft(case: dict[str, Any]) -> tuple[float | None, float | None]:
    prefill_values, ttft_values = _mtplx_case_prefill_ttft_samples(case)
    return median_or_none(prefill_values), median_or_none(ttft_values)


def _lightning_case_prefill_ttft_samples(
    case: dict[str, Any],
) -> tuple[list[float], list[float]]:
    measured_runs = [r for r in case.get("runs", []) if r.get("measured")]
    ttft_values = [
        float(r["ttft_s"])
        for r in measured_runs
        if r.get("ttft_s") is not None and float(r["ttft_s"]) > 0
    ]
    if not ttft_values:
        return [], []
    prompt_tokens_values = [
        float(r["prompt_tokens"])
        for r in measured_runs
        if r.get("prompt_tokens") and r.get("ttft_s") and float(r["ttft_s"]) > 0
    ]
    prefill_values = (
        [p / t for p, t in zip(prompt_tokens_values, ttft_values) if t > 0]
        if len(prompt_tokens_values) == len(ttft_values)
        else []
    )
    return prefill_values, [t * 1000 for t in ttft_values]


def _lightning_case_prefill_ttft(case: dict[str, Any]) -> tuple[float | None, float | None]:
    prefill_values, ttft_values = _lightning_case_prefill_ttft_samples(case)
    return median_or_none(prefill_values), median_or_none(ttft_values)


def extract_case_prefill_ttft_samples(
    engine: str, case_or_row: dict[str, Any]
) -> tuple[list[float], list[float]]:
    if engine in AX_ENGINES:
        return _ax_case_prefill_ttft_samples(case_or_row)
    if engine == "mtplx":
        return _mtplx_case_prefill_ttft_samples(case_or_row)
    if engine in LIGHTNING_ENGINES:
        return _lightning_case_prefill_ttft_samples(case_or_row)
    return [], []


def extract_case_prefill_ttft(
    engine: str, case_or_row: dict[str, Any]
) -> tuple[float | None, float | None]:
    if engine in AX_ENGINES:
        return _ax_case_prefill_ttft(case_or_row)
    if engine == "mtplx":
        return _mtplx_case_prefill_ttft(case_or_row)
    if engine in LIGHTNING_ENGINES:
        return _lightning_case_prefill_ttft(case_or_row)
    return None, None


# ---------------------------------------------------------------------------
# Artifact-level extraction  (returns median across all cases)
# ---------------------------------------------------------------------------

AX_MTP_ENGINES = {"ax_engine_mlx_ngram_accel", "ax_engine_mlx_pure_mtp"}


def _iter_ax_rows(artifact: dict[str, Any]):
    for row in artifact.get("results", []):
        if row.get("engine") in AX_MTP_ENGINES and row.get("prompt_case_id"):
            yield row


def _iter_mtplx_cases(artifact: dict[str, Any]):
    for case in artifact.get("results", []):
        if case.get("prompt_id"):
            yield case


def _iter_lightning_cases(artifact: dict[str, Any]):
    for case in artifact.get("results", []):
        if case.get("prompt_id"):
            yield case


def engine_prefill_ttft(engine: str, artifact_path: Path) -> dict[str, Any]:
    if not artifact_path.is_file():
        return {"status": "missing", "prefill_tok_s": None, "ttft_ms": None}
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("schema") == "ax.mtp_engine_error.v1":
        return {"status": "error", "prefill_tok_s": None, "ttft_ms": None}

    if engine in AX_ENGINES:
        rows = list(_iter_ax_rows(artifact))
    elif engine == "mtplx":
        rows = list(_iter_mtplx_cases(artifact))
    else:
        rows = list(_iter_lightning_cases(artifact))

    prefill_values: list[float] = []
    ttft_values: list[float] = []
    prefill_samples: list[float] = []
    ttft_samples: list[float] = []
    for item in rows:
        p, t = extract_case_prefill_ttft(engine, item)
        if p is not None:
            prefill_values.append(p)
        if t is not None:
            ttft_values.append(t)
        p_samples, t_samples = extract_case_prefill_ttft_samples(engine, item)
        prefill_samples.extend(p_samples)
        ttft_samples.extend(t_samples)

    return {
        "status": "ok" if ttft_values else "no_data",
        "prefill_tok_s": median_or_none(prefill_values),
        "ttft_ms": median_or_none(ttft_values),
        "prefill_tok_s_samples": prefill_samples or prefill_values,
        "ttft_ms_samples": ttft_samples or ttft_values,
        "case_count": len(rows),
        "prefill_note": "approx_via_ttft" if engine in LIGHTNING_ENGINES else "measured",
    }


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------


def build_report(result_dir: Path) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    summary = json.loads(summary_path.read_text())

    engines: list[str] = summary["contract"]["engines"]
    rows: list[dict[str, Any]] = []

    for row in summary["rows"]:
        model = row["model"]
        suite = row["suite"]
        engine_stats: dict[str, dict[str, Any]] = {}
        for engine in engines:
            artifact_path = result_dir / model / suite / f"{engine}.json"
            engine_stats[engine] = engine_prefill_ttft(engine, artifact_path)
        rows.append(
            {
                "model": model,
                "model_label": row["model_label"],
                "suite": suite,
                "depth": row["depth"],
                "engines": engine_stats,
            }
        )

    return {
        "schema": "ax.mtp_prefill_ttft_report.v1",
        "created_at": summary.get("created_at"),
        "contract": summary["contract"],
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def write_prefill_ttft_markdown(path: Path, report: dict[str, Any]) -> None:
    engines = [e for e in ENGINE_ORDER if e in report["contract"]["engines"]]
    lines = [
        "# MTP Prefill Rate and TTFT Report",
        "",
        "## Notes",
        "",
        "- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).",
        "- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).",
        "- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).",
        "  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.",
        "",
        "## Prefill Rate (tok/s, higher is better)",
        "",
    ]
    header = ["Model", "Suite"] + [ENGINE_LABELS.get(e, e) for e in engines]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * 2 + ["---:"] * len(engines)) + " |")
    for row in report["rows"]:
        cells = [row["model_label"], row["suite"]]
        for engine in engines:
            stats = row["engines"].get(engine, {})
            note = " *" if stats.get("prefill_note") == "approx_via_ttft" else ""
            cells.append(_fmt(stats.get("prefill_tok_s")) + note)
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "\\* approx: Lightning prefill = prompt\\_tokens / ttft\\_s (includes HTTP overhead)",
        "",
        "## TTFT (ms, lower is better)",
        "",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * 2 + ["---:"] * len(engines)) + " |")
    for row in report["rows"]:
        cells = [row["model_label"], row["suite"]]
        for engine in engines:
            stats = row["engines"].get(engine, {})
            note = " *" if engine in LIGHTNING_ENGINES else ""
            cells.append(_fmt(stats.get("ttft_ms")) + note)
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "\\* Lightning TTFT includes local HTTP socket overhead",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# SVG charts
# ---------------------------------------------------------------------------

_CHART_FONT = "Inter,Segoe UI,Arial,sans-serif"


def _render_subtitle(subtitle: str, x: int, y: int) -> str:
    lower_marker = "lower is better"
    idx = subtitle.lower().find(lower_marker)
    if idx >= 0:
        before = subtitle[:idx]
        mid = subtitle[idx : idx + len(lower_marker)]
        after = subtitle[idx + len(lower_marker) :]
        inner = ""
        if before:
            inner += f"<tspan>{html.escape(before)}</tspan>"
        inner += f'<tspan fill="#dc2626" font-weight="700">{html.escape(mid)}</tspan>'
        if after:
            inner += f"<tspan>{html.escape(after)}</tspan>"
        return (
            f'<text x="{x}" y="{y}" font-family="{_CHART_FONT}"'
            f' font-size="11" fill="#4b5563">{inner}</text>'
        )
    return (
        f'<text x="{x}" y="{y}" font-family="{_CHART_FONT}"'
        f' font-size="11" fill="#4b5563">{html.escape(subtitle)}</text>'
    )


def _nice_axis_ceiling(max_value: float) -> float:
    if not math.isfinite(max_value) or max_value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(max_value))
    scaled = max_value / magnitude
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        if scaled <= step:
            return step * magnitude
    return 10.0 * magnitude


def _axis_label(value: float, unit: str) -> str:
    if unit == "ms":
        return f"{value:.0f}"
    if value >= 1000:
        return f"{value / 1000:g}k"
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _point_label(value: float, unit: str) -> str:
    if unit == "ms":
        return f"{value:.0f}"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _chart_samples(stats: dict[str, Any], metric: str) -> list[float]:
    samples = stats.get(f"{metric}_samples")
    values = (
        [float(v) for v in samples if isinstance(v, int | float)]
        if isinstance(samples, list)
        else []
    )
    if values:
        return values
    value = stats.get(metric)
    return [float(value)] if isinstance(value, int | float) else []


def _model_short_label(label: str) -> str:
    return label.removeprefix("Qwen3.6 ").removesuffix(" 4-bit")


def _combined_suite_chart_group(
    rows: list[dict[str, Any]],
    active_engines: list[str],
    metric: str,
    *,
    label: str = "all suites",
) -> list[dict[str, Any]]:
    values_by_engine = {engine: [] for engine in active_engines}
    for row in rows:
        for engine in active_engines:
            values_by_engine[engine].extend(
                _chart_samples(row["engines"].get(engine) or {}, metric)
            )
    return [{"label": label, "values": values_by_engine}]


def _model_combined_suite_chart_groups(
    rows: list[dict[str, Any]],
    active_engines: list[str],
    metric: str,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    model_keys: list[str] = []
    for row in rows:
        model_key = row["model"]
        if model_key not in model_keys:
            model_keys.append(model_key)
    for model_key in model_keys:
        model_rows = [row for row in rows if row["model"] == model_key]
        groups.extend(
            _combined_suite_chart_group(
                model_rows,
                active_engines,
                metric,
                label=_model_short_label(model_rows[0]["model_label"]),
            )
        )
    return groups


def _box_whisker_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    unit: str,
    groups: list[dict[str, Any]],
    active_engines: list[str],
    max_value: float | None = None,
    width: int = 900,
    lower_is_better: bool = False,
) -> None:
    height = 460
    left = 64
    right = 160
    header_right = 34
    top = 86
    bottom = 106
    plot_right = width - right
    plot_w = plot_right - left
    plot_h = height - top - bottom
    group_w = plot_w / max(len(groups), 1)
    gap = 8.0
    box_w = min(
        34.0,
        max(
            12.0,
            (group_w - 56 - gap * (len(active_engines) - 1))
            / max(len(active_engines), 1),
        ),
    )

    stats_by_key: dict[tuple[int, str], BoxStats] = {}
    all_values: list[float] = []
    all_medians: list[float] = []
    for group_index, group in enumerate(groups):
        for engine in active_engines:
            values = [
                float(v) for v in group["values"].get(engine, []) if v is not None
            ]
            if not values:
                continue
            stats = box_stats(values)
            stats_by_key[(group_index, engine)] = stats
            all_values.extend(values)
            all_medians.append(stats.median)

    axis_max = (
        max_value
        if max_value is not None
        else _nice_axis_ceiling(max(all_values or [1.0]) * 1.08)
    )
    best_value = (
        min(all_medians) if lower_is_better and all_medians else max(all_medians or [0.0])
    )

    def fy(value: float) -> float:
        clamped = max(0.0, min(value, axis_max))
        return top + plot_h - (clamped / axis_max) * plot_h

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

    direction_label = "Lower is better" if lower_is_better else "Higher is better"
    direction_fill = "#dc2626" if lower_is_better else "#374151"
    best_line_label = "lowest median" if lower_is_better else "highest median"
    best_side_label = "lowest" if lower_is_better else "highest"
    best_label = f"{best_side_label}: {_point_label(best_value, unit)}"
    unit_w = max(48, len(unit) * 7 + 24)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>Grouped box-and-whisker plot comparing "
        f"{html.escape(', '.join(ENGINE_LABELS.get(e, e) for e in active_engines))}.</desc>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="24" y="24" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="16" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        _render_subtitle(subtitle, 24, 46),
        f'<rect x="{width - header_right - unit_w}" y="13" width="{unit_w}" height="22" '
        f'rx="11" fill="#eef2ff" stroke="#c7d2fe"/>',
        f'<text x="{width - header_right - unit_w / 2:.1f}" y="28" text-anchor="middle" '
        f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" font-weight="700" '
        f'fill="#3730a3">{html.escape(unit)}</text>',
        f'<text x="{width - header_right}" y="52" text-anchor="end" '
        f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" font-weight="700" '
        f'fill="{direction_fill}">{html.escape(direction_label)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="6" '
        f'fill="#ffffff" stroke="#dbe3ef"/>',
    ]
    for i in range(5):
        value = axis_max * i / 4
        y = fy(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" fill="#6b7280">'
            f'{_axis_label(value, unit)}</text>'
        )
    if best_value > 0:
        best_y = fy(best_value)
        parts.append(
            f'<line x1="{left}" y1="{best_y:.1f}" x2="{plot_right}" y2="{best_y:.1f}" '
            f'stroke="#dc2626" stroke-width="1.2" stroke-dasharray="1 4" '
            f'stroke-linecap="round"/>'
        )
        parts.append(
            f'<text x="{plot_right + 8}" y="{max(top + 11, best_y - 5):.1f}" '
            f'text-anchor="start" font-family="Inter,Segoe UI,Arial,sans-serif" '
            f'font-size="11" font-weight="700" fill="#dc2626" '
            f'data-label="{html.escape(best_line_label)}">'
            f'{html.escape(best_label)}</text>'
        )
    dot_slots = 9
    dot_jitter = [
        (-0.36 + 0.72 * i / (dot_slots - 1)) * box_w for i in range(dot_slots)
    ]
    for group_index, group in enumerate(groups):
        group_x = left + group_w * group_index
        centers = engine_centers(group_x, len(active_engines))
        parts.append(
            f'<text x="{group_x + group_w / 2:.1f}" y="{height - 62}" text-anchor="middle" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" font-weight="700" '
            f'fill="#111827">{html.escape(group["label"])}</text>'
        )
        for engine_index, engine in enumerate(active_engines):
            stats = stats_by_key.get((group_index, engine))
            if stats is None:
                continue
            x = centers[engine_index]
            color = ENGINE_COLORS.get(engine, "#9ca3af")
            approx = engine in LIGHTNING_ENGINES
            stroke_opacity = "0.55" if approx else "0.82"
            fill_opacity = "0.12" if approx else "0.18"
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
                    f'cy="{fy(value):.1f}" r="1.45" fill="{color}" '
                    f'fill-opacity="{"0.34" if approx else "0.46"}"/>'
                )
            parts.extend(
                [
                    f'<line x1="{x:.1f}" y1="{y_max:.1f}" x2="{x:.1f}" y2="{y_min:.1f}" '
                    f'stroke="{color}" stroke-opacity="{stroke_opacity}" stroke-width="1.8"/>',
                    f'<line x1="{cap_left:.1f}" y1="{y_max:.1f}" x2="{cap_right:.1f}" y2="{y_max:.1f}" '
                    f'stroke="{color}" stroke-opacity="{stroke_opacity}" stroke-width="1.8"/>',
                    f'<line x1="{cap_left:.1f}" y1="{y_min:.1f}" x2="{cap_right:.1f}" y2="{y_min:.1f}" '
                    f'stroke="{color}" stroke-opacity="{stroke_opacity}" stroke-width="1.8"/>',
                    f'<rect x="{box_left:.1f}" y="{box_top:.1f}" width="{box_w:.1f}" '
                    f'height="{box_h:.1f}" rx="2" fill="{color}" fill-opacity="{fill_opacity}" '
                    f'stroke="{color}" stroke-opacity="{stroke_opacity}" stroke-width="1.8"/>',
                    f'<line x1="{box_left:.1f}" y1="{y_med:.1f}" x2="{box_left + box_w:.1f}" '
                    f'y2="{y_med:.1f}" stroke="{color}" stroke-opacity="{stroke_opacity}" '
                    f'stroke-width="2.6"/>',
                    f'<text x="{box_left + box_w + 6:.1f}" y="{y_med + 4:.1f}" '
                    f'text-anchor="start" font-family="Inter,Segoe UI,Arial,sans-serif" '
                    f'font-size="11" font-weight="700" fill="#111827" '
                    f'stroke="#ffffff" stroke-width="3" paint-order="stroke">'
                    f'{html.escape(_point_label(stats.median, unit))}</text>',
                ]
            )
    legend_y = height - 18
    legend_x = left
    legend_step = max(118.0, (width - left - right) / max(len(active_engines), 1))
    for engine in active_engines:
        color = ENGINE_COLORS.get(engine, "#9ca3af")
        label = ENGINE_LABELS.get(engine, engine)
        approx = engine in LIGHTNING_ENGINES
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" rx="2" '
            f'fill="{color}" fill-opacity="{"0.60" if approx else "0.86"}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14}" y="{legend_y}" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" fill="#374151">'
            f'{html.escape(label)}</text>'
        )
        legend_x += legend_step
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def write_prefill_svg(path: Path, report: dict[str, Any]) -> None:
    active_engines = [e for e in ENGINE_ORDER if e in report["contract"]["engines"]]
    groups = _model_combined_suite_chart_groups(
        report["rows"], active_engines, "prefill_tok_s"
    )
    _box_whisker_chart(
        path,
        title="MTP prefill throughput",
        subtitle="All suites combined per model | box=IQR | dots=runs | Lightning ~= TTFT-derived",
        unit="tok/s",
        groups=groups,
        active_engines=active_engines,
        width=max(900, len(groups) * 260),
    )


def write_ttft_svg(path: Path, report: dict[str, Any]) -> None:
    active_engines = [e for e in ENGINE_ORDER if e in report["contract"]["engines"]]
    groups = _model_combined_suite_chart_groups(report["rows"], active_engines, "ttft_ms")
    _box_whisker_chart(
        path,
        title="MTP time-to-first-token (TTFT)",
        subtitle="All suites combined per model | lower is better | box=IQR | dots=runs",
        unit="ms",
        groups=groups,
        active_engines=active_engines,
        width=max(900, len(groups) * 260),
        lower_is_better=True,
    )


def write_prefill_model_svg(path: Path, report: dict[str, Any], model_key: str) -> None:
    rows = [r for r in report["rows"] if r["model"] == model_key]
    if not rows:
        return
    model_label = rows[0]["model_label"]
    active_engines = [e for e in ENGINE_ORDER if e in report["contract"]["engines"]]
    _box_whisker_chart(
        path,
        title=f"{model_label} MTP prefill throughput",
        subtitle="All suites combined | box=IQR | dots=runs | Lightning ~= TTFT-derived",
        unit="tok/s",
        groups=_combined_suite_chart_group(rows, active_engines, "prefill_tok_s"),
        active_engines=active_engines,
    )


def write_ttft_model_svg(path: Path, report: dict[str, Any], model_key: str) -> None:
    rows = [r for r in report["rows"] if r["model"] == model_key]
    if not rows:
        return
    model_label = rows[0]["model_label"]
    active_engines = [e for e in ENGINE_ORDER if e in report["contract"]["engines"]]
    _box_whisker_chart(
        path,
        title=f"{model_label} MTP TTFT",
        subtitle="All suites combined | lower is better | box=IQR | dots=runs",
        unit="ms",
        groups=_combined_suite_chart_group(rows, active_engines, "ttft_ms"),
        active_engines=active_engines,
        lower_is_better=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Path to a bench_qwen36_mtp_fair output directory containing summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for report files (default: same as --result-dir)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = (args.output_dir or result_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading artifacts from: {result_dir}", flush=True)
    report = build_report(result_dir)

    report_json = output_dir / "prefill-ttft-report.json"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Saved report JSON: {report_json}")

    report_md = output_dir / "prefill-ttft-report.md"
    write_prefill_ttft_markdown(report_md, report)
    print(f"Saved report markdown: {report_md}")

    prefill_svg = output_dir / "prefill-tok-s.svg"
    write_prefill_svg(prefill_svg, report)
    print(f"Saved prefill chart: {prefill_svg}")

    ttft_svg = output_dir / "ttft-ms.svg"
    write_ttft_svg(ttft_svg, report)
    print(f"Saved TTFT chart: {ttft_svg}")

    model_keys = sorted({row["model"] for row in report["rows"]})
    for model_key in model_keys:
        p = output_dir / f"prefill-tok-s-{model_key}.svg"
        write_prefill_model_svg(p, report, model_key)
        print(f"Saved prefill chart: {p}")
        t = output_dir / f"ttft-ms-{model_key}.svg"
        write_ttft_model_svg(t, report, model_key)
        print(f"Saved TTFT chart: {t}")

    # print summary table to stdout
    print()
    engines = [e for e in ENGINE_ORDER if e in report["contract"]["engines"]]
    header = ["Model", "Suite"] + [ENGINE_LABELS.get(e, e) for e in engines]
    col_w = [max(len(h), 14) for h in header]
    sep = "  ".join("-" * w for w in col_w)
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print("Prefill tok/s (median of cases, ~ = approx via ttft_s)")
    print(row_fmt.format(*header))
    print(sep)
    for row in report["rows"]:
        cells: list[str] = [row["model_label"], row["suite"]]
        for engine in engines:
            stats = row["engines"].get(engine, {})
            val = stats.get("prefill_tok_s")
            note = "~" if stats.get("prefill_note") == "approx_via_ttft" else " "
            cells.append(_fmt(val) + note if val is not None else "-")
        print(row_fmt.format(*cells))
    print()
    print("TTFT ms (median of cases, ~ = includes HTTP overhead)")
    print(row_fmt.format(*header))
    print(sep)
    for row in report["rows"]:
        cells = [row["model_label"], row["suite"]]
        for engine in engines:
            stats = row["engines"].get(engine, {})
            val = stats.get("ttft_ms")
            note = "~" if engine in LIGHTNING_ENGINES else " "
            cells.append(_fmt(val) + note if val is not None else "-")
        print(row_fmt.format(*cells))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        import sys
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
