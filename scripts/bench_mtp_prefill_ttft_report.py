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
import statistics
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
    "mtplx": "#a78bfa",
    "lightning_mlx": "#60a5fa",
    "lightning_mtp_ngram": "#1d4ed8",
    "ax_engine": "#86efac",
    "ax_engine_ngram": "#15803d",
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


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


# ---------------------------------------------------------------------------
# Per-case extraction
# ---------------------------------------------------------------------------


def _ax_case_prefill_ttft(row: dict[str, Any]) -> tuple[float | None, float | None]:
    pfr = row.get("prefill_tok_s")
    ttft = row.get("ttft_ms")
    prefill_tok_s = float(pfr["median"]) if isinstance(pfr, dict) else None
    ttft_ms = float(ttft["median"]) if isinstance(ttft, dict) else None
    return prefill_tok_s, ttft_ms


def _mtplx_case_prefill_ttft(case: dict[str, Any]) -> tuple[float | None, float | None]:
    prompt_tokens = case.get("prompt_tokens")
    if not prompt_tokens:
        return None, None
    measured_runs = [r for r in case.get("runs", []) if r.get("measured")]
    prefill_times = [
        float(r["prompt_eval_time_s"])
        for r in measured_runs
        if r.get("prompt_eval_time_s") and float(r["prompt_eval_time_s"]) > 0
    ]
    if not prefill_times:
        return None, None
    prefill_tok_s = median_or_none([prompt_tokens / t for t in prefill_times])
    ttft_ms = median_or_none([t * 1000 for t in prefill_times])
    return prefill_tok_s, ttft_ms


def _lightning_case_prefill_ttft(case: dict[str, Any]) -> tuple[float | None, float | None]:
    measured_runs = [r for r in case.get("runs", []) if r.get("measured")]
    ttft_values = [
        float(r["ttft_s"])
        for r in measured_runs
        if r.get("ttft_s") is not None and float(r["ttft_s"]) > 0
    ]
    if not ttft_values:
        return None, None
    prompt_tokens_values = [
        float(r["prompt_tokens"])
        for r in measured_runs
        if r.get("prompt_tokens") and r.get("ttft_s") and float(r["ttft_s"]) > 0
    ]
    ttft_ms = median_or_none([t * 1000 for t in ttft_values])
    prefill_tok_s = (
        median_or_none(
            [p / t for p, t in zip(prompt_tokens_values, ttft_values) if t > 0]
        )
        if len(prompt_tokens_values) == len(ttft_values)
        else None
    )
    return prefill_tok_s, ttft_ms


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
    for item in rows:
        p, t = extract_case_prefill_ttft(engine, item)
        if p is not None:
            prefill_values.append(p)
        if t is not None:
            ttft_values.append(t)

    return {
        "status": "ok" if ttft_values else "no_data",
        "prefill_tok_s": median_or_none(prefill_values),
        "ttft_ms": median_or_none(ttft_values),
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
            f'<text x="{x}" y="{y}" text-anchor="end" font-family="{_CHART_FONT}"'
            f' font-size="11" fill="#4b5563">{inner}</text>'
        )
    return (
        f'<text x="{x}" y="{y}" text-anchor="end" font-family="{_CHART_FONT}"'
        f' font-size="11" fill="#4b5563">{html.escape(subtitle)}</text>'
    )


def _vertical_bar_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    groups: list[dict[str, Any]],
    active_engines: list[str],
    max_value: float,
    width: int = 760,
) -> None:
    height = 430
    left = 60
    right = 28
    top = 68
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(len(groups), 1)
    bar_gap = 8
    bar_w = min(
        54, (group_w - 44 - bar_gap * (len(active_engines) - 1)) / max(len(active_engines), 1)
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="18" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        _render_subtitle(subtitle, width - right, 30),
    ]
    for i in range(4):
        y = top + plot_h - plot_h * i / 3
        value = max_value * i / 3
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" fill="#6b7280">'
            f'{value:.0f}</text>'
        )
    for group_index, group in enumerate(groups):
        group_x = left + group_w * group_index
        bars_w = bar_w * len(active_engines) + bar_gap * max(len(active_engines) - 1, 0)
        start_x = group_x + (group_w - bars_w) / 2
        parts.append(
            f'<text x="{group_x + group_w / 2:.1f}" y="{height - 34}" text-anchor="middle" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" font-weight="700" '
            f'fill="#111827">{html.escape(group["label"])}</text>'
        )
        for engine_index, engine in enumerate(active_engines):
            value = group["values"].get(engine)
            bar_h = 0.0 if value is None else plot_h * float(value) / max_value
            x = start_x + engine_index * (bar_w + bar_gap)
            y = top + plot_h - bar_h
            color = ENGINE_COLORS.get(engine, "#9ca3af")
            approx = engine in LIGHTNING_ENGINES
            fill_opacity = "0.60" if approx else "0.86"
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'rx="2" fill="{color}" fill-opacity="{fill_opacity}"/>'
            )
            label_txt = "-" if value is None else f"{float(value):.0f}"
            if approx and value is not None:
                label_txt += "~"
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{max(y - 5, top + 10):.1f}" '
                f'text-anchor="middle" font-family="Inter,Segoe UI,Arial,sans-serif" '
                f'font-size="10" font-weight="700" fill="#111827">{label_txt}</text>'
            )
    legend_y = height - 14
    legend_x = left
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
        legend_x += 130
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def write_prefill_svg(path: Path, report: dict[str, Any]) -> None:
    active_engines = [e for e in ENGINE_ORDER if e in report["contract"]["engines"]]
    groups = [
        {
            "label": f"{row['model'].replace('-4bit', '')} {row['suite']}",
            "values": {
                e: (row["engines"].get(e) or {}).get("prefill_tok_s")
                for e in active_engines
            },
        }
        for row in report["rows"]
    ]
    all_vals = [float(v) for g in groups for v in g["values"].values() if v is not None]
    max_value = max(all_vals or [1.0])
    _vertical_bar_chart(
        path,
        title="MTP prefill throughput",
        subtitle="tok/s — higher is better  (~ = via ttft_s, includes HTTP overhead)",
        groups=groups,
        active_engines=active_engines,
        max_value=max_value,
        width=max(760, len(groups) * 120),
    )


def write_ttft_svg(path: Path, report: dict[str, Any]) -> None:
    active_engines = [e for e in ENGINE_ORDER if e in report["contract"]["engines"]]
    groups = [
        {
            "label": f"{row['model'].replace('-4bit', '')} {row['suite']}",
            "values": {
                e: (row["engines"].get(e) or {}).get("ttft_ms")
                for e in active_engines
            },
        }
        for row in report["rows"]
    ]
    all_vals = [float(v) for g in groups for v in g["values"].values() if v is not None]
    max_value = max(all_vals or [1.0])
    _vertical_bar_chart(
        path,
        title="MTP time-to-first-token (TTFT)",
        subtitle="ms — lower is better  (~ = client-side via ttft_s, includes HTTP overhead)",
        groups=groups,
        active_engines=active_engines,
        max_value=max_value,
        width=max(760, len(groups) * 120),
    )


def write_prefill_model_svg(path: Path, report: dict[str, Any], model_key: str) -> None:
    rows = [r for r in report["rows"] if r["model"] == model_key]
    if not rows:
        return
    model_label = rows[0]["model_label"]
    active_engines = [e for e in ENGINE_ORDER if e in report["contract"]["engines"]]
    groups = [
        {
            "label": row["suite"],
            "values": {
                e: (row["engines"].get(e) or {}).get("prefill_tok_s")
                for e in active_engines
            },
        }
        for row in rows
    ]
    all_vals = [float(v) for g in groups for v in g["values"].values() if v is not None]
    max_value = max(all_vals or [1.0])
    _vertical_bar_chart(
        path,
        title=f"{model_label} MTP prefill throughput",
        subtitle="tok/s — higher is better  (~ = via ttft_s, includes HTTP overhead)",
        groups=groups,
        active_engines=active_engines,
        max_value=max_value,
    )


def write_ttft_model_svg(path: Path, report: dict[str, Any], model_key: str) -> None:
    rows = [r for r in report["rows"] if r["model"] == model_key]
    if not rows:
        return
    model_label = rows[0]["model_label"]
    active_engines = [e for e in ENGINE_ORDER if e in report["contract"]["engines"]]
    groups = [
        {
            "label": row["suite"],
            "values": {
                e: (row["engines"].get(e) or {}).get("ttft_ms")
                for e in active_engines
            },
        }
        for row in rows
    ]
    all_vals = [float(v) for g in groups for v in g["values"].values() if v is not None]
    max_value = max(all_vals or [1.0])
    _vertical_bar_chart(
        path,
        title=f"{model_label} MTP TTFT",
        subtitle="ms — lower is better  (~ = client-side via ttft_s, includes HTTP overhead)",
        groups=groups,
        active_engines=active_engines,
        max_value=max_value,
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
