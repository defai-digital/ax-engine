#!/usr/bin/env python3
"""Render validated MLX prefill/TTFT scaling artifacts as Markdown reports."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from check_mlx_prefill_scaling_artifact import (
    PrefillScalingArtifactError,
    load_json,
    validate_prefill_scaling_artifact,
)


class PrefillScalingReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReportRow:
    context_tokens: int
    generation_tokens: int
    mlx_prefill_tok_s: float
    ax_prefill_tok_s: float
    prefill_ratio: float
    mlx_ttft_ms: float
    ax_ttft_ms: float
    ttft_ratio: float
    ax_peak_memory_gb: float
    bend_note: str


def metric_value(row: dict[str, Any], metric: str, stat: str) -> float:
    payload = row.get(metric)
    if not isinstance(payload, dict) or not isinstance(payload.get(stat), (int, float)):
        raise PrefillScalingReportError(
            f"{row.get('engine')} context={row.get('context_tokens')} lacks {metric}.{stat}"
        )
    return float(payload[stat])


def ratio_value(row: dict[str, Any], key: str) -> float:
    ratios = row.get("ratios_to_mlx_lm")
    if not isinstance(ratios, dict) or not isinstance(ratios.get(key), (int, float)):
        raise PrefillScalingReportError(
            f"{row.get('engine')} context={row.get('context_tokens')} lacks ratio {key}"
        )
    return float(ratios[key])


def row_by_shape(artifact: dict[str, Any]) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    groups: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in artifact.get("rows", []):
        if not isinstance(row, dict):
            continue
        shape = (int(row["context_tokens"]), int(row["generation_tokens"]))
        groups.setdefault(shape, {})[str(row["engine"])] = row
    return groups


def build_report_rows(
    artifact: dict[str, Any],
    *,
    bend_drop_ratio: float,
) -> list[ReportRow]:
    report_rows: list[ReportRow] = []
    previous_ax_prefill: float | None = None
    for (context_tokens, generation_tokens), engines in sorted(row_by_shape(artifact).items()):
        mlx_row = engines.get("mlx_lm")
        ax_row = engines.get("ax_engine_mlx")
        if mlx_row is None or ax_row is None:
            raise PrefillScalingReportError(
                f"context={context_tokens} generation={generation_tokens} lacks required rows"
            )

        ax_prefill = metric_value(ax_row, "prefill_tok_s", "median")
        bend_note = ""
        if previous_ax_prefill is not None and ax_prefill / previous_ax_prefill <= bend_drop_ratio:
            bend_note = f"drop <= {bend_drop_ratio:.2f}x previous"
        previous_ax_prefill = ax_prefill

        report_rows.append(
            ReportRow(
                context_tokens=context_tokens,
                generation_tokens=generation_tokens,
                mlx_prefill_tok_s=metric_value(mlx_row, "prefill_tok_s", "median"),
                ax_prefill_tok_s=ax_prefill,
                prefill_ratio=ratio_value(ax_row, "prefill_tok_s"),
                mlx_ttft_ms=metric_value(mlx_row, "ttft_ms", "median"),
                ax_ttft_ms=metric_value(ax_row, "ttft_ms", "median"),
                ttft_ratio=ratio_value(ax_row, "ttft_ms"),
                ax_peak_memory_gb=metric_value(ax_row, "peak_memory_gb", "max"),
                bend_note=bend_note,
            )
        )
    return report_rows


def fmt_number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def render_report(
    artifact_path: Path,
    *,
    bend_drop_ratio: float = 0.75,
) -> str:
    validate_prefill_scaling_artifact(artifact_path)
    artifact = load_json(artifact_path)
    model = artifact.get("model", {})
    host = artifact.get("host", {})
    benchmark = artifact.get("benchmark", {})
    rows = build_report_rows(artifact, bend_drop_ratio=bend_drop_ratio)

    lines = [
        "# MLX Prefill Scaling Report",
        "",
        f"- Artifact: `{artifact_path}`",
        f"- Model: `{model.get('id', 'unknown')}`",
        f"- Host: {host.get('chip', 'unknown')} / {host.get('memory_gb', 'unknown')} GB",
        (
            f"- Benchmark: batch={benchmark.get('batch_size', 'unknown')}, "
            f"generation={rows[0].generation_tokens if rows else 'unknown'}, "
            f"repetitions={benchmark.get('repetitions', 'unknown')}, "
            f"prefill_step_size={benchmark.get('prefill_step_size', 'unknown')}"
        ),
        "",
        (
            "TTFT scope: AX rows use runner prefill timing when available; "
            "`mlx_lm` rows may be derived from reported prefill throughput."
        ),
        "",
        "| Context tok | Gen tok | mlx_lm prefill tok/s | AX prefill tok/s | AX/MLX prefill | mlx_lm TTFT ms | AX TTFT ms | AX/MLX TTFT | AX peak GB | Bend |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.context_tokens:,} | "
            f"{row.generation_tokens:,} | "
            f"{fmt_number(row.mlx_prefill_tok_s)} | "
            f"{fmt_number(row.ax_prefill_tok_s)} | "
            f"{row.prefill_ratio:.3f}x | "
            f"{fmt_number(row.mlx_ttft_ms)} | "
            f"{fmt_number(row.ax_ttft_ms)} | "
            f"{row.ttft_ratio:.3f}x | "
            f"{fmt_number(row.ax_peak_memory_gb)} | "
            f"{row.bend_note} |"
        )

    bend_rows = [row for row in rows if row.bend_note]
    lines.extend(["", "## Interpretation Guardrails", ""])
    if bend_rows:
        first = bend_rows[0]
        lines.append(
            f"- First AX prefill throughput bend: {first.context_tokens:,} context tokens."
        )
    else:
        lines.append("- No AX prefill throughput bend crossed the configured threshold.")
    lines.extend(
        [
            "- This report covers direct AX prefill/TTFT scaling, not n-gram decode acceleration.",
            "- Treat host, model artifact identity, prompt hashes, and temperature as part of the claim.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a validated ax.mlx_prefill_scaling.v1 artifact as Markdown."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bend-drop-ratio", type=float, default=0.75)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = render_report(args.artifact, bend_drop_ratio=args.bend_drop_ratio)
    except (PrefillScalingArtifactError, PrefillScalingReportError) as error:
        print(f"MLX prefill scaling report failed: {error}", file=sys.stderr)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"MLX prefill scaling report written: {args.output}")
    else:
        print(report)
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
