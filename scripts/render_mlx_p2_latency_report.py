#!/usr/bin/env python3
"""Render validated MLX P2 latency artifacts as a Markdown report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from check_mlx_concurrent_prefill_artifact import (
    ConcurrentPrefillArtifactError,
    load_json as load_concurrent_json,
    validate_mlx_concurrent_prefill_artifact,
)
from check_mlx_startup_latency_artifact import (
    StartupLatencyArtifactError,
    load_json as load_startup_json,
    validate_mlx_startup_latency_artifact,
)


class P2LatencyReportError(RuntimeError):
    pass


def metric_value(row: dict[str, Any], metric: str, stat: str) -> float:
    payload = row.get(metric)
    if not isinstance(payload, dict) or not isinstance(payload.get(stat), (int, float)):
        raise P2LatencyReportError(
            f"row lacks numeric {metric}.{stat}: {row.get('phase') or row.get('concurrent_requests')}"
        )
    return float(payload[stat])


def ratio_value(row: dict[str, Any], field: str, ratio_key: str) -> float | None:
    ratios = row.get(field)
    if ratios is None:
        return None
    if not isinstance(ratios, dict) or not isinstance(ratios.get(ratio_key), (int, float)):
        raise P2LatencyReportError(
            f"row lacks numeric {field}.{ratio_key}: "
            f"{row.get('phase') or row.get('concurrent_requests')}"
        )
    return float(ratios[ratio_key])


def fmt_number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}x"


def render_startup_section(path: Path) -> list[str]:
    validate_mlx_startup_latency_artifact(path)
    artifact = load_startup_json(path)
    model = artifact.get("model", {})
    host = artifact.get("host", {})
    benchmark = artifact.get("benchmark", {})
    rows = {
        str(row.get("phase")): row
        for row in artifact.get("rows", [])
        if isinstance(row, dict)
    }

    lines = [
        "## Startup Latency",
        "",
        f"- Artifact: `{path}`",
        f"- Model: `{model.get('id', 'unknown')}`",
        f"- Host: {host.get('chip', 'unknown')}",
        (
            f"- Shape: context={benchmark.get('context_tokens', 'unknown')}, "
            f"generation={benchmark.get('generation_tokens', 'unknown')}, "
            f"repetitions={benchmark.get('repetitions', 'unknown')}"
        ),
        "",
        "| Phase | Server ready ms | Model load ms | TTFT ms | TTFT vs warm | Decode tok/s | Decode vs warm | Peak GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in ("process_cold", "model_warm", "benchmark_warm"):
        row = rows[phase]
        server_ready = (
            fmt_number(metric_value(row, "server_ready_ms", "median"))
            if "server_ready_ms" in row
            else "-"
        )
        model_load = (
            fmt_number(metric_value(row, "model_load_ms", "median"))
            if "model_load_ms" in row
            else "-"
        )
        lines.append(
            "| "
            f"{phase} | "
            f"{server_ready} | "
            f"{model_load} | "
            f"{fmt_number(metric_value(row, 'ttft_ms', 'median'))} | "
            f"{fmt_ratio(ratio_value(row, 'ratios_to_benchmark_warm', 'ttft_ms'))} | "
            f"{fmt_number(metric_value(row, 'decode_tok_s', 'median'))} | "
            f"{fmt_ratio(ratio_value(row, 'ratios_to_benchmark_warm', 'decode_tok_s'))} | "
            f"{fmt_number(metric_value(row, 'peak_memory_gb', 'max'))} |"
        )
    lines.extend(
        [
            "",
            "Startup guardrail: cold rows include process/server/model-load costs; the benchmark-warm row must not mix those costs into warm throughput.",
            "",
        ]
    )
    return lines


def render_concurrent_section(path: Path) -> list[str]:
    validate_mlx_concurrent_prefill_artifact(path)
    artifact = load_concurrent_json(path)
    model = artifact.get("model", {})
    host = artifact.get("host", {})
    benchmark = artifact.get("benchmark", {})
    rows = sorted(
        [row for row in artifact.get("rows", []) if isinstance(row, dict)],
        key=lambda row: int(row.get("concurrent_requests", 0)),
    )

    lines = [
        "## Concurrent Prefill",
        "",
        f"- Artifact: `{path}`",
        f"- Model: `{model.get('id', 'unknown')}`",
        f"- Host: {host.get('chip', 'unknown')}",
        (
            f"- Shape: context={benchmark.get('context_tokens', 'unknown')}, "
            f"generation={benchmark.get('generation_tokens', 'unknown')}, "
            f"repetitions={benchmark.get('repetitions', 'unknown')}"
        ),
        "",
        "| Requests | Request TTFT ms | TTFT vs single | Total wall ms | Wall vs single | Queue delay ms | Failures max | Peak GB | Memory vs single | Overlap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        overlap = row.get("prefill_overlap", {})
        classification = overlap.get("classification", "unknown") if isinstance(overlap, dict) else "unknown"
        lines.append(
            "| "
            f"{int(row['concurrent_requests'])} | "
            f"{fmt_number(metric_value(row, 'request_ttft_ms', 'median'))} | "
            f"{fmt_ratio(ratio_value(row, 'ratios_to_single_request', 'request_ttft_ms'))} | "
            f"{fmt_number(metric_value(row, 'total_wall_ms', 'median'))} | "
            f"{fmt_ratio(ratio_value(row, 'ratios_to_single_request', 'total_wall_ms'))} | "
            f"{fmt_number(metric_value(row, 'queue_delay_ms', 'median'))} | "
            f"{fmt_number(metric_value(row, 'failure_count', 'max'), digits=0)} | "
            f"{fmt_number(metric_value(row, 'peak_memory_gb', 'max'))} | "
            f"{fmt_ratio(ratio_value(row, 'ratios_to_single_request', 'peak_memory_gb'))} | "
            f"{classification} |"
        )
    lines.extend(
        [
            "",
            "Concurrency guardrail: this is server-path long-prompt evidence, not proof of continuous batching or production multi-user throughput.",
            "",
        ]
    )
    return lines


def render_report(
    *,
    startup_artifact: Path | None,
    concurrent_artifact: Path | None,
) -> str:
    if startup_artifact is None and concurrent_artifact is None:
        raise P2LatencyReportError("at least one artifact path is required")

    lines = [
        "# MLX P2 Latency Report",
        "",
        "This report is rendered only from validated P2 artifacts.",
        "",
    ]
    if startup_artifact is not None:
        lines.extend(render_startup_section(startup_artifact))
    if concurrent_artifact is not None:
        lines.extend(render_concurrent_section(concurrent_artifact))
    lines.extend(
        [
            "## Interpretation Guardrails",
            "",
            "- Keep P2 startup/concurrency evidence separate from README batch=1 throughput rows.",
            "- These rows use direct AX MLX policy; they do not measure n-gram decode acceleration.",
            "- Treat host, model artifact identity, prompt hashes, and generation shape as part of the claim.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-artifact", type=Path)
    parser.add_argument("--concurrent-artifact", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = render_report(
            startup_artifact=args.startup_artifact,
            concurrent_artifact=args.concurrent_artifact,
        )
    except (
        ConcurrentPrefillArtifactError,
        StartupLatencyArtifactError,
        P2LatencyReportError,
    ) as error:
        print(f"MLX P2 latency report failed: {error}", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"MLX P2 latency report written: {args.output}")
    else:
        print(report)
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
