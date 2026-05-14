#!/usr/bin/env python3
"""Render AX serving benchmark artifacts as Markdown reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from check_ax_serving_benchmark_artifact import (
    ArtifactCheckError,
    load_json,
    validate_serving_benchmark_artifact,
)


class ServingBenchmarkReportError(RuntimeError):
    pass


def fmt(value: Any, digits: int = 1) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):,.{digits}f}"
    return "n/a"


def fmt_dist(dist: Any, key: str, digits: int = 1) -> str:
    if not isinstance(dist, dict):
        return "n/a"
    return fmt(dist.get(key), digits=digits)


def fmt_slo(value: Any) -> str:
    if value is None:
        return "not set"
    return fmt(value)


def require_summary(payload: dict[str, Any], owner: str) -> dict[str, Any]:
    summary = payload.get(owner)
    if not isinstance(summary, dict):
        raise ServingBenchmarkReportError(f"artifact lacks {owner}")
    return summary


def render_distribution_table(summary: dict[str, Any]) -> list[str]:
    rows = [
        ("TTFT ms", summary.get("ttft_ms")),
        ("Client TPOT ms", summary.get("client_tpot_ms")),
        ("Step interval ms", summary.get("stream_step_interval_ms")),
        ("E2E latency ms", summary.get("e2e_latency_ms")),
        ("Queue delay ms", summary.get("queue_delay_ms")),
        ("Input tokens", summary.get("input_tokens")),
        ("Output tokens", summary.get("output_tokens")),
    ]
    lines = [
        "| Metric | Count | p50 | p75 | p95 | p99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, dist in rows:
        lines.append(
            "| "
            f"{label} | "
            f"{fmt_dist(dist, 'count', digits=0)} | "
            f"{fmt_dist(dist, 'p50')} | "
            f"{fmt_dist(dist, 'p75')} | "
            f"{fmt_dist(dist, 'p95')} | "
            f"{fmt_dist(dist, 'p99')} | "
            f"{fmt_dist(dist, 'max')} |"
        )
    return lines


def render_category_table(by_category: dict[str, Any]) -> list[str]:
    lines = [
        "| Category | Requests | Errors | TTFT p50 ms | TTFT p95 ms | TPOT p95 ms | E2E p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, raw_summary in sorted(by_category.items()):
        if not isinstance(raw_summary, dict):
            continue
        lines.append(
            "| "
            f"{category} | "
            f"{fmt(raw_summary.get('requests'), digits=0)} | "
            f"{fmt(raw_summary.get('error_requests'), digits=0)} | "
            f"{fmt_dist(raw_summary.get('ttft_ms'), 'p50')} | "
            f"{fmt_dist(raw_summary.get('ttft_ms'), 'p95')} | "
            f"{fmt_dist(raw_summary.get('client_tpot_ms'), 'p95')} | "
            f"{fmt_dist(raw_summary.get('e2e_latency_ms'), 'p95')} |"
        )
    return lines


def render_report(
    artifact_path: Path,
    *,
    min_requests: int = 1,
    min_concurrency: int = 1,
    require_zero_errors: bool = True,
    require_slo: bool = False,
    min_goodput_ratio: float | None = None,
    min_input_tokens_p95: int | None = None,
) -> str:
    validate_serving_benchmark_artifact(
        artifact_path,
        min_requests=min_requests,
        min_concurrency=min_concurrency,
        require_zero_errors=require_zero_errors,
        require_slo=require_slo,
        min_goodput_ratio=min_goodput_ratio,
        min_input_tokens_p95=min_input_tokens_p95,
    )
    artifact = load_json(artifact_path)
    target = require_summary(artifact, "target")
    load = require_summary(artifact, "load")
    corpus = require_summary(artifact, "corpus")
    summary = require_summary(artifact, "summary")
    goodput = require_summary(summary, "goodput")
    by_category = require_summary(artifact, "by_category")

    lines = [
        "# AX Serving Benchmark Report",
        "",
        f"- Artifact: `{artifact_path}`",
        f"- Target: `{target.get('model_id', 'unknown')}` at `{target.get('base_url', 'unknown')}`",
        (
            f"- Load: concurrency={load.get('concurrency', 'unknown')}, "
            f"request_rate_rps={load.get('request_rate_rps')}, "
            f"measured_requests={load.get('measured_requests', 'unknown')}, "
            f"warmup_requests={load.get('warmup_requests', 'unknown')}"
        ),
        (
            f"- Corpus: `{corpus.get('path', 'unknown')}` "
            f"({corpus.get('prompt_count', 'unknown')} prompts, sha256={corpus.get('sha256', 'unknown')})"
        ),
        (
            f"- Goodput: {fmt(goodput.get('ratio'), digits=3)} "
            f"({fmt(goodput.get('requests'), digits=0)} requests); "
            f"SLO TTFT={fmt_slo(goodput.get('ttft_slo_ms'))} ms, "
            f"TPOT={fmt_slo(goodput.get('client_tpot_slo_ms'))} ms, "
            f"E2E={fmt_slo(goodput.get('e2e_slo_ms'))} ms"
        ),
        (
            f"- Throughput: {fmt(summary.get('request_throughput_rps'), digits=2)} req/s, "
            f"{fmt(summary.get('output_token_throughput_tok_s'), digits=2)} output tok/s"
        ),
        "",
        "Scope: online serving latency over `/v1/generate/stream`. This report does not replace fixed-shape prefill/decode microbenchmarks.",
        "",
        "## Latency Summary",
        "",
    ]
    lines.extend(render_distribution_table(summary))
    lines.extend(["", "## Category Breakdown", ""])
    lines.extend(render_category_table(by_category))
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- TTFT, TPOT, queue delay, and E2E latency are client-observed SSE timings.",
            "- Use `--request-rate-rps` artifacts for open-loop serving claims.",
            "- For long-prompt serving claims, validate with `--min-input-tokens-p95` before citing the report.",
            "- Compare this report with vLLM or llama-server only when corpus, load policy, output length, and SLOs are stated together.",
            "",
        ]
    )
    return "\n".join(lines)


def ratio_arg(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an ax.serving_benchmark.v1 artifact as Markdown."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-requests", type=positive_int_arg, default=1)
    parser.add_argument("--min-concurrency", type=positive_int_arg, default=1)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--require-slo", action="store_true")
    parser.add_argument("--min-goodput-ratio", type=ratio_arg)
    parser.add_argument("--min-input-tokens-p95", type=positive_int_arg)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = render_report(
            args.artifact,
            min_requests=args.min_requests,
            min_concurrency=args.min_concurrency,
            require_zero_errors=not args.allow_errors,
            require_slo=args.require_slo,
            min_goodput_ratio=args.min_goodput_ratio,
            min_input_tokens_p95=args.min_input_tokens_p95,
        )
    except (ArtifactCheckError, ServingBenchmarkReportError) as error:
        print(f"AX serving benchmark report failed: {error}", file=sys.stderr)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"AX serving benchmark report written: {args.output}")
    else:
        print(report)
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
