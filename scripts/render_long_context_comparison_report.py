#!/usr/bin/env python3
"""Render long-context comparison artifacts as Markdown reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from check_long_context_comparison_artifact import (
    LongContextComparisonArtifactError,
    load_json,
    validate_long_context_comparison_artifact,
)


class LongContextComparisonReportError(RuntimeError):
    pass


def metric_value(row: dict[str, Any], key: str, stat: str = "median") -> float:
    metric = row.get(key)
    if not isinstance(metric, dict) or not isinstance(metric.get(stat), (int, float)):
        raise LongContextComparisonReportError(
            f"{row.get('engine')} context={row.get('context_tokens')} lacks {key}.{stat}"
        )
    return float(metric[stat])


def ratio_value(row: dict[str, Any], key: str) -> float:
    ratios = row.get("ratios_to_mlx_lm")
    if not isinstance(ratios, dict) or not isinstance(ratios.get(key), (int, float)):
        raise LongContextComparisonReportError(
            f"{row.get('engine')} context={row.get('context_tokens')} lacks ratio {key}"
        )
    return float(ratios[key])


def grouped_rows(artifact: dict[str, Any]) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    groups: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in artifact.get("rows", []):
        if not isinstance(row, dict):
            continue
        shape = (int(row["context_tokens"]), int(row["generation_tokens"]))
        groups.setdefault(shape, {})[str(row["engine"])] = row
    return groups


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def fmt_ratio(row: dict[str, Any] | None, key: str) -> str:
    if row is None:
        return "n/a"
    return f"{ratio_value(row, key):.3f}x"


def fmt_metric(row: dict[str, Any] | None, key: str) -> str:
    if row is None:
        return "n/a"
    return fmt(metric_value(row, key))


def render_report(
    artifact_path: Path,
    *,
    require_llama_cpp: bool = False,
) -> str:
    validate_long_context_comparison_artifact(
        artifact_path,
        require_llama_cpp=require_llama_cpp,
    )
    artifact = load_json(artifact_path)
    model = artifact.get("model", {})
    host = artifact.get("host", {})
    benchmark = artifact.get("benchmark", {})

    lines = [
        "# Long-Context Comparison Report",
        "",
        f"- Artifact: `{artifact_path}`",
        f"- Model: `{model.get('id', 'unknown')}`",
        f"- Host: {host.get('chip', 'unknown')} / {host.get('memory_gb', 'unknown')} GB",
        (
            f"- Benchmark: batch={benchmark.get('batch_size', 'unknown')}, "
            f"repetitions={benchmark.get('repetitions', 'unknown')}, "
            f"prefill_step_size={benchmark.get('prefill_step_size', 'unknown')}"
        ),
        "",
        (
            "Scope: cold long-prefill comparison. AX and `mlx_lm` share the same "
            "prompt-token hash. `llama.cpp Metal` rows are external GGUF "
            "shape-compatible rows, not prompt-hash parity evidence."
        ),
        "",
        "| Context tok | Gen tok | mlx_lm prefill tok/s | AX prefill tok/s | AX/MLX prefill | llama.cpp prefill tok/s | llama/MLX prefill | AX TTFT ms | llama.cpp TTFT ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (context_tokens, generation_tokens), engines in sorted(grouped_rows(artifact).items()):
        mlx = engines["mlx_lm"]
        ax = engines["ax_engine_mlx"]
        llama = engines.get("llama_cpp_metal")
        lines.append(
            "| "
            f"{context_tokens:,} | "
            f"{generation_tokens:,} | "
            f"{fmt_metric(mlx, 'prefill_tok_s')} | "
            f"{fmt_metric(ax, 'prefill_tok_s')} | "
            f"{fmt_ratio(ax, 'prefill_tok_s')} | "
            f"{fmt_metric(llama, 'prefill_tok_s')} | "
            f"{fmt_ratio(llama, 'prefill_tok_s')} | "
            f"{fmt_metric(ax, 'ttft_ms')} | "
            f"{fmt_metric(llama, 'ttft_ms')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- This report compares cold prefill/derived TTFT, not prefix-cache reuse.",
            "- AX-vs-`mlx_lm` rows are prompt-hash parity rows.",
            "- `llama.cpp Metal` rows use `llama-bench` internal synthetic tokens and must stay in an external baseline column.",
            "- Decode-at-depth and server prefix-reuse need separate artifacts before claiming long-session serving superiority.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an ax.long_context_comparison.v1 artifact as Markdown."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-llama-cpp", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = render_report(args.artifact, require_llama_cpp=args.require_llama_cpp)
    except (LongContextComparisonArtifactError, LongContextComparisonReportError) as error:
        print(f"Long-context comparison report failed: {error}", file=sys.stderr)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"Long-context comparison report written: {args.output}")
    else:
        print(report)
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
