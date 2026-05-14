#!/usr/bin/env python3
"""Render decode-at-depth artifacts as Markdown reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from check_long_context_decode_at_depth_artifact import (
    LongContextDecodeAtDepthArtifactError,
    load_json,
    validate_long_context_decode_at_depth_artifact,
)


class LongContextDecodeAtDepthReportError(RuntimeError):
    pass


def metric_value(row: dict[str, Any], key: str, stat: str = "median") -> float:
    metric = row.get(key)
    if not isinstance(metric, dict) or not isinstance(metric.get(stat), (int, float)):
        raise LongContextDecodeAtDepthReportError(
            f"{row.get('engine')} depth={row.get('context_depth_tokens')} lacks {key}.{stat}"
        )
    return float(metric[stat])


def ratio_value(row: dict[str, Any], key: str) -> float:
    ratios = row.get("ratios_to_mlx_lm")
    if not isinstance(ratios, dict) or not isinstance(ratios.get(key), (int, float)):
        raise LongContextDecodeAtDepthReportError(
            f"{row.get('engine')} depth={row.get('context_depth_tokens')} lacks ratio {key}"
        )
    return float(ratios[key])


def grouped_rows(artifact: dict[str, Any]) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    groups: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in artifact.get("rows", []):
        if not isinstance(row, dict):
            continue
        shape = (int(row["context_depth_tokens"]), int(row["generation_tokens"]))
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
    validate_long_context_decode_at_depth_artifact(
        artifact_path,
        require_llama_cpp=require_llama_cpp,
    )
    artifact = load_json(artifact_path)
    model = artifact.get("model", {})
    host = artifact.get("host", {})
    benchmark = artifact.get("benchmark", {})

    lines = [
        "# Long-Context Decode-at-Depth Report",
        "",
        f"- Artifact: `{artifact_path}`",
        f"- Model: `{model.get('id', 'unknown')}`",
        f"- Host: {host.get('chip', 'unknown')} / {host.get('memory_gb', 'unknown')} GB",
        (
            f"- Benchmark: batch={benchmark.get('batch_size', 'unknown')}, "
            f"repetitions={benchmark.get('repetitions', 'unknown')}"
        ),
        "",
        (
            "Scope: decode throughput after a context depth already exists. "
            "AX and `mlx_lm` rows share the prompt-token hash. `llama.cpp Metal` "
            "rows must be explicit `llama-bench n_depth` evidence and remain "
            "shape-compatible external rows."
        ),
        "",
        "| Context depth tok | Gen tok | mlx_lm decode tok/s | AX decode tok/s | AX/MLX decode | llama.cpp decode tok/s | llama/MLX decode |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (context_depth_tokens, generation_tokens), engines in sorted(
        grouped_rows(artifact).items()
    ):
        mlx = engines["mlx_lm"]
        ax = engines["ax_engine_mlx"]
        llama = engines.get("llama_cpp_metal")
        lines.append(
            "| "
            f"{context_depth_tokens:,} | "
            f"{generation_tokens:,} | "
            f"{fmt_metric(mlx, 'decode_tok_s')} | "
            f"{fmt_metric(ax, 'decode_tok_s')} | "
            f"{fmt_ratio(ax, 'decode_tok_s')} | "
            f"{fmt_metric(llama, 'decode_tok_s')} | "
            f"{fmt_ratio(llama, 'decode_tok_s')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- This report compares decode throughput after context depth, not cold prefill.",
            "- AX-vs-`mlx_lm` rows are prompt-hash parity rows.",
            "- `llama.cpp Metal` rows are included only when they carry explicit `llama-bench n_depth` evidence.",
            "- Serving TTFT, queue delay, and prefix-cache reuse require separate artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an ax.long_context_decode_at_depth.v1 artifact as Markdown."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-llama-cpp", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = render_report(args.artifact, require_llama_cpp=args.require_llama_cpp)
    except (LongContextDecodeAtDepthArtifactError, LongContextDecodeAtDepthReportError) as error:
        print(f"Long-context decode-at-depth report failed: {error}", file=sys.stderr)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"Long-context decode-at-depth report written: {args.output}")
    else:
        print(report)
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
