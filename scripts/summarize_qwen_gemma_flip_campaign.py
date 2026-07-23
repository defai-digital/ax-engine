#!/usr/bin/env python3
"""Aggregate a repeated Qwen/Gemma campaign and emit the explicit flip decision."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import compare_qwen_gemma_flip as comparator

CAMPAIGN_SCHEMA_VERSION = "ax.qwen_gemma_flip_campaign.v1"
SCHEMA_VERSION = "ax.qwen_gemma_flip_campaign_summary.v1"


class SummaryError(RuntimeError):
    """Raised when a campaign cannot support an aggregate decision."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SummaryError(f"{path}: expected a JSON object")
    return payload


def resolve_artifact_path(campaign_path: Path, recorded: str) -> Path:
    path = Path(recorded)
    if path.is_file():
        return path
    fallback = campaign_path.parent / "artifacts" / path.name
    if fallback.is_file():
        return fallback
    raise SummaryError(f"campaign artifact is missing: {recorded}")


def metric(artifact: dict[str, Any], name: str) -> float:
    summary = artifact.get("summary")
    if not isinstance(summary, dict):
        raise SummaryError("artifact summary is missing")
    if name == "throughput":
        value = summary.get("output_token_throughput_tok_s")
    elif name == "ttft_p95_ms":
        distribution = summary.get("ttft_ms")
        value = distribution.get("p95") if isinstance(distribution, dict) else None
    elif name == "stream_gap_p95_ms":
        value = comparator.interactive_gap(artifact)
    else:
        raise SummaryError(f"unknown metric {name!r}")
    if not isinstance(value, (int, float)):
        raise SummaryError(f"artifact metric {name} is missing")
    return float(value)


def median_metric(artifacts: list[dict[str, Any]], name: str) -> float:
    if len(artifacts) < 3:
        raise SummaryError(f"{name} requires at least three artifacts")
    return float(statistics.median(metric(artifact, name) for artifact in artifacts))


def safe_ratio(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        raise SummaryError("baseline metric must be positive")
    return candidate / baseline


def add_gate(
    gates: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    gates.append({"name": name, "passed": passed, "required": True, "detail": detail})


def aggregate_scenario(
    *,
    scenario_id: str,
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    min_throughput_ratio: float,
    max_ttft_p95_ratio: float,
    max_stream_gap_p95_ratio: float,
    max_stream_gap_p95_ms: float | None,
) -> dict[str, Any]:
    if len(candidate) != len(baseline) or len(candidate) < 3:
        raise SummaryError(
            f"{scenario_id}: candidate/baseline need the same count with at least 3 runs"
        )
    contracts = [
        comparator.evaluate_comparison_contract(cand, base)
        for cand, base in zip(candidate, baseline, strict=True)
    ]
    cand_throughput = median_metric(candidate, "throughput")
    base_throughput = median_metric(baseline, "throughput")
    cand_ttft = median_metric(candidate, "ttft_p95_ms")
    base_ttft = median_metric(baseline, "ttft_p95_ms")
    cand_gap = median_metric(candidate, "stream_gap_p95_ms")
    base_gap = median_metric(baseline, "stream_gap_p95_ms")
    throughput_ratio = safe_ratio(cand_throughput, base_throughput)
    ttft_ratio = safe_ratio(cand_ttft, base_ttft)
    gap_ratio = safe_ratio(cand_gap, base_gap)
    candidate_error_rates = [
        float(artifact.get("availability", {}).get("request_error_rate", 1.0))
        for artifact in candidate
    ]
    candidate_http_503 = [
        int(artifact.get("availability", {}).get("request_http_503", 1)) for artifact in candidate
    ]
    candidate_lifecycle_errors = [
        int(artifact.get("lifecycle", {}).get("error_events", 1)) for artifact in candidate
    ]

    gates: list[dict[str, Any]] = []
    add_gate(
        gates,
        "comparison_contract",
        all(contract["passed"] for contract in contracts),
        (
            "all paired trials matched"
            if all(contract["passed"] for contract in contracts)
            else "; ".join(
                mismatch for contract in contracts for mismatch in contract.get("mismatches", [])
            )
        ),
    )
    add_gate(
        gates,
        "median_throughput_ratio",
        throughput_ratio >= min_throughput_ratio,
        f"{throughput_ratio:.4f} >= {min_throughput_ratio:.4f}",
    )
    add_gate(
        gates,
        "median_ttft_p95_ratio",
        ttft_ratio <= max_ttft_p95_ratio,
        f"{ttft_ratio:.4f} <= {max_ttft_p95_ratio:.4f}",
    )
    add_gate(
        gates,
        "median_stream_gap_p95_ratio",
        gap_ratio <= max_stream_gap_p95_ratio,
        f"{gap_ratio:.4f} <= {max_stream_gap_p95_ratio:.4f}",
    )
    if max_stream_gap_p95_ms is not None:
        add_gate(
            gates,
            "absolute_stream_gap_p95",
            cand_gap <= max_stream_gap_p95_ms,
            f"{cand_gap:.3f}ms <= {max_stream_gap_p95_ms:.3f}ms",
        )
    add_gate(
        gates,
        "candidate_availability",
        max(candidate_error_rates) == 0.0
        and max(candidate_http_503) == 0
        and max(candidate_lifecycle_errors) == 0,
        (
            f"max_error_rate={max(candidate_error_rates):.4f}, "
            f"max_http_503={max(candidate_http_503)}, "
            f"max_lifecycle_errors={max(candidate_lifecycle_errors)}"
        ),
    )
    failed = [gate["name"] for gate in gates if not gate["passed"]]
    return {
        "scenario": scenario_id,
        "repetitions": len(candidate),
        "candidate": {
            "throughput_median_tok_s": cand_throughput,
            "ttft_p95_median_ms": cand_ttft,
            "stream_gap_p95_median_ms": cand_gap,
        },
        "baseline": {
            "throughput_median_tok_s": base_throughput,
            "ttft_p95_median_ms": base_ttft,
            "stream_gap_p95_median_ms": base_gap,
        },
        "ratios": {
            "throughput": throughput_ratio,
            "ttft_p95": ttft_ratio,
            "stream_gap_p95": gap_ratio,
        },
        "gates": gates,
        "failed_required_gates": failed,
        "passed": not failed,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen 3 + Gemma 4 vs mlxcel flip decision",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        (
            f"Candidate `{summary['candidate_label']}` vs baseline "
            f"`{summary['baseline_label']}`; medians over "
            f"{summary['repetitions']} cache-isolated repetitions."
        ),
        "",
        "| Scenario | AX tok/s | mlxcel tok/s | Throughput ratio | TTFT ratio | "
        "Stream-gap ratio | Result |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["scenarios"]:
        lines.append(
            "| {scenario} | {cand:.2f} | {base:.2f} | {throughput:.3f}x | "
            "{ttft:.3f}x | {gap:.3f}x | {result} |".format(
                scenario=row["scenario"].upper(),
                cand=row["candidate"]["throughput_median_tok_s"],
                base=row["baseline"]["throughput_median_tok_s"],
                throughput=row["ratios"]["throughput"],
                ttft=row["ratios"]["ttft_p95"],
                gap=row["ratios"]["stream_gap_p95"],
                result="PASS" if row["passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "Locked gates:",
            "",
            f"- throughput ratio ≥ {summary['thresholds']['min_throughput_ratio']:.2f}",
            f"- p95 TTFT ratio ≤ {summary['thresholds']['max_ttft_p95_ratio']:.2f}",
            (
                "- interactive p95 stream-gap ratio ≤ "
                f"{summary['thresholds']['max_stream_gap_p95_ratio']:.2f}"
            ),
            (
                "- absolute interactive p95 stream gap ≤ "
                f"{summary['thresholds']['max_stream_gap_p95_ms']:.2f} ms"
                if summary["thresholds"]["max_stream_gap_p95_ms"] is not None
                else "- no absolute interactive p95 stream-gap cap supplied"
            ),
            "- zero candidate request errors, HTTP 503s, and lifecycle errors",
            "",
        ]
    )
    failed = [
        f"{row['scenario'].upper()}: {', '.join(row['failed_required_gates'])}"
        for row in summary["scenarios"]
        if not row["passed"]
    ]
    if failed:
        lines.extend(["Failed gates:", "", *[f"- {item}" for item in failed], ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--candidate-label", default="ax-engine")
    parser.add_argument("--baseline-label", default="mlxcel")
    parser.add_argument("--min-throughput-ratio", type=float, default=1.15)
    parser.add_argument("--max-ttft-p95-ratio", type=float, default=0.90)
    parser.add_argument("--max-stream-gap-p95-ratio", type=float, default=0.90)
    parser.add_argument("--max-stream-gap-p95-ms", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--report-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        campaign_path = args.campaign.resolve()
        campaign = load_json(campaign_path)
        if campaign.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
            raise SummaryError(f"campaign schema must be {CAMPAIGN_SCHEMA_VERSION}")
        if campaign.get("passed") is not True:
            raise SummaryError("campaign did not pass its per-trial and contract checks")
        runs = campaign.get("runs")
        if not isinstance(runs, list):
            raise SummaryError("campaign runs are missing")
        scenario_ids = [str(item["id"]) for item in campaign.get("scenarios") or []]
        repetitions = int(campaign.get("methodology", {}).get("repetitions") or 0)
        if repetitions < 3:
            raise SummaryError("campaign needs at least three repetitions")

        artifacts: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
        for run in runs:
            if not isinstance(run, dict) or run.get("passed") is not True:
                raise SummaryError("campaign contains a failed run")
            label = str(run.get("target"))
            scenario_id = str(run.get("scenario"))
            repetition = int(run.get("repetition"))
            path = resolve_artifact_path(campaign_path, str(run.get("artifact")))
            artifacts.setdefault((label, scenario_id), []).append((repetition, load_json(path)))

        rows: list[dict[str, Any]] = []
        for scenario_id in scenario_ids:
            candidate_rows = sorted(artifacts.get((args.candidate_label, scenario_id), []))
            baseline_rows = sorted(artifacts.get((args.baseline_label, scenario_id), []))
            if [item[0] for item in candidate_rows] != [item[0] for item in baseline_rows]:
                raise SummaryError(f"{scenario_id}: candidate/baseline repetitions differ")
            rows.append(
                aggregate_scenario(
                    scenario_id=scenario_id,
                    candidate=[item[1] for item in candidate_rows],
                    baseline=[item[1] for item in baseline_rows],
                    min_throughput_ratio=args.min_throughput_ratio,
                    max_ttft_p95_ratio=args.max_ttft_p95_ratio,
                    max_stream_gap_p95_ratio=args.max_stream_gap_p95_ratio,
                    max_stream_gap_p95_ms=args.max_stream_gap_p95_ms,
                )
            )
    except (SummaryError, ValueError, KeyError) as error:
        print(f"flip campaign summary failed: {error}", file=sys.stderr)
        return 2

    passed = all(row["passed"] for row in rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": "flip" if passed else "not_yet",
        "passed": passed,
        "campaign": str(campaign_path),
        "candidate_label": args.candidate_label,
        "baseline_label": args.baseline_label,
        "repetitions": repetitions,
        "thresholds": {
            "min_throughput_ratio": args.min_throughput_ratio,
            "max_ttft_p95_ratio": args.max_ttft_p95_ratio,
            "max_stream_gap_p95_ratio": args.max_stream_gap_p95_ratio,
            "max_stream_gap_p95_ms": args.max_stream_gap_p95_ms,
        },
        "scenarios": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(summary) + "\n")
    print(f"flip decision {summary['decision']}: {args.output}", file=sys.stderr)
    return 0 if passed or args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
