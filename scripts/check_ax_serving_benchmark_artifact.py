#!/usr/bin/env python3
"""Validate AX online serving benchmark artifacts.

This checker is intentionally about online serving behavior: TTFT, client TPOT,
end-to-end latency, scheduling policy, failures, and corpus shape. It should not
be used as a replacement for fixed-shape model-runtime benchmarks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ax.serving_benchmark.v1"
REQUIRED_METHODOLOGY = {
    "scope": "online_serving_streaming_latency",
    "endpoint": "/v1/generate/stream",
    "timing_scope": "client_observed_sse",
}
SUMMARY_DISTRIBUTIONS = [
    "ttft_ms",
    "client_tpot_ms",
    "stream_step_interval_ms",
    "e2e_latency_ms",
    "queue_delay_ms",
    "input_tokens",
    "output_tokens",
]
PERCENTILE_KEYS = ["count", "min", "mean", "p50", "p75", "p90", "p95", "p99", "max"]


class ArtifactCheckError(RuntimeError):
    """Raised when a serving benchmark artifact is not claim-ready."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ArtifactCheckError(f"artifact does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ArtifactCheckError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ArtifactCheckError(f"{path}: artifact root must be an object")
    return payload


def require_object(value: Any, owner: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactCheckError(f"{owner} must be an object")
    return value


def require_list(value: Any, owner: str) -> list[Any]:
    if not isinstance(value, list):
        raise ArtifactCheckError(f"{owner} must be a list")
    return value


def require_number(value: Any, owner: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ArtifactCheckError(f"{owner} must be a finite number")
    parsed = float(value)
    if positive and parsed <= 0.0:
        raise ArtifactCheckError(f"{owner} must be positive")
    if not positive and parsed < 0.0:
        raise ArtifactCheckError(f"{owner} must be non-negative")
    return parsed


def require_int(value: Any, owner: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactCheckError(f"{owner} must be an integer")
    if positive and value <= 0:
        raise ArtifactCheckError(f"{owner} must be positive")
    if not positive and value < 0:
        raise ArtifactCheckError(f"{owner} must be non-negative")
    return value


def require_string(value: Any, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactCheckError(f"{owner} must be a non-empty string")
    return value


def validate_distribution(value: Any, owner: str, *, allow_null: bool = False) -> dict[str, Any] | None:
    if value is None and allow_null:
        return None
    dist = require_object(value, owner)
    for key in PERCENTILE_KEYS:
        require_number(dist.get(key), f"{owner}.{key}")
    count = require_int(dist.get("count"), f"{owner}.count", positive=True)
    if dist["min"] > dist["p50"] or dist["p50"] > dist["p95"] or dist["p95"] > dist["max"]:
        raise ArtifactCheckError(f"{owner} percentiles are not monotonic")
    if count < 1:
        raise ArtifactCheckError(f"{owner}.count must be at least 1")
    return dist


def validate_observation(row: Any, index: int) -> dict[str, Any]:
    obs = require_object(row, f"observations[{index}]")
    phase = obs.get("phase")
    if phase not in {"warmup", "measured"}:
        raise ArtifactCheckError(f"observations[{index}].phase must be warmup or measured")
    require_string(obs.get("prompt_id"), f"observations[{index}].prompt_id")
    require_string(obs.get("category"), f"observations[{index}].category")
    if not isinstance(obs.get("ok"), bool):
        raise ArtifactCheckError(f"observations[{index}].ok must be a boolean")
    require_number(obs.get("scheduled_at_s"), f"observations[{index}].scheduled_at_s")
    require_number(obs.get("started_at_s"), f"observations[{index}].started_at_s")
    require_number(obs.get("queue_delay_ms"), f"observations[{index}].queue_delay_ms")
    require_number(obs.get("e2e_latency_ms"), f"observations[{index}].e2e_latency_ms")
    intervals = require_list(
        obs.get("stream_step_interval_ms"), f"observations[{index}].stream_step_interval_ms"
    )
    for interval_index, interval in enumerate(intervals):
        require_number(
            interval,
            f"observations[{index}].stream_step_interval_ms[{interval_index}]",
        )
    if obs["ok"]:
        require_number(obs.get("ttft_ms"), f"observations[{index}].ttft_ms")
        output_tokens = require_int(
            obs.get("output_tokens"), f"observations[{index}].output_tokens", positive=True
        )
        if output_tokens > 1:
            require_number(obs.get("client_tpot_ms"), f"observations[{index}].client_tpot_ms")
    route_decisions = obs.get("route_decisions")
    if route_decisions is not None:
        validate_route_decisions(route_decisions, f"observations[{index}].route_decisions")
    return obs


def validate_route_decisions(value: Any, owner: str) -> dict[str, Any]:
    decisions = require_object(value, owner)
    for key, route_value in decisions.items():
        if not isinstance(key, str) or not key:
            raise ArtifactCheckError(f"{owner} keys must be non-empty strings")
        if not isinstance(route_value, (int, float, str, bool)) or (
            isinstance(route_value, float) and not math.isfinite(route_value)
        ):
            raise ArtifactCheckError(
                f"{owner}.{key} must be a finite number, string, or boolean"
            )
    return decisions


def validate_counts(summary: dict[str, Any], observations: list[dict[str, Any]]) -> None:
    measured = [row for row in observations if row.get("phase") == "measured"]
    ok = [row for row in measured if row.get("ok")]
    requests = require_int(summary.get("requests"), "summary.requests")
    ok_requests = require_int(summary.get("ok_requests"), "summary.ok_requests")
    error_requests = require_int(summary.get("error_requests"), "summary.error_requests")
    if requests != len(measured):
        raise ArtifactCheckError(
            f"summary.requests={requests} does not match measured observations={len(measured)}"
        )
    if ok_requests != len(ok):
        raise ArtifactCheckError(
            f"summary.ok_requests={ok_requests} does not match ok observations={len(ok)}"
        )
    if error_requests != requests - ok_requests:
        raise ArtifactCheckError("summary.error_requests is inconsistent with request counts")


def validate_goodput(summary: dict[str, Any], *, require_slo: bool, min_goodput_ratio: float | None) -> None:
    goodput = require_object(summary.get("goodput"), "summary.goodput")
    require_int(goodput.get("requests"), "summary.goodput.requests")
    ratio = require_number(goodput.get("ratio"), "summary.goodput.ratio")
    if ratio > 1.0:
        raise ArtifactCheckError("summary.goodput.ratio must be <= 1")
    slo_values = [
        goodput.get("ttft_slo_ms"),
        goodput.get("client_tpot_slo_ms"),
        goodput.get("e2e_slo_ms"),
    ]
    if require_slo and not any(value is not None for value in slo_values):
        raise ArtifactCheckError("--require-slo needs at least one configured goodput SLO")
    for name, value in zip(
        ["ttft_slo_ms", "client_tpot_slo_ms", "e2e_slo_ms"],
        slo_values,
        strict=True,
    ):
        if value is not None:
            require_number(value, f"summary.goodput.{name}", positive=True)
    if min_goodput_ratio is not None and ratio < min_goodput_ratio:
        raise ArtifactCheckError(
            f"summary.goodput.ratio={ratio:.6f} is below required {min_goodput_ratio:.6f}"
        )


def validate_serving_benchmark_artifact(
    path: Path,
    *,
    min_requests: int,
    min_concurrency: int,
    require_zero_errors: bool,
    require_slo: bool,
    min_goodput_ratio: float | None,
    min_input_tokens_p95: int | None,
    required_route_decision_mins: dict[str, float] | None = None,
) -> dict[str, Any]:
    artifact = load_json(path)
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactCheckError(f"{path}: schema_version must be {SCHEMA_VERSION}")

    methodology = require_object(artifact.get("methodology"), "methodology")
    for key, expected in REQUIRED_METHODOLOGY.items():
        if methodology.get(key) != expected:
            raise ArtifactCheckError(f"methodology.{key} must be {expected!r}")

    target = require_object(artifact.get("target"), "target")
    require_string(target.get("base_url"), "target.base_url")
    require_string(target.get("model_id"), "target.model_id")
    if target.get("input_kind") not in {"auto", "text", "tokens"}:
        raise ArtifactCheckError("target.input_kind must be auto, text, or tokens")

    load = require_object(artifact.get("load"), "load")
    concurrency = require_int(load.get("concurrency"), "load.concurrency", positive=True)
    measured_requests = require_int(
        load.get("measured_requests"), "load.measured_requests", positive=True
    )
    require_int(load.get("warmup_requests"), "load.warmup_requests")
    request_rate = load.get("request_rate_rps")
    if request_rate is not None:
        require_number(request_rate, "load.request_rate_rps", positive=True)
    if concurrency < min_concurrency:
        raise ArtifactCheckError(
            f"load.concurrency={concurrency} is below required {min_concurrency}"
        )
    if measured_requests < min_requests:
        raise ArtifactCheckError(
            f"load.measured_requests={measured_requests} is below required {min_requests}"
        )

    corpus = require_object(artifact.get("corpus"), "corpus")
    require_string(corpus.get("path"), "corpus.path")
    require_string(corpus.get("sha256"), "corpus.sha256")
    require_int(corpus.get("prompt_count"), "corpus.prompt_count", positive=True)
    categories = require_list(corpus.get("categories"), "corpus.categories")
    if not categories or not all(isinstance(category, str) and category for category in categories):
        raise ArtifactCheckError("corpus.categories must contain non-empty strings")

    observations = [
        validate_observation(row, index)
        for index, row in enumerate(require_list(artifact.get("observations"), "observations"))
    ]
    summary = require_object(artifact.get("summary"), "summary")
    validate_counts(summary, observations)
    validate_goodput(
        summary,
        require_slo=require_slo,
        min_goodput_ratio=min_goodput_ratio,
    )
    require_number(summary.get("request_throughput_rps"), "summary.request_throughput_rps")
    require_number(
        summary.get("output_token_throughput_tok_s"), "summary.output_token_throughput_tok_s"
    )
    for key in SUMMARY_DISTRIBUTIONS:
        validate_distribution(summary.get(key), f"summary.{key}", allow_null=False)
    summary_route_decisions = summary.get("route_decisions")
    if summary_route_decisions is not None:
        validate_route_decisions(summary_route_decisions, "summary.route_decisions")
    required_route_decision_mins = required_route_decision_mins or {}
    if required_route_decision_mins:
        route_decisions = validate_route_decisions(
            summary.get("route_decisions"), "summary.route_decisions"
        )
        for key, minimum in required_route_decision_mins.items():
            actual = require_number(route_decisions.get(key), f"summary.route_decisions.{key}")
            if actual < minimum:
                raise ArtifactCheckError(
                    f"summary.route_decisions.{key}={actual:.6f} "
                    f"is below required {minimum:.6f}"
                )

    if require_zero_errors and int(summary["error_requests"]) != 0:
        raise ArtifactCheckError("serving artifact contains failed measured requests")

    input_tokens = require_object(summary.get("input_tokens"), "summary.input_tokens")
    if min_input_tokens_p95 is not None:
        p95 = require_number(input_tokens.get("p95"), "summary.input_tokens.p95")
        if p95 < min_input_tokens_p95:
            raise ArtifactCheckError(
                f"summary.input_tokens.p95={p95:.0f} is below required {min_input_tokens_p95}"
            )

    by_category = require_object(artifact.get("by_category"), "by_category")
    for category in categories:
        if category not in by_category:
            raise ArtifactCheckError(f"by_category lacks corpus category {category!r}")

    return artifact


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


def route_decision_min_arg(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must use KEY=MIN")
    key, raw_minimum = value.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("KEY must be non-empty")
    try:
        minimum = float(raw_minimum)
    except ValueError as error:
        raise argparse.ArgumentTypeError("MIN must be numeric") from error
    if not math.isfinite(minimum) or minimum < 0.0:
        raise argparse.ArgumentTypeError("MIN must be a non-negative finite number")
    return key, minimum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--min-requests", type=positive_int_arg, default=1)
    parser.add_argument("--min-concurrency", type=positive_int_arg, default=1)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--require-slo", action="store_true")
    parser.add_argument("--min-goodput-ratio", type=ratio_arg)
    parser.add_argument(
        "--min-input-tokens-p95",
        type=positive_int_arg,
        help="Require p95 input tokens at or above this value for long-prompt serving claims.",
    )
    parser.add_argument(
        "--require-route-decision-min",
        action="append",
        default=[],
        metavar="KEY=MIN",
        type=route_decision_min_arg,
        help=(
            "Require summary.route_decisions[KEY] to be at least MIN. Repeat for "
            "promotion gates that must prove a runtime route was exercised."
        ),
    )
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        artifact = validate_serving_benchmark_artifact(
            args.artifact,
            min_requests=args.min_requests,
            min_concurrency=args.min_concurrency,
            require_zero_errors=not args.allow_errors,
            require_slo=args.require_slo,
            min_goodput_ratio=args.min_goodput_ratio,
            min_input_tokens_p95=args.min_input_tokens_p95,
            required_route_decision_mins=dict(args.require_route_decision_min),
        )
    except ArtifactCheckError as error:
        print(f"AX serving benchmark artifact check failed: {error}", file=sys.stderr)
        return 1
    summary = artifact["summary"]
    load = artifact["load"]
    print(
        "AX serving benchmark artifact check passed: "
        f"{summary['requests']} measured requests, "
        f"concurrency={load['concurrency']}, "
        f"goodput={summary['goodput']['ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
