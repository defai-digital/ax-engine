#!/usr/bin/env python3
"""Validate AX multi-model serving benchmark artifacts.

Intended for Qwen 3 + Gemma 4 flip work: schema correctness, focus-family
presence, availability, and optional absolute stream-gap caps on interactive
decode. This is not a substitute for fixed-shape model-runtime benches.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ax.multimodel_serving_benchmark.v1"
REQUIRED_METHODOLOGY = {
    "scope": "timed_multi_model_serving_and_lifecycle",
    "request_endpoint": "/v1/generate/stream",
    "timing_scope": "client_observed",
}
PERCENTILE_KEYS = ["count", "min", "mean", "p50", "p75", "p90", "p95", "p99", "max"]
SUMMARY_DISTRIBUTIONS = [
    "ttft_ms",
    "client_tpot_ms",
    "stream_step_interval_ms",
    "e2e_latency_ms",
    "queue_delay_ms",
    "input_tokens",
    "output_tokens",
]

# Focus product lines for the mlxcel flip schedule.
FOCUS_FAMILY_PATTERNS: dict[str, re.Pattern[str]] = {
    "qwen3": re.compile(r"qwen3", re.IGNORECASE),
    "gemma4": re.compile(r"gemma[-_]?4", re.IGNORECASE),
}


class ArtifactCheckError(RuntimeError):
    """Raised when a multi-model artifact is not claim-ready."""


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


def validate_distribution(
    value: Any, owner: str, *, allow_null: bool = False
) -> dict[str, Any] | None:
    if value is None and allow_null:
        return None
    if value is None:
        raise ArtifactCheckError(f"{owner} must be present")
    dist = require_object(value, owner)
    for key in PERCENTILE_KEYS:
        require_number(dist.get(key), f"{owner}.{key}")
    count = require_int(dist.get("count"), f"{owner}.count", positive=True)
    if dist["min"] > dist["p50"] or dist["p50"] > dist["p95"] or dist["p95"] > dist["max"]:
        raise ArtifactCheckError(f"{owner} percentiles are not monotonic")
    if count < 1:
        raise ArtifactCheckError(f"{owner}.count must be at least 1")
    return dist


def classify_focus_model(model_id: str) -> str | None:
    for family, pattern in FOCUS_FAMILY_PATTERNS.items():
        if pattern.search(model_id):
            return family
    return None


def linear_percentile(values: list[float], q: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def interactive_stream_gap_p95(
    artifact: dict[str, Any],
    observations: list[dict[str, Any]],
    summary: dict[str, Any],
) -> float:
    """Return interactive stream-step p95 from focus fields or observations."""
    interactive_gap = artifact.get("interactive_stream_gap_ms")
    if isinstance(interactive_gap, dict) and interactive_gap.get("p95") is not None:
        return require_number(interactive_gap.get("p95"), "interactive_stream_gap_ms.p95")

    intervals: list[float] = []
    for obs in observations:
        if obs.get("kind") != "request":
            continue
        category = obs.get("category")
        if category is not None and category != "interactive_decode":
            continue
        for value in obs.get("stream_step_interval_ms") or []:
            intervals.append(float(value))
    if intervals:
        return linear_percentile(intervals, 0.95)

    dist = summary.get("stream_step_interval_ms")
    if not isinstance(dist, dict) or dist.get("p95") is None:
        raise ArtifactCheckError(
            "cannot enforce stream-gap cap without interactive intervals"
        )
    return require_number(dist.get("p95"), "summary.stream_step_interval_ms.p95")


def validate_observation(row: Any, index: int) -> dict[str, Any]:
    obs = require_object(row, f"observations[{index}]")
    require_string(obs.get("event_id"), f"observations[{index}].event_id")
    kind = obs.get("kind")
    if kind not in {"request", "load", "unload"}:
        raise ArtifactCheckError(
            f"observations[{index}].kind must be request, load, or unload"
        )
    require_string(obs.get("model_id"), f"observations[{index}].model_id")
    if not isinstance(obs.get("ok"), bool):
        raise ArtifactCheckError(f"observations[{index}].ok must be a boolean")
    require_number(obs.get("scheduled_at_s"), f"observations[{index}].scheduled_at_s")
    require_number(obs.get("started_at_s"), f"observations[{index}].started_at_s")
    if kind == "request":
        require_number(obs.get("e2e_latency_ms"), f"observations[{index}].e2e_latency_ms")
        intervals = obs.get("stream_step_interval_ms")
        if intervals is not None:
            require_list(intervals, f"observations[{index}].stream_step_interval_ms")
            for i, value in enumerate(intervals):
                require_number(value, f"observations[{index}].stream_step_interval_ms[{i}]")
    else:
        require_number(obs.get("latency_ms"), f"observations[{index}].latency_ms")
    return obs


def validate_summary_block(summary: dict[str, Any], owner: str, *, allow_empty: bool) -> None:
    requests = require_int(summary.get("requests"), f"{owner}.requests")
    if requests == 0:
        if not allow_empty:
            raise ArtifactCheckError(f"{owner}.requests must be positive")
        return
    require_int(summary.get("ok_requests"), f"{owner}.ok_requests")
    require_int(summary.get("error_requests"), f"{owner}.error_requests")
    if summary["ok_requests"] + summary["error_requests"] != summary["requests"]:
        raise ArtifactCheckError(f"{owner} ok+error must equal requests")
    require_number(summary.get("request_throughput_rps"), f"{owner}.request_throughput_rps")
    require_number(
        summary.get("output_token_throughput_tok_s"),
        f"{owner}.output_token_throughput_tok_s",
    )
    # client_tpot / stream intervals are absent for single-token or failed
    # streams; ttft/e2e/queue/token counts should still materialize on ok rows.
    nullable_when_partial = {
        "client_tpot_ms",
        "stream_step_interval_ms",
    }
    for key in SUMMARY_DISTRIBUTIONS:
        allow_null = summary["ok_requests"] == 0 or key in nullable_when_partial
        validate_distribution(summary.get(key), f"{owner}.{key}", allow_null=allow_null)


def validate_multimodel_serving_artifact(
    path: Path,
    *,
    min_requests: int,
    require_zero_errors: bool,
    require_focus_families: list[str],
    max_interactive_stream_gap_p95_ms: float | None,
    max_request_http_503: int | None,
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
    models = require_list(target.get("models"), "target.models")
    if not all(isinstance(model, str) and model for model in models):
        raise ArtifactCheckError("target.models must contain non-empty strings")

    scenario = require_object(artifact.get("scenario"), "scenario")
    require_string(scenario.get("path"), "scenario.path")
    require_string(scenario.get("sha256"), "scenario.sha256")
    require_int(scenario.get("events"), "scenario.events", positive=True)

    sampling = require_object(artifact.get("sampling"), "sampling")
    require_number(sampling.get("temperature"), "sampling.temperature")
    require_number(sampling.get("top_p"), "sampling.top_p")
    require_int(sampling.get("seed"), "sampling.seed")

    observations = [
        validate_observation(row, index)
        for index, row in enumerate(require_list(artifact.get("observations"), "observations"))
    ]
    if len(observations) != scenario["events"]:
        raise ArtifactCheckError(
            f"observations count {len(observations)} != scenario.events {scenario['events']}"
        )

    summary = require_object(artifact.get("summary"), "summary")
    validate_summary_block(summary, "summary", allow_empty=False)
    if summary["requests"] < min_requests:
        raise ArtifactCheckError(
            f"summary.requests={summary['requests']} is below required {min_requests}"
        )
    if require_zero_errors and int(summary["error_requests"]) != 0:
        raise ArtifactCheckError("multi-model artifact contains failed request observations")

    by_model = require_object(artifact.get("by_model"), "by_model")
    request_models = sorted(
        {
            str(item["model_id"])
            for item in observations
            if item.get("kind") == "request"
        }
    )
    for model_id in request_models:
        if model_id not in by_model:
            raise ArtifactCheckError(f"by_model lacks model {model_id!r}")
        validate_summary_block(
            require_object(by_model[model_id], f"by_model[{model_id}]"),
            f"by_model[{model_id}]",
            allow_empty=False,
        )

    lifecycle = require_object(artifact.get("lifecycle"), "lifecycle")
    require_int(lifecycle.get("events"), "lifecycle.events")
    require_int(lifecycle.get("ok_events"), "lifecycle.ok_events")
    require_int(lifecycle.get("error_events"), "lifecycle.error_events")
    if lifecycle["events"] > 0:
        validate_distribution(
            lifecycle.get("latency_ms"), "lifecycle.latency_ms", allow_null=False
        )

    present_families = {
        family
        for model_id in request_models
        if (family := classify_focus_model(model_id)) is not None
    }
    for family in require_focus_families:
        if family not in FOCUS_FAMILY_PATTERNS:
            raise ArtifactCheckError(
                f"unknown focus family {family!r}; expected one of "
                f"{sorted(FOCUS_FAMILY_PATTERNS)}"
            )
        if family not in present_families:
            raise ArtifactCheckError(
                f"focus family {family!r} missing from request models {request_models}"
            )

    focus = artifact.get("focus")
    if focus is not None:
        focus_obj = require_object(focus, "focus")
        families = require_list(focus_obj.get("families"), "focus.families")
        if not all(isinstance(item, str) and item for item in families):
            raise ArtifactCheckError("focus.families must contain non-empty strings")

    if max_interactive_stream_gap_p95_ms is not None:
        p95 = interactive_stream_gap_p95(artifact, observations, summary)
        if p95 > max_interactive_stream_gap_p95_ms:
            raise ArtifactCheckError(
                f"interactive stream-gap p95={p95:.3f}ms exceeds cap "
                f"{max_interactive_stream_gap_p95_ms:.3f}ms"
            )

    if max_request_http_503 is not None:
        http_503 = 0
        for obs in observations:
            if obs.get("kind") != "request":
                continue
            if obs.get("status") == 503:
                http_503 += 1
        availability = artifact.get("availability")
        if isinstance(availability, dict) and availability.get("request_http_503") is not None:
            http_503 = require_int(
                availability.get("request_http_503"), "availability.request_http_503"
            )
        if http_503 > max_request_http_503:
            raise ArtifactCheckError(
                f"request HTTP 503 count={http_503} exceeds budget {max_request_http_503}"
            )

    return artifact


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_float_arg(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--min-requests", type=positive_int_arg, default=1)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument(
        "--require-focus-family",
        action="append",
        default=[],
        choices=sorted(FOCUS_FAMILY_PATTERNS),
        help="Require at least one request model matching this focus family. Repeatable.",
    )
    parser.add_argument(
        "--max-interactive-stream-gap-p95-ms",
        type=positive_float_arg,
        help="Fail if interactive decode stream-step p95 exceeds this many milliseconds.",
    )
    parser.add_argument(
        "--max-request-http-503",
        type=non_negative_int_arg,
        help="Fail if request observations include more than this many HTTP 503s.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        artifact = validate_multimodel_serving_artifact(
            args.artifact,
            min_requests=args.min_requests,
            require_zero_errors=not args.allow_errors,
            require_focus_families=list(args.require_focus_family),
            max_interactive_stream_gap_p95_ms=args.max_interactive_stream_gap_p95_ms,
            max_request_http_503=args.max_request_http_503,
        )
    except ArtifactCheckError as error:
        print(f"AX multi-model serving artifact check failed: {error}", file=sys.stderr)
        return 1

    summary = artifact["summary"]
    models = artifact["target"]["models"]
    print(
        "AX multi-model serving artifact check passed: "
        f"{summary['requests']} requests, models={models}, "
        f"output_tok_s={summary['output_token_throughput_tok_s']:.3f}, "
        f"errors={summary['error_requests']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
