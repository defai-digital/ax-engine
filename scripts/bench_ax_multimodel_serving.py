#!/usr/bin/env python3
"""Replay a timed multi-model AX Engine serving and lifecycle scenario.

Each JSONL row is scheduled relative to one benchmark start:

  {"id":"qwen-long","kind":"request","at_ms":0,"model":"qwen",
   "input_tokens":[...],"max_output_tokens":128,"category":"long_prefill"}
  {"id":"gemma-chat","kind":"request","at_ms":100,"model":"gemma",
   "input_text":"Hello","max_output_tokens":64}
  {"id":"remove-embed","kind":"unload","at_ms":500,"model":"embed"}

`load` rows additionally require `model_path` and accept `load_mode`,
`load_policy`, and `make_default`. The artifact keeps request latency/goodput
separate from lifecycle latency and records per-model summaries.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bench_ax_serving as serving

SCHEMA_VERSION = "ax.multimodel_serving_benchmark.v1"
EVENT_KINDS = {"request", "load", "unload"}
# Product focus for mlxcel flip work (ADR-010). Other models remain nice-to-have.
FOCUS_FAMILY_MARKERS = (
    ("qwen3", ("qwen3", "qwen-3")),
    ("gemma4", ("gemma-4", "gemma4", "gemma_4")),
)


@dataclass(frozen=True)
class ScenarioEvent:
    id: str
    kind: str
    at_s: float
    model_id: str
    category: str
    raw: dict[str, Any]


def load_scenario(path: Path) -> list[ScenarioEvent]:
    events: list[ScenarioEvent] = []
    ids: set[str] = set()
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_no}: invalid JSONL row: {error}") from error
        if not isinstance(raw, dict):
            raise SystemExit(f"{path}:{line_no}: row must be an object")
        event_id = raw.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise SystemExit(f"{path}:{line_no}: id must be a non-empty string")
        if event_id in ids:
            raise SystemExit(f"{path}:{line_no}: duplicate id {event_id!r}")
        ids.add(event_id)
        kind = raw.get("kind", "request")
        if kind not in EVENT_KINDS:
            raise SystemExit(f"{path}:{line_no}: kind must be one of {sorted(EVENT_KINDS)}")
        model_id = raw.get("model")
        if not isinstance(model_id, str) or not model_id:
            raise SystemExit(f"{path}:{line_no}: model must be a non-empty string")
        at_ms = raw.get("at_ms", 0)
        if not isinstance(at_ms, (int, float)) or at_ms < 0:
            raise SystemExit(f"{path}:{line_no}: at_ms must be non-negative")
        category = raw.get("category", kind)
        if not isinstance(category, str) or not category:
            raise SystemExit(f"{path}:{line_no}: category must be a non-empty string")
        if kind == "request":
            _validate_request_row(path, line_no, raw)
        elif kind == "load":
            model_path = raw.get("model_path")
            model_path_env = raw.get("model_path_env")
            if not (isinstance(model_path, str) and model_path) and not (
                isinstance(model_path_env, str) and model_path_env
            ):
                raise SystemExit(f"{path}:{line_no}: load requires model_path or model_path_env")
        events.append(
            ScenarioEvent(
                id=event_id,
                kind=kind,
                at_s=float(at_ms) / 1000.0,
                model_id=model_id,
                category=category,
                raw=raw,
            )
        )
    if not events:
        raise SystemExit(f"{path}: scenario has no events")
    return sorted(events, key=lambda event: (event.at_s, event.id))


def _validate_request_row(path: Path, line_no: int, raw: dict[str, Any]) -> None:
    input_text = raw.get("input_text")
    input_text_pattern = raw.get("input_text_pattern")
    input_text_repeats = raw.get("input_text_repeats")
    input_tokens = raw.get("input_tokens")
    input_token_pattern = raw.get("input_token_pattern")
    input_tokens_path = raw.get("input_tokens_path")
    input_tokens_count = raw.get("input_tokens_count")
    has_tokens = (
        isinstance(input_tokens, list)
        and input_tokens
        and all(isinstance(token, int) and token >= 0 for token in input_tokens)
    )
    has_pattern = (
        isinstance(input_token_pattern, list)
        and input_token_pattern
        and all(isinstance(token, int) and token >= 0 for token in input_token_pattern)
        and isinstance(input_tokens_count, int)
        and input_tokens_count > 0
    )
    has_token_file = isinstance(input_tokens_path, str) and bool(input_tokens_path)
    has_text = isinstance(input_text, str) and bool(input_text)
    has_text_pattern = (
        isinstance(input_text_pattern, str)
        and bool(input_text_pattern)
        and isinstance(input_text_repeats, int)
        and input_text_repeats > 0
    )
    if not any((has_text, has_text_pattern, has_tokens, has_pattern, has_token_file)):
        raise SystemExit(
            f"{path}:{line_no}: request needs non-empty input_text, input_text_pattern "
            "with positive input_text_repeats, non-empty input_tokens, input_tokens_path, "
            "or input_token_pattern with positive input_tokens_count"
        )
    max_output_tokens = raw.get("max_output_tokens", 128)
    if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
        raise SystemExit(f"{path}:{line_no}: max_output_tokens must be positive")


def prompt_for_event(
    event: ScenarioEvent,
    *,
    scenario_dir: Path | None = None,
) -> serving.PromptItem:
    input_text = event.raw.get("input_text")
    input_text_pattern = event.raw.get("input_text_pattern")
    input_text_repeats = event.raw.get("input_text_repeats")
    if (
        input_text is None
        and isinstance(input_text_pattern, str)
        and input_text_pattern
        and isinstance(input_text_repeats, int)
        and input_text_repeats > 0
    ):
        input_text = input_text_pattern * input_text_repeats
    input_tokens = event.raw.get("input_tokens")
    input_token_pattern = event.raw.get("input_token_pattern")
    input_tokens_path = event.raw.get("input_tokens_path")
    requested_count = event.raw.get("input_tokens_count")
    if input_tokens is None and isinstance(input_tokens_path, str) and input_tokens_path:
        token_path = Path(input_tokens_path)
        if not token_path.is_absolute():
            token_path = (scenario_dir or Path.cwd()) / token_path
        try:
            token_payload = json.loads(token_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot read input_tokens_path {token_path}: {error}") from error
        token_field = event.raw.get("input_tokens_field", "token_ids")
        if not isinstance(token_field, str) or not token_field:
            raise RuntimeError("input_tokens_field must be a non-empty string")
        input_tokens = token_payload.get(token_field)
        if not (
            isinstance(input_tokens, list)
            and input_tokens
            and all(isinstance(token, int) and token >= 0 for token in input_tokens)
        ):
            raise RuntimeError(
                f"input_tokens_path {token_path} field {token_field!r} "
                "must contain non-negative token ids"
            )
    if (
        input_tokens is None
        and isinstance(input_token_pattern, list)
        and input_token_pattern
        and isinstance(requested_count, int)
        and requested_count > 0
    ):
        repeats = (requested_count + len(input_token_pattern) - 1) // len(input_token_pattern)
        input_tokens = (input_token_pattern * repeats)[:requested_count]
    input_count = len(input_tokens) if isinstance(input_tokens, list) else None
    return serving.PromptItem(
        id=event.id,
        category=event.category,
        input_text=input_text,
        input_tokens=input_tokens,
        input_tokens_count=requested_count if requested_count is not None else input_count,
        max_output_tokens=event.raw.get("max_output_tokens", 128),
        metadata=event.raw.get("metadata", {}),
    )


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            payload_out: Any = json.loads(body)
        except json.JSONDecodeError:
            payload_out = body
        return error.code, payload_out


def run_control_event(
    event: ScenarioEvent,
    *,
    base_url: str,
    timeout: float,
    benchmark_started: float,
) -> dict[str, Any]:
    target = benchmark_started + event.at_s
    delay = target - time.perf_counter()
    if delay > 0:
        time.sleep(delay)
    started = time.perf_counter()
    if event.kind == "load":
        endpoint = "/v1/model/load"
        model_path = event.raw.get("model_path")
        model_path_env = event.raw.get("model_path_env")
        if model_path is None and isinstance(model_path_env, str):
            model_path = os.environ.get(model_path_env)
        if not isinstance(model_path, str) or not model_path:
            return {
                "event_id": event.id,
                "kind": event.kind,
                "model_id": event.model_id,
                "category": event.category,
                "scheduled_at_s": event.at_s,
                "started_at_s": started - benchmark_started,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "status": None,
                "ok": False,
                "error": f"model path environment variable {model_path_env!r} is unset",
                "response": None,
            }
        payload = {
            "model_id": event.model_id,
            "model_path": model_path,
            "load_mode": event.raw.get("load_mode", "add"),
            "load_policy": event.raw.get("load_policy", "availability_first"),
            "make_default": event.raw.get("make_default", False),
        }
    else:
        endpoint = "/v1/model/unload"
        payload = {"model_id": event.model_id}
    try:
        status, response = post_json(f"{base_url.rstrip('/')}{endpoint}", payload, timeout)
        error = None
    except Exception as caught:  # noqa: BLE001 - artifacts capture control failures.
        status, response, error = None, None, str(caught)
    completed = time.perf_counter()
    return {
        "event_id": event.id,
        "kind": event.kind,
        "model_id": event.model_id,
        "category": event.category,
        "scheduled_at_s": event.at_s,
        "started_at_s": started - benchmark_started,
        "latency_ms": (completed - started) * 1000.0,
        "status": status,
        "ok": status is not None and 200 <= status < 300 and error is None,
        "error": error,
        "response": response,
    }


def parse_route_requirement(value: str) -> tuple[str, int]:
    name, separator, minimum_text = value.partition("=")
    if not name:
        raise argparse.ArgumentTypeError("route counter name must not be empty")
    if not separator:
        return name, 1
    try:
        minimum = int(minimum_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("route counter minimum must be an integer") from error
    if minimum < 0:
        raise argparse.ArgumentTypeError("route counter minimum must be non-negative")
    return name, minimum


def route_contract(
    summary: dict[str, Any],
    requirements: list[tuple[str, int]],
) -> dict[str, Any]:
    observed = summary.get("route_decisions", {})
    if not isinstance(observed, dict):
        observed = {}
    failures: list[str] = []
    required: dict[str, int] = {}
    selected_observed: dict[str, int | float | None] = {}
    for name, minimum in requirements:
        required[name] = max(required.get(name, 0), minimum)
    for name, minimum in sorted(required.items()):
        value = observed.get(name)
        selected_observed[name] = value if isinstance(value, (int, float)) else None
        if not isinstance(value, (int, float)) or value < minimum:
            failures.append(f"{name}: observed {value!r}, required >= {minimum}")
    return {
        "passed": not failures,
        "required": required,
        "observed": selected_observed,
        "failures": failures,
    }


def run_benchmark(
    args: argparse.Namespace,
    *,
    request_runner: Callable[..., dict[str, Any]] = serving.run_one_request,
    control_runner: Callable[..., dict[str, Any]] = run_control_event,
) -> dict[str, Any]:
    scenario_path = Path(args.scenario)
    events = load_scenario(scenario_path)
    benchmark_started = time.perf_counter()

    def run(event: ScenarioEvent) -> dict[str, Any]:
        if event.kind == "request":
            observation = request_runner(
                prompt=prompt_for_event(event, scenario_dir=scenario_path.parent),
                model_id=event.model_id,
                base_url=args.base_url,
                input_kind=args.input_kind,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                scheduled_offset_s=event.at_s,
                benchmark_started=benchmark_started,
                timeout=args.timeout,
            )
            observation["event_id"] = event.id
            observation["kind"] = event.kind
            observation["model_id"] = event.model_id
            return observation
        return control_runner(
            event,
            base_url=args.base_url,
            timeout=args.timeout,
            benchmark_started=benchmark_started,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        observations = list(executor.map(run, events))
    observations.sort(
        key=lambda item: (
            float(item.get("scheduled_at_s", 0.0)),
            str(item.get("event_id", "")),
        )
    )
    request_observations = [item for item in observations if item.get("kind") == "request"]
    lifecycle_observations = [item for item in observations if item.get("kind") != "request"]
    wall_duration_s = max(
        (
            float(item.get("started_at_s", 0.0))
            + float(item.get("e2e_latency_ms", item.get("latency_ms", 0.0))) / 1000.0
            for item in observations
        ),
        default=0.0,
    )
    model_ids = sorted({str(item["model_id"]) for item in request_observations})

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return serving.summarize_observations(
            items,
            wall_duration_s=wall_duration_s,
            ttft_slo_ms=args.slo_ttft_ms,
            tpot_slo_ms=args.slo_tpot_ms,
            e2e_slo_ms=args.slo_e2e_ms,
        )

    focus_families = sorted(
        {
            family
            for model_id in model_ids
            if (family := classify_focus_family(model_id)) is not None
        }
    )
    interactive_requests = [
        item for item in request_observations if item.get("category") == "interactive_decode"
    ]
    interactive_intervals = [
        float(value)
        for item in interactive_requests
        for value in (item.get("stream_step_interval_ms") or [])
        if value is not None
    ]
    request_http_503 = sum(1 for item in request_observations if item.get("status") == 503)
    request_http_5xx = sum(
        1
        for item in request_observations
        if isinstance(item.get("status"), int) and 500 <= int(item["status"]) < 600
    )

    summary = summarize(request_observations)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "created_at": serving.utc_now(),
        "methodology": {
            "scope": "timed_multi_model_serving_and_lifecycle",
            "request_endpoint": "/v1/generate/stream",
            "timing_scope": "client_observed",
        },
        "focus": {
            "families": focus_families,
            "policy": "qwen3_gemma4_primary",
            "note": (
                "AX flip work optimizes Qwen 3 and Gemma 4; other models are "
                "nice-to-have and do not own flip gates."
            ),
        },
        "target": {"base_url": args.base_url.rstrip("/"), "models": model_ids},
        "scenario": {
            "path": str(scenario_path),
            "sha256": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
            "events": len(events),
        },
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "summary": summary,
        "by_model": {
            model_id: summarize(
                [item for item in request_observations if item.get("model_id") == model_id]
            )
            for model_id in model_ids
        },
        "interactive_stream_gap_ms": serving.summarize_values(interactive_intervals),
        "availability": {
            "request_http_503": request_http_503,
            "request_http_5xx": request_http_5xx,
            "request_error_rate": (
                (
                    sum(1 for item in request_observations if not item.get("ok"))
                    / len(request_observations)
                )
                if request_observations
                else 0.0
            ),
        },
        "lifecycle": {
            "events": len(lifecycle_observations),
            "ok_events": sum(bool(item.get("ok")) for item in lifecycle_observations),
            "error_events": sum(not bool(item.get("ok")) for item in lifecycle_observations),
            "latency_ms": serving.summarize_values(
                item.get("latency_ms") for item in lifecycle_observations
            ),
        },
        "observations": observations,
    }
    artifact["route_contract"] = route_contract(
        summary,
        list(getattr(args, "require_route_counter", [])),
    )
    return artifact


def classify_focus_family(model_id: str) -> str | None:
    lowered = model_id.lower()
    for family, markers in FOCUS_FAMILY_MARKERS:
        if any(marker in lowered for marker in markers):
            return family
    return None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def optional_positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31418")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--workers", type=positive_int, default=16)
    parser.add_argument("--input-kind", choices=["auto", "text", "tokens"], default="auto")
    parser.add_argument("--timeout", type=optional_positive_float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slo-ttft-ms", type=optional_positive_float)
    parser.add_argument("--slo-tpot-ms", type=optional_positive_float)
    parser.add_argument("--slo-e2e-ms", type=optional_positive_float)
    parser.add_argument(
        "--require-route-counter",
        action="append",
        default=[],
        type=parse_route_requirement,
        metavar="NAME[=MIN]",
        help="fail after writing the artifact unless an aggregate route counter reaches MIN",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_benchmark(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0 if result["route_contract"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
