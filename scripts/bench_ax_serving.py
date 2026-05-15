#!/usr/bin/env python3
"""Benchmark AX Engine as an online serving endpoint.

This harness measures user-visible streaming latency through
`/v1/generate/stream`. It complements the MLX inference-stack microbenchmark:
use this for serving claims such as TTFT percentiles, request throughput, output
token throughput, TPOT, E2E latency, and SLO goodput over a prompt corpus.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = "ax.serving_benchmark.v1"


@dataclass(frozen=True)
class PromptItem:
    id: str
    category: str
    input_text: str | None
    input_tokens: list[int] | None
    input_tokens_count: int | None
    max_output_tokens: int
    metadata: dict[str, Any]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_values(values: Iterable[float | int | None]) -> dict[str, float | int] | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return {
        "count": len(clean),
        "min": min(clean),
        "mean": statistics.fmean(clean),
        "p50": percentile(clean, 0.50),
        "p75": percentile(clean, 0.75),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "max": max(clean),
    }


def corpus_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus(path: Path) -> list[PromptItem]:
    prompts: list[PromptItem] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_no}: invalid JSONL row: {error}") from error

        prompt_id = raw.get("id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise SystemExit(f"{path}:{line_no}: prompt id must be a non-empty string")
        category = raw.get("category", "uncategorized")
        if not isinstance(category, str) or not category:
            raise SystemExit(f"{path}:{line_no}: category must be a non-empty string")

        input_text = raw.get("input_text")
        if input_text is not None and not isinstance(input_text, str):
            raise SystemExit(f"{path}:{line_no}: input_text must be a string")
        if input_text is not None and not input_text:
            raise SystemExit(f"{path}:{line_no}: input_text must not be empty")

        input_tokens = raw.get("input_tokens")
        if input_tokens is not None:
            if not isinstance(input_tokens, list) or not all(
                isinstance(token, int) and token >= 0 for token in input_tokens
            ):
                raise SystemExit(f"{path}:{line_no}: input_tokens must be non-negative ints")
            if not input_tokens:
                raise SystemExit(f"{path}:{line_no}: input_tokens must not be empty")

        if input_text is None and input_tokens is None:
            raise SystemExit(f"{path}:{line_no}: row needs input_text or input_tokens")

        input_tokens_count = raw.get("input_tokens_count")
        if input_tokens_count is not None and (
            not isinstance(input_tokens_count, int) or input_tokens_count <= 0
        ):
            raise SystemExit(f"{path}:{line_no}: input_tokens_count must be positive")
        if input_tokens is not None:
            input_tokens_count = len(input_tokens)

        max_output_tokens = raw.get("max_output_tokens")
        if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            raise SystemExit(f"{path}:{line_no}: max_output_tokens must be positive")

        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise SystemExit(f"{path}:{line_no}: metadata must be an object")

        prompts.append(
            PromptItem(
                id=prompt_id,
                category=category,
                input_text=input_text,
                input_tokens=input_tokens,
                input_tokens_count=input_tokens_count,
                max_output_tokens=max_output_tokens,
                metadata=metadata,
            )
        )

    if not prompts:
        raise SystemExit(f"{path}: corpus is empty")
    return prompts


def select_prompt_input(prompt: PromptItem, input_kind: str) -> tuple[str, str | list[int]]:
    if input_kind == "tokens":
        if prompt.input_tokens is None:
            raise RuntimeError(f"prompt {prompt.id} does not include input_tokens")
        return "input_tokens", prompt.input_tokens
    if input_kind == "text":
        if prompt.input_text is None:
            raise RuntimeError(f"prompt {prompt.id} does not include input_text")
        return "input_text", prompt.input_text
    if prompt.input_tokens is not None:
        return "input_tokens", prompt.input_tokens
    if prompt.input_text is not None:
        return "input_text", prompt.input_text
    raise RuntimeError(f"prompt {prompt.id} does not include a supported input")


def build_payload(
    prompt: PromptItem,
    *,
    model_id: str,
    input_kind: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    input_name, input_value = select_prompt_input(prompt, input_kind)
    return {
        "model": model_id,
        input_name: input_value,
        "max_output_tokens": prompt.max_output_tokens,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": 1.0,
            "seed": seed,
        },
    }


def sse_field_value(line: str, field: str) -> str | None:
    prefix = f"{field}:"
    if not line.startswith(prefix):
        return None
    value = line[len(prefix):]
    if value.startswith(" "):
        value = value[1:]
    return value


def parse_sse_frame(lines: list[str]) -> tuple[str | None, str] | None:
    event_name: str | None = None
    data_parts: list[str] = []
    for line in lines:
        event_value = sse_field_value(line, "event")
        if event_value is not None:
            event_name = event_value
            continue
        data_value = sse_field_value(line, "data")
        if data_value is not None:
            data_parts.append(data_value)
    if not data_parts:
        return None
    return event_name, "\n".join(data_parts)


def parse_sse_text(text: str) -> list[tuple[str | None, str]]:
    frames: list[tuple[str | None, str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line == "":
            frame = parse_sse_frame(current)
            if frame is not None:
                frames.append(frame)
            current = []
        else:
            current.append(line)
    frame = parse_sse_frame(current)
    if frame is not None:
        frames.append(frame)
    return frames


def decode_sse_data(data: str) -> Any:
    if data == "[DONE]":
        return {"done": True}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"raw": data}


def http_sse_events(
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> Iterator[tuple[str | None, Any, float]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            yield "__http_status__", {"status": response.status}, 0.0
            current: list[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line == "":
                    frame = parse_sse_frame(current)
                    current = []
                    if frame is not None:
                        event_name, data = frame
                        yield event_name, decode_sse_data(data), time.perf_counter() - started
                    continue
                current.append(line)
            frame = parse_sse_frame(current)
            if frame is not None:
                event_name, data = frame
                yield event_name, decode_sse_data(data), time.perf_counter() - started
    except urllib.error.HTTPError as error:
        data = error.read().decode("utf-8", errors="replace")
        yield "__http_status__", {"status": error.code, "error": data}, time.perf_counter() - started


def delta_token_count(event_name: str | None, payload: Any) -> int:
    if event_name != "step" or not isinstance(payload, dict):
        return 0
    delta_tokens = payload.get("delta_tokens")
    if isinstance(delta_tokens, list):
        return len(delta_tokens)
    delta_text = payload.get("delta_text")
    if isinstance(delta_text, str) and delta_text:
        return 1
    return 0


def final_output_token_count(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    response = payload.get("response")
    if not isinstance(response, dict):
        return None
    output_token_count = response.get("output_token_count")
    if isinstance(output_token_count, int):
        return output_token_count
    output_tokens = response.get("output_tokens")
    if isinstance(output_tokens, list):
        return len(output_tokens)
    return None


def final_route_decisions(payload: Any) -> dict[str, int | float | str | bool]:
    if not isinstance(payload, dict):
        return {}
    response = payload.get("response")
    if not isinstance(response, dict):
        return {}
    route = response.get("route")
    if not isinstance(route, dict):
        return {}
    decisions = route.get("crossover_decisions")
    if not isinstance(decisions, dict):
        return {}
    out: dict[str, int | float | str | bool] = {}
    for key, value in decisions.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, (int, float, str, bool)):
            out[key] = value
    return out


def observe_stream(
    events: Iterable[tuple[str | None, Any, float]],
    *,
    prompt: PromptItem,
    scheduled_at_s: float,
    started_at_s: float,
    completed_at_s: float | None = None,
) -> dict[str, Any]:
    status = 200
    error: str | None = None
    first_token_s: float | None = None
    output_chunk_times: list[float] = []
    observed_output_tokens = 0
    final_output_tokens: int | None = None
    route_decisions: dict[str, int | float | str | bool] = {}
    event_count = 0

    for event_name, payload, elapsed_s in events:
        event_count += 1
        if event_name == "__http_status__" and isinstance(payload, dict):
            status = int(payload.get("status", status))
            if "error" in payload:
                error = str(payload["error"])
            continue
        if event_name == "error":
            error = json.dumps(payload, sort_keys=True)
            continue
        count = delta_token_count(event_name, payload)
        if count > 0:
            observed_output_tokens += count
            output_chunk_times.append(elapsed_s)
            if first_token_s is None:
                first_token_s = elapsed_s
        if event_name == "response":
            final_output_tokens = final_output_token_count(payload)
            route_decisions = final_route_decisions(payload)

    if completed_at_s is None:
        completed_at_s = time.perf_counter()

    output_tokens = final_output_tokens
    if output_tokens is None:
        output_tokens = observed_output_tokens if observed_output_tokens > 0 else None

    e2e_s = max(completed_at_s - started_at_s, 0.0)
    ttft_ms = first_token_s * 1000.0 if first_token_s is not None else None
    tpot_ms = None
    if first_token_s is not None and output_tokens is not None and output_tokens > 1:
        tpot_ms = max((e2e_s - first_token_s) * 1000.0 / (output_tokens - 1), 0.0)

    intervals_ms = [
        (later - earlier) * 1000.0
        for earlier, later in zip(output_chunk_times, output_chunk_times[1:])
        if later >= earlier
    ]

    return {
        "prompt_id": prompt.id,
        "category": prompt.category,
        "phase": "measured",
        "status": status,
        "ok": 200 <= status < 300 and error is None,
        "error": error,
        "scheduled_at_s": scheduled_at_s,
        "started_at_s": started_at_s,
        "queue_delay_ms": max((started_at_s - scheduled_at_s) * 1000.0, 0.0),
        "e2e_latency_ms": e2e_s * 1000.0,
        "ttft_ms": ttft_ms,
        "client_tpot_ms": tpot_ms,
        "stream_step_interval_ms": intervals_ms,
        "input_tokens": prompt.input_tokens_count,
        "max_output_tokens": prompt.max_output_tokens,
        "output_tokens": output_tokens,
        "output_chunks": len(output_chunk_times),
        "events": event_count,
        "route_decisions": route_decisions,
        "metadata": prompt.metadata,
    }


def run_one_request(
    *,
    prompt: PromptItem,
    model_id: str,
    base_url: str,
    input_kind: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    scheduled_offset_s: float,
    benchmark_started: float,
    timeout: float,
    stream_func=http_sse_events,
) -> dict[str, Any]:
    scheduled_at_s = scheduled_offset_s
    target = benchmark_started + scheduled_offset_s
    delay = target - time.perf_counter()
    if delay > 0:
        time.sleep(delay)

    started = time.perf_counter()
    relative_started = started - benchmark_started
    payload = build_payload(
        prompt,
        model_id=model_id,
        input_kind=input_kind,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    url = f"{base_url.rstrip('/')}/v1/generate/stream"
    try:
        events = list(stream_func(url, payload, timeout))
        completed = time.perf_counter()
        return observe_stream(
            events,
            prompt=prompt,
            scheduled_at_s=scheduled_at_s,
            started_at_s=relative_started,
            completed_at_s=completed - benchmark_started,
        )
    except Exception as error:  # noqa: BLE001 - benchmark artifacts should capture failures.
        completed = time.perf_counter()
        return {
            "prompt_id": prompt.id,
            "category": prompt.category,
            "phase": "measured",
            "status": None,
            "ok": False,
            "error": str(error),
            "scheduled_at_s": scheduled_at_s,
            "started_at_s": relative_started,
            "queue_delay_ms": max((relative_started - scheduled_at_s) * 1000.0, 0.0),
            "e2e_latency_ms": (completed - started) * 1000.0,
            "ttft_ms": None,
            "client_tpot_ms": None,
            "stream_step_interval_ms": [],
            "input_tokens": prompt.input_tokens_count,
            "max_output_tokens": prompt.max_output_tokens,
            "output_tokens": None,
            "output_chunks": 0,
            "events": 0,
            "route_decisions": {},
            "metadata": prompt.metadata,
        }


def goodput_pass(
    observation: dict[str, Any],
    *,
    ttft_slo_ms: float | None,
    tpot_slo_ms: float | None,
    e2e_slo_ms: float | None,
) -> bool:
    if not observation.get("ok"):
        return False
    checks = [
        (ttft_slo_ms, observation.get("ttft_ms")),
        (tpot_slo_ms, observation.get("client_tpot_ms")),
        (e2e_slo_ms, observation.get("e2e_latency_ms")),
    ]
    for slo, value in checks:
        if slo is not None and (value is None or float(value) > slo):
            return False
    return True


def summarize_observations(
    observations: list[dict[str, Any]],
    *,
    wall_duration_s: float,
    ttft_slo_ms: float | None,
    tpot_slo_ms: float | None,
    e2e_slo_ms: float | None,
) -> dict[str, Any]:
    measured = [item for item in observations if item.get("phase") == "measured"]
    ok = [item for item in measured if item.get("ok")]
    output_tokens = [item.get("output_tokens") or 0 for item in ok]
    input_tokens = [item.get("input_tokens") for item in ok]
    route_decisions = summarize_route_decisions(ok)
    intervals = list(
        itertools.chain.from_iterable(item.get("stream_step_interval_ms", []) for item in ok)
    )
    good = [
        item
        for item in measured
        if goodput_pass(
            item,
            ttft_slo_ms=ttft_slo_ms,
            tpot_slo_ms=tpot_slo_ms,
            e2e_slo_ms=e2e_slo_ms,
        )
    ]
    duration = max(wall_duration_s, 0.001)
    return {
        "requests": len(measured),
        "ok_requests": len(ok),
        "error_requests": len(measured) - len(ok),
        "request_throughput_rps": len(ok) / duration,
        "output_token_throughput_tok_s": sum(output_tokens) / duration,
        "ttft_ms": summarize_values(item.get("ttft_ms") for item in ok),
        "client_tpot_ms": summarize_values(item.get("client_tpot_ms") for item in ok),
        "stream_step_interval_ms": summarize_values(intervals),
        "e2e_latency_ms": summarize_values(item.get("e2e_latency_ms") for item in ok),
        "queue_delay_ms": summarize_values(item.get("queue_delay_ms") for item in measured),
        "input_tokens": summarize_values(input_tokens),
        "output_tokens": summarize_values(output_tokens),
        "route_decisions": route_decisions,
        "goodput": {
            "requests": len(good),
            "ratio": len(good) / len(measured) if measured else 0.0,
            "ttft_slo_ms": ttft_slo_ms,
            "client_tpot_slo_ms": tpot_slo_ms,
            "e2e_slo_ms": e2e_slo_ms,
        },
    }


def summarize_route_decisions(observations: list[dict[str, Any]]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    for item in observations:
        decisions = item.get("route_decisions")
        if not isinstance(decisions, dict):
            continue
        for key, value in decisions.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, bool):
                totals[key] = totals.get(key, 0) + int(value)
            elif isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
            elif isinstance(value, float) and math.isfinite(value):
                totals[key] = float(totals.get(key, 0)) + value
    return totals


def summarize_by_category(
    observations: list[dict[str, Any]],
    *,
    wall_duration_s: float,
    ttft_slo_ms: float | None,
    tpot_slo_ms: float | None,
    e2e_slo_ms: float | None,
) -> dict[str, Any]:
    categories = sorted({str(item.get("category", "uncategorized")) for item in observations})
    return {
        category: summarize_observations(
            [item for item in observations if item.get("category") == category],
            wall_duration_s=wall_duration_s,
            ttft_slo_ms=ttft_slo_ms,
            tpot_slo_ms=tpot_slo_ms,
            e2e_slo_ms=e2e_slo_ms,
        )
        for category in categories
    }


def scheduled_offsets(requests: int, request_rate_rps: float | None) -> list[float]:
    if requests <= 0:
        return []
    if request_rate_rps is None or request_rate_rps <= 0:
        return [0.0 for _ in range(requests)]
    return [index / request_rate_rps for index in range(requests)]


def run_benchmark(args: argparse.Namespace, *, stream_func=http_sse_events) -> dict[str, Any]:
    corpus_path = Path(args.corpus)
    prompts = load_corpus(corpus_path)
    prompt_cycle = itertools.cycle(prompts)

    warmup_prompts = [next(prompt_cycle) for _ in range(args.warmup_requests)]
    measured_prompts = [next(prompt_cycle) for _ in range(args.requests)]

    observations: list[dict[str, Any]] = []
    for prompt in warmup_prompts:
        started = time.perf_counter()
        observation = run_one_request(
            prompt=prompt,
            model_id=args.model_id,
            base_url=args.base_url,
            input_kind=args.input_kind,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            scheduled_offset_s=0.0,
            benchmark_started=started,
            timeout=args.timeout,
            stream_func=stream_func,
        )
        observation["phase"] = "warmup"
        observations.append(observation)

    benchmark_started = time.perf_counter()
    offsets = scheduled_offsets(args.requests, args.request_rate_rps)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_one_request,
                prompt=prompt,
                model_id=args.model_id,
                base_url=args.base_url,
                input_kind=args.input_kind,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                scheduled_offset_s=offset,
                benchmark_started=benchmark_started,
                timeout=args.timeout,
                stream_func=stream_func,
            )
            for prompt, offset in zip(measured_prompts, offsets)
        ]
        for future in concurrent.futures.as_completed(futures):
            observations.append(future.result())

    wall_duration_s = max(
        (
            item["started_at_s"] + item["e2e_latency_ms"] / 1000.0
            for item in observations
            if item.get("phase") == "measured"
        ),
        default=0.0,
    )
    measured_observations = sorted(
        [item for item in observations if item.get("phase") == "measured"],
        key=lambda item: (item["scheduled_at_s"], item["prompt_id"]),
    )
    warmup_observations = [item for item in observations if item.get("phase") == "warmup"]
    ordered_observations = warmup_observations + measured_observations

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "methodology": {
            "scope": "online_serving_streaming_latency",
            "endpoint": "/v1/generate/stream",
            "timing_scope": "client_observed_sse",
            "notes": [
                "TTFT is measured from client request start to first non-empty step event.",
                "client_tpot_ms is measured from first token/chunk to stream completion divided by output tokens after the first token.",
                "stream_step_interval_ms is based on observed non-empty SSE step arrivals; multi-token steps are not split into synthetic per-token timestamps.",
                "Use this artifact for serving behavior, not raw model-kernel throughput.",
            ],
        },
        "target": {
            "base_url": args.base_url.rstrip("/"),
            "model_id": args.model_id,
            "input_kind": args.input_kind,
        },
        "load": {
            "concurrency": args.concurrency,
            "request_rate_rps": args.request_rate_rps,
            "warmup_requests": args.warmup_requests,
            "measured_requests": args.requests,
        },
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "corpus": {
            "path": str(corpus_path),
            "sha256": corpus_sha256(corpus_path),
            "prompt_count": len(prompts),
            "categories": sorted({prompt.category for prompt in prompts}),
        },
        "summary": summarize_observations(
            measured_observations,
            wall_duration_s=wall_duration_s,
            ttft_slo_ms=args.slo_ttft_ms,
            tpot_slo_ms=args.slo_tpot_ms,
            e2e_slo_ms=args.slo_e2e_ms,
        ),
        "by_category": summarize_by_category(
            measured_observations,
            wall_duration_s=wall_duration_s,
            ttft_slo_ms=args.slo_ttft_ms,
            tpot_slo_ms=args.slo_tpot_ms,
            e2e_slo_ms=args.slo_e2e_ms,
        ),
        "observations": ordered_observations,
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def optional_positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--input-kind", choices=["auto", "text", "tokens"], default="auto")
    parser.add_argument("--requests", type=positive_int, default=12)
    parser.add_argument("--warmup-requests", type=non_negative_int, default=2)
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument("--request-rate-rps", type=optional_positive_float)
    parser.add_argument("--timeout", type=optional_positive_float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slo-ttft-ms", type=optional_positive_float)
    parser.add_argument("--slo-tpot-ms", type=optional_positive_float)
    parser.add_argument("--slo-e2e-ms", type=optional_positive_float)
    parser.add_argument("--output", type=Path)
    return parser


def main_with_args_for_test(argv: list[str], *, stream_func=http_sse_events) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_benchmark(args, stream_func=stream_func)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


def main() -> None:
    raise SystemExit(main_with_args_for_test(sys.argv[1:]))


if __name__ == "__main__":
    main()
