#!/usr/bin/env python3
"""Exercise native generation with normal, disconnected, and slow consumers."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import bench_ax_serving as serving_bench
import bench_mlx_inference_stack as bench


SCHEMA_VERSION = "ax.native_generation_fault_soak.v1"
QUIESCENT_GAUGES = (
    "ax_engine_jobs_in_flight",
    "ax_engine_generation_jobs_pending",
    "ax_engine_generation_commands_queued",
    "ax_engine_generation_active_streams",
    "ax_engine_generation_buffered_stream_events",
)
SATURATION_COUNTER = "ax_engine_generation_saturated_commands_total"
BACKLOG_OVERFLOW_COUNTER = "ax_engine_generation_stream_backlog_overflows_total"


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    kind: str
    input_tokens: list[int]
    max_output_tokens: int


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 2 or "{" in fields[0]:
            continue
        try:
            metrics[fields[0]] = float(fields[1])
        except ValueError:
            continue
    return metrics


def request_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def request_json(url: str, timeout: float) -> dict[str, Any]:
    payload = json.loads(request_text(url, timeout))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return payload


def fetch_metrics(base_url: str, timeout: float) -> dict[str, float]:
    metrics = parse_prometheus_metrics(
        request_text(f"{base_url.rstrip('/')}/metrics", timeout)
    )
    missing = [
        name
        for name in (*QUIESCENT_GAUGES, SATURATION_COUNTER, BACKLOG_OVERFLOW_COUNTER)
        if name not in metrics
    ]
    if missing:
        raise RuntimeError(f"metrics endpoint is missing required fields: {', '.join(missing)}")
    return metrics


def wait_for_quiescence(
    base_url: str,
    *,
    timeout: float,
    poll_interval: float = 0.05,
) -> tuple[bool, dict[str, float]]:
    deadline = time.monotonic() + timeout
    latest: dict[str, float] = {}
    while time.monotonic() < deadline:
        latest = fetch_metrics(base_url, timeout=min(timeout, 5.0))
        if all(latest[name] == 0.0 for name in QUIESCENT_GAUGES):
            return True, latest
        time.sleep(poll_interval)
    return False, latest


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return after[name] - before[name]


def sample_metrics_until(
    stop: threading.Event,
    *,
    base_url: str,
    timeout: float,
    interval: float,
    started: float,
    samples: list[dict[str, Any]],
) -> None:
    while not stop.wait(interval):
        try:
            values = fetch_metrics(base_url, timeout=min(timeout, 5.0))
            samples.append(
                {
                    "elapsed_seconds": time.perf_counter() - started,
                    "values": values,
                }
            )
        except Exception as error:  # noqa: BLE001 - retain monitor failures in evidence.
            samples.append(
                {
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": str(error),
                }
            )


def summarize_metric_peaks(samples: list[dict[str, Any]]) -> dict[str, float]:
    names = (*QUIESCENT_GAUGES, SATURATION_COUNTER, BACKLOG_OVERFLOW_COUNTER)
    return {
        name: max(
            (
                float(sample["values"][name])
                for sample in samples
                if isinstance(sample.get("values"), dict)
                and name in sample["values"]
            ),
            default=0.0,
        )
        for name in names
    }


def build_payload(spec: RequestSpec, model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "input_tokens": spec.input_tokens,
        "max_output_tokens": spec.max_output_tokens,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "seed": 0,
            "ignore_eos": True,
        },
    }


def run_stalled_request(
    spec: RequestSpec,
    *,
    base_url: str,
    model_id: str,
    timeout: float,
    hold_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/generate/stream",
        data=json.dumps(build_payload(spec, model_id)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    http_status: int | None = None
    error_payload: Any = None
    outcome = "client_error"
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            http_status = response.status
            time.sleep(hold_s)
            outcome = "expected_stall_disconnect"
    except Exception as error:  # noqa: BLE001 - retain transport failures in evidence.
        error_payload = str(error)
    return {
        "request_id": spec.request_id,
        "kind": spec.kind,
        "outcome": outcome,
        "http_status": http_status,
        "output_events": 0,
        "output_tokens": 0,
        "error": error_payload,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def run_fault_request(
    spec: RequestSpec,
    *,
    base_url: str,
    model_id: str,
    timeout: float,
    disconnect_after_output_events: int,
    slow_delay_s: float,
    stream_func: Callable[
        [str, dict[str, Any], float], Iterator[tuple[str | None, Any, float]]
    ] = serving_bench.http_sse_events,
) -> dict[str, Any]:
    started = time.perf_counter()
    iterator = stream_func(
        f"{base_url.rstrip('/')}/v1/generate/stream",
        build_payload(spec, model_id),
        timeout,
    )
    http_status: int | None = None
    output_events = 0
    output_tokens = 0
    response_seen = False
    error_payload: Any = None
    outcome = "incomplete"
    try:
        for event_name, payload, _elapsed_s in iterator:
            if event_name == "__http_status__" and isinstance(payload, dict):
                http_status = int(payload.get("status", 0))
                if payload.get("error") is not None:
                    error_payload = payload["error"]
                continue
            if event_name == "error":
                error_payload = payload
                outcome = "stream_error"
                continue
            delta_tokens = serving_bench.delta_token_count(event_name, payload)
            if delta_tokens > 0:
                output_events += 1
                output_tokens += delta_tokens
                if (
                    spec.kind == "disconnect"
                    and output_events >= disconnect_after_output_events
                ):
                    outcome = "expected_disconnect"
                    break
                if spec.kind == "slow" and slow_delay_s > 0.0:
                    time.sleep(slow_delay_s)
            if event_name == "response":
                response_seen = True
                outcome = "completed"
    except Exception as error:  # noqa: BLE001 - the artifact must preserve client failures.
        error_payload = str(error)
        outcome = "client_error"
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    if outcome == "incomplete" and response_seen:
        outcome = "completed"
    if (
        spec.kind == "disconnect"
        and outcome == "completed"
        and output_events < disconnect_after_output_events
    ):
        outcome = "completed_before_disconnect_window"
    return {
        "request_id": spec.request_id,
        "kind": spec.kind,
        "outcome": outcome,
        "http_status": http_status,
        "output_events": output_events,
        "output_tokens": output_tokens,
        "error": error_payload,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def evaluate_run(
    outcomes: list[dict[str, Any]],
    *,
    quiescent: bool,
    counter_deltas: dict[str, float],
    metric_peaks: dict[str, float] | None = None,
    require_backpressure: bool = False,
    min_buffered_events: int = 1,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    for outcome in outcomes:
        kind = outcome["kind"]
        observed = outcome["outcome"]
        if kind == "normal" and observed != "completed":
            reasons.append(f"normal request {outcome['request_id']} ended as {observed}")
        elif kind == "disconnect" and observed not in {
            "expected_disconnect",
            "completed_before_disconnect_window",
        }:
            reasons.append(
                f"disconnect request {outcome['request_id']} ended as {observed}"
            )
        elif kind == "slow" and observed not in {"completed", "stream_error"}:
            reasons.append(f"slow request {outcome['request_id']} ended as {observed}")
        elif kind == "stalled" and observed != "expected_stall_disconnect":
            reasons.append(f"stalled request {outcome['request_id']} ended as {observed}")

    slow_errors = sum(
        1
        for outcome in outcomes
        if outcome["kind"] == "slow" and outcome["outcome"] == "stream_error"
    )
    if slow_errors > 0 and counter_deltas[BACKLOG_OVERFLOW_COUNTER] < slow_errors:
        reasons.append("slow-consumer stream errors lacked matching backlog-overflow evidence")
    disconnect_requests = [
        outcome for outcome in outcomes if outcome["kind"] == "disconnect"
    ]
    if disconnect_requests and not any(
        outcome["outcome"] == "expected_disconnect"
        for outcome in disconnect_requests
    ):
        reasons.append("no disconnect request reached the configured cancellation window")
    if counter_deltas[SATURATION_COUNTER] < 0.0 or counter_deltas[BACKLOG_OVERFLOW_COUNTER] < 0.0:
        reasons.append("generation counters moved backwards during the run")
    if not quiescent:
        reasons.append("generation lifecycle gauges did not return to zero")
    buffered_peak = float(
        (metric_peaks or {}).get("ax_engine_generation_buffered_stream_events", 0.0)
    )
    if (
        require_backpressure
        and buffered_peak < float(min_buffered_events)
        and counter_deltas[BACKLOG_OVERFLOW_COUNTER] <= 0.0
    ):
        reasons.append(
            "required backpressure was not observed in buffered-event or overflow metrics"
        )
    return ("pass" if not reasons else "fail"), reasons


def build_request_specs(
    *,
    rounds: int,
    normal_per_round: int,
    disconnect_per_round: int,
    slow_per_round: int,
    stalled_per_round: int,
    base_tokens: list[int],
    normal_output_tokens: int,
    fault_output_tokens: int,
) -> list[list[RequestSpec]]:
    rounds_out: list[list[RequestSpec]] = []
    request_index = 0
    for round_index in range(rounds):
        specs: list[RequestSpec] = []
        for kind, count, max_output_tokens in (
            ("normal", normal_per_round, normal_output_tokens),
            ("disconnect", disconnect_per_round, fault_output_tokens),
            ("slow", slow_per_round, fault_output_tokens),
            ("stalled", stalled_per_round, fault_output_tokens),
        ):
            for _ in range(count):
                input_tokens = list(base_tokens)
                specs.append(
                    RequestSpec(
                        request_id=f"r{round_index}-{kind}-{request_index}",
                        kind=kind,
                        input_tokens=input_tokens,
                        max_output_tokens=max_output_tokens,
                    )
                )
                request_index += 1
        rounds_out.append(specs)
    return rounds_out


def repo_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(bench.REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def tracked_dirty_status() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(bench.REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in output.splitlines() if line]


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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=positive_int, default=19110)
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument("--normal-per-round", type=non_negative_int, default=2)
    parser.add_argument("--disconnect-per-round", type=non_negative_int, default=1)
    parser.add_argument("--slow-per-round", type=non_negative_int, default=1)
    parser.add_argument("--stalled-per-round", type=non_negative_int, default=1)
    parser.add_argument("--prompt-tokens", type=positive_int, default=128)
    parser.add_argument("--normal-output-tokens", type=positive_int, default=32)
    parser.add_argument("--fault-output-tokens", type=positive_int, default=512)
    parser.add_argument("--disconnect-after-output-events", type=positive_int, default=1)
    parser.add_argument("--slow-delay-ms", type=positive_float, default=20.0)
    parser.add_argument("--stalled-hold-ms", type=positive_float, default=8000.0)
    parser.add_argument("--metrics-sample-interval", type=positive_float, default=0.25)
    parser.add_argument(
        "--require-backpressure",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-buffered-events", type=positive_int, default=1)
    parser.add_argument("--request-timeout", type=positive_float, default=600.0)
    parser.add_argument("--cleanup-timeout", type=positive_float, default=30.0)
    parser.add_argument("--warmup-requests", type=non_negative_int, default=2)
    return parser


def main_with_args(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.normal_per_round
        + args.disconnect_per_round
        + args.slow_per_round
        + args.stalled_per_round
        == 0
    ):
        raise SystemExit("at least one request per round is required")
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {model_dir}")
    if not bench.AX_ENGINE_SERVER.exists():
        raise SystemExit(
            f"release server binary not found: {bench.AX_ENGINE_SERVER}; run cargo build --release -p ax-engine-server"
        )

    vocab_size = bench.model_vocab_size(model_dir)
    base_tokens = bench.mlx_lm_reference_prompt_tokens(vocab_size, args.prompt_tokens)
    rounds = build_request_specs(
        rounds=args.rounds,
        normal_per_round=args.normal_per_round,
        disconnect_per_round=args.disconnect_per_round,
        slow_per_round=args.slow_per_round,
        stalled_per_round=args.stalled_per_round,
        base_tokens=base_tokens,
        normal_output_tokens=args.normal_output_tokens,
        fault_output_tokens=args.fault_output_tokens,
    )

    started_at = utc_now()
    process = bench.start_axengine(
        bench.AX_ENGINE_SERVER,
        model_dir,
        args.port,
        model_id=args.model_id,
        direct_mode=True,
    )
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        if not bench.wait_for_server(f"{base_url}/health", proc=process):
            raise RuntimeError(
                "ax-engine-server did not become ready:\n"
                + bench.process_stderr_snapshot(process)
            )

        warmup_specs = [
            RequestSpec(
                request_id=f"warmup-{index}",
                kind="normal",
                input_tokens=base_tokens,
                max_output_tokens=args.normal_output_tokens,
            )
            for index in range(args.warmup_requests)
        ]
        for spec in warmup_specs:
            outcome = run_fault_request(
                spec,
                base_url=base_url,
                model_id=args.model_id,
                timeout=args.request_timeout,
                disconnect_after_output_events=args.disconnect_after_output_events,
                slow_delay_s=args.slow_delay_ms / 1000.0,
            )
            if outcome["outcome"] != "completed":
                raise RuntimeError(f"warmup request failed: {outcome}")

        warmup_quiescent, before_metrics = wait_for_quiescence(
            base_url,
            timeout=args.cleanup_timeout,
        )
        if not warmup_quiescent:
            raise RuntimeError("generation gauges did not quiesce after warmup")

        rss_samples_gb = [bench.process_rss_gb(process.pid) or 0.0]
        outcomes: list[dict[str, Any]] = []
        run_started = time.perf_counter()
        metric_samples: list[dict[str, Any]] = []
        stop_sampling = threading.Event()
        metric_sampler = threading.Thread(
            target=sample_metrics_until,
            kwargs={
                "stop": stop_sampling,
                "base_url": base_url,
                "timeout": args.request_timeout,
                "interval": args.metrics_sample_interval,
                "started": run_started,
                "samples": metric_samples,
            },
            name="ax-fault-soak-metrics",
            daemon=True,
        )
        metric_sampler.start()
        try:
            for specs in rounds:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.concurrency
                ) as executor:
                    futures = []
                    for spec in specs:
                        if spec.kind == "stalled":
                            future = executor.submit(
                                run_stalled_request,
                                spec,
                                base_url=base_url,
                                model_id=args.model_id,
                                timeout=args.request_timeout,
                                hold_s=args.stalled_hold_ms / 1000.0,
                            )
                        else:
                            future = executor.submit(
                                run_fault_request,
                                spec,
                                base_url=base_url,
                                model_id=args.model_id,
                                timeout=args.request_timeout,
                                disconnect_after_output_events=args.disconnect_after_output_events,
                                slow_delay_s=args.slow_delay_ms / 1000.0,
                            )
                        futures.append(future)
                    for future in concurrent.futures.as_completed(futures):
                        outcomes.append(future.result())
                rss_samples_gb.append(bench.process_rss_gb(process.pid) or 0.0)
        finally:
            stop_sampling.set()
            metric_sampler.join(timeout=5.0)

        quiescent, after_metrics = wait_for_quiescence(
            base_url,
            timeout=args.cleanup_timeout,
        )
        health = request_json(f"{base_url}/health", timeout=5.0)
        counter_deltas = {
            SATURATION_COUNTER: metric_delta(
                before_metrics, after_metrics, SATURATION_COUNTER
            ),
            BACKLOG_OVERFLOW_COUNTER: metric_delta(
                before_metrics, after_metrics, BACKLOG_OVERFLOW_COUNTER
            ),
        }
        metric_peaks = summarize_metric_peaks(metric_samples)
        verdict, failure_reasons = evaluate_run(
            outcomes,
            quiescent=quiescent,
            counter_deltas=counter_deltas,
            metric_peaks=metric_peaks,
            require_backpressure=args.require_backpressure,
            min_buffered_events=args.min_buffered_events,
        )
        dirty_status = tracked_dirty_status()
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "started_at": started_at,
            "provenance": {
                "runner": "scripts/run_native_generation_fault_soak.py",
                "repo_revision": repo_revision(),
                "server_binary": str(bench.AX_ENGINE_SERVER),
                "git_tracked_dirty": bool(dirty_status),
                "git_tracked_status": dirty_status,
            },
            "model": {"id": args.model_id, "model_dir": str(model_dir)},
            "workload": {
                "rounds": args.rounds,
                "concurrency": args.concurrency,
                "normal_per_round": args.normal_per_round,
                "disconnect_per_round": args.disconnect_per_round,
                "slow_per_round": args.slow_per_round,
                "stalled_per_round": args.stalled_per_round,
                "prompt_tokens": args.prompt_tokens,
                "normal_output_tokens": args.normal_output_tokens,
                "fault_output_tokens": args.fault_output_tokens,
                "disconnect_after_output_events": args.disconnect_after_output_events,
                "slow_delay_ms": args.slow_delay_ms,
                "stalled_hold_ms": args.stalled_hold_ms,
                "metrics_sample_interval": args.metrics_sample_interval,
                "require_backpressure": args.require_backpressure,
                "min_buffered_events": args.min_buffered_events,
                "warmup_requests": args.warmup_requests,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(base_tokens, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
            "elapsed_seconds": time.perf_counter() - run_started,
            "health_after": health,
            "metrics_before": before_metrics,
            "metrics_after": after_metrics,
            "metric_samples": metric_samples,
            "metric_peaks": metric_peaks,
            "counter_deltas": counter_deltas,
            "quiescent": quiescent,
            "rss_gb": {
                "samples": rss_samples_gb,
                "start": rss_samples_gb[0],
                "end": rss_samples_gb[-1],
                "peak": max(rss_samples_gb),
                "delta": rss_samples_gb[-1] - rss_samples_gb[0],
            },
            "outcomes": sorted(outcomes, key=lambda item: item["request_id"]),
            "verdict": verdict,
            "failure_reasons": failure_reasons,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(f"Native generation fault-soak artifact: {args.output}")
        print(f"Verdict: {verdict}")
        if failure_reasons:
            for reason in failure_reasons:
                print(f"  - {reason}")
        return 0 if verdict == "pass" else 1
    finally:
        bench.kill_proc(process)


def main() -> None:
    raise SystemExit(main_with_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
