#!/usr/bin/env python3
"""Capture P2 MLX startup and concurrent-prefill latency artifacts.

This runner is intentionally conservative: it measures AX server-path latency
with direct AX MLX policy and emits artifacts that are immediately validated by
the fail-closed P2 checkers. It does not compare against mlx_lm.benchmark and
does not make raw kernel-throughput claims.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bench_mlx_inference_stack as bench
from check_mlx_concurrent_prefill_artifact import (
    validate_mlx_concurrent_prefill_artifact,
)
from check_mlx_startup_latency_artifact import validate_mlx_startup_latency_artifact
from render_mlx_p2_latency_report import render_report


STARTUP_SCHEMA_VERSION = "ax.mlx_startup_latency.v1"
CONCURRENT_SCHEMA_VERSION = "ax.mlx_concurrent_prefill.v1"


@dataclass(frozen=True)
class PromptDoc:
    prompt_tokens: int
    generation_tokens: int
    vocab_size: int
    token_ids: list[int]
    token_ids_sha256: str
    token_ids_path: str


def metric(values: list[float]) -> dict[str, float]:
    if not values:
        raise RuntimeError("cannot summarize an empty metric")
    sorted_values = sorted(float(value) for value in values)
    return {
        "mean": sum(sorted_values) / len(sorted_values),
        "median": statistics.median(sorted_values),
        "p75": percentile(sorted_values, 0.75),
        "min": min(sorted_values),
        "max": max(sorted_values),
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def ratio(value: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise RuntimeError("ratio baseline must be positive")
    return value / baseline


def repo_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(bench.REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def build_prompt_docs(
    *,
    model_dir: Path,
    artifact_root: Path,
    prompt_tokens: int,
    generation_tokens: int,
    request_count: int,
) -> list[PromptDoc]:
    vocab_size = bench.model_vocab_size(model_dir)
    base_tokens = bench.mlx_lm_reference_prompt_tokens(vocab_size, prompt_tokens)
    docs: list[PromptDoc] = []
    for index in range(request_count):
        tokens = [int((token + index) % vocab_size) for token in base_tokens]
        doc = bench.write_prompt_tokens(
            artifact_root,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            vocab_size=vocab_size,
            tokens=tokens,
        )
        docs.append(
            PromptDoc(
                prompt_tokens=int(doc["prompt_tokens"]),
                generation_tokens=int(doc["generation_tokens"]),
                vocab_size=int(doc["vocab_size"]),
                token_ids=tokens,
                token_ids_sha256=str(doc["token_ids_sha256"]),
                token_ids_path=str(doc["token_ids_path"]),
            )
        )
    return docs


def common_artifact_metadata(
    *,
    model_id: str,
    model_dir: Path,
    model_metadata: dict[str, Any],
    host_label: str,
    prompt_tokens: int,
    generation_tokens: int,
    repetitions: int,
) -> dict[str, Any]:
    return {
        "model": {
            "id": model_id,
            "model_dir": str(model_dir),
            **{
                key: value
                for key, value in model_metadata.items()
                if key in {"model_type", "model_family", "quantization"}
            },
        },
        "host": {
            "chip": host_label,
            "source": "operator_provided_or_default",
        },
        "benchmark": {
            "batch_size": 1,
            "temperature": 0.0,
            "context_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
            "repetitions": repetitions,
        },
        "provenance": {
            "runner": "scripts/run_mlx_p2_latency_artifacts.py",
            "repo_revision": repo_revision(),
            "server_binary": str(bench.AX_ENGINE_SERVER),
        },
    }


def start_direct_server(model_dir: Path, port: int) -> tuple[subprocess.Popen[Any], float, float]:
    started = time.perf_counter()
    proc = bench.start_axengine(
        bench.AX_ENGINE_SERVER,
        model_dir,
        port,
        direct_mode=True,
    )
    process_spawn_ms = (time.perf_counter() - started) * 1000.0
    if not bench.wait_for_server(f"http://127.0.0.1:{port}/health", proc=proc):
        stderr = bench.process_stderr_snapshot(proc)
        bench.kill_proc(proc)
        raise RuntimeError(f"ax-engine-server did not become ready:\n{stderr}")
    server_ready_ms = (time.perf_counter() - started) * 1000.0
    return proc, process_spawn_ms, server_ready_ms


def run_one_request(port: int, prompt: PromptDoc, server_pid: int | None) -> dict[str, Any]:
    started = time.perf_counter()
    run = bench.axengine_one_run(
        port,
        prompt.token_ids,
        prompt.generation_tokens,
        server_pid=server_pid,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "ttft_ms": float(run["ttft_ms"]),
        "decode_tok_s": float(run["decode_tok_s"]),
        "wall_ms": wall_ms,
        "peak_memory_gb": float(run.get("peak_memory_gb", 0.0)),
        "scheduler_telemetry": run.get("scheduler_telemetry", {}),
    }


def startup_phase_row(
    *,
    phase: str,
    prompt: PromptDoc,
    observations: list[dict[str, float]],
    process_start_ms: list[float] | None = None,
    server_ready_ms: list[float] | None = None,
    model_load_ms: list[float] | None = None,
    ratios_to_benchmark_warm: dict[str, float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "engine": "ax_engine_mlx",
        "method": "server_sse_runner_time_us",
        "timing_scope": "ax_engine_runner_time_us",
        "ax_decode_policy": "direct_no_ngram_acceleration",
        "route": {"selected_backend": "mlx"},
        "context_tokens": prompt.prompt_tokens,
        "generation_tokens": prompt.generation_tokens,
        "prompt_token_ids_sha256": prompt.token_ids_sha256,
        "repetitions": len(observations),
        "ttft_ms": metric([item["ttft_ms"] for item in observations]),
        "first_request_ttft_ms": metric([item["ttft_ms"] for item in observations])
        if phase == "process_cold"
        else None,
        "decode_tok_s": metric([item["decode_tok_s"] for item in observations]),
        "peak_memory_gb": metric([item["peak_memory_gb"] for item in observations]),
        "trials": observations,
    }
    if process_start_ms is not None:
        row["process_start_ms"] = metric(process_start_ms)
    if server_ready_ms is not None:
        row["server_ready_ms"] = metric(server_ready_ms)
    if model_load_ms is not None:
        row["model_load_ms"] = metric(model_load_ms)
        row["model_load_source"] = "server_ready_elapsed_minus_process_spawn_elapsed"
    if ratios_to_benchmark_warm is not None:
        row["ratios_to_benchmark_warm"] = ratios_to_benchmark_warm
    return {key: value for key, value in row.items() if value is not None}


def capture_startup_artifact(
    *,
    model_dir: Path,
    model_id: str,
    model_metadata: dict[str, Any],
    host_label: str,
    port: int,
    prompt: PromptDoc,
    repetitions: int,
    cooldown: float,
) -> dict[str, Any]:
    process_observations: list[dict[str, float]] = []
    process_start_ms: list[float] = []
    server_ready_ms: list[float] = []
    model_load_ms: list[float] = []

    for index in range(repetitions):
        proc, spawn_ms, ready_ms = start_direct_server(model_dir, port)
        try:
            process_start_ms.append(max(spawn_ms, 0.001))
            server_ready_ms.append(ready_ms)
            model_load_ms.append(max(ready_ms - spawn_ms, 0.001))
            process_observations.append(run_one_request(port, prompt, proc.pid))
        finally:
            bench.kill_proc(proc)
        if cooldown > 0 and index < repetitions - 1:
            time.sleep(cooldown)

    model_warm_observations: list[dict[str, float]] = []
    model_warm_load_ms: list[float] = []
    for index in range(repetitions):
        proc, spawn_ms, ready_ms = start_direct_server(model_dir, port)
        try:
            bench.axengine_one_run(port, prompt.token_ids, prompt.generation_tokens, server_pid=proc.pid)
            model_warm_load_ms.append(max(ready_ms - spawn_ms, 0.001))
            model_warm_observations.append(run_one_request(port, prompt, proc.pid))
        finally:
            bench.kill_proc(proc)
        if cooldown > 0 and index < repetitions - 1:
            time.sleep(cooldown)

    proc, _spawn_ms, _ready_ms = start_direct_server(model_dir, port)
    try:
        bench.axengine_one_run(port, prompt.token_ids, prompt.generation_tokens, server_pid=proc.pid)
        benchmark_warm_observations = [
            run_one_request(port, prompt, proc.pid) for _ in range(repetitions)
        ]
    finally:
        bench.kill_proc(proc)

    warm_ttft = statistics.median(item["ttft_ms"] for item in benchmark_warm_observations)
    warm_decode = statistics.median(item["decode_tok_s"] for item in benchmark_warm_observations)

    process_row = startup_phase_row(
        phase="process_cold",
        prompt=prompt,
        observations=process_observations,
        process_start_ms=process_start_ms,
        server_ready_ms=server_ready_ms,
        model_load_ms=model_load_ms,
        ratios_to_benchmark_warm={
            "ttft_ms": ratio(
                statistics.median(item["ttft_ms"] for item in process_observations),
                warm_ttft,
            ),
            "decode_tok_s": ratio(
                statistics.median(item["decode_tok_s"] for item in process_observations),
                warm_decode,
            ),
        },
    )
    model_warm_row = startup_phase_row(
        phase="model_warm",
        prompt=prompt,
        observations=model_warm_observations,
        model_load_ms=model_warm_load_ms,
        ratios_to_benchmark_warm={
            "ttft_ms": ratio(
                statistics.median(item["ttft_ms"] for item in model_warm_observations),
                warm_ttft,
            ),
            "decode_tok_s": ratio(
                statistics.median(item["decode_tok_s"] for item in model_warm_observations),
                warm_decode,
            ),
        },
    )
    benchmark_warm_row = startup_phase_row(
        phase="benchmark_warm",
        prompt=prompt,
        observations=benchmark_warm_observations,
    )

    artifact = {
        "schema_version": STARTUP_SCHEMA_VERSION,
        **common_artifact_metadata(
            model_id=model_id,
            model_dir=model_dir,
            model_metadata=model_metadata,
            host_label=host_label,
            prompt_tokens=prompt.prompt_tokens,
            generation_tokens=prompt.generation_tokens,
            repetitions=repetitions,
        ),
        "claim_scope": "cold_warm_startup_latency",
        "interpretation": (
            "process_cold measures fresh process startup through first request; "
            "model_warm and benchmark_warm are warmed server-path rows. Keep this "
            "separate from raw model throughput evidence."
        ),
        "benchmark": {
            **common_artifact_metadata(
                model_id=model_id,
                model_dir=model_dir,
                model_metadata=model_metadata,
                host_label=host_label,
                prompt_tokens=prompt.prompt_tokens,
                generation_tokens=prompt.generation_tokens,
                repetitions=repetitions,
            )["benchmark"],
            "prompt_token_ids_sha256": prompt.token_ids_sha256,
        },
        "prompt_artifacts": [prompt.token_ids_path],
        "rows": [process_row, model_warm_row, benchmark_warm_row],
    }
    return artifact


def run_concurrent_trial(
    *,
    port: int,
    prompts: list[PromptDoc],
    server_pid: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    failures = 0
    observations: list[dict[str, float]] = []

    def invoke(prompt: PromptDoc) -> dict[str, float]:
        return run_one_request(port, prompt, server_pid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(invoke, prompt) for prompt in prompts]
        for future in concurrent.futures.as_completed(futures):
            try:
                observations.append(future.result())
            except Exception:
                failures += 1
    total_wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "total_wall_ms": total_wall_ms,
        "failure_count": failures,
        "request_ttft_ms": statistics.median(
            item["ttft_ms"] for item in observations
        )
        if observations
        else 0.0,
        "queue_delay_ms": statistics.median(
            max(item["wall_ms"] - item["ttft_ms"], 0.0) for item in observations
        )
        if observations
        else 0.0,
        "peak_memory_gb": bench.process_rss_gb(server_pid) or 0.0,
        "observations": observations,
    }


def overlap_efficiency(
    *,
    concurrent_requests: int,
    single_request_total_wall_ms: float,
    total_wall_ms: float,
) -> float:
    if concurrent_requests <= 1:
        return 0.0
    serial_ms = single_request_total_wall_ms * concurrent_requests
    denominator = serial_ms - single_request_total_wall_ms
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (serial_ms - total_wall_ms) / denominator))


def overlap_classification(value: float) -> str:
    if value >= 0.66:
        return "overlapped"
    if value >= 0.15:
        return "partial_overlap"
    return "serialized"


def summarize_scheduler_evidence(trials: list[dict[str, Any]]) -> dict[str, int]:
    evidence = {
        "scheduled_prefill_tokens": 0,
        "scheduled_decode_tokens": 0,
        "skipped_prefill_tokens": 0,
        "skipped_decode_tokens": 0,
        "mixed_prefill_decode_batches": 0,
    }
    key_map = {
        "ax_scheduler_scheduled_prefill_tokens": "scheduled_prefill_tokens",
        "ax_scheduler_scheduled_decode_tokens": "scheduled_decode_tokens",
        "ax_scheduler_skipped_prefill_tokens": "skipped_prefill_tokens",
        "ax_scheduler_skipped_decode_tokens": "skipped_decode_tokens",
        "ax_scheduler_mixed_prefill_decode_batches": "mixed_prefill_decode_batches",
    }
    for trial in trials:
        for observation in trial.get("observations", []):
            telemetry = observation.get("scheduler_telemetry") or {}
            for source_key, evidence_key in key_map.items():
                evidence[evidence_key] += int(telemetry.get(source_key, 0))
    return evidence


def concurrent_row(
    *,
    prompts: list[PromptDoc],
    trials: list[dict[str, Any]],
    single_row: dict[str, Any] | None,
) -> dict[str, Any]:
    request_count = len(prompts)
    request_ttft_metric = metric([trial["request_ttft_ms"] for trial in trials])
    total_wall_metric = metric([trial["total_wall_ms"] for trial in trials])
    peak_memory_metric = metric([trial["peak_memory_gb"] for trial in trials])
    overlap_values = [
        overlap_efficiency(
            concurrent_requests=request_count,
            single_request_total_wall_ms=float(single_row["total_wall_ms"]["median"])
            if single_row
            else trial["total_wall_ms"],
            total_wall_ms=trial["total_wall_ms"],
        )
        for trial in trials
    ]
    row: dict[str, Any] = {
        "engine": "ax_engine_mlx",
        "method": "server_sse_runner_time_us_concurrent",
        "timing_scope": "end_to_end_server_path_plus_runner_ttft",
        "ax_decode_policy": "direct_no_ngram_acceleration",
        "route": {"selected_backend": "mlx"},
        "concurrent_requests": request_count,
        "context_tokens": prompts[0].prompt_tokens,
        "generation_tokens": prompts[0].generation_tokens,
        "prompt_token_ids_sha256": [prompt.token_ids_sha256 for prompt in prompts],
        "repetitions": len(trials),
        "request_ttft_ms": request_ttft_metric,
        "total_wall_ms": total_wall_metric,
        "queue_delay_ms": metric([trial["queue_delay_ms"] for trial in trials]),
        "failure_count": metric([float(trial["failure_count"]) for trial in trials]),
        "peak_memory_gb": peak_memory_metric,
        "prefill_overlap": {
            "classification": overlap_classification(statistics.median(overlap_values)),
            "overlap_efficiency": metric(overlap_values),
        },
        "scheduler_evidence": summarize_scheduler_evidence(trials),
        "trials": trials,
    }
    if single_row is not None:
        row["ratios_to_single_request"] = {
            "request_ttft_ms": ratio(
                request_ttft_metric["median"],
                float(single_row["request_ttft_ms"]["median"]),
            ),
            "total_wall_ms": ratio(
                total_wall_metric["median"],
                float(single_row["total_wall_ms"]["median"]),
            ),
            "peak_memory_gb": ratio(
                peak_memory_metric["max"],
                float(single_row["peak_memory_gb"]["max"]),
            ),
        }
    return row


def capture_concurrent_artifact(
    *,
    model_dir: Path,
    model_id: str,
    model_metadata: dict[str, Any],
    host_label: str,
    port: int,
    prompt_groups: dict[int, list[PromptDoc]],
    repetitions: int,
    cooldown: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    proc, _spawn_ms, _ready_ms = start_direct_server(model_dir, port)
    try:
        baseline_prompt = prompt_groups[1][0]
        bench.axengine_one_run(
            port,
            baseline_prompt.token_ids,
            baseline_prompt.generation_tokens,
            server_pid=proc.pid,
        )
        single_row: dict[str, Any] | None = None
        for concurrency in sorted(prompt_groups):
            trials = []
            for index in range(repetitions):
                trials.append(
                    run_concurrent_trial(
                        port=port,
                        prompts=prompt_groups[concurrency],
                        server_pid=proc.pid,
                    )
                )
                if cooldown > 0 and index < repetitions - 1:
                    time.sleep(cooldown)
            row = concurrent_row(
                prompts=prompt_groups[concurrency],
                trials=trials,
                single_row=single_row,
            )
            if concurrency == 1:
                single_row = row
            rows.append(row)
    finally:
        bench.kill_proc(proc)

    first_prompt = prompt_groups[1][0]
    artifact = {
        "schema_version": CONCURRENT_SCHEMA_VERSION,
        **common_artifact_metadata(
            model_id=model_id,
            model_dir=model_dir,
            model_metadata=model_metadata,
            host_label=host_label,
            prompt_tokens=first_prompt.prompt_tokens,
            generation_tokens=first_prompt.generation_tokens,
            repetitions=repetitions,
        ),
        "claim_scope": "concurrent_long_context_prefill",
        "interpretation": (
            "This is AX server-path concurrent request evidence. Keep it separate "
            "from batch=1 runner throughput and from continuous-batching claims."
        ),
        "prompt_artifacts": [
            prompt.token_ids_path
            for prompts in prompt_groups.values()
            for prompt in prompts
        ],
        "rows": rows,
    }
    return artifact


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_concurrency_levels(value: str) -> list[int]:
    levels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not levels:
        raise argparse.ArgumentTypeError("at least one concurrency level is required")
    if 1 not in levels:
        levels.insert(0, 1)
    if any(level <= 0 for level in levels):
        raise argparse.ArgumentTypeError("concurrency levels must be positive")
    return sorted(set(levels))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host-label", default="unknown_apple_silicon")
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--startup-generation-tokens", type=int, default=128)
    parser.add_argument("--concurrent-generation-tokens", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--axengine-port", type=int, default=8092)
    parser.add_argument(
        "--concurrency-levels",
        type=parse_concurrency_levels,
        default=parse_concurrency_levels("1,2,4"),
    )
    parser.add_argument("--skip-startup", action="store_true")
    parser.add_argument("--skip-concurrent", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    if not args.model_dir.is_dir():
        parser.error(f"--model-dir does not exist: {args.model_dir}")
    if not bench.AX_ENGINE_SERVER.exists() and not args.dry_run:
        parser.error(
            f"ax-engine-server not found at {bench.AX_ENGINE_SERVER}; "
            "run cargo build -p ax-engine-server --release"
        )
    if args.skip_startup and args.skip_concurrent:
        parser.error("at least one artifact kind must be enabled")

    model_id = args.model_id or str(args.model_dir)
    model_metadata = bench.collect_model_metadata(args.model_dir)

    startup_output = args.output_dir / "startup-latency.json"
    concurrent_output = args.output_dir / "concurrent-prefill.json"
    report_output = args.output_dir / "p2-latency.md"
    print(f"Output directory: {args.output_dir}")
    if not args.skip_startup:
        print(f"Startup artifact: {startup_output}")
    if not args.skip_concurrent:
        print(f"Concurrent artifact: {concurrent_output}")
    print(f"Markdown report: {report_output}")

    if args.dry_run:
        print("Dry run only; no server will be started.")
        return 0

    prompt_root = args.output_dir / "prompts"
    if not args.skip_startup:
        startup_prompt = build_prompt_docs(
            model_dir=args.model_dir,
            artifact_root=prompt_root / "startup",
            prompt_tokens=args.context_tokens,
            generation_tokens=args.startup_generation_tokens,
            request_count=1,
        )[0]
        startup_artifact = capture_startup_artifact(
            model_dir=args.model_dir,
            model_id=model_id,
            model_metadata=model_metadata,
            host_label=args.host_label,
            port=args.axengine_port,
            prompt=startup_prompt,
            repetitions=args.repetitions,
            cooldown=args.cooldown,
        )
        write_json(startup_output, startup_artifact)
        validate_mlx_startup_latency_artifact(startup_output)

    if not args.skip_concurrent:
        max_concurrency = max(args.concurrency_levels)
        concurrent_prompts = build_prompt_docs(
            model_dir=args.model_dir,
            artifact_root=prompt_root / "concurrent",
            prompt_tokens=args.context_tokens,
            generation_tokens=args.concurrent_generation_tokens,
            request_count=max_concurrency,
        )
        prompt_groups = {
            level: concurrent_prompts[:level] for level in args.concurrency_levels
        }
        concurrent_artifact = capture_concurrent_artifact(
            model_dir=args.model_dir,
            model_id=model_id,
            model_metadata=model_metadata,
            host_label=args.host_label,
            port=args.axengine_port,
            prompt_groups=prompt_groups,
            repetitions=args.repetitions,
            cooldown=args.cooldown,
        )
        write_json(concurrent_output, concurrent_artifact)
        validate_mlx_concurrent_prefill_artifact(
            concurrent_output,
            min_concurrency_levels=min(2, len(args.concurrency_levels)),
            min_max_concurrent_requests=max(args.concurrency_levels),
        )

    report = render_report(
        startup_artifact=None if args.skip_startup else startup_output,
        concurrent_artifact=None if args.skip_concurrent else concurrent_output,
    )
    report_output.write_text(report + "\n")
    print(f"P2 latency report: {report_output}")

    return 0


def main_with_args_for_test(args: list[str]) -> int:
    return main(args)


if __name__ == "__main__":
    raise SystemExit(main())
