#!/usr/bin/env python3
"""Validate concurrent MLX prefill latency artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ax.mlx_concurrent_prefill.v1"
PROMPT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
OVERLAP_CLASSIFICATIONS = {"serialized", "partial_overlap", "overlapped"}
SCHEDULER_EVIDENCE_KEYS = {
    "scheduled_prefill_tokens",
    "scheduled_decode_tokens",
    "skipped_prefill_tokens",
    "skipped_decode_tokens",
    "mixed_prefill_decode_batches",
}


class ConcurrentPrefillArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConcurrentPrefillRow:
    artifact_path: Path
    concurrent_requests: int
    context_tokens: int
    generation_tokens: int
    request_ttft_ms_median: float
    total_wall_ms_median: float
    peak_memory_gb_max: float
    payload: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ConcurrentPrefillArtifactError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ConcurrentPrefillArtifactError(f"{path} must contain a JSON object")
    return payload


def require_mapping(payload: dict[str, Any], key: str, *, owner: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ConcurrentPrefillArtifactError(f"{owner} lacks object field {key!r}")
    return value


def require_non_empty_str(payload: dict[str, Any], key: str, *, owner: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConcurrentPrefillArtifactError(f"{owner} lacks non-empty string field {key!r}")
    return value


def require_positive_int(payload: dict[str, Any], key: str, *, owner: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ConcurrentPrefillArtifactError(f"{owner} lacks positive integer field {key!r}")
    return value


def require_non_negative_float(payload: dict[str, Any], key: str, *, owner: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ConcurrentPrefillArtifactError(f"{owner} lacks non-negative numeric field {key!r}")
    return float(value)


def require_positive_float(payload: dict[str, Any], key: str, *, owner: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ConcurrentPrefillArtifactError(f"{owner} lacks positive numeric field {key!r}")
    return float(value)


def require_metric(
    payload: dict[str, Any],
    key: str,
    required_stat: str,
    *,
    owner: str,
    positive: bool = True,
) -> float:
    metric = payload.get(key)
    if not isinstance(metric, dict):
        raise ConcurrentPrefillArtifactError(f"{owner} lacks metric object {key!r}")
    if positive:
        return require_positive_float(metric, required_stat, owner=f"{owner}.{key}")
    return require_non_negative_float(metric, required_stat, owner=f"{owner}.{key}")


def validate_top_level(path: Path, artifact: dict[str, Any]) -> tuple[int, int]:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ConcurrentPrefillArtifactError(
            f"{path} has schema_version={artifact.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )

    model = require_mapping(artifact, "model", owner=str(path))
    require_non_empty_str(model, "id", owner=f"{path}.model")

    host = require_mapping(artifact, "host", owner=str(path))
    require_non_empty_str(host, "chip", owner=f"{path}.host")

    benchmark = require_mapping(artifact, "benchmark", owner=str(path))
    if require_positive_int(benchmark, "batch_size", owner=f"{path}.benchmark") != 1:
        raise ConcurrentPrefillArtifactError(f"{path}.benchmark.batch_size must be 1")
    repetitions = require_positive_int(benchmark, "repetitions", owner=f"{path}.benchmark")
    if repetitions < 3:
        raise ConcurrentPrefillArtifactError(f"{path}.benchmark.repetitions must be at least 3")
    context_tokens = require_positive_int(benchmark, "context_tokens", owner=f"{path}.benchmark")
    generation_tokens = require_positive_int(
        benchmark, "generation_tokens", owner=f"{path}.benchmark"
    )
    require_non_negative_float(benchmark, "temperature", owner=f"{path}.benchmark")
    return context_tokens, generation_tokens


def validate_prompt_hashes(
    row: dict[str, Any],
    *,
    owner: str,
    concurrent_requests: int,
) -> None:
    hashes = row.get("prompt_token_ids_sha256")
    if not isinstance(hashes, list):
        raise ConcurrentPrefillArtifactError(f"{owner} lacks prompt_token_ids_sha256 list")
    if len(hashes) != concurrent_requests:
        raise ConcurrentPrefillArtifactError(
            f"{owner}.prompt_token_ids_sha256 must contain one hash per request"
        )
    for index, prompt_hash in enumerate(hashes):
        if not isinstance(prompt_hash, str) or PROMPT_HASH_RE.fullmatch(prompt_hash) is None:
            raise ConcurrentPrefillArtifactError(
                f"{owner}.prompt_token_ids_sha256[{index}] is not a valid sha256"
            )


def parse_row(
    path: Path,
    row: dict[str, Any],
    index: int,
    *,
    expected_context_tokens: int,
    expected_generation_tokens: int,
) -> ConcurrentPrefillRow:
    owner = f"{path}.rows[{index}]"
    if require_non_empty_str(row, "engine", owner=owner) != "ax_engine_mlx":
        raise ConcurrentPrefillArtifactError(f"{owner}.engine must be ax_engine_mlx")
    if row.get("ax_decode_policy") != "direct_no_ngram_acceleration":
        raise ConcurrentPrefillArtifactError(
            f"{owner} must use direct_no_ngram_acceleration"
        )
    route = require_mapping(row, "route", owner=owner)
    require_non_empty_str(route, "selected_backend", owner=f"{owner}.route")

    concurrent_requests = require_positive_int(row, "concurrent_requests", owner=owner)
    context_tokens = require_positive_int(row, "context_tokens", owner=owner)
    generation_tokens = require_positive_int(row, "generation_tokens", owner=owner)
    if context_tokens != expected_context_tokens:
        raise ConcurrentPrefillArtifactError(f"{owner}.context_tokens does not match benchmark")
    if generation_tokens != expected_generation_tokens:
        raise ConcurrentPrefillArtifactError(f"{owner}.generation_tokens does not match benchmark")
    validate_prompt_hashes(row, owner=owner, concurrent_requests=concurrent_requests)

    repetitions = require_positive_int(row, "repetitions", owner=owner)
    if repetitions < 3:
        raise ConcurrentPrefillArtifactError(f"{owner}.repetitions must be at least 3")

    require_metric(row, "request_ttft_ms", "median", owner=owner)
    require_metric(row, "request_ttft_ms", "p75", owner=owner)
    require_metric(row, "total_wall_ms", "median", owner=owner)
    require_metric(row, "queue_delay_ms", "median", owner=owner, positive=False)
    require_metric(row, "failure_count", "max", owner=owner, positive=False)
    if require_metric(row, "failure_count", "max", owner=owner, positive=False) != 0.0:
        raise ConcurrentPrefillArtifactError(f"{owner}.failure_count.max must be 0")
    peak_memory_gb_max = require_metric(row, "peak_memory_gb", "max", owner=owner)

    overlap = require_mapping(row, "prefill_overlap", owner=owner)
    classification = require_non_empty_str(overlap, "classification", owner=f"{owner}.prefill_overlap")
    if classification not in OVERLAP_CLASSIFICATIONS:
        raise ConcurrentPrefillArtifactError(
            f"{owner}.prefill_overlap.classification must be one of {sorted(OVERLAP_CLASSIFICATIONS)}"
        )
    require_metric(overlap, "overlap_efficiency", "median", owner=f"{owner}.prefill_overlap", positive=False)
    scheduler_evidence = require_mapping(row, "scheduler_evidence", owner=owner)
    for key in sorted(SCHEDULER_EVIDENCE_KEYS):
        value = scheduler_evidence.get(key)
        if not isinstance(value, int) or value < 0:
            raise ConcurrentPrefillArtifactError(
                f"{owner}.scheduler_evidence.{key} must be a non-negative integer"
            )

    return ConcurrentPrefillRow(
        artifact_path=path,
        concurrent_requests=concurrent_requests,
        context_tokens=context_tokens,
        generation_tokens=generation_tokens,
        request_ttft_ms_median=require_metric(row, "request_ttft_ms", "median", owner=owner),
        total_wall_ms_median=require_metric(row, "total_wall_ms", "median", owner=owner),
        peak_memory_gb_max=peak_memory_gb_max,
        payload=row,
    )


def assert_ratio_matches(
    *,
    row: ConcurrentPrefillRow,
    single_row: ConcurrentPrefillRow,
    ratio_key: str,
    expected: float,
    tolerance: float,
) -> None:
    ratios = row.payload.get("ratios_to_single_request")
    if not isinstance(ratios, dict):
        raise ConcurrentPrefillArtifactError(
            f"{row.artifact_path} concurrency={row.concurrent_requests} lacks ratios_to_single_request"
        )
    value = ratios.get(ratio_key)
    if not isinstance(value, (int, float)):
        raise ConcurrentPrefillArtifactError(
            f"{row.artifact_path} concurrency={row.concurrent_requests} "
            f"lacks numeric ratios_to_single_request.{ratio_key}"
        )
    if abs(float(value) - expected) > tolerance:
        raise ConcurrentPrefillArtifactError(
            f"{row.artifact_path} stale {ratio_key} ratio for "
            f"concurrency={row.concurrent_requests}: artifact={float(value):.6f} "
            f"expected={expected:.6f} against concurrency=1"
        )


def validate_concurrency_coverage(
    path: Path,
    rows: list[ConcurrentPrefillRow],
    *,
    min_concurrency_levels: int,
    min_max_concurrent_requests: int,
    ratio_tolerance: float,
) -> None:
    by_concurrency: dict[int, ConcurrentPrefillRow] = {}
    for row in rows:
        if row.concurrent_requests in by_concurrency:
            raise ConcurrentPrefillArtifactError(
                f"{path} has duplicate concurrency={row.concurrent_requests} row"
            )
        by_concurrency[row.concurrent_requests] = row

    if 1 not in by_concurrency:
        raise ConcurrentPrefillArtifactError(f"{path} lacks required concurrency=1 baseline")
    if len(by_concurrency) < min_concurrency_levels:
        raise ConcurrentPrefillArtifactError(
            f"{path} has {len(by_concurrency)} concurrency levels, expected at least "
            f"{min_concurrency_levels}"
        )
    if max(by_concurrency) < min_max_concurrent_requests:
        raise ConcurrentPrefillArtifactError(
            f"{path} largest concurrency={max(by_concurrency)}, expected at least "
            f"{min_max_concurrent_requests}"
        )

    single_row = by_concurrency[1]
    for concurrency, row in sorted(by_concurrency.items()):
        if concurrency == 1:
            continue
        assert_ratio_matches(
            row=row,
            single_row=single_row,
            ratio_key="request_ttft_ms",
            expected=row.request_ttft_ms_median / single_row.request_ttft_ms_median,
            tolerance=ratio_tolerance,
        )
        assert_ratio_matches(
            row=row,
            single_row=single_row,
            ratio_key="total_wall_ms",
            expected=row.total_wall_ms_median / single_row.total_wall_ms_median,
            tolerance=ratio_tolerance,
        )
        assert_ratio_matches(
            row=row,
            single_row=single_row,
            ratio_key="peak_memory_gb",
            expected=row.peak_memory_gb_max / single_row.peak_memory_gb_max,
            tolerance=ratio_tolerance,
        )


def validate_mlx_concurrent_prefill_artifact(
    path: Path,
    *,
    min_concurrency_levels: int = 2,
    min_max_concurrent_requests: int = 2,
    ratio_tolerance: float = 0.0005,
) -> list[str]:
    artifact = load_json(path)
    context_tokens, generation_tokens = validate_top_level(path, artifact)
    rows_payload = artifact.get("rows")
    if not isinstance(rows_payload, list) or not rows_payload:
        raise ConcurrentPrefillArtifactError(f"{path} lacks non-empty rows array")

    rows = [
        parse_row(
            path,
            row,
            index,
            expected_context_tokens=context_tokens,
            expected_generation_tokens=generation_tokens,
        )
        for index, row in enumerate(rows_payload)
        if isinstance(row, dict)
    ]
    if len(rows) != len(rows_payload):
        raise ConcurrentPrefillArtifactError(f"{path}.rows must contain only objects")

    validate_concurrency_coverage(
        path,
        rows,
        min_concurrency_levels=min_concurrency_levels,
        min_max_concurrent_requests=min_max_concurrent_requests,
        ratio_tolerance=ratio_tolerance,
    )
    return [f"concurrency={row.concurrent_requests}" for row in sorted(rows, key=lambda item: item.concurrent_requests)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--min-concurrency-levels", type=int, default=2)
    parser.add_argument("--min-max-concurrent-requests", type=int, default=2)
    parser.add_argument("--ratio-tolerance", type=float, default=0.0005)
    args = parser.parse_args(argv)

    try:
        checked = validate_mlx_concurrent_prefill_artifact(
            args.artifact,
            min_concurrency_levels=args.min_concurrency_levels,
            min_max_concurrent_requests=args.min_max_concurrent_requests,
            ratio_tolerance=args.ratio_tolerance,
        )
    except ConcurrentPrefillArtifactError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"validated {args.artifact} "
        f"({SCHEMA_VERSION}; {', '.join(checked)})"
    )
    return 0


def main_with_args_for_test(args: list[str]) -> int:
    return main(args)


if __name__ == "__main__":
    raise SystemExit(main())
