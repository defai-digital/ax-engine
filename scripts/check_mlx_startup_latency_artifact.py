#!/usr/bin/env python3
"""Validate cold-vs-warm MLX startup latency artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ax.mlx_startup_latency.v1"
REQUIRED_PHASES = ["process_cold", "model_warm", "benchmark_warm"]
PROMPT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class StartupLatencyArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class StartupRow:
    artifact_path: Path
    phase: str
    prompt_hash: str
    context_tokens: int
    generation_tokens: int
    ttft_ms_median: float
    decode_tok_s_median: float
    payload: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise StartupLatencyArtifactError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise StartupLatencyArtifactError(f"{path} must contain a JSON object")
    return payload


def require_mapping(payload: dict[str, Any], key: str, *, owner: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise StartupLatencyArtifactError(f"{owner} lacks object field {key!r}")
    return value


def require_non_empty_str(payload: dict[str, Any], key: str, *, owner: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise StartupLatencyArtifactError(f"{owner} lacks non-empty string field {key!r}")
    return value


def require_positive_int(payload: dict[str, Any], key: str, *, owner: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value <= 0:
        raise StartupLatencyArtifactError(f"{owner} lacks positive integer field {key!r}")
    return value


def require_non_negative_float(payload: dict[str, Any], key: str, *, owner: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise StartupLatencyArtifactError(f"{owner} lacks non-negative numeric field {key!r}")
    return float(value)


def require_positive_float(payload: dict[str, Any], key: str, *, owner: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise StartupLatencyArtifactError(f"{owner} lacks positive numeric field {key!r}")
    return float(value)


def require_metric(
    payload: dict[str, Any],
    key: str,
    required_stat: str,
    *,
    owner: str,
) -> float:
    metric = payload.get(key)
    if not isinstance(metric, dict):
        raise StartupLatencyArtifactError(f"{owner} lacks metric object {key!r}")
    return require_positive_float(metric, required_stat, owner=f"{owner}.{key}")


def validate_top_level(path: Path, artifact: dict[str, Any]) -> tuple[int, int, str]:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise StartupLatencyArtifactError(
            f"{path} has schema_version={artifact.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )

    model = require_mapping(artifact, "model", owner=str(path))
    require_non_empty_str(model, "id", owner=f"{path}.model")

    host = require_mapping(artifact, "host", owner=str(path))
    require_non_empty_str(host, "chip", owner=f"{path}.host")

    benchmark = require_mapping(artifact, "benchmark", owner=str(path))
    if require_positive_int(benchmark, "batch_size", owner=f"{path}.benchmark") != 1:
        raise StartupLatencyArtifactError(f"{path}.benchmark.batch_size must be 1")
    require_non_negative_float(benchmark, "temperature", owner=f"{path}.benchmark")
    repetitions = require_positive_int(benchmark, "repetitions", owner=f"{path}.benchmark")
    if repetitions < 3:
        raise StartupLatencyArtifactError(f"{path}.benchmark.repetitions must be at least 3")

    context_tokens = require_positive_int(benchmark, "context_tokens", owner=f"{path}.benchmark")
    generation_tokens = require_positive_int(
        benchmark, "generation_tokens", owner=f"{path}.benchmark"
    )
    prompt_hash = require_non_empty_str(
        benchmark, "prompt_token_ids_sha256", owner=f"{path}.benchmark"
    )
    if PROMPT_HASH_RE.fullmatch(prompt_hash) is None:
        raise StartupLatencyArtifactError(
            f"{path}.benchmark has invalid prompt_token_ids_sha256"
        )
    return context_tokens, generation_tokens, prompt_hash


def parse_row(
    path: Path,
    row: dict[str, Any],
    index: int,
    *,
    expected_context_tokens: int,
    expected_generation_tokens: int,
    expected_prompt_hash: str,
) -> StartupRow:
    owner = f"{path}.rows[{index}]"
    phase = require_non_empty_str(row, "phase", owner=owner)
    if phase not in REQUIRED_PHASES:
        raise StartupLatencyArtifactError(f"{owner} has unsupported phase {phase!r}")

    if require_non_empty_str(row, "engine", owner=owner) != "ax_engine_mlx":
        raise StartupLatencyArtifactError(f"{owner}.engine must be ax_engine_mlx")
    if row.get("ax_decode_policy") != "direct_no_ngram_acceleration":
        raise StartupLatencyArtifactError(
            f"{owner} must use direct_no_ngram_acceleration"
        )
    route = require_mapping(row, "route", owner=owner)
    require_non_empty_str(route, "selected_backend", owner=f"{owner}.route")

    context_tokens = require_positive_int(row, "context_tokens", owner=owner)
    generation_tokens = require_positive_int(row, "generation_tokens", owner=owner)
    if context_tokens != expected_context_tokens:
        raise StartupLatencyArtifactError(f"{owner}.context_tokens does not match benchmark")
    if generation_tokens != expected_generation_tokens:
        raise StartupLatencyArtifactError(f"{owner}.generation_tokens does not match benchmark")

    prompt_hash = require_non_empty_str(row, "prompt_token_ids_sha256", owner=owner)
    if prompt_hash != expected_prompt_hash:
        raise StartupLatencyArtifactError(f"{owner} does not reuse benchmark prompt hash")

    repetitions = require_positive_int(row, "repetitions", owner=owner)
    if repetitions < 3:
        raise StartupLatencyArtifactError(f"{owner}.repetitions must be at least 3")

    if phase == "process_cold":
        require_metric(row, "process_start_ms", "median", owner=owner)
        require_metric(row, "server_ready_ms", "median", owner=owner)
        require_metric(row, "model_load_ms", "median", owner=owner)
        require_metric(row, "first_request_ttft_ms", "median", owner=owner)
    elif phase == "model_warm":
        require_metric(row, "model_load_ms", "median", owner=owner)
    else:
        if "process_start_ms" in row or "model_load_ms" in row:
            raise StartupLatencyArtifactError(
                f"{owner} benchmark_warm must not mix load/startup metrics into warm rows"
            )

    require_metric(row, "ttft_ms", "median", owner=owner)
    require_metric(row, "decode_tok_s", "median", owner=owner)
    require_metric(row, "peak_memory_gb", "max", owner=owner)

    return StartupRow(
        artifact_path=path,
        phase=phase,
        prompt_hash=prompt_hash,
        context_tokens=context_tokens,
        generation_tokens=generation_tokens,
        ttft_ms_median=require_metric(row, "ttft_ms", "median", owner=owner),
        decode_tok_s_median=require_metric(row, "decode_tok_s", "median", owner=owner),
        payload=row,
    )


def assert_ratio_matches(
    *,
    row: StartupRow,
    warm_row: StartupRow,
    ratio_key: str,
    expected: float,
    tolerance: float,
) -> None:
    ratios = row.payload.get("ratios_to_benchmark_warm")
    if not isinstance(ratios, dict):
        raise StartupLatencyArtifactError(
            f"{row.artifact_path} {row.phase} lacks ratios_to_benchmark_warm"
        )
    value = ratios.get(ratio_key)
    if not isinstance(value, (int, float)):
        raise StartupLatencyArtifactError(
            f"{row.artifact_path} {row.phase} lacks numeric ratios_to_benchmark_warm.{ratio_key}"
        )
    if abs(float(value) - expected) > tolerance:
        raise StartupLatencyArtifactError(
            f"{row.artifact_path} stale {ratio_key} ratio for {row.phase}: "
            f"artifact={float(value):.6f} expected={expected:.6f} "
            f"against benchmark_warm"
        )


def validate_phase_set(
    path: Path,
    rows: list[StartupRow],
    *,
    ratio_tolerance: float,
) -> None:
    by_phase: dict[str, StartupRow] = {}
    for row in rows:
        if row.phase in by_phase:
            raise StartupLatencyArtifactError(f"{path} has duplicate {row.phase} row")
        by_phase[row.phase] = row

    missing = [phase for phase in REQUIRED_PHASES if phase not in by_phase]
    if missing:
        raise StartupLatencyArtifactError(f"{path} lacks required startup phases: {missing}")

    warm_row = by_phase["benchmark_warm"]
    for phase in ("process_cold", "model_warm"):
        row = by_phase[phase]
        assert_ratio_matches(
            row=row,
            warm_row=warm_row,
            ratio_key="ttft_ms",
            expected=row.ttft_ms_median / warm_row.ttft_ms_median,
            tolerance=ratio_tolerance,
        )
        assert_ratio_matches(
            row=row,
            warm_row=warm_row,
            ratio_key="decode_tok_s",
            expected=row.decode_tok_s_median / warm_row.decode_tok_s_median,
            tolerance=ratio_tolerance,
        )


def validate_mlx_startup_latency_artifact(
    path: Path,
    *,
    ratio_tolerance: float = 0.0005,
) -> list[str]:
    artifact = load_json(path)
    context_tokens, generation_tokens, prompt_hash = validate_top_level(path, artifact)
    rows_payload = artifact.get("rows")
    if not isinstance(rows_payload, list) or not rows_payload:
        raise StartupLatencyArtifactError(f"{path} lacks non-empty rows array")

    rows = [
        parse_row(
            path,
            row,
            index,
            expected_context_tokens=context_tokens,
            expected_generation_tokens=generation_tokens,
            expected_prompt_hash=prompt_hash,
        )
        for index, row in enumerate(rows_payload)
        if isinstance(row, dict)
    ]
    if len(rows) != len(rows_payload):
        raise StartupLatencyArtifactError(f"{path}.rows must contain only objects")

    validate_phase_set(path, rows, ratio_tolerance=ratio_tolerance)
    return [f"phase={phase}" for phase in REQUIRED_PHASES]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--ratio-tolerance", type=float, default=0.0005)
    args = parser.parse_args(argv)

    try:
        checked = validate_mlx_startup_latency_artifact(
            args.artifact,
            ratio_tolerance=args.ratio_tolerance,
        )
    except StartupLatencyArtifactError as error:
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
