#!/usr/bin/env python3
"""Validate long-context MLX prefill and TTFT scaling artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ax.mlx_prefill_scaling.v1"
REQUIRED_ENGINES = {"mlx_lm", "ax_engine_mlx"}
PROMPT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class PrefillScalingArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScalingShape:
    context_tokens: int
    generation_tokens: int


@dataclass(frozen=True)
class ScalingRow:
    artifact_path: Path
    engine: str
    shape: ScalingShape
    prompt_hash: str
    prefill_tok_s_median: float
    ttft_ms_median: float
    peak_memory_gb_max: float
    payload: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise PrefillScalingArtifactError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise PrefillScalingArtifactError(f"{path} must contain a JSON object")
    return payload


def require_mapping(payload: dict[str, Any], key: str, *, owner: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PrefillScalingArtifactError(f"{owner} lacks object field {key!r}")
    return value


def require_non_empty_str(payload: dict[str, Any], key: str, *, owner: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PrefillScalingArtifactError(f"{owner} lacks non-empty string field {key!r}")
    return value


def require_positive_int(payload: dict[str, Any], key: str, *, owner: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value <= 0:
        raise PrefillScalingArtifactError(f"{owner} lacks positive integer field {key!r}")
    return value


def require_positive_float(payload: dict[str, Any], key: str, *, owner: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise PrefillScalingArtifactError(f"{owner} lacks positive numeric field {key!r}")
    return float(value)


def require_non_negative_float(payload: dict[str, Any], key: str, *, owner: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise PrefillScalingArtifactError(f"{owner} lacks non-negative numeric field {key!r}")
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
        raise PrefillScalingArtifactError(f"{owner} lacks metric object {key!r}")
    return require_positive_float(metric, required_stat, owner=f"{owner}.{key}")


def validate_top_level(path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise PrefillScalingArtifactError(
            f"{path} has schema_version={artifact.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )

    model = require_mapping(artifact, "model", owner=str(path))
    require_non_empty_str(model, "id", owner=f"{path}.model")

    host = require_mapping(artifact, "host", owner=str(path))
    require_non_empty_str(host, "chip", owner=f"{path}.host")

    benchmark = require_mapping(artifact, "benchmark", owner=str(path))
    batch_size = require_positive_int(benchmark, "batch_size", owner=f"{path}.benchmark")
    if batch_size != 1:
        raise PrefillScalingArtifactError(f"{path}.benchmark.batch_size must be 1")
    require_positive_int(benchmark, "prefill_step_size", owner=f"{path}.benchmark")
    require_non_negative_float(benchmark, "temperature", owner=f"{path}.benchmark")
    repetitions = require_positive_int(benchmark, "repetitions", owner=f"{path}.benchmark")
    if repetitions < 3:
        raise PrefillScalingArtifactError(f"{path}.benchmark.repetitions must be at least 3")


def parse_row(path: Path, row: dict[str, Any], index: int) -> ScalingRow:
    owner = f"{path}.rows[{index}]"
    engine = require_non_empty_str(row, "engine", owner=owner)
    if engine not in {"mlx_lm", "ax_engine_mlx", "ax_engine_mlx_ngram_accel"}:
        raise PrefillScalingArtifactError(f"{owner} has unsupported engine {engine!r}")

    context_tokens = require_positive_int(row, "context_tokens", owner=owner)
    if context_tokens < 1024:
        raise PrefillScalingArtifactError(f"{owner}.context_tokens must be at least 1024")
    generation_tokens = require_positive_int(row, "generation_tokens", owner=owner)
    prompt_hash = require_non_empty_str(row, "prompt_token_ids_sha256", owner=owner)
    if PROMPT_HASH_RE.fullmatch(prompt_hash) is None:
        raise PrefillScalingArtifactError(f"{owner} has invalid prompt_token_ids_sha256")
    repetitions = require_positive_int(row, "repetitions", owner=owner)
    if repetitions < 3:
        raise PrefillScalingArtifactError(f"{owner}.repetitions must be at least 3")

    if engine == "mlx_lm":
        baseline = require_mapping(row, "baseline", owner=owner)
        if baseline.get("role") != "primary_reference":
            raise PrefillScalingArtifactError(f"{owner} mlx_lm row lacks primary reference role")
    elif engine == "ax_engine_mlx":
        if row.get("ax_decode_policy") != "direct_no_ngram_acceleration":
            raise PrefillScalingArtifactError(
                f"{owner} ax_engine_mlx row must use direct_no_ngram_acceleration"
            )
        route = require_mapping(row, "route", owner=owner)
        require_non_empty_str(route, "selected_backend", owner=f"{owner}.route")
    elif engine == "ax_engine_mlx_ngram_accel":
        if "ngram" not in str(row.get("ax_decode_policy", "")):
            raise PrefillScalingArtifactError(f"{owner} n-gram row lacks n-gram decode policy")
        if not isinstance(row.get("ngram_acceleration_telemetry"), dict):
            raise PrefillScalingArtifactError(f"{owner} n-gram row lacks telemetry")

    return ScalingRow(
        artifact_path=path,
        engine=engine,
        shape=ScalingShape(
            context_tokens=context_tokens,
            generation_tokens=generation_tokens,
        ),
        prompt_hash=prompt_hash,
        prefill_tok_s_median=require_metric(row, "prefill_tok_s", "median", owner=owner),
        ttft_ms_median=require_metric(row, "ttft_ms", "median", owner=owner),
        peak_memory_gb_max=require_metric(row, "peak_memory_gb", "max", owner=owner),
        payload=row,
    )


def assert_ratio_matches(
    *,
    row: ScalingRow,
    baseline: ScalingRow,
    ratio_key: str,
    expected: float,
    tolerance: float,
) -> None:
    ratios = row.payload.get("ratios_to_mlx_lm")
    if not isinstance(ratios, dict):
        raise PrefillScalingArtifactError(
            f"{row.artifact_path} {row.engine} context={row.shape.context_tokens} "
            "lacks ratios_to_mlx_lm"
        )
    value = ratios.get(ratio_key)
    if not isinstance(value, (int, float)):
        raise PrefillScalingArtifactError(
            f"{row.artifact_path} {row.engine} context={row.shape.context_tokens} "
            f"lacks numeric ratios_to_mlx_lm.{ratio_key}"
        )
    if abs(float(value) - expected) > tolerance:
        raise PrefillScalingArtifactError(
            f"{row.artifact_path} stale {ratio_key} ratio for "
            f"context={row.shape.context_tokens} generation={row.shape.generation_tokens}: "
            f"artifact={float(value):.6f} expected={expected:.6f} "
            f"against mlx_lm baseline at context={baseline.shape.context_tokens}"
        )


def validate_shape_groups(
    path: Path,
    rows: list[ScalingRow],
    *,
    ratio_tolerance: float,
) -> None:
    by_shape: dict[ScalingShape, dict[str, ScalingRow]] = {}
    prompt_hashes: dict[ScalingShape, set[str]] = {}
    for row in rows:
        engines = by_shape.setdefault(row.shape, {})
        if row.engine in engines:
            raise PrefillScalingArtifactError(
                f"{path} has duplicate {row.engine} row for "
                f"context={row.shape.context_tokens} generation={row.shape.generation_tokens}"
            )
        engines[row.engine] = row
        prompt_hashes.setdefault(row.shape, set()).add(row.prompt_hash)

    for shape, engines in by_shape.items():
        missing = REQUIRED_ENGINES - set(engines)
        if missing:
            raise PrefillScalingArtifactError(
                f"{path} context={shape.context_tokens} generation={shape.generation_tokens} "
                f"lacks required engines: {sorted(missing)}"
            )
        if len(prompt_hashes[shape]) != 1:
            raise PrefillScalingArtifactError(
                f"{path} context={shape.context_tokens} generation={shape.generation_tokens} "
                "does not reuse one prompt hash across engines"
            )

        baseline = engines["mlx_lm"]
        candidate = engines["ax_engine_mlx"]
        assert_ratio_matches(
            row=candidate,
            baseline=baseline,
            ratio_key="prefill_tok_s",
            expected=candidate.prefill_tok_s_median / baseline.prefill_tok_s_median,
            tolerance=ratio_tolerance,
        )
        assert_ratio_matches(
            row=candidate,
            baseline=baseline,
            ratio_key="ttft_ms",
            expected=candidate.ttft_ms_median / baseline.ttft_ms_median,
            tolerance=ratio_tolerance,
        )


def validate_context_coverage(
    path: Path,
    rows: list[ScalingRow],
    *,
    min_context_count: int,
    min_largest_context_tokens: int,
) -> None:
    contexts = sorted({row.shape.context_tokens for row in rows if row.engine == "ax_engine_mlx"})
    if len(contexts) < min_context_count:
        raise PrefillScalingArtifactError(
            f"{path} has {len(contexts)} AX context points, expected at least {min_context_count}"
        )
    if contexts[-1] < min_largest_context_tokens:
        raise PrefillScalingArtifactError(
            f"{path} largest AX context is {contexts[-1]}, expected at least "
            f"{min_largest_context_tokens}"
        )


def validate_prefill_scaling_artifact(
    path: Path,
    *,
    min_context_count: int = 2,
    min_largest_context_tokens: int = 8192,
    ratio_tolerance: float = 0.005,
) -> list[str]:
    artifact = load_json(path)
    validate_top_level(path, artifact)

    raw_rows = artifact.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise PrefillScalingArtifactError(f"{path} lacks non-empty rows list")
    rows = [
        parse_row(path, row, index)
        for index, row in enumerate(raw_rows)
        if isinstance(row, dict)
    ]
    if len(rows) != len(raw_rows):
        raise PrefillScalingArtifactError(f"{path} rows must all be JSON objects")

    validate_shape_groups(path, rows, ratio_tolerance=ratio_tolerance)
    validate_context_coverage(
        path,
        rows,
        min_context_count=min_context_count,
        min_largest_context_tokens=min_largest_context_tokens,
    )
    shapes = sorted({(row.shape.context_tokens, row.shape.generation_tokens) for row in rows})
    return [f"context={context}:generation={generation}" for context, generation in shapes]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate AX long-context prefill/TTFT scaling artifact contracts."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--min-context-count", type=int, default=2)
    parser.add_argument("--min-largest-context-tokens", type=int, default=8192)
    parser.add_argument("--ratio-tolerance", type=float, default=0.005)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checked = 0
    try:
        for artifact in args.artifacts:
            checked += len(
                validate_prefill_scaling_artifact(
                    artifact,
                    min_context_count=args.min_context_count,
                    min_largest_context_tokens=args.min_largest_context_tokens,
                    ratio_tolerance=args.ratio_tolerance,
                )
            )
    except PrefillScalingArtifactError as error:
        print(f"MLX prefill scaling artifact check failed: {error}", file=sys.stderr)
        return 1
    print(f"MLX prefill scaling artifact check passed: {checked} shape groups validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
