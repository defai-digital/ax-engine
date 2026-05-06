#!/usr/bin/env python3
"""Validate TurboQuant fused cold-decode microbenchmark artifacts.

The microbenchmark is not a production promotion gate by itself. This checker
keeps the standalone fused-kernel evidence fail-closed so regressions in the
compressed decode path are caught before the heavier model-level quality gate is
run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ax.turboquant_fused_decode_microbench.v1"
DECODE_PATH = "fused_compressed_decode"
PRESET = "k8v4"
REQUIRED_VARIANT = "two_stage_scores"
DEFAULT_MIN_COLD_TOKENS = 8192
DEFAULT_MAX_ABS_DIFF = 1e-5
DEFAULT_MIN_COSINE_SIMILARITY = 0.999999
DEFAULT_MIN_SPEEDUP_VS_CPU = 1.0
DEFAULT_MIN_SPEEDUP_VS_DIM_PARALLEL = 1.0


class MicrobenchArtifactValidationError(RuntimeError):
    """Raised when a fused-kernel microbenchmark artifact is not valid evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MicrobenchArtifactValidationError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _number(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be numeric",
    )
    return float(value)


def _integer(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be an integer")
    return value


def _positive_number(value: Any, field: str) -> float:
    number = _number(value, field)
    _require(number > 0, f"{field} must be positive")
    return number


def _validate_top_level(doc: dict[str, Any]) -> None:
    _require(doc.get("schema_version") == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}")
    _require(doc.get("decode_path") == DECODE_PATH, f"decode_path must be {DECODE_PATH}")
    _require(doc.get("preset") == PRESET, f"preset must be {PRESET}")
    _require(_integer(doc.get("key_bits"), "key_bits") == 8, "key_bits must be 8")
    _require(_integer(doc.get("value_bits"), "value_bits") == 4, "value_bits must be 4")


def _variant_by_name(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants = row.get("kernel_variants")
    _require(isinstance(variants, list) and variants, "row.kernel_variants must be a non-empty array")

    by_name: dict[str, dict[str, Any]] = {}
    for index, variant in enumerate(variants):
        entry = _mapping(variant, f"row.kernel_variants[{index}]")
        name = entry.get("name")
        _require(isinstance(name, str) and name, f"row.kernel_variants[{index}].name must be a string")
        by_name[name] = entry
    return by_name


def _validate_quality(
    variant: dict[str, Any],
    *,
    max_abs_diff: float,
    min_cosine_similarity: float,
) -> None:
    quality = _mapping(variant.get("quality"), "two_stage_scores.quality")
    _require(
        _number(quality.get("max_abs_diff"), "two_stage_scores.quality.max_abs_diff")
        <= max_abs_diff,
        "two_stage_scores max_abs_diff exceeds fused-kernel limit",
    )
    _require(
        _number(
            quality.get("min_cosine_similarity"),
            "two_stage_scores.quality.min_cosine_similarity",
        )
        >= min_cosine_similarity,
        "two_stage_scores min_cosine_similarity is below fused-kernel limit",
    )


def _validate_hot_tail_merge(
    doc: dict[str, Any],
    row: dict[str, Any],
    *,
    max_abs_diff: float,
    min_cosine_similarity: float,
) -> None:
    hot_tail_merge = row.get("hot_tail_merge")
    if hot_tail_merge is None:
        return
    entry = _mapping(hot_tail_merge, "row.hot_tail_merge")
    _require(
        entry.get("contract") == "shared_logsumexp_partition_merge",
        "row.hot_tail_merge.contract must be shared_logsumexp_partition_merge",
    )
    _require(
        _integer(entry.get("hot_tokens"), "row.hot_tail_merge.hot_tokens") > 0,
        "row.hot_tail_merge.hot_tokens must be positive",
    )
    config = _mapping(doc.get("config"), "config")
    _require(
        _integer(config.get("hot_tokens"), "config.hot_tokens")
        == _integer(entry.get("hot_tokens"), "row.hot_tail_merge.hot_tokens"),
        "config.hot_tokens must match row.hot_tail_merge.hot_tokens",
    )
    quality = _mapping(entry.get("quality"), "row.hot_tail_merge.quality")
    _require(
        _number(quality.get("max_abs_diff"), "row.hot_tail_merge.quality.max_abs_diff")
        <= max_abs_diff,
        "hot-tail merge max_abs_diff exceeds fused-kernel limit",
    )
    _require(
        _number(
            quality.get("min_cosine_similarity"),
            "row.hot_tail_merge.quality.min_cosine_similarity",
        )
        >= min_cosine_similarity,
        "hot-tail merge min_cosine_similarity is below fused-kernel limit",
    )


def _median_us(variant: dict[str, Any], field: str) -> float:
    metal_wall_us = _mapping(variant.get("metal_wall_us"), f"{field}.metal_wall_us")
    return _positive_number(metal_wall_us.get("median"), f"{field}.metal_wall_us.median")


def _validate_row(
    doc: dict[str, Any],
    row: dict[str, Any],
    *,
    min_cold_tokens: int,
    min_speedup_vs_cpu: float,
    min_speedup_vs_dim_parallel: float,
    max_abs_diff: float,
    min_cosine_similarity: float,
) -> None:
    cold_tokens = _integer(row.get("cold_tokens"), "row.cold_tokens")
    _require(cold_tokens >= min_cold_tokens, f"row.cold_tokens must be >= {min_cold_tokens}")
    _require(_integer(row.get("key_bits", 8), "row.key_bits") == 8, "row key_bits must be 8")
    _require(_integer(row.get("value_bits", 4), "row.value_bits") == 4, "row value_bits must be 4")
    _require(
        _integer(row.get("estimated_cold_saved_bytes"), "row.estimated_cold_saved_bytes") > 0,
        "row.estimated_cold_saved_bytes must be positive",
    )
    _require(
        _integer(row.get("full_precision_cold_kv_bytes"), "row.full_precision_cold_kv_bytes") > 0,
        "row.full_precision_cold_kv_bytes must be positive",
    )
    _require(
        _integer(row.get("compressed_buffer_bytes"), "row.compressed_buffer_bytes") > 0,
        "row.compressed_buffer_bytes must be positive",
    )

    variants = _variant_by_name(row)
    _require(REQUIRED_VARIANT in variants, f"row missing required variant {REQUIRED_VARIANT}")

    candidate = variants[REQUIRED_VARIANT]
    _validate_quality(
        candidate,
        max_abs_diff=max_abs_diff,
        min_cosine_similarity=min_cosine_similarity,
    )
    _validate_hot_tail_merge(
        doc,
        row,
        max_abs_diff=max_abs_diff,
        min_cosine_similarity=min_cosine_similarity,
    )

    cpu_reference_us = _positive_number(row.get("cpu_reference_wall_us"), "row.cpu_reference_wall_us")
    candidate_us = _median_us(candidate, REQUIRED_VARIANT)
    _require(
        cpu_reference_us / candidate_us >= min_speedup_vs_cpu,
        f"{REQUIRED_VARIANT} speedup versus CPU reference is below {min_speedup_vs_cpu}",
    )

    if "dim_parallel" in variants:
        dim_parallel_us = _median_us(variants["dim_parallel"], "dim_parallel")
        _require(
            dim_parallel_us / candidate_us >= min_speedup_vs_dim_parallel,
            f"{REQUIRED_VARIANT} speedup versus dim_parallel is below {min_speedup_vs_dim_parallel}",
        )


def validate_artifact(
    doc: dict[str, Any],
    *,
    min_cold_tokens: int = DEFAULT_MIN_COLD_TOKENS,
    min_speedup_vs_cpu: float = DEFAULT_MIN_SPEEDUP_VS_CPU,
    min_speedup_vs_dim_parallel: float = DEFAULT_MIN_SPEEDUP_VS_DIM_PARALLEL,
    max_abs_diff: float = DEFAULT_MAX_ABS_DIFF,
    min_cosine_similarity: float = DEFAULT_MIN_COSINE_SIMILARITY,
) -> None:
    _validate_top_level(doc)

    rows = doc.get("rows")
    _require(isinstance(rows, list) and rows, "rows must be a non-empty array")

    eligible_rows = [
        _mapping(row, f"rows[{index}]")
        for index, row in enumerate(rows)
        if isinstance(row, dict) and isinstance(row.get("cold_tokens"), int)
        and row.get("cold_tokens") >= min_cold_tokens
    ]
    _require(eligible_rows, f"artifact must include a row with cold_tokens >= {min_cold_tokens}")

    largest_row = max(eligible_rows, key=lambda row: row["cold_tokens"])
    _validate_row(
        doc,
        largest_row,
        min_cold_tokens=min_cold_tokens,
        min_speedup_vs_cpu=min_speedup_vs_cpu,
        min_speedup_vs_dim_parallel=min_speedup_vs_dim_parallel,
        max_abs_diff=max_abs_diff,
        min_cosine_similarity=min_cosine_similarity,
    )


def load_and_validate(
    path: Path,
    *,
    min_cold_tokens: int = DEFAULT_MIN_COLD_TOKENS,
    min_speedup_vs_cpu: float = DEFAULT_MIN_SPEEDUP_VS_CPU,
    min_speedup_vs_dim_parallel: float = DEFAULT_MIN_SPEEDUP_VS_DIM_PARALLEL,
    max_abs_diff: float = DEFAULT_MAX_ABS_DIFF,
    min_cosine_similarity: float = DEFAULT_MIN_COSINE_SIMILARITY,
) -> None:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise MicrobenchArtifactValidationError(f"{path}: invalid JSON: {exc}") from exc
    _require(isinstance(doc, dict), f"{path}: top-level JSON must be an object")
    validate_artifact(
        doc,
        min_cold_tokens=min_cold_tokens,
        min_speedup_vs_cpu=min_speedup_vs_cpu,
        min_speedup_vs_dim_parallel=min_speedup_vs_dim_parallel,
        max_abs_diff=max_abs_diff,
        min_cosine_similarity=min_cosine_similarity,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--min-cold-tokens", type=int, default=DEFAULT_MIN_COLD_TOKENS)
    parser.add_argument("--min-speedup-vs-cpu", type=float, default=DEFAULT_MIN_SPEEDUP_VS_CPU)
    parser.add_argument(
        "--min-speedup-vs-dim",
        type=float,
        default=DEFAULT_MIN_SPEEDUP_VS_DIM_PARALLEL,
        help="minimum speedup versus dim_parallel when that variant is present",
    )
    parser.add_argument("--max-abs-diff", type=float, default=DEFAULT_MAX_ABS_DIFF)
    parser.add_argument(
        "--min-cosine-similarity",
        type=float,
        default=DEFAULT_MIN_COSINE_SIMILARITY,
    )
    args = parser.parse_args(argv)

    for artifact in args.artifacts:
        try:
            load_and_validate(
                artifact,
                min_cold_tokens=args.min_cold_tokens,
                min_speedup_vs_cpu=args.min_speedup_vs_cpu,
                min_speedup_vs_dim_parallel=args.min_speedup_vs_dim,
                max_abs_diff=args.max_abs_diff,
                min_cosine_similarity=args.min_cosine_similarity,
            )
        except MicrobenchArtifactValidationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"ok: {artifact}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
