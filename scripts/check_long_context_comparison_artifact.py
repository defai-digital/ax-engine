#!/usr/bin/env python3
"""Validate long-context AX/MLX/llama.cpp comparison artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ax.long_context_comparison.v1"
REQUIRED_PARITY_ENGINES = {"mlx_lm", "ax_engine_mlx"}
OPTIONAL_EXTERNAL_ENGINES = {"llama_cpp_metal"}
PROMPT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class LongContextComparisonArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class Shape:
    context_tokens: int
    generation_tokens: int


@dataclass(frozen=True)
class Row:
    engine: str
    shape: Shape
    prompt_hash: str | None
    prefill_tok_s_median: float
    ttft_ms_median: float
    payload: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise LongContextComparisonArtifactError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise LongContextComparisonArtifactError(f"{path} must contain a JSON object")
    return payload


def require_mapping(payload: dict[str, Any], key: str, *, owner: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise LongContextComparisonArtifactError(f"{owner} lacks object field {key!r}")
    return value


def require_positive_int(payload: dict[str, Any], key: str, *, owner: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LongContextComparisonArtifactError(f"{owner} lacks positive integer field {key!r}")
    return int(value)


def require_positive_float(payload: dict[str, Any], key: str, *, owner: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
        raise LongContextComparisonArtifactError(f"{owner} lacks positive numeric field {key!r}")
    return float(value)


def metric_median(payload: dict[str, Any], key: str, *, owner: str) -> float:
    metric = payload.get(key)
    if not isinstance(metric, dict):
        raise LongContextComparisonArtifactError(f"{owner} lacks metric object {key!r}")
    return require_positive_float(metric, "median", owner=f"{owner}.{key}")


def validate_top_level(path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise LongContextComparisonArtifactError(
            f"{path} has schema_version={artifact.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    if artifact.get("claim_scope") != "long_context_cold_prefill_comparison":
        raise LongContextComparisonArtifactError(f"{path} has unexpected claim_scope")

    model = require_mapping(artifact, "model", owner=str(path))
    if not isinstance(model.get("id"), str) or not model["id"].strip():
        raise LongContextComparisonArtifactError(f"{path}.model lacks id")

    benchmark = require_mapping(artifact, "benchmark", owner=str(path))
    batch_size = require_positive_int(benchmark, "batch_size", owner=f"{path}.benchmark")
    if batch_size != 1:
        raise LongContextComparisonArtifactError(f"{path}.benchmark.batch_size must be 1")
    repetitions = require_positive_int(benchmark, "repetitions", owner=f"{path}.benchmark")
    if repetitions < 3:
        raise LongContextComparisonArtifactError(f"{path}.benchmark.repetitions must be at least 3")


def parse_row(path: Path, row: dict[str, Any], index: int) -> Row:
    owner = f"{path}.rows[{index}]"
    engine = row.get("engine")
    if engine not in REQUIRED_PARITY_ENGINES | OPTIONAL_EXTERNAL_ENGINES:
        raise LongContextComparisonArtifactError(f"{owner} has unsupported engine {engine!r}")

    shape = Shape(
        context_tokens=require_positive_int(row, "context_tokens", owner=owner),
        generation_tokens=require_positive_int(row, "generation_tokens", owner=owner),
    )
    if shape.context_tokens < 1024:
        raise LongContextComparisonArtifactError(f"{owner}.context_tokens must be at least 1024")

    prompt_hash_value = row.get("prompt_token_ids_sha256")
    prompt_hash = prompt_hash_value if isinstance(prompt_hash_value, str) else None
    if engine in REQUIRED_PARITY_ENGINES:
        if prompt_hash is None or PROMPT_HASH_RE.fullmatch(prompt_hash) is None:
            raise LongContextComparisonArtifactError(f"{owner} has invalid prompt_token_ids_sha256")
    elif engine == "llama_cpp_metal":
        if row.get("prompt_contract") != "shape_compatible_llama_bench_internal_tokens":
            raise LongContextComparisonArtifactError(
                f"{owner} llama.cpp row must use shape_compatible prompt contract"
            )
        claim_boundary = str(row.get("claim_boundary", ""))
        if "not prompt-hash parity" not in claim_boundary:
            raise LongContextComparisonArtifactError(
                f"{owner} llama.cpp row lacks external claim boundary"
            )

    return Row(
        engine=str(engine),
        shape=shape,
        prompt_hash=prompt_hash,
        prefill_tok_s_median=metric_median(row, "prefill_tok_s", owner=owner),
        ttft_ms_median=metric_median(row, "ttft_ms", owner=owner),
        payload=row,
    )


def assert_ratio(
    *,
    row: Row,
    baseline: Row,
    ratio_key: str,
    expected: float,
    tolerance: float,
) -> None:
    ratios = row.payload.get("ratios_to_mlx_lm")
    if not isinstance(ratios, dict):
        raise LongContextComparisonArtifactError(
            f"{row.engine} context={row.shape.context_tokens} lacks ratios_to_mlx_lm"
        )
    value = ratios.get(ratio_key)
    if not isinstance(value, (int, float)):
        raise LongContextComparisonArtifactError(
            f"{row.engine} context={row.shape.context_tokens} lacks {ratio_key}"
        )
    if abs(float(value) - expected) > tolerance:
        raise LongContextComparisonArtifactError(
            f"stale {ratio_key} for {row.engine} context={row.shape.context_tokens}: "
            f"artifact={float(value):.6f} expected={expected:.6f} "
            f"against mlx_lm context={baseline.shape.context_tokens}"
        )


def validate_shape_groups(
    path: Path,
    rows: list[Row],
    *,
    require_llama_cpp: bool,
    ratio_tolerance: float,
) -> None:
    groups: dict[Shape, dict[str, Row]] = {}
    for row in rows:
        engines = groups.setdefault(row.shape, {})
        if row.engine in engines:
            raise LongContextComparisonArtifactError(
                f"{path} has duplicate {row.engine} row for "
                f"context={row.shape.context_tokens} generation={row.shape.generation_tokens}"
            )
        engines[row.engine] = row

    for shape, engines in groups.items():
        missing = REQUIRED_PARITY_ENGINES - set(engines)
        if missing:
            raise LongContextComparisonArtifactError(
                f"{path} context={shape.context_tokens} generation={shape.generation_tokens} "
                f"lacks required engines: {sorted(missing)}"
            )
        if require_llama_cpp and "llama_cpp_metal" not in engines:
            raise LongContextComparisonArtifactError(
                f"{path} context={shape.context_tokens} generation={shape.generation_tokens} "
                "lacks required llama_cpp_metal external row"
            )

        prompt_hashes = {
            engines[engine].prompt_hash
            for engine in REQUIRED_PARITY_ENGINES
            if engines[engine].prompt_hash is not None
        }
        if len(prompt_hashes) != 1:
            raise LongContextComparisonArtifactError(
                f"{path} context={shape.context_tokens} generation={shape.generation_tokens} "
                "does not reuse one prompt hash across AX and mlx_lm rows"
            )

        baseline = engines["mlx_lm"]
        for engine in ("ax_engine_mlx", "llama_cpp_metal"):
            row = engines.get(engine)
            if row is None:
                continue
            assert_ratio(
                row=row,
                baseline=baseline,
                ratio_key="prefill_tok_s",
                expected=row.prefill_tok_s_median / baseline.prefill_tok_s_median,
                tolerance=ratio_tolerance,
            )
            assert_ratio(
                row=row,
                baseline=baseline,
                ratio_key="ttft_ms",
                expected=row.ttft_ms_median / baseline.ttft_ms_median,
                tolerance=ratio_tolerance,
            )


def validate_context_coverage(
    path: Path,
    rows: list[Row],
    *,
    min_context_count: int,
    min_largest_context_tokens: int,
) -> None:
    contexts = sorted({row.shape.context_tokens for row in rows if row.engine == "ax_engine_mlx"})
    if len(contexts) < min_context_count:
        raise LongContextComparisonArtifactError(
            f"{path} has {len(contexts)} AX context points, expected at least {min_context_count}"
        )
    if contexts[-1] < min_largest_context_tokens:
        raise LongContextComparisonArtifactError(
            f"{path} largest AX context is {contexts[-1]}, expected at least "
            f"{min_largest_context_tokens}"
        )


def validate_long_context_comparison_artifact(
    path: Path,
    *,
    require_llama_cpp: bool = False,
    min_context_count: int = 2,
    min_largest_context_tokens: int = 8192,
    ratio_tolerance: float = 0.005,
) -> list[str]:
    artifact = load_json(path)
    validate_top_level(path, artifact)
    raw_rows = artifact.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise LongContextComparisonArtifactError(f"{path} lacks non-empty rows list")
    rows = [
        parse_row(path, row, index)
        for index, row in enumerate(raw_rows)
        if isinstance(row, dict)
    ]
    if len(rows) != len(raw_rows):
        raise LongContextComparisonArtifactError(f"{path} rows must all be JSON objects")

    validate_shape_groups(
        path,
        rows,
        require_llama_cpp=require_llama_cpp,
        ratio_tolerance=ratio_tolerance,
    )
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
        description="Validate ax.long_context_comparison.v1 artifacts."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--require-llama-cpp", action="store_true")
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
                validate_long_context_comparison_artifact(
                    artifact,
                    require_llama_cpp=args.require_llama_cpp,
                    min_context_count=args.min_context_count,
                    min_largest_context_tokens=args.min_largest_context_tokens,
                    ratio_tolerance=args.ratio_tolerance,
                )
            )
    except LongContextComparisonArtifactError as error:
        print(f"Long-context comparison artifact check failed: {error}", file=sys.stderr)
        return 1
    print(f"Long-context comparison artifact check passed: {checked} shape groups validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
