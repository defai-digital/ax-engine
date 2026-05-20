#!/usr/bin/env python3
"""Validate direct-MLX hotpath probe artifacts.

The probe is an evidence gate before any direct C++ path is promoted into the
production model runner. This checker keeps that evidence fail-closed: the JSON
must prove correctness, include both portable and direct measurements, and show
that the direct shim actually reduced Rust-side MLX dispatch count.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "ax.microbench.v1"
SURFACE = "direct-mlx-hotpath"
CANDIDATE_MEASUREMENTS = {
    "gelu_approx_mul": (
        "portable_gelu_approx_mul",
        "direct_cpp_gelu_approx_mul",
    ),
    "gelu_approx_mul_matmul": (
        "portable_gelu_approx_mul_matmul",
        "direct_cpp_gelu_approx_mul_matmul",
    ),
    "gelu_approx_quantized_ffn": (
        "portable_gelu_approx_quantized_ffn",
        "direct_cpp_gelu_approx_quantized_ffn",
    ),
    "qk_norm_rope": (
        "portable_qk_norm_rope",
        "direct_cpp_qk_norm_rope",
    ),
    "gemma4_post_attn_ffn_block": (
        "portable_gemma4_post_attn_ffn_block",
        "direct_cpp_gemma4_post_attn_ffn_block",
    ),
    "qwen_linear_attention_inputs_packed": (
        "portable_qwen_linear_attention_inputs_packed",
        "direct_cpp_qwen_linear_attention_inputs_packed",
    ),
}
LARGE_BLOCK_CANDIDATES = {
    "gemma4_post_attn_ffn_block",
}
SPEEDUP_MEASUREMENT = "direct_cpp_speedup_ratio"
DEFAULT_MAX_ABS_ERROR = 1e-6
DEFAULT_MIN_SAMPLES = 1
DEFAULT_MIN_SPEEDUP = 0.0
SPEEDUP_RATIO_TOLERANCE = 1e-6


class DirectMlxHotpathProbeArtifactError(RuntimeError):
    """Raised when a direct-MLX hotpath probe artifact is not valid evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DirectMlxHotpathProbeArtifactError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be an array")
    return value


def _string(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value, f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be an integer")
    return value


def _positive_integer(value: Any, field: str) -> int:
    number = _integer(value, field)
    _require(number > 0, f"{field} must be positive")
    return number


def _number(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be numeric",
    )
    return float(value)


def _positive_number(value: Any, field: str) -> float:
    number = _number(value, field)
    _require(number > 0.0, f"{field} must be positive")
    return number


def _shape(value: Any, field: str) -> list[int]:
    shape = _array(value, field)
    for index, dim in enumerate(shape):
        _positive_integer(dim, f"{field}[{index}]")
    return shape


def _measurement_map(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _array(doc.get("measurements"), "measurements")
    _require(rows, "measurements must be non-empty")
    by_name: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        entry = _mapping(row, f"measurements[{index}]")
        name = _string(entry.get("name"), f"measurements[{index}].name")
        _require(name not in by_name, f"duplicate measurement {name!r}")
        by_name[name] = entry
    return by_name


def _validate_timing_measurement(
    entry: dict[str, Any],
    *,
    name: str,
    min_samples: int,
) -> tuple[float, int]:
    _require(entry.get("unit") == "microseconds", f"{name}.unit must be microseconds")
    samples = _positive_integer(entry.get("samples"), f"{name}.samples")
    _require(samples >= min_samples, f"{name}.samples must be >= {min_samples}")
    _positive_number(entry.get("mean"), f"{name}.mean")
    median = _positive_number(entry.get("median"), f"{name}.median")
    _positive_number(entry.get("min"), f"{name}.min")
    _positive_number(entry.get("max"), f"{name}.max")
    _require(
        _number(entry.get("min"), f"{name}.min") <= _number(entry.get("median"), f"{name}.median"),
        f"{name}.min must be <= median",
    )
    _require(
        _number(entry.get("median"), f"{name}.median")
        <= _number(entry.get("max"), f"{name}.max"),
        f"{name}.median must be <= max",
    )
    return median, _positive_integer(
        entry.get("op_count_median"), f"{name}.op_count_median"
    )


def validate_artifact(
    doc: dict[str, Any],
    *,
    max_abs_error: float = DEFAULT_MAX_ABS_ERROR,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    large_block_only: bool = False,
    require_clean_git: bool = False,
) -> None:
    _require(doc.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    _require(doc.get("surface") == SURFACE, f"surface must be {SURFACE}")
    _string(doc.get("command"), "command")

    config = _mapping(doc.get("config"), "config")
    candidate = _string(config.get("candidate"), "config.candidate")
    _require(
        candidate in CANDIDATE_MEASUREMENTS,
        f"config.candidate must be one of {sorted(CANDIDATE_MEASUREMENTS)}",
    )
    if large_block_only:
        _require(
            candidate in LARGE_BLOCK_CANDIDATES,
            f"config.candidate must be one of {sorted(LARGE_BLOCK_CANDIDATES)} for large-block gate",
        )
    portable_measurement, direct_measurement = CANDIDATE_MEASUREMENTS[candidate]
    if require_clean_git:
        git = _mapping(doc.get("git"), "git")
        _string(git.get("commit"), "git.commit")
        _require(git.get("dirty") is False, "git.dirty must be false")
    rows = _positive_integer(config.get("rows"), "config.rows")
    cols = _positive_integer(config.get("cols"), "config.cols")
    output_cols = cols
    if candidate in {
        "gelu_approx_mul_matmul",
        "gelu_approx_quantized_ffn",
        "gemma4_post_attn_ffn_block",
    }:
        output_cols = _positive_integer(config.get("down_cols"), "config.down_cols")
    if candidate in {
        "gelu_approx_quantized_ffn",
        "gemma4_post_attn_ffn_block",
        "qwen_linear_attention_inputs_packed",
    }:
        group_size = _positive_integer(config.get("group_size"), "config.group_size")
        bits = _positive_integer(config.get("bits"), "config.bits")
        _require(group_size in {32, 64, 128}, "config.group_size must be one of 32, 64, 128")
        if candidate == "qwen_linear_attention_inputs_packed":
            _require(2 <= bits <= 8, "config.bits must be between 2 and 8")
        else:
            _require(bits == 4, "config.bits must be 4")
    expected_shape: list[int]
    expected_component_shapes: dict[str, list[int]] | None = None
    if candidate == "qk_norm_rope":
        head_dim = _positive_integer(config.get("head_dim"), "config.head_dim")
        n_heads = _positive_integer(config.get("n_heads"), "config.n_heads")
        _require(
            n_heads * head_dim == cols,
            "config.n_heads * config.head_dim must equal config.cols for qk_norm_rope",
        )
        expected_shape = [1, n_heads, rows, head_dim]
    elif candidate == "qwen_linear_attention_inputs_packed":
        num_key_heads = _positive_integer(
            config.get("linear_num_key_heads"), "config.linear_num_key_heads"
        )
        num_value_heads = _positive_integer(
            config.get("linear_num_value_heads"), "config.linear_num_value_heads"
        )
        key_head_dim = _positive_integer(
            config.get("linear_key_head_dim"), "config.linear_key_head_dim"
        )
        value_head_dim = _positive_integer(
            config.get("linear_value_head_dim"), "config.linear_value_head_dim"
        )
        _require(
            num_value_heads % num_key_heads == 0,
            "config.linear_num_value_heads must be divisible by config.linear_num_key_heads",
        )
        key_dim = num_key_heads * key_head_dim
        value_dim = num_value_heads * value_head_dim
        expected_shape = [1, rows, key_dim * 2 + value_dim]
        expected_component_shapes = {
            "qkv": expected_shape,
            "z": [1, rows, num_value_heads, value_head_dim],
            "a": [1, rows, num_value_heads],
            "b": [1, rows, num_value_heads],
        }
    else:
        expected_shape = [rows, output_cols]
    _require(config.get("dtype") == "float32", "config.dtype must be float32")
    _positive_integer(config.get("iterations"), "config.iterations")

    correctness = _mapping(doc.get("correctness"), "correctness")
    _require(correctness.get("passed") is True, "correctness.passed must be true")
    _require(
        _number(correctness.get("max_abs_error"), "correctness.max_abs_error") <= max_abs_error,
        f"correctness.max_abs_error must be <= {max_abs_error}",
    )
    shape = _shape(correctness.get("shape"), "correctness.shape")
    _require(shape == expected_shape, "correctness.shape must match expected output shape")
    if expected_component_shapes is not None:
        component_shapes = _mapping(
            correctness.get("component_shapes"), "correctness.component_shapes"
        )
        for name, expected in expected_component_shapes.items():
            observed = _shape(
                component_shapes.get(name), f"correctness.component_shapes.{name}"
            )
            _require(
                observed == expected,
                f"correctness.component_shapes.{name} must match expected output shape",
            )

    measurements = _measurement_map(doc)
    for name in (portable_measurement, direct_measurement, SPEEDUP_MEASUREMENT):
        _require(name in measurements, f"missing measurement {name}")

    portable_median, portable_ops = _validate_timing_measurement(
        measurements[portable_measurement],
        name=portable_measurement,
        min_samples=min_samples,
    )
    direct_median, direct_ops = _validate_timing_measurement(
        measurements[direct_measurement],
        name=direct_measurement,
        min_samples=min_samples,
    )
    _require(
        direct_ops < portable_ops,
        f"{direct_measurement}.op_count_median must be lower than {portable_measurement}",
    )

    speedup = measurements[SPEEDUP_MEASUREMENT]
    _require(speedup.get("unit") == "ratio", f"{SPEEDUP_MEASUREMENT}.unit must be ratio")
    speedup_samples = _positive_integer(
        speedup.get("samples"), f"{SPEEDUP_MEASUREMENT}.samples"
    )
    _require(
        speedup_samples == 1,
        f"{SPEEDUP_MEASUREMENT}.samples must be 1 because speedup is a derived scalar",
    )
    speedup_mean = _positive_number(speedup.get("mean"), f"{SPEEDUP_MEASUREMENT}.mean")
    speedup_median = _positive_number(speedup.get("median"), f"{SPEEDUP_MEASUREMENT}.median")
    speedup_min = _positive_number(speedup.get("min"), f"{SPEEDUP_MEASUREMENT}.min")
    speedup_max = _positive_number(speedup.get("max"), f"{SPEEDUP_MEASUREMENT}.max")
    expected_speedup = portable_median / direct_median
    for field, value in {
        "mean": speedup_mean,
        "median": speedup_median,
        "min": speedup_min,
        "max": speedup_max,
    }.items():
        _require(
            abs(value - expected_speedup) <= SPEEDUP_RATIO_TOLERANCE,
            f"{SPEEDUP_MEASUREMENT}.{field} must equal portable/direct median ratio",
        )
    _require(speedup_median >= min_speedup, f"{SPEEDUP_MEASUREMENT}.median must be >= {min_speedup}")


def load_artifact(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise DirectMlxHotpathProbeArtifactError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(doc, dict):
        raise DirectMlxHotpathProbeArtifactError(f"{path} must contain a JSON object")
    return doc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Path to direct-mlx-hotpath-probe JSON")
    parser.add_argument("--max-abs-error", type=float, default=DEFAULT_MAX_ABS_ERROR)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP)
    parser.add_argument(
        "--large-block-only",
        action="store_true",
        help="Accept only direct-MLX large-block candidates for P0 evidence",
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Require git.dirty=false for promotion-grade evidence",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        validate_artifact(
            load_artifact(args.artifact),
            max_abs_error=args.max_abs_error,
            min_samples=args.min_samples,
            min_speedup=args.min_speedup,
            large_block_only=args.large_block_only,
            require_clean_git=args.require_clean_git,
        )
    except DirectMlxHotpathProbeArtifactError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"ok: {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
