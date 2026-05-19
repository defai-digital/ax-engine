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
CANDIDATE = "gelu_approx_mul"
PORTABLE_MEASUREMENT = "portable_gelu_approx_mul"
DIRECT_MEASUREMENT = "direct_cpp_gelu_approx_mul"
SPEEDUP_MEASUREMENT = "direct_cpp_speedup_ratio"
DEFAULT_MAX_ABS_ERROR = 1e-6
DEFAULT_MIN_SAMPLES = 1
DEFAULT_MIN_SPEEDUP = 0.0


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
) -> int:
    _require(entry.get("unit") == "microseconds", f"{name}.unit must be microseconds")
    samples = _positive_integer(entry.get("samples"), f"{name}.samples")
    _require(samples >= min_samples, f"{name}.samples must be >= {min_samples}")
    _positive_number(entry.get("mean"), f"{name}.mean")
    _positive_number(entry.get("median"), f"{name}.median")
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
    return _positive_integer(entry.get("op_count_median"), f"{name}.op_count_median")


def validate_artifact(
    doc: dict[str, Any],
    *,
    max_abs_error: float = DEFAULT_MAX_ABS_ERROR,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
) -> None:
    _require(doc.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    _require(doc.get("surface") == SURFACE, f"surface must be {SURFACE}")
    _string(doc.get("command"), "command")

    config = _mapping(doc.get("config"), "config")
    _require(config.get("candidate") == CANDIDATE, f"config.candidate must be {CANDIDATE}")
    rows = _positive_integer(config.get("rows"), "config.rows")
    cols = _positive_integer(config.get("cols"), "config.cols")
    _require(config.get("dtype") == "float32", "config.dtype must be float32")
    _positive_integer(config.get("iterations"), "config.iterations")

    correctness = _mapping(doc.get("correctness"), "correctness")
    _require(correctness.get("passed") is True, "correctness.passed must be true")
    _require(
        _number(correctness.get("max_abs_error"), "correctness.max_abs_error") <= max_abs_error,
        f"correctness.max_abs_error must be <= {max_abs_error}",
    )
    shape = _array(correctness.get("shape"), "correctness.shape")
    _require(shape == [rows, cols], "correctness.shape must match config rows/cols")

    measurements = _measurement_map(doc)
    for name in (PORTABLE_MEASUREMENT, DIRECT_MEASUREMENT, SPEEDUP_MEASUREMENT):
        _require(name in measurements, f"missing measurement {name}")

    portable_ops = _validate_timing_measurement(
        measurements[PORTABLE_MEASUREMENT],
        name=PORTABLE_MEASUREMENT,
        min_samples=min_samples,
    )
    direct_ops = _validate_timing_measurement(
        measurements[DIRECT_MEASUREMENT],
        name=DIRECT_MEASUREMENT,
        min_samples=min_samples,
    )
    _require(
        direct_ops < portable_ops,
        f"{DIRECT_MEASUREMENT}.op_count_median must be lower than {PORTABLE_MEASUREMENT}",
    )

    speedup = measurements[SPEEDUP_MEASUREMENT]
    _require(speedup.get("unit") == "ratio", f"{SPEEDUP_MEASUREMENT}.unit must be ratio")
    _positive_integer(speedup.get("samples"), f"{SPEEDUP_MEASUREMENT}.samples")
    speedup_median = _positive_number(speedup.get("median"), f"{SPEEDUP_MEASUREMENT}.median")
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
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        validate_artifact(
            load_artifact(args.artifact),
            max_abs_error=args.max_abs_error,
            min_samples=args.min_samples,
            min_speedup=args.min_speedup,
        )
    except DirectMlxHotpathProbeArtifactError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"ok: {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
