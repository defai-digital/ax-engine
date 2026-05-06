#!/usr/bin/env python3
"""Build TurboQuant decode quality metrics from baseline/candidate vectors."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ax.turboquant_quality_metrics.v1"


class QualityMetricsBuildError(RuntimeError):
    """Raised when decode-output vectors cannot be compared safely."""


def _load_vectors(path: Path) -> list[list[float]]:
    doc = json.loads(path.read_text())
    payload = doc.get("decode_outputs", doc.get("outputs")) if isinstance(doc, dict) else doc
    if not isinstance(payload, list) or not payload:
        raise QualityMetricsBuildError(f"{path}: expected a non-empty vector array")

    vectors: list[list[float]] = []
    for row_index, row in enumerate(payload):
        if not isinstance(row, list) or not row:
            raise QualityMetricsBuildError(f"{path}: vector {row_index} must be non-empty")
        vector = []
        for col_index, value in enumerate(row):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise QualityMetricsBuildError(
                    f"{path}: vector {row_index} value {col_index} must be numeric"
                )
            number = float(value)
            if not math.isfinite(number):
                raise QualityMetricsBuildError(
                    f"{path}: vector {row_index} value {col_index} must be finite"
                )
            vector.append(number)
        vectors.append(vector)
    return vectors


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def compare_vectors(
    baseline: list[list[float]],
    candidate: list[list[float]],
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        raise QualityMetricsBuildError(
            f"head/output count mismatch: baseline={len(baseline)} candidate={len(candidate)}"
        )

    max_abs_diff = 0.0
    total_abs_diff = 0.0
    total_elements = 0
    min_cosine = 1.0
    rows = []

    for index, (expected, actual) in enumerate(zip(baseline, candidate)):
        if len(expected) != len(actual):
            raise QualityMetricsBuildError(
                f"vector {index} dimension mismatch: baseline={len(expected)} candidate={len(actual)}"
            )
        diffs = [abs(a - b) for a, b in zip(expected, actual)]
        row_max = max(diffs) if diffs else 0.0
        row_mean = sum(diffs) / len(diffs)
        row_cosine = _cosine(expected, actual)
        max_abs_diff = max(max_abs_diff, row_max)
        total_abs_diff += sum(diffs)
        total_elements += len(diffs)
        min_cosine = min(min_cosine, row_cosine)
        rows.append(
            {
                "index": index,
                "max_abs_diff": row_max,
                "mean_abs_diff": row_mean,
                "cosine_similarity": row_cosine,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "metrics": {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": total_abs_diff / total_elements,
            "min_cosine_similarity": min_cosine,
        },
        "rows": rows,
    }


def build_metrics(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    return compare_vectors(_load_vectors(baseline_path), _load_vectors(candidate_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-outputs", required=True, type=Path)
    parser.add_argument("--candidate-outputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        metrics = build_metrics(args.baseline_outputs, args.candidate_outputs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    except (OSError, json.JSONDecodeError, QualityMetricsBuildError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"ok: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
