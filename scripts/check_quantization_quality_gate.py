#!/usr/bin/env python3
"""Validate quantization quality gate artifacts.

Validates that a quantized model maintains quality metrics compared to a
reference. Intentionally validates evidence artifacts instead of running the
model directly, since model runs are expensive and host-specific.

Output schema: ax.quantization_quality_gate.v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ax.quantization_quality_gate.v1"
MIN_COSINE_SIMILARITY = 0.995
MAX_MEAN_ABS_DIFF = 0.05
MAX_MAX_ABS_DIFF = 0.15
MIN_DECODE_RATIO_TO_BASELINE = 0.90


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "artifact",
        type=Path,
        help="Path to the quantization quality artifact to validate.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write gate decision JSON to this path.",
    )
    return p.parse_args()


def validate_quality_metrics(metrics: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    cosine = metrics.get("cosine_similarity")
    if cosine is not None and cosine < MIN_COSINE_SIMILARITY:
        failures.append(f"cosine_similarity {cosine:.6f} < {MIN_COSINE_SIMILARITY}")

    mean_abs = metrics.get("mean_abs_diff")
    if mean_abs is not None and mean_abs > MAX_MEAN_ABS_DIFF:
        failures.append(f"mean_abs_diff {mean_abs:.6f} > {MAX_MEAN_ABS_DIFF}")

    max_abs = metrics.get("max_abs_diff")
    if max_abs is not None and max_abs > MAX_MAX_ABS_DIFF:
        failures.append(f"max_abs_diff {max_abs:.6f} > {MAX_MAX_ABS_DIFF}")

    return failures


def validate_speed_metrics(
    metrics: dict[str, Any],
    baseline_decode_tok_s: float | None,
) -> list[str]:
    failures: list[str] = []

    if baseline_decode_tok_s is None or baseline_decode_tok_s <= 0:
        return failures

    candidate_decode = metrics.get("decode_tok_s")
    if candidate_decode is not None and candidate_decode > 0:
        ratio = candidate_decode / baseline_decode_tok_s
        if ratio < MIN_DECODE_RATIO_TO_BASELINE:
            failures.append(
                f"decode_tok_s ratio {ratio:.4f} < {MIN_DECODE_RATIO_TO_BASELINE} "
                f"({candidate_decode:.1f} vs baseline {baseline_decode_tok_s:.1f})"
            )

    return failures


def validate_artifact(artifact_path: Path) -> dict[str, Any]:
    if not artifact_path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "reject",
            "reason": "artifact_not_found",
            "artifact_path": str(artifact_path),
        }

    try:
        artifact = json.loads(artifact_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "reject",
            "reason": f"failed_to_read_artifact: {e}",
            "artifact_path": str(artifact_path),
        }

    artifact_schema = artifact.get("schema_version", "")
    if artifact_schema != SCHEMA_VERSION:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "reject",
            "reason": f"schema_version mismatch: expected {SCHEMA_VERSION}, got {artifact_schema}",
            "artifact_path": str(artifact_path),
        }

    quality_metrics = artifact.get("quality_metrics", {})
    quality_failures = validate_quality_metrics(quality_metrics)

    speed_metrics = artifact.get("speed_metrics", {})
    baseline_decode = artifact.get("baseline_decode_tok_s")
    speed_failures = validate_speed_metrics(speed_metrics, baseline_decode)

    all_failures = quality_failures + speed_failures
    decision = "approve" if not all_failures else "reject"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "artifact_path": str(artifact_path),
        "quality_failures": quality_failures,
        "speed_failures": speed_failures,
        "total_failures": len(all_failures),
        "quality_metrics": quality_metrics,
        "speed_metrics": speed_metrics,
    }


def main() -> int:
    args = parse_args()
    result = validate_artifact(args.artifact)

    output_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)

    if result["decision"] == "reject":
        print(
            f"\nGATE REJECTED: {result.get('reason', 'quality/speed failures')}",
            file=sys.stderr,
        )
        for f in result.get("quality_failures", []):
            print(f"  quality: {f}", file=sys.stderr)
        for f in result.get("speed_failures", []):
            print(f"  speed: {f}", file=sys.stderr)
        return 1

    print("\nGATE APPROVED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
