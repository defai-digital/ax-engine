#!/usr/bin/env python3
"""Fail-closed checker for durable tiered prefix-cache promotion artifacts.

Validates schema ``ax.disk_prefix_cache_promotion.v2`` fields and PRD §9.2
gates when a complete artifact is present. Does not rewrite source data.

Exit codes:
  0  gates pass (or --allow-incomplete with only schema-valid incomplete work)
  2  usage / IO error
  3  schema or correctness failure
  4  performance gate failure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "ax.disk_prefix_cache_promotion.v2"
REQUIRED_TOP = (
    "schema",
    "run_id",
    "ax_commit",
    "modes",
    "models",
    "correctness",
)
REQUIRED_MODE_KEYS = (
    "cold_prefill",
    "l1_hit",
    "l2_hit_warm_fs",
    "l2_hit_cold_fs",
    "producer_l2_enabled",
)
PREFIX_BUCKETS = ("256", "2k", "8k", "32k", "gte_32k")


class DiskPrefixCachePromotionError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = 3) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def load_artifact(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DiskPrefixCachePromotionError(f"cannot read {path}: {exc}", exit_code=2) from exc
    except json.JSONDecodeError as exc:
        raise DiskPrefixCachePromotionError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DiskPrefixCachePromotionError(f"{path}: root must be an object")
    return data


def require_keys(obj: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise DiskPrefixCachePromotionError(f"{where}: missing keys {missing}")


def check_schema(artifact: dict[str, Any], *, path: Path) -> None:
    require_keys(artifact, REQUIRED_TOP, str(path))
    if artifact.get("schema") != SCHEMA:
        raise DiskPrefixCachePromotionError(
            f"{path}: schema must be {SCHEMA!r}, got {artifact.get('schema')!r}"
        )
    modes = artifact["modes"]
    if not isinstance(modes, dict):
        raise DiskPrefixCachePromotionError(f"{path}: modes must be an object")
    for mode in REQUIRED_MODE_KEYS:
        if mode not in modes:
            raise DiskPrefixCachePromotionError(f"{path}: modes missing {mode!r}")
    fs_state = modes.get("l2_hit_cold_fs", {})
    if isinstance(fs_state, dict):
        provenance = fs_state.get("filesystem_cache_state")
        if provenance not in ("cold_fs", "warm_fs", "unknown", None):
            raise DiskPrefixCachePromotionError(
                f"{path}: l2_hit_cold_fs.filesystem_cache_state invalid: {provenance!r}"
            )


def check_correctness(artifact: dict[str, Any], *, path: Path) -> None:
    correctness = artifact["correctness"]
    if not isinstance(correctness, dict):
        raise DiskPrefixCachePromotionError(f"{path}: correctness must be an object")
    if correctness.get("deterministic_match") is not True:
        raise DiskPrefixCachePromotionError(
            f"{path}: correctness.deterministic_match must be true"
        )
    if correctness.get("wrong_prefix_hits", 0) not in (0, 0.0):
        raise DiskPrefixCachePromotionError(f"{path}: wrong_prefix_hits must be 0")
    if correctness.get("corrupt_restores", 0) not in (0, 0.0):
        raise DiskPrefixCachePromotionError(f"{path}: corrupt_restores must be 0")


def _p95_ttft(bucket: dict[str, Any]) -> float | None:
    for key in ("ttft_p95_ms", "p95_ttft_ms", "ttft_p95_us"):
        if key in bucket and bucket[key] is not None:
            value = float(bucket[key])
            if key.endswith("_us"):
                return value / 1000.0
            return value
    return None


def check_performance_gates(artifact: dict[str, Any], *, path: Path) -> None:
    """PRD §9.2: admitted 8k/32k L2 p95 TTFT ≥25% better than cold; no losses."""
    gates = artifact.get("performance_gates") or artifact.get("promotion")
    if not isinstance(gates, dict):
        raise DiskPrefixCachePromotionError(
            f"{path}: performance_gates object required for promotion",
            exit_code=4,
        )
    if gates.get("decision") not in ("promote", "promoted", True):
        # Allow explicit not_promoted with reason for incomplete runs.
        if gates.get("decision") in ("not_promoted", "hold", False):
            raise DiskPrefixCachePromotionError(
                f"{path}: decision is not promote ({gates.get('decision')!r}): "
                f"{gates.get('summary') or gates.get('reason') or 'no reason'}",
                exit_code=4,
            )
        raise DiskPrefixCachePromotionError(
            f"{path}: performance_gates.decision must be promote or not_promoted",
            exit_code=4,
        )

    improvements = gates.get("admitted_bucket_improvements") or []
    if not improvements:
        raise DiskPrefixCachePromotionError(
            f"{path}: admitted_bucket_improvements required",
            exit_code=4,
        )
    for row in improvements:
        if not isinstance(row, dict):
            raise DiskPrefixCachePromotionError(f"{path}: improvement row must be object")
        bucket = str(row.get("prefix_bucket", ""))
        if bucket not in ("8k", "8k_32k", "32k", "gte_32k") and bucket not in PREFIX_BUCKETS:
            continue
        cold = row.get("cold_prefill_p95_ttft_ms")
        warm = row.get("l2_hit_p95_ttft_ms")
        if cold is None or warm is None:
            raise DiskPrefixCachePromotionError(
                f"{path}: bucket {bucket} missing cold/l2 p95 ttft",
                exit_code=4,
            )
        cold_f = float(cold)
        warm_f = float(warm)
        if warm_f > cold_f:
            raise DiskPrefixCachePromotionError(
                f"{path}: admitted bucket {bucket} loses to cold prefill "
                f"(l2={warm_f} > cold={cold_f})",
                exit_code=4,
            )
        if cold_f > 0 and (cold_f - warm_f) / cold_f < 0.25:
            # Only enforce 25% on declared 8k / 32k+ promotion buckets.
            if bucket in ("8k", "8k_32k", "32k", "gte_32k"):
                raise DiskPrefixCachePromotionError(
                    f"{path}: admitted bucket {bucket} needs ≥25% p95 TTFT gain; "
                    f"cold={cold_f} l2={warm_f}",
                    exit_code=4,
                )

    cold_fs = (artifact.get("modes") or {}).get("l2_hit_cold_fs") or {}
    if isinstance(cold_fs, dict) and cold_fs.get("filesystem_cache_state") == "unknown":
        raise DiskPrefixCachePromotionError(
            f"{path}: cold_fs provenance is unknown; cannot claim cold-fs promotion",
            exit_code=4,
        )


def check_disk_prefix_cache_promotion(
    path: Path,
    *,
    require_performance: bool = True,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    artifact = load_artifact(path)
    check_schema(artifact, path=path)
    if artifact.get("status") == "incomplete":
        if allow_incomplete:
            return {"path": str(path), "status": "incomplete", "schema_ok": True}
        raise DiskPrefixCachePromotionError(
            f"{path}: artifact status is incomplete (pass --allow-incomplete)",
            exit_code=4,
        )
    check_correctness(artifact, path=path)
    if require_performance:
        check_performance_gates(artifact, path=path)
    return {"path": str(path), "status": "ok", "schema_ok": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="+",
        type=Path,
        help="Promotion artifact JSON path(s)",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Validate schema without performance gates",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Accept artifacts with status=incomplete after schema check",
    )
    args = parser.parse_args(argv)
    try:
        for path in args.artifacts:
            result = check_disk_prefix_cache_promotion(
                path,
                require_performance=not args.schema_only,
                allow_incomplete=args.allow_incomplete,
            )
            print(json.dumps(result, sort_keys=True))
    except DiskPrefixCachePromotionError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
