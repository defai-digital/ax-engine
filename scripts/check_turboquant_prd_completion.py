#!/usr/bin/env python3
"""Report TurboQuant codec/kernel PRD completion status.

This is a PRD completion checker, not a benchmark runner. It keeps the final
"complete" decision fail-closed by joining the existing TurboQuant evidence
surfaces:

- long-context quality/promotion readiness;
- fused cold-decode microbenchmark evidence; and
- explicit short-decode speedup evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ax.turboquant_prd_completion.v1"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks/results/turboquant"
DEFAULT_MODELS_ROOT = REPO_ROOT / ".internal/models"
DEFAULT_MIN_MICROBENCH_COLD_TOKENS = 8192
DEFAULT_MIN_D3_SPEEDUP_VS_DIM = 1.5
DEFAULT_MIN_D4_SPEEDUP = 2.0


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality_checker = _load_module(
    "check_turboquant_quality_artifact",
    Path(__file__).with_name("check_turboquant_quality_artifact.py"),
)
microbench_checker = _load_module(
    "check_turboquant_microbench_artifact",
    Path(__file__).with_name("check_turboquant_microbench_artifact.py"),
)
promotion_checker = _load_module(
    "check_turboquant_promotion_readiness",
    Path(__file__).with_name("check_turboquant_promotion_readiness.py"),
)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def discover_json_artifacts(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(results_root.rglob("*.json"))


def discover_microbench_artifacts(results_root: Path) -> list[Path]:
    artifacts: list[Path] = []
    for path in discover_json_artifacts(results_root):
        try:
            doc = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if doc.get("schema_version") == microbench_checker.SCHEMA_VERSION:
            artifacts.append(path)
    return artifacts


def inspect_microbench_artifact(
    path: Path,
    *,
    min_cold_tokens: int,
    min_speedup_vs_dim: float,
) -> dict[str, Any]:
    doc: dict[str, Any] | None = None
    try:
        doc = _load_json(path)
        microbench_checker.validate_artifact(
            doc,
            min_cold_tokens=min_cold_tokens,
            min_speedup_vs_dim_parallel=min_speedup_vs_dim,
        )
    except Exception as exc:  # noqa: BLE001 - readiness reports fail-closed reasons.
        return {
            "path": _relative(path),
            "passes_gate": False,
            "blocker": str(exc),
            "largest_cold_tokens": largest_cold_tokens(doc),
            "head_dims": row_head_dims(doc),
        }
    return {
        "path": _relative(path),
        "passes_gate": True,
        "blocker": None,
        "largest_cold_tokens": largest_cold_tokens(doc),
        "head_dims": row_head_dims(doc),
    }


def largest_cold_tokens(doc: dict[str, Any] | None) -> int | None:
    if not isinstance(doc, dict):
        return None
    rows = doc.get("rows")
    if not isinstance(rows, list):
        return None
    values = [
        row.get("cold_tokens")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("cold_tokens"), int)
    ]
    return max(values) if values else None


def row_head_dims(doc: dict[str, Any] | None) -> list[int]:
    if not isinstance(doc, dict):
        return []
    rows = doc.get("rows")
    if not isinstance(rows, list):
        return []
    dims = {
        row.get("head_dim")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("head_dim"), int)
    }
    return sorted(dims)


def inspect_short_decode_artifact(path: Path, *, min_speedup: float) -> dict[str, Any]:
    try:
        doc = _load_json(path)
        speedup = doc.get("short_decode_speedup")
        if not isinstance(speedup, (int, float)) or isinstance(speedup, bool):
            raise ValueError("short_decode_speedup must be numeric")
        if speedup < min_speedup:
            raise ValueError(f"short_decode_speedup must be >= {min_speedup}")
        new_tokens = doc.get("new_tokens")
        cold_tokens = doc.get("cold_tokens")
        if new_tokens != 1:
            raise ValueError("new_tokens must be 1 for D4 PRD evidence")
        if cold_tokens != 1024:
            raise ValueError("cold_tokens must be 1024 for D4 PRD evidence")
    except Exception as exc:  # noqa: BLE001 - readiness reports fail-closed reasons.
        return {
            "path": _relative(path),
            "passes_gate": False,
            "blocker": str(exc),
        }
    return {
        "path": _relative(path),
        "passes_gate": True,
        "blocker": None,
        "short_decode_speedup": float(doc["short_decode_speedup"]),
    }


def quality_families(quality_artifacts: list[dict[str, Any]]) -> list[str]:
    families: set[str] = set()
    for item in quality_artifacts:
        if item.get("passes_quality_gate") is not True:
            continue
        path_value = item.get("path")
        if not isinstance(path_value, str):
            continue
        try:
            doc = _load_json(REPO_ROOT / path_value)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        model = doc.get("model")
        family = model.get("family") if isinstance(model, dict) else None
        if isinstance(family, str) and family:
            families.add(family)
    return sorted(families)


def build_report(
    *,
    models_root: Path,
    results_root: Path,
    quality_artifacts: list[Path],
    microbench_artifacts: list[Path],
    short_decode_artifacts: list[Path],
    required_model_families: list[str],
    require_artifact_files: bool,
    min_microbench_cold_tokens: int,
    min_d3_speedup_vs_dim: float,
    min_d4_speedup: float,
) -> dict[str, Any]:
    promotion_report = promotion_checker.build_report(
        models_root=models_root,
        results_root=results_root,
        artifacts=quality_artifacts,
        require_artifact_files=require_artifact_files,
        root=REPO_ROOT,
    )
    discovered_microbench = microbench_artifacts or discover_microbench_artifacts(results_root)
    microbench = [
        inspect_microbench_artifact(
            path,
            min_cold_tokens=min_microbench_cold_tokens,
            min_speedup_vs_dim=min_d3_speedup_vs_dim,
        )
        for path in discovered_microbench
    ]
    short_decode = [
        inspect_short_decode_artifact(path, min_speedup=min_d4_speedup)
        for path in short_decode_artifacts
    ]

    families = quality_families(promotion_report["quality_artifacts"])
    missing_families = [
        family for family in required_model_families if family not in set(families)
    ]

    blockers: list[str] = []
    if missing_families:
        blockers.append(
            "missing D1+D2 model quality evidence for families: "
            + ", ".join(missing_families)
        )
    if not any(item.get("passes_gate") is True for item in microbench):
        blockers.append(
            "missing D3 fused decode microbench evidence "
            f"(cold_tokens >= {min_microbench_cold_tokens}, speedup_vs_dim >= {min_d3_speedup_vs_dim})"
        )
    if not any(item.get("passes_gate") is True for item in short_decode):
        blockers.append(
            "missing D4 short decode speedup evidence "
            f"(1 new token, 1024 cold, speedup >= {min_d4_speedup})"
        )
    promotion_blockers = promotion_report["decision"]["blockers"]
    if promotion_blockers:
        blockers.extend(str(blocker) for blocker in promotion_blockers)

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": {
            "prd_complete": not blockers,
            "blockers": blockers,
            "public_docs_should_remain_experimental": bool(blockers),
        },
        "requirements": {
            "required_model_families": required_model_families,
            "min_microbench_cold_tokens": min_microbench_cold_tokens,
            "min_d3_speedup_vs_dim_parallel": min_d3_speedup_vs_dim,
            "min_d4_short_decode_speedup": min_d4_speedup,
        },
        "evidence": {
            "quality_families": families,
            "microbench_artifacts": microbench,
            "short_decode_artifacts": short_decode,
            "promotion_readiness": promotion_report,
        },
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--quality-artifact", type=Path, action="append", default=[])
    parser.add_argument("--microbench-artifact", type=Path, action="append", default=[])
    parser.add_argument("--short-decode-artifact", type=Path, action="append", default=[])
    parser.add_argument(
        "--required-model-families",
        default="gemma4,qwen3,qwen3_5",
        help="comma-separated model families required for D1+D2 quality evidence",
    )
    parser.add_argument(
        "--min-microbench-cold-tokens",
        type=int,
        default=DEFAULT_MIN_MICROBENCH_COLD_TOKENS,
    )
    parser.add_argument(
        "--min-d3-speedup-vs-dim",
        type=float,
        default=DEFAULT_MIN_D3_SPEEDUP_VS_DIM,
    )
    parser.add_argument("--min-d4-speedup", type=float, default=DEFAULT_MIN_D4_SPEEDUP)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-blockers", action="store_true")
    parser.add_argument("--no-require-artifact-files", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = build_report(
            models_root=args.models_root,
            results_root=args.results_root,
            quality_artifacts=args.quality_artifact,
            microbench_artifacts=args.microbench_artifact,
            short_decode_artifacts=args.short_decode_artifact,
            required_model_families=parse_csv(args.required_model_families),
            require_artifact_files=not args.no_require_artifact_files,
            min_microbench_cold_tokens=args.min_microbench_cold_tokens,
            min_d3_speedup_vs_dim=args.min_d3_speedup_vs_dim,
            min_d4_speedup=args.min_d4_speedup,
        )
        text = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            sys.stdout.write(text)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    has_blockers = bool(report["decision"]["blockers"])
    if args.fail_on_blockers and has_blockers:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
