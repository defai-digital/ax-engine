#!/usr/bin/env python3
"""Build and validate a TurboQuant long-context quality artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = Path(__file__).with_name("check_turboquant_quality_artifact.py")
CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_turboquant_quality_artifact",
    CHECKER_PATH,
)
assert CHECKER_SPEC and CHECKER_SPEC.loader
checker = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(checker)


class QualityArtifactBuildError(RuntimeError):
    """Raised when benchmark inputs cannot produce promotion evidence."""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise QualityArtifactBuildError(f"{path} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_label(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _metric_median(row: dict[str, Any], metric: str) -> float:
    data = row.get(metric)
    if not isinstance(data, dict):
        raise QualityArtifactBuildError(f"row missing {metric} summary")
    value = data.get("median", data.get("mean"))
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise QualityArtifactBuildError(f"row {metric} summary must be numeric")
    return float(value)


def _matching_rows(
    doc: dict[str, Any],
    *,
    engine: str,
    context_tokens: int,
    generation_tokens: int,
) -> list[dict[str, Any]]:
    rows = []
    for row in doc.get("results", []):
        if not isinstance(row, dict):
            continue
        if row.get("engine") != engine:
            continue
        if int(row.get("prompt_tokens", -1)) != context_tokens:
            continue
        if int(row.get("generation_tokens", -1)) != generation_tokens:
            continue
        rows.append(row)
    return rows


def _select_baseline_row(
    doc: dict[str, Any],
    *,
    engine: str,
    context_tokens: int,
    generation_tokens: int,
) -> dict[str, Any]:
    candidates = [
        row
        for row in _matching_rows(
            doc,
            engine=engine,
            context_tokens=context_tokens,
            generation_tokens=generation_tokens,
        )
        if row.get("experimental_mlx_kv_compression") in {None, "disabled"}
    ]
    if len(candidates) != 1:
        raise QualityArtifactBuildError(
            "expected exactly one full-precision baseline row for "
            f"engine={engine} prompt_tokens={context_tokens} "
            f"generation_tokens={generation_tokens}, got {len(candidates)}"
        )
    return candidates[0]


def _select_candidate_row(
    doc: dict[str, Any],
    *,
    engine: str,
    context_tokens: int,
    generation_tokens: int,
) -> dict[str, Any]:
    candidates = [
        row
        for row in _matching_rows(
            doc,
            engine=engine,
            context_tokens=context_tokens,
            generation_tokens=generation_tokens,
        )
        if row.get("experimental_mlx_kv_compression") not in {None, "disabled"}
        or row.get("kv_compression_telemetry")
    ]
    if len(candidates) != 1:
        raise QualityArtifactBuildError(
            "expected exactly one TurboQuant candidate row for "
            f"engine={engine} prompt_tokens={context_tokens} "
            f"generation_tokens={generation_tokens}, got {len(candidates)}"
        )
    return candidates[0]


def _quality_metrics(doc: dict[str, Any]) -> dict[str, Any]:
    metrics = doc.get("metrics", doc)
    if not isinstance(metrics, dict):
        raise QualityArtifactBuildError("quality metrics must be a JSON object")
    return {
        "max_abs_diff": metrics.get("max_abs_diff"),
        "mean_abs_diff": metrics.get("mean_abs_diff"),
        "min_cosine_similarity": metrics.get("min_cosine_similarity"),
    }


def build_quality_artifact(
    *,
    baseline_benchmark: Path,
    candidate_benchmark: Path,
    quality_metrics: Path,
    manifest: Path,
    model_id: str,
    model_family: str,
    model_revision: str,
    head_dim: int,
    context_tokens: int,
    generation_tokens: int,
    baseline_engine: str,
    candidate_engine: str,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    baseline_doc = _load_json(baseline_benchmark)
    candidate_doc = _load_json(candidate_benchmark)
    metrics_doc = _load_json(quality_metrics)
    baseline_row = _select_baseline_row(
        baseline_doc,
        engine=baseline_engine,
        context_tokens=context_tokens,
        generation_tokens=generation_tokens,
    )
    candidate_row = _select_candidate_row(
        candidate_doc,
        engine=candidate_engine,
        context_tokens=context_tokens,
        generation_tokens=generation_tokens,
    )

    prompt_hash = candidate_row.get("prompt_token_ids_sha256")
    if not isinstance(prompt_hash, str):
        raise QualityArtifactBuildError("candidate row missing prompt_token_ids_sha256")
    if baseline_row.get("prompt_token_ids_sha256") != prompt_hash:
        raise QualityArtifactBuildError("baseline and candidate prompt hashes differ")

    route_decisions = candidate_row.get("kv_compression_telemetry")
    if not isinstance(route_decisions, dict):
        raise QualityArtifactBuildError("candidate row missing kv_compression_telemetry")

    baseline_decode = _metric_median(baseline_row, "decode_tok_s")
    candidate_decode = _metric_median(candidate_row, "decode_tok_s")
    if baseline_decode <= 0:
        raise QualityArtifactBuildError("baseline decode throughput must be positive")

    kv_saved_kib = route_decisions.get("ax_mlx_kv_compression_estimated_saved_kib")
    if not isinstance(kv_saved_kib, int):
        raise QualityArtifactBuildError("candidate route metadata missing saved KiB")

    metrics = _quality_metrics(metrics_doc)
    metrics.update(
        {
            "decode_tok_s_ratio_to_baseline": candidate_decode / baseline_decode,
            "kv_saved_kib": kv_saved_kib,
        }
    )
    performance_blockers = checker.performance_gate_blockers(metrics)

    artifact = {
        "schema_version": checker.SCHEMA_VERSION,
        "model": {
            "id": model_id,
            "family": model_family,
            "revision": model_revision,
            "head_dim": head_dim,
        },
        "workload": {
            "manifest": _path_label(manifest, root),
            "context_tokens": context_tokens,
            "generation_tokens": generation_tokens,
            "prompt_sha256": prompt_hash,
        },
        "baseline": {
            "backend": "mlx",
            "kv_compression_mode": "disabled",
            "engine": baseline_row.get("engine"),
            "decode_tok_s": baseline_decode,
        },
        "candidate": {
            "backend": "mlx",
            "kv_compression_mode": candidate_row.get("experimental_mlx_kv_compression"),
            "preset": "k8v4",
            "quality_profile": "reference_k8v4",
            "decode_path": candidate_row.get("kv_compression_decode_path"),
            "engine": candidate_row.get("engine"),
            "decode_tok_s": candidate_decode,
        },
        "metrics": metrics,
        "route_metadata": {"crossover_decisions": route_decisions},
        "artifacts": [
            {
                "role": "baseline",
                "path": _path_label(baseline_benchmark, root),
                "sha256": _sha256_file(baseline_benchmark),
            },
            {
                "role": "candidate",
                "path": _path_label(candidate_benchmark, root),
                "sha256": _sha256_file(candidate_benchmark),
            },
            {
                "role": "quality_metrics",
                "path": _path_label(quality_metrics, root),
                "sha256": _sha256_file(quality_metrics),
            },
        ],
        "decision": {
            "passed": True,
            "quality_gate_passed": True,
            "performance_promotion_ready": not performance_blockers,
            "performance_blockers": performance_blockers,
            "public_support_docs_approved": False,
        },
    }
    checker.validate_artifact(artifact, root=root)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-benchmark", required=True, type=Path)
    parser.add_argument("--candidate-benchmark", required=True, type=Path)
    parser.add_argument("--quality-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "benchmarks/manifests/scenario/long_context_qwen_8k.json",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--generation-tokens", type=int, default=256)
    parser.add_argument("--baseline-engine", default="ax_engine_mlx")
    parser.add_argument("--candidate-engine", default="ax_engine_mlx")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    try:
        artifact = build_quality_artifact(
            baseline_benchmark=args.baseline_benchmark,
            candidate_benchmark=args.candidate_benchmark,
            quality_metrics=args.quality_metrics,
            manifest=args.manifest,
            model_id=args.model_id,
            model_family=args.model_family,
            model_revision=args.model_revision,
            head_dim=args.head_dim,
            context_tokens=args.context_tokens,
            generation_tokens=args.generation_tokens,
            baseline_engine=args.baseline_engine,
            candidate_engine=args.candidate_engine,
            root=args.root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    except (
        OSError,
        json.JSONDecodeError,
        QualityArtifactBuildError,
        checker.ArtifactValidationError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"ok: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
