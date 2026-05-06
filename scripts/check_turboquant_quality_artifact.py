#!/usr/bin/env python3
"""Validate TurboQuant long-context quality gate artifacts.

This script intentionally validates an evidence artifact instead of running a
model. The model run can be expensive and host-specific; the contract here keeps
the promotion decision fail-closed once such a run has been produced.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "ax.turboquant_quality_gate.v1"
MIN_CONTEXT_TOKENS = 8192
MIN_GENERATION_TOKENS = 128
MIN_DECODE_RATIO_TO_BASELINE = 0.85
MIN_SAVED_KIB = 1
MAX_RUNTIME_PRODUCTION_BLOCKERS = 2

QUALITY_GATES = {
    "reference_k8v4": {
        "max_abs_diff": 0.04,
        "mean_abs_diff": 0.02,
        "min_cosine_similarity": 0.998,
    }
}

REQUIRED_ROUTE_KEYS = {
    "ax_mlx_kv_compression_route_metadata_schema",
    "ax_mlx_kv_compression_production_ready",
    "ax_mlx_kv_compression_production_blockers",
    "ax_mlx_kv_compression_preset",
    "ax_mlx_kv_compression_key_bits",
    "ax_mlx_kv_compression_value_bits",
    "ax_mlx_kv_compression_eligible_layers",
    "ax_mlx_kv_compression_candidate_token_layers",
    "ax_mlx_kv_compression_estimated_saved_kib",
    "ax_mlx_kv_compression_runtime_storage_written_slots",
}

HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactValidationError(RuntimeError):
    """Raised when a TurboQuant quality artifact is not promotion evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactValidationError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _string(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be a non-empty string")
    return value


def _number(value: Any, field: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} must be numeric")
    return float(value)


def _integer(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be an integer")
    return value


def _sha256(value: Any, field: str) -> str:
    digest = _string(value, field)
    _require(bool(HEX_SHA256_RE.match(digest)), f"{field} must be a lowercase sha256 hex digest")
    return digest


def _resolve_artifact_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def _validate_referenced_file(entry: dict[str, Any], field: str, root: Path, require_files: bool) -> None:
    path = _string(entry.get("path"), f"{field}.path")
    _sha256(entry.get("sha256"), f"{field}.sha256")
    if require_files:
        resolved = _resolve_artifact_path(path, root)
        _require(resolved.exists(), f"{field}.path does not exist: {path}")


def _validate_quality_gate(metrics: dict[str, Any], profile: str) -> None:
    gates = QUALITY_GATES.get(profile)
    _require(gates is not None, f"unsupported quality_profile {profile!r}")

    max_abs = _number(metrics.get("max_abs_diff"), "metrics.max_abs_diff")
    mean_abs = _number(metrics.get("mean_abs_diff"), "metrics.mean_abs_diff")
    min_cos = _number(
        metrics.get("min_cosine_similarity"),
        "metrics.min_cosine_similarity",
    )

    _require(
        max_abs <= gates["max_abs_diff"],
        "metrics.max_abs_diff exceeds reference_k8v4 limit",
    )
    _require(
        mean_abs <= gates["mean_abs_diff"],
        "metrics.mean_abs_diff exceeds reference_k8v4 limit",
    )
    _require(
        min_cos >= gates["min_cosine_similarity"],
        "metrics.min_cosine_similarity is below reference_k8v4 limit",
    )


def _validate_route_metadata(route_metadata: dict[str, Any]) -> None:
    decisions = route_metadata.get("crossover_decisions", route_metadata)
    _require(isinstance(decisions, dict), "route_metadata.crossover_decisions must be an object")

    missing = sorted(REQUIRED_ROUTE_KEYS - decisions.keys())
    _require(not missing, f"route metadata missing required keys: {', '.join(missing)}")

    _require(
        _integer(decisions["ax_mlx_kv_compression_route_metadata_schema"], "route schema") >= 1,
        "route metadata schema must be >= 1",
    )
    _require(
        _integer(decisions["ax_mlx_kv_compression_production_ready"], "production ready") == 0,
        "quality artifacts must not mark TurboQuant production-ready before public docs approval",
    )
    _require(
        _integer(decisions["ax_mlx_kv_compression_production_blockers"], "production blockers")
        <= MAX_RUNTIME_PRODUCTION_BLOCKERS,
        "quality artifact route metadata reports unexpected production blockers",
    )
    _require(
        _integer(decisions["ax_mlx_kv_compression_preset"], "compression preset") == 1,
        "initial quality gate only accepts K8V4 route preset code 1",
    )
    _require(
        _integer(decisions["ax_mlx_kv_compression_key_bits"], "compression key bits") == 8,
        "initial quality gate only accepts 8-bit keys",
    )
    _require(
        _integer(decisions["ax_mlx_kv_compression_value_bits"], "compression value bits") == 4,
        "initial quality gate only accepts 4-bit values",
    )
    _require(
        _integer(decisions["ax_mlx_kv_compression_eligible_layers"], "eligible layers") > 0,
        "route metadata must show eligible TurboQuant layers",
    )
    _require(
        _integer(decisions["ax_mlx_kv_compression_candidate_token_layers"], "candidate token layers")
        > 0,
        "route metadata must show cold candidate token-layers",
    )
    _require(
        _integer(decisions["ax_mlx_kv_compression_estimated_saved_kib"], "estimated saved KiB")
        >= MIN_SAVED_KIB,
        "route metadata must show positive estimated saved KiB",
    )
    _require(
        _integer(
            decisions["ax_mlx_kv_compression_runtime_storage_written_slots"],
            "runtime storage written slots",
        )
        > 0,
        "route metadata must show runtime compressed slot writes",
    )


def validate_artifact(doc: dict[str, Any], *, root: Path = REPO_ROOT, require_files: bool = True) -> None:
    _require(doc.get("schema_version") == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}")

    model = _mapping(doc.get("model"), "model")
    _string(model.get("id"), "model.id")
    _string(model.get("family"), "model.family")
    _string(model.get("revision"), "model.revision")
    _require(_integer(model.get("head_dim"), "model.head_dim") == 128, "initial gate requires head_dim=128")

    workload = _mapping(doc.get("workload"), "workload")
    manifest = _string(workload.get("manifest"), "workload.manifest")
    if require_files:
        _require(
            _resolve_artifact_path(manifest, root).exists(),
            f"workload.manifest does not exist: {manifest}",
        )
    _require(
        _integer(workload.get("context_tokens"), "workload.context_tokens") >= MIN_CONTEXT_TOKENS,
        f"workload.context_tokens must be >= {MIN_CONTEXT_TOKENS}",
    )
    _require(
        _integer(workload.get("generation_tokens"), "workload.generation_tokens")
        >= MIN_GENERATION_TOKENS,
        f"workload.generation_tokens must be >= {MIN_GENERATION_TOKENS}",
    )
    _sha256(workload.get("prompt_sha256"), "workload.prompt_sha256")

    baseline = _mapping(doc.get("baseline"), "baseline")
    _require(baseline.get("backend") == "mlx", "baseline.backend must be mlx")
    _require(
        baseline.get("kv_compression_mode") == "disabled",
        "baseline.kv_compression_mode must be disabled",
    )

    candidate = _mapping(doc.get("candidate"), "candidate")
    _require(candidate.get("backend") == "mlx", "candidate.backend must be mlx")
    _require(candidate.get("preset") == "k8v4", "candidate.preset must be k8v4")
    _require(
        candidate.get("quality_profile") == "reference_k8v4",
        "candidate.quality_profile must be reference_k8v4",
    )
    _require(
        candidate.get("decode_path") == "fused_compressed_decode",
        "candidate.decode_path must be fused_compressed_decode",
    )

    metrics = _mapping(doc.get("metrics"), "metrics")
    _validate_quality_gate(metrics, candidate["quality_profile"])
    _require(
        _number(metrics.get("decode_tok_s_ratio_to_baseline"), "metrics.decode_tok_s_ratio_to_baseline")
        >= MIN_DECODE_RATIO_TO_BASELINE,
        f"metrics.decode_tok_s_ratio_to_baseline must be >= {MIN_DECODE_RATIO_TO_BASELINE}",
    )
    _require(
        _integer(metrics.get("kv_saved_kib"), "metrics.kv_saved_kib") >= MIN_SAVED_KIB,
        "metrics.kv_saved_kib must be positive",
    )

    route_metadata = _mapping(doc.get("route_metadata"), "route_metadata")
    _validate_route_metadata(route_metadata)

    artifacts = doc.get("artifacts")
    _require(isinstance(artifacts, list) and artifacts, "artifacts must be a non-empty array")
    roles = set()
    for index, artifact in enumerate(artifacts):
        entry = _mapping(artifact, f"artifacts[{index}]")
        role = _string(entry.get("role"), f"artifacts[{index}].role")
        roles.add(role)
        _validate_referenced_file(entry, f"artifacts[{index}]", root, require_files)
    _require({"baseline", "candidate"}.issubset(roles), "artifacts must include baseline and candidate roles")

    decision = _mapping(doc.get("decision"), "decision")
    _require(decision.get("passed") is True, "decision.passed must be true")
    _require(
        decision.get("public_support_docs_approved") is False,
        "decision.public_support_docs_approved must remain false for this internal gate",
    )


def load_and_validate(path: Path, *, root: Path = REPO_ROOT, require_files: bool = True) -> None:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"{path}: invalid JSON: {exc}") from exc
    _require(isinstance(doc, dict), f"{path}: top-level JSON must be an object")
    validate_artifact(doc, root=root, require_files=require_files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument(
        "--no-require-files",
        action="store_true",
        help="validate artifact shape without checking referenced paths exist",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="root used to resolve relative manifest and artifact paths",
    )
    args = parser.parse_args(argv)

    for artifact in args.artifacts:
        try:
            load_and_validate(
                artifact,
                root=args.root,
                require_files=not args.no_require_files,
            )
        except ArtifactValidationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"ok: {artifact}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
