#!/usr/bin/env python3
"""Validate TurboQuant long-context quality/path gate artifacts.

This script intentionally validates an evidence artifact instead of running a
model. The model run can be expensive and host-specific; the contract here keeps
the fused-path quality decision fail-closed once such a run has been produced.
Public-support promotion is stricter: the readiness checker also requires the
separate decode-throughput performance gate to pass.
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
MAX_RUNTIME_PRODUCTION_BLOCKERS = 1
REQUIRED_CANDIDATE_COMPRESSION_MODE = "turboquant-fused-experimental"
SUPPORTED_HEAD_DIMS = {128, 256, 512}
DECODE_PATH_LABELS = {
    1: "full_precision_shadow",
    2: "fused_compressed_decode",
    3: "cpu_oracle_compressed_decode",
}
FUSED_DECODE_FALLBACK_REASON_LABELS = {
    0: "none",
    1: "shadow_only",
    2: "missing_runtime_storage",
    3: "unsupported_preset",
    4: "runner_not_integrated",
    5: "cpu_oracle_unavailable",
}
FUSED_DECODE_BLOCKED_COUNTERS = {
    "prefill_only": "ax_mlx_kv_compression_fused_decode_blocked_prefill_only",
    "attention_kind": "ax_mlx_kv_compression_fused_decode_blocked_attention_kind",
    "ineligible_layer": "ax_mlx_kv_compression_fused_decode_blocked_ineligible_layer",
    "unsupported_preset": "ax_mlx_kv_compression_fused_decode_blocked_unsupported_preset",
    "unsupported_head_dim": "ax_mlx_kv_compression_fused_decode_blocked_unsupported_head_dim",
    "gqa": "ax_mlx_kv_compression_fused_decode_blocked_gqa",
    "missing_storage": "ax_mlx_kv_compression_fused_decode_blocked_missing_storage",
}

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
    "ax_mlx_kv_compression_decode_path",
    "ax_mlx_kv_compression_fused_decode_candidates",
    "ax_mlx_kv_compression_fused_decode_attempts",
    "ax_mlx_kv_compression_fused_decode_successes",
    "ax_mlx_kv_compression_fused_decode_metal_successes",
    "ax_mlx_kv_compression_fused_decode_fallbacks",
    "ax_mlx_kv_compression_fused_decode_fallback_reason",
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


def _decisions_from_route_metadata(route_metadata: dict[str, Any]) -> dict[str, Any]:
    decisions = route_metadata.get("crossover_decisions", route_metadata)
    _require(isinstance(decisions, dict), "route_metadata.crossover_decisions must be an object")
    return decisions


def _optional_int(decisions: dict[str, Any], key: str) -> int | None:
    value = decisions.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def route_truth_surface(route_metadata: dict[str, Any]) -> dict[str, Any]:
    """Return promotion-facing labels for the real TurboQuant decode route."""

    decisions = _decisions_from_route_metadata(route_metadata)
    decode_path = _optional_int(decisions, "ax_mlx_kv_compression_decode_path")
    attempts = _optional_int(decisions, "ax_mlx_kv_compression_fused_decode_attempts")
    successes = _optional_int(decisions, "ax_mlx_kv_compression_fused_decode_successes")
    metal_successes = _optional_int(
        decisions,
        "ax_mlx_kv_compression_fused_decode_metal_successes",
    )
    fallbacks = _optional_int(decisions, "ax_mlx_kv_compression_fused_decode_fallbacks")
    fallback_reason = _optional_int(
        decisions,
        "ax_mlx_kv_compression_fused_decode_fallback_reason",
    )
    blocked_counters = {
        label: _optional_int(decisions, key)
        for label, key in FUSED_DECODE_BLOCKED_COUNTERS.items()
    }
    blocked_total = sum(value for value in blocked_counters.values() if isinstance(value, int))
    blocked_reasons = [
        label
        for label, value in blocked_counters.items()
        if isinstance(value, int) and value > 0
    ]
    fused_path_selected = decode_path == 2
    fused_success_observed = (
        isinstance(successes, int)
        and successes > 0
        and isinstance(metal_successes, int)
        and metal_successes > 0
    )
    zero_fallbacks = fallbacks == 0 and fallback_reason == 0

    return {
        "decode_path_code": decode_path,
        "decode_path_label": DECODE_PATH_LABELS.get(
            decode_path,
            f"unknown_{decode_path}" if decode_path is not None else "missing",
        ),
        "fused_decode_attempts": attempts,
        "fused_decode_successes": successes,
        "fused_decode_metal_successes": metal_successes,
        "fused_decode_fallbacks": fallbacks,
        "fused_decode_fallback_reason_code": fallback_reason,
        "fused_decode_fallback_reason_label": FUSED_DECODE_FALLBACK_REASON_LABELS.get(
            fallback_reason,
            f"unknown_{fallback_reason}" if fallback_reason is not None else "missing",
        ),
        "fused_decode_blocked_counters": blocked_counters,
        "fused_decode_blocked_total": blocked_total,
        "fused_decode_blocked_reasons": blocked_reasons,
        "fused_path_selected": fused_path_selected,
        "fused_success_observed": fused_success_observed,
        "zero_fallbacks": zero_fallbacks,
        "promotion_path_ready": fused_path_selected and fused_success_observed and zero_fallbacks,
    }


def performance_gate_blockers(metrics: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    try:
        ratio = _number(
            metrics.get("decode_tok_s_ratio_to_baseline"),
            "metrics.decode_tok_s_ratio_to_baseline",
        )
    except ArtifactValidationError as exc:
        return [str(exc)]
    if ratio < MIN_DECODE_RATIO_TO_BASELINE:
        blockers.append(
            "metrics.decode_tok_s_ratio_to_baseline must be >= "
            f"{MIN_DECODE_RATIO_TO_BASELINE}"
        )
    return blockers


def _validate_route_metadata(route_metadata: dict[str, Any]) -> None:
    decisions = _decisions_from_route_metadata(route_metadata)

    missing = sorted(REQUIRED_ROUTE_KEYS - decisions.keys())
    _require(not missing, f"route metadata missing required keys: {', '.join(missing)}")

    _require(
        _integer(decisions["ax_mlx_kv_compression_route_metadata_schema"], "route schema") >= 2,
        "route metadata schema must be >= 2",
    )
    _require(
        _integer(decisions["ax_mlx_kv_compression_production_ready"], "production ready") == 0,
        "quality artifacts must not mark TurboQuant production-ready before artifact approval",
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
    _require(
        _integer(decisions["ax_mlx_kv_compression_decode_path"], "compression decode path") == 2,
        "route metadata must report fused_compressed_decode path code 2",
    )
    _require(
        _integer(
            decisions["ax_mlx_kv_compression_fused_decode_candidates"],
            "fused decode candidates",
        )
        > 0,
        "route metadata must show fused decode candidate snapshots",
    )
    _require(
        _integer(
            decisions["ax_mlx_kv_compression_fused_decode_attempts"],
            "fused decode attempts",
        )
        > 0,
        "route metadata must show fused decode attempts",
    )
    _require(
        _integer(
            decisions["ax_mlx_kv_compression_fused_decode_successes"],
            "fused decode successes",
        )
        > 0,
        "route metadata must show fused decode successes",
    )
    _require(
        _integer(
            decisions["ax_mlx_kv_compression_fused_decode_metal_successes"],
            "fused decode Metal successes",
        )
        > 0,
        "route metadata must show Metal fused decode successes",
    )
    _require(
        _integer(
            decisions["ax_mlx_kv_compression_fused_decode_fallbacks"],
            "fused decode fallbacks",
        )
        == 0,
        "route metadata must show zero fused decode fallbacks",
    )
    _require(
        _integer(
            decisions["ax_mlx_kv_compression_fused_decode_fallback_reason"],
            "fused decode fallback reason",
        )
        == 0,
        "route metadata must report no fused decode fallback reason",
    )


def validate_artifact(doc: dict[str, Any], *, root: Path = REPO_ROOT, require_files: bool = True) -> None:
    _require(doc.get("schema_version") == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}")

    model = _mapping(doc.get("model"), "model")
    _string(model.get("id"), "model.id")
    _string(model.get("family"), "model.family")
    _string(model.get("revision"), "model.revision")
    _require(
        _integer(model.get("head_dim"), "model.head_dim") in SUPPORTED_HEAD_DIMS,
        "initial gate requires head_dim=128, head_dim=256, or head_dim=512",
    )

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
    _require(
        candidate.get("kv_compression_mode") == REQUIRED_CANDIDATE_COMPRESSION_MODE,
        f"candidate.kv_compression_mode must be {REQUIRED_CANDIDATE_COMPRESSION_MODE}",
    )
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
    _number(
        metrics.get("decode_tok_s_ratio_to_baseline"),
        "metrics.decode_tok_s_ratio_to_baseline",
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
        decision.get("quality_gate_passed", True) is True,
        "decision.quality_gate_passed must be true",
    )
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
