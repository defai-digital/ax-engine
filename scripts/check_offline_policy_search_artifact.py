#!/usr/bin/env python3
"""Validate AX offline policy-search artifacts.

The checker validates evidence produced by the internal Quantum OP plan while
keeping implementation/public names neutral. It does not run a model or bless a
runtime policy; it only verifies that a search artifact is complete enough to be
used as diagnostic or promotion-review evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ax.offline_policy_search.v1"

RUNTIME_POLICY_TARGETS = {
    "turboquant_kv_policy",
    "ngram_speculation_policy",
    "prefix_cache_retention_policy",
    "moe_locality_policy",
}

FALLBACK_REQUIRED_TARGETS = {
    "turboquant_kv_policy",
    "ngram_speculation_policy",
}

PROMOTION_REVIEW_CLASSIFICATION = "promotion_ready_for_companion_prd_review"
CONFIRMATION_REQUIRED_CLASSIFICATIONS = {
    "candidate_win_needs_repeat",
    "negative_result",
    "rejected_noise",
}

ALLOWED_DECISIONS = {
    "diagnostic_only",
    "negative_result",
    "candidate_win_needs_repeat",
    PROMOTION_REVIEW_CLASSIFICATION,
    "rejected_quality",
    "rejected_fallbacks",
    "rejected_route_boundary",
    "rejected_noise",
}

DELEGATED_BACKENDS = {"llama_cpp", "mlx_lm_delegated"}


class OfflinePolicySearchArtifactError(RuntimeError):
    """Raised when an offline policy-search artifact violates the contract."""


@dataclass(frozen=True)
class OfflinePolicySearchCheckResult:
    artifact_count: int
    promotion_review_count: int
    diagnostic_count: int
    negative_count: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OfflinePolicySearchArtifactError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be a list")
    return value


def _string(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be an integer")
    return value


def _number(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be a number",
    )
    return float(value)


def _boolean(value: Any, field: str) -> bool:
    _require(isinstance(value, bool), f"{field} must be a boolean")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except OSError as error:
        raise OfflinePolicySearchArtifactError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise OfflinePolicySearchArtifactError(f"{path} is not valid JSON: {error}") from error
    return _mapping(payload, str(path))


def _route_selected_backend(route: dict[str, Any], field: str) -> str:
    return _string(route.get("selected_backend"), f"{field}.selected_backend")


def _validate_repo(repo: dict[str, Any]) -> None:
    _string(repo.get("commit"), "repo.commit")
    dirty = _boolean(repo.get("dirty"), "repo.dirty")
    if dirty:
        changed_files = _list(repo.get("changed_files"), "repo.changed_files")
        _require(changed_files, "repo.changed_files must be listed when repo.dirty=true")
        for index, item in enumerate(changed_files):
            _string(item, f"repo.changed_files[{index}]")


def _validate_model(model: dict[str, Any]) -> None:
    for field in ("id", "family", "artifacts_dir", "manifest_digest"):
        _string(model.get(field), f"model.{field}")


def _validate_search(search: dict[str, Any]) -> None:
    _string(search.get("algorithm"), "search.algorithm")
    _integer(search.get("seed"), "search.seed")
    space = _mapping(search.get("space"), "search.space")
    _require(space, "search.space must define at least one search dimension")
    for field, value in space.items():
        field_name = _string(field, "search.space key")
        values = _list(value, f"search.space.{field_name}")
        _require(values, f"search.space.{field_name} must not be empty")
        seen: set[str] = set()
        for index, item in enumerate(values):
            if isinstance(item, str):
                normalized = _string(item, f"search.space.{field_name}[{index}]")
            elif isinstance(item, int) and not isinstance(item, bool):
                normalized = str(item)
            else:
                raise OfflinePolicySearchArtifactError(
                    f"search.space.{field_name}[{index}] must be a non-empty string or integer"
                )
            _require(
                normalized not in seen,
                f"search.space.{field_name}[{index}] duplicates an earlier value",
            )
            seen.add(normalized)
    budget = _mapping(search.get("budget"), "search.budget")
    _require(
        _integer(budget.get("max_candidates"), "search.budget.max_candidates") > 0,
        "search.budget.max_candidates must be positive",
    )
    _require(
        _integer(budget.get("max_wall_time_seconds"), "search.budget.max_wall_time_seconds")
        > 0,
        "search.budget.max_wall_time_seconds must be positive",
    )


def _validate_objective(objective: dict[str, Any]) -> None:
    for field in ("maximize", "minimize", "hard_constraints"):
        values = _list(objective.get(field), f"objective.{field}")
        for index, item in enumerate(values):
            _string(item, f"objective.{field}[{index}]")


def _validate_baseline(baseline: dict[str, Any], target: str) -> None:
    _string(baseline.get("policy_id"), "baseline.policy_id")
    route = _mapping(baseline.get("route"), "baseline.route")
    backend = _route_selected_backend(route, "baseline.route")
    if target in RUNTIME_POLICY_TARGETS:
        _require(backend == "mlx", "baseline.route.selected_backend must be mlx")
    _integer(baseline.get("prompt_tokens"), "baseline.prompt_tokens")
    _integer(baseline.get("generation_tokens"), "baseline.generation_tokens")


def _validate_candidate_common(candidate: dict[str, Any], index: int, target: str) -> None:
    prefix = f"candidates[{index}]"
    _string(candidate.get("policy_id"), f"{prefix}.policy_id")
    _integer(candidate.get("prompt_tokens"), f"{prefix}.prompt_tokens")
    _integer(candidate.get("generation_tokens"), f"{prefix}.generation_tokens")
    _integer(candidate.get("seed"), f"{prefix}.seed")
    _mapping(candidate.get("policy"), f"{prefix}.policy")
    route = _mapping(candidate.get("route"), f"{prefix}.route")
    backend = _route_selected_backend(route, f"{prefix}.route")
    if target in RUNTIME_POLICY_TARGETS:
        _require(backend == "mlx", f"{prefix}.route.selected_backend must be mlx")
    _require(
        backend not in DELEGATED_BACKENDS,
        f"{prefix} uses delegated backend {backend!r} in AX-owned policy search",
    )
    _mapping(candidate.get("metrics"), f"{prefix}.metrics")
    _mapping(candidate.get("route_metadata"), f"{prefix}.route_metadata")


def _validate_candidate_fallbacks(candidate: dict[str, Any], index: int, target: str) -> None:
    if target not in FALLBACK_REQUIRED_TARGETS:
        return
    prefix = f"candidates[{index}]"
    fallback_count = _integer(candidate.get("fallback_count"), f"{prefix}.fallback_count")
    fallback_tokens = _integer(candidate.get("fallback_tokens"), f"{prefix}.fallback_tokens")
    _require(fallback_count >= 0, f"{prefix}.fallback_count must be non-negative")
    _require(fallback_tokens >= 0, f"{prefix}.fallback_tokens must be non-negative")


def _validate_candidate_rows(candidates: list[Any], target: str) -> list[dict[str, Any]]:
    _require(candidates, "candidates must keep raw candidate rows")
    checked: list[dict[str, Any]] = []
    policy_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        row = _mapping(candidate, f"candidates[{index}]")
        _validate_candidate_common(row, index, target)
        policy_id = str(row["policy_id"])
        _require(
            policy_id not in policy_ids,
            f"candidates[{index}].policy_id {policy_id!r} duplicates an earlier candidate",
        )
        policy_ids.add(policy_id)
        _validate_candidate_fallbacks(row, index, target)
        checked.append(row)
    return checked


def _validate_candidate_shapes(
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> None:
    baseline_prompt = _integer(baseline.get("prompt_tokens"), "baseline.prompt_tokens")
    baseline_generation = _integer(baseline.get("generation_tokens"), "baseline.generation_tokens")
    for index, candidate in enumerate(candidates):
        prompt = _integer(candidate.get("prompt_tokens"), f"candidates[{index}].prompt_tokens")
        generation = _integer(
            candidate.get("generation_tokens"),
            f"candidates[{index}].generation_tokens",
        )
        _require(
            prompt == baseline_prompt and generation == baseline_generation,
            f"candidates[{index}] prompt/generation shape must match baseline",
        )


def _validate_best_candidate(best_candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    policy_id = _string(best_candidate.get("policy_id"), "best_candidate.policy_id")
    candidate_ids = {str(candidate.get("policy_id")) for candidate in candidates}
    _require(policy_id in candidate_ids, "best_candidate.policy_id must refer to a candidate row")


def _validate_promotion_evidence(artifact: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    evidence = _mapping(artifact.get("promotion_evidence"), "promotion_evidence")
    _require(
        _boolean(
            evidence.get("deterministic_replay_passed"),
            "promotion_evidence.deterministic_replay_passed",
        )
        is True,
        "promotion_evidence.deterministic_replay_passed must be true",
    )
    _require(
        _boolean(evidence.get("quality_gate_passed"), "promotion_evidence.quality_gate_passed")
        is True,
        "promotion_evidence.quality_gate_passed must be true",
    )
    repeated = _mapping(evidence.get("repeated_measurements"), "promotion_evidence.repeated_measurements")
    _require(
        _integer(repeated.get("runs"), "promotion_evidence.repeated_measurements.runs") >= 2,
        "promotion_evidence.repeated_measurements.runs must be >= 2",
    )
    _require(
        _integer(
            repeated.get("cooldown_seconds"),
            "promotion_evidence.repeated_measurements.cooldown_seconds",
        )
        > 0,
        "promotion_evidence.repeated_measurements.cooldown_seconds must be positive",
    )
    companion_gates = _list(evidence.get("companion_prd_gates"), "promotion_evidence.companion_prd_gates")
    _require(companion_gates, "promotion_evidence.companion_prd_gates must not be empty")
    for index, gate in enumerate(companion_gates):
        _string(gate, f"promotion_evidence.companion_prd_gates[{index}]")
    best_policy_id = str(_mapping(artifact.get("best_candidate"), "best_candidate").get("policy_id"))
    best_rows = [candidate for candidate in candidates if str(candidate.get("policy_id")) == best_policy_id]
    _require(best_rows, "promotion evidence requires a valid best_candidate")
    best = best_rows[0]
    _require(
        best.get("quality_gate_passed") is True,
        "best candidate must pass quality gate for promotion review",
    )
    _require(
        best.get("deterministic_replay_passed") is True,
        "best candidate must pass deterministic replay for promotion review",
    )
    if "fallback_count" in best:
        _require(best["fallback_count"] == 0, "best candidate must have zero fallbacks for promotion review")
    if "fallback_tokens" in best:
        _require(best["fallback_tokens"] == 0, "best candidate must have zero fallback tokens for promotion review")


def _validate_confirmation_evidence(
    artifact: dict[str, Any],
    candidates: list[dict[str, Any]],
    classification: str,
) -> None:
    evidence = _mapping(artifact.get("confirmation_evidence"), "confirmation_evidence")
    baseline_policy_id = _string(
        evidence.get("baseline_policy_id"),
        "confirmation_evidence.baseline_policy_id",
    )
    baseline = _mapping(artifact.get("baseline"), "baseline")
    expected_baseline_policy_id = _string(baseline.get("policy_id"), "baseline.policy_id")
    _require(
        baseline_policy_id == expected_baseline_policy_id,
        "confirmation_evidence.baseline_policy_id must match baseline.policy_id",
    )
    candidate_policy_id = _string(
        evidence.get("candidate_policy_id"),
        "confirmation_evidence.candidate_policy_id",
    )
    candidate_ids = {str(candidate.get("policy_id")) for candidate in candidates}
    _require(
        candidate_policy_id in candidate_ids,
        "confirmation_evidence.candidate_policy_id must refer to a candidate row",
    )
    repeated = _mapping(evidence.get("repeated_measurements"), "confirmation_evidence.repeated_measurements")
    _require(
        _integer(repeated.get("runs"), "confirmation_evidence.repeated_measurements.runs") >= 2,
        "confirmation_evidence.repeated_measurements.runs must be >= 2",
    )
    _require(
        _integer(
            repeated.get("cooldown_seconds"),
            "confirmation_evidence.repeated_measurements.cooldown_seconds",
        )
        > 0,
        "confirmation_evidence.repeated_measurements.cooldown_seconds must be positive",
    )
    _string(evidence.get("decision_metric"), "confirmation_evidence.decision_metric")
    _number(evidence.get("baseline_median"), "confirmation_evidence.baseline_median")
    _number(evidence.get("candidate_median"), "confirmation_evidence.candidate_median")
    _number(evidence.get("relative_delta"), "confirmation_evidence.relative_delta")
    _require(
        _number(evidence.get("noise_band"), "confirmation_evidence.noise_band") >= 0.0,
        "confirmation_evidence.noise_band must be non-negative",
    )
    classification_hint = _string(
        evidence.get("classification_hint"),
        "confirmation_evidence.classification_hint",
    )
    _require(
        classification_hint == classification,
        "confirmation_evidence.classification_hint must match decision.classification",
    )


def _validate_decision(
    artifact: dict[str, Any],
    decision: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    classification = _string(decision.get("classification"), "decision.classification")
    _require(
        classification in ALLOWED_DECISIONS,
        f"decision.classification {classification!r} is not supported",
    )
    _string(decision.get("reason"), "decision.reason")
    if classification == PROMOTION_REVIEW_CLASSIFICATION:
        _validate_promotion_evidence(artifact, candidates)
    if classification in CONFIRMATION_REQUIRED_CLASSIFICATIONS:
        _validate_confirmation_evidence(artifact, candidates, classification)
    return classification


def validate_offline_policy_search_artifact(path: Path) -> str:
    artifact = _load_json(path)
    _require(
        artifact.get("schema") == SCHEMA_VERSION,
        f"{path} has schema={artifact.get('schema')!r}, expected {SCHEMA_VERSION}",
    )
    target = _string(artifact.get("target"), "target")
    _string(artifact.get("status"), "status")
    _string(artifact.get("created_at"), "created_at")

    _validate_repo(_mapping(artifact.get("repo"), "repo"))
    _validate_model(_mapping(artifact.get("model"), "model"))

    route = _mapping(artifact.get("route"), "route")
    backend = _route_selected_backend(route, "route")
    if target in RUNTIME_POLICY_TARGETS:
        _require(backend == "mlx", "route.selected_backend must be mlx")
    _require(
        backend not in DELEGATED_BACKENDS,
        f"route.selected_backend {backend!r} is delegated evidence, not AX-owned policy evidence",
    )

    _validate_search(_mapping(artifact.get("search"), "search"))
    _validate_objective(_mapping(artifact.get("objective"), "objective"))
    baseline = _mapping(artifact.get("baseline"), "baseline")
    _validate_baseline(baseline, target)
    candidates = _validate_candidate_rows(_list(artifact.get("candidates"), "candidates"), target)
    _validate_candidate_shapes(baseline, candidates)
    max_candidates = _integer(
        _mapping(_mapping(artifact.get("search"), "search").get("budget"), "search.budget").get(
            "max_candidates"
        ),
        "search.budget.max_candidates",
    )
    _require(
        max_candidates >= len(candidates),
        "search.budget.max_candidates must be >= number of candidate rows",
    )
    _validate_best_candidate(_mapping(artifact.get("best_candidate"), "best_candidate"), candidates)
    return _validate_decision(artifact, _mapping(artifact.get("decision"), "decision"), candidates)


def check_offline_policy_search_artifacts(paths: list[Path]) -> OfflinePolicySearchCheckResult:
    promotion_review_count = 0
    diagnostic_count = 0
    negative_count = 0
    for path in paths:
        classification = validate_offline_policy_search_artifact(path)
        if classification == PROMOTION_REVIEW_CLASSIFICATION:
            promotion_review_count += 1
        elif classification == "negative_result":
            negative_count += 1
        else:
            diagnostic_count += 1
    return OfflinePolicySearchCheckResult(
        artifact_count=len(paths),
        promotion_review_count=promotion_review_count,
        diagnostic_count=diagnostic_count,
        negative_count=negative_count,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = check_offline_policy_search_artifacts(args.artifacts)
    except OfflinePolicySearchArtifactError as error:
        print(f"offline policy search artifact check failed: {error}", file=sys.stderr)
        return 1
    print(
        "offline policy search artifact check passed: "
        f"{result.artifact_count} artifacts, "
        f"{result.promotion_review_count} promotion-review, "
        f"{result.diagnostic_count} diagnostic, "
        f"{result.negative_count} negative"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
