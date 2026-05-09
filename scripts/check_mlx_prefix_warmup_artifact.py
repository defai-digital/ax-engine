#!/usr/bin/env python3
"""Validate MLX physical prefix snapshot miss/warmup correctness artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ax.mlx_prefix_warmup.v1"
PROMPT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
CORRECTNESS_STATUSES = {"passed"}
PROMPT_DIGEST_FIELDS = {
    "prompt_token_ids_sha256": "token_ids",
    "prompt_ref_sha256": "prompt_ref_bytes",
}


class PrefixWarmupArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class PrefixWarmupObservation:
    artifact_path: Path
    request_id: str
    matched_token_count: int
    miss_count: int
    warmup_token_count: int
    payload: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise PrefixWarmupArtifactError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise PrefixWarmupArtifactError(f"{path} must contain a JSON object")
    return payload


def require_mapping(payload: dict[str, Any], key: str, *, owner: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PrefixWarmupArtifactError(f"{owner} lacks object field {key!r}")
    return value


def require_list(payload: dict[str, Any], key: str, *, owner: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise PrefixWarmupArtifactError(f"{owner} lacks list field {key!r}")
    return value


def require_non_empty_str(payload: dict[str, Any], key: str, *, owner: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PrefixWarmupArtifactError(f"{owner} lacks non-empty string field {key!r}")
    return value


def require_bool(payload: dict[str, Any], key: str, *, owner: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise PrefixWarmupArtifactError(f"{owner} lacks boolean field {key!r}")
    return value


def require_non_negative_int(payload: dict[str, Any], key: str, *, owner: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PrefixWarmupArtifactError(
            f"{owner} lacks non-negative integer field {key!r}"
        )
    return value


def require_positive_int(payload: dict[str, Any], key: str, *, owner: str) -> int:
    value = require_non_negative_int(payload, key, owner=owner)
    if value <= 0:
        raise PrefixWarmupArtifactError(f"{owner}.{key} must be positive")
    return value


def validate_prompt_hash(value: str, *, owner: str) -> None:
    if PROMPT_HASH_RE.fullmatch(value) is None:
        raise PrefixWarmupArtifactError(f"{owner} is not a valid sha256")


def validate_prompt_digest(observation: dict[str, Any], *, owner: str) -> None:
    digest_fields = [
        (field, kind)
        for field, kind in PROMPT_DIGEST_FIELDS.items()
        if field in observation
    ]
    if len(digest_fields) != 1:
        raise PrefixWarmupArtifactError(
            f"{owner} must contain exactly one prompt digest field"
        )
    field, expected_kind = digest_fields[0]
    prompt_hash = require_non_empty_str(observation, field, owner=owner)
    validate_prompt_hash(prompt_hash, owner=f"{owner}.{field}")
    kind = require_non_empty_str(observation, "prompt_digest_kind", owner=owner)
    if kind != expected_kind:
        raise PrefixWarmupArtifactError(
            f"{owner}.prompt_digest_kind must be {expected_kind}"
        )


def validate_top_level(path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise PrefixWarmupArtifactError(
            f"{path} has schema_version={artifact.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    if artifact.get("claim_scope") != "physical_prefix_miss_warmup_correctness":
        raise PrefixWarmupArtifactError(
            f"{path}.claim_scope must be physical_prefix_miss_warmup_correctness"
        )

    model = require_mapping(artifact, "model", owner=str(path))
    require_non_empty_str(model, "id", owner=f"{path}.model")

    host = require_mapping(artifact, "host", owner=str(path))
    require_non_empty_str(host, "chip", owner=f"{path}.host")

    benchmark = require_mapping(artifact, "benchmark", owner=str(path))
    require_positive_int(benchmark, "shared_prefix_tokens", owner=f"{path}.benchmark")
    require_positive_int(benchmark, "generation_tokens", owner=f"{path}.benchmark")
    repetitions = require_positive_int(benchmark, "repetitions", owner=f"{path}.benchmark")
    if repetitions < 1:
        raise PrefixWarmupArtifactError(f"{path}.benchmark.repetitions must be at least 1")


def parse_observation(
    path: Path,
    observation: dict[str, Any],
    index: int,
) -> PrefixWarmupObservation:
    owner = f"{path}.observations[{index}]"
    request_id = require_non_empty_str(observation, "request_id", owner=owner)
    validate_prompt_digest(observation, owner=owner)

    route = require_mapping(observation, "route", owner=owner)
    if route.get("selected_backend") != "mlx":
        raise PrefixWarmupArtifactError(f"{owner}.route.selected_backend must be mlx")
    if route.get("route_identity") != "repo_owned_mlx":
        raise PrefixWarmupArtifactError(f"{owner}.route.route_identity must be repo_owned_mlx")

    logical = require_mapping(observation, "logical_prefix_reuse", owner=owner)
    matched_token_count = require_positive_int(
        logical, "matched_token_count", owner=f"{owner}.logical_prefix_reuse"
    )
    require_positive_int(
        logical, "reused_block_count", owner=f"{owner}.logical_prefix_reuse"
    )

    physical = require_mapping(observation, "physical_prefix_snapshot", owner=owner)
    hit_count = require_non_negative_int(
        physical, "hit_count", owner=f"{owner}.physical_prefix_snapshot"
    )
    miss_count = require_positive_int(
        physical, "miss_count", owner=f"{owner}.physical_prefix_snapshot"
    )
    warmup_token_count = require_positive_int(
        physical, "warmup_token_count", owner=f"{owner}.physical_prefix_snapshot"
    )
    reused_token_count = require_non_negative_int(
        physical, "reused_token_count", owner=f"{owner}.physical_prefix_snapshot"
    )
    blocked_count = require_non_negative_int(
        physical, "blocked_count", owner=f"{owner}.physical_prefix_snapshot"
    )
    if hit_count != 0:
        raise PrefixWarmupArtifactError(
            f"{owner}.physical_prefix_snapshot.hit_count must be 0 for miss/warmup evidence"
        )
    if reused_token_count != 0:
        raise PrefixWarmupArtifactError(
            f"{owner}.physical_prefix_snapshot.reused_token_count must be 0 for miss/warmup evidence"
        )
    if blocked_count != 0:
        raise PrefixWarmupArtifactError(
            f"{owner}.physical_prefix_snapshot.blocked_count must be 0 for eligible miss/warmup evidence"
        )
    if physical.get("physical_snapshot_coverage") != "miss_warmup_only":
        raise PrefixWarmupArtifactError(
            f"{owner}.physical_prefix_snapshot.physical_snapshot_coverage must be miss_warmup_only"
        )

    correctness = require_mapping(observation, "correctness", owner=owner)
    status = require_non_empty_str(correctness, "status", owner=f"{owner}.correctness")
    if status not in CORRECTNESS_STATUSES:
        raise PrefixWarmupArtifactError(f"{owner}.correctness.status must be passed")
    if not require_bool(
        correctness, "deterministic_replay", owner=f"{owner}.correctness"
    ):
        raise PrefixWarmupArtifactError(
            f"{owner}.correctness.deterministic_replay must be true"
        )
    require_non_empty_str(correctness, "output_token_ids_sha256", owner=f"{owner}.correctness")
    validate_prompt_hash(
        str(correctness["output_token_ids_sha256"]),
        owner=f"{owner}.correctness.output_token_ids_sha256",
    )

    return PrefixWarmupObservation(
        artifact_path=path,
        request_id=request_id,
        matched_token_count=matched_token_count,
        miss_count=miss_count,
        warmup_token_count=warmup_token_count,
        payload=observation,
    )


def validate_observation_coverage(
    path: Path,
    observations: list[PrefixWarmupObservation],
) -> None:
    if not observations:
        raise PrefixWarmupArtifactError(f"{path} lacks prefix warmup observations")
    request_ids: set[str] = set()
    total_warmup_tokens = 0
    for observation in observations:
        if observation.request_id in request_ids:
            raise PrefixWarmupArtifactError(
                f"{path} has duplicate request_id {observation.request_id!r}"
            )
        request_ids.add(observation.request_id)
        total_warmup_tokens += observation.warmup_token_count
    if total_warmup_tokens <= 0:
        raise PrefixWarmupArtifactError(f"{path} lacks physical prefix warmup tokens")


def validate_prefix_warmup_artifact(path: Path) -> list[str]:
    artifact = load_json(path)
    validate_top_level(path, artifact)
    raw_observations = require_list(artifact, "observations", owner=str(path))
    observations = [
        parse_observation(path, observation, index)
        for index, observation in enumerate(raw_observations)
        if isinstance(observation, dict)
    ]
    if len(observations) != len(raw_observations):
        raise PrefixWarmupArtifactError(f"{path}.observations must contain objects")
    validate_observation_coverage(path, observations)
    return [
        f"{observation.request_id}:matched={observation.matched_token_count}:warmup={observation.warmup_token_count}"
        for observation in observations
    ]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    try:
        checked = validate_prefix_warmup_artifact(args.artifact)
    except PrefixWarmupArtifactError as error:
        print(f"MLX prefix warmup artifact check failed: {error}", file=sys.stderr)
        return 1
    print(
        "MLX prefix warmup artifact check passed: "
        f"{args.artifact} ({SCHEMA_VERSION}; {', '.join(checked)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
