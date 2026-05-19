#!/usr/bin/env python3
"""Validate MLA prefix-restore retirement evidence artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "ax.prefix_reuse_equivalence.v1"
DEFAULT_ARTIFACTS = (
    Path(
        "benchmarks/results/prefix-reuse-equivalence/"
        "glm47-warm-extend-default-mla-chunk16-2026-05-14.json"
    ),
)


class MlaPrefixRestoreEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceSummary:
    path: Path
    model_id: str
    prompts_total: int
    warm_hit_count: int
    warm_reused_tokens: int


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MlaPrefixRestoreEvidenceError(f"artifact not found: {path}") from error
    except json.JSONDecodeError as error:
        raise MlaPrefixRestoreEvidenceError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise MlaPrefixRestoreEvidenceError(f"{path} must contain a JSON object")
    return payload


def _flag(artifact: dict[str, Any], name: str) -> dict[str, Any]:
    flags = artifact.get("environment_flags")
    if not isinstance(flags, dict):
        raise MlaPrefixRestoreEvidenceError("artifact lacks environment_flags")
    value = flags.get(name)
    if not isinstance(value, dict):
        raise MlaPrefixRestoreEvidenceError(f"artifact lacks {name} provenance")
    return value


def _flag_truthy(artifact: dict[str, Any], name: str) -> bool:
    return bool(_flag(artifact, name).get("truthy"))


def _flag_set(artifact: dict[str, Any], name: str) -> bool:
    return bool(_flag(artifact, name).get("set"))


def _model_id(artifact: dict[str, Any]) -> str:
    model = artifact.get("model")
    if isinstance(model, dict):
        value = model.get("model_id")
        if isinstance(value, str):
            return value
    value = artifact.get("model_id")
    return value if isinstance(value, str) else ""


def _warm_telemetry_sum(artifact: dict[str, Any], key: str) -> int:
    total = 0
    per_prompt = artifact.get("per_prompt")
    if not isinstance(per_prompt, list):
        raise MlaPrefixRestoreEvidenceError("artifact lacks per_prompt rows")
    for row in per_prompt:
        if not isinstance(row, dict):
            raise MlaPrefixRestoreEvidenceError("per_prompt rows must be objects")
        telemetry = row.get("warm_telemetry")
        if not isinstance(telemetry, dict):
            raise MlaPrefixRestoreEvidenceError("per_prompt row lacks warm_telemetry")
        total += int(telemetry.get(key, 0))
    return total


def validate_artifact(
    path: Path,
    *,
    min_prompts: int,
    require_default_path: bool,
    model_substring: str,
) -> EvidenceSummary:
    artifact = _load_json(path)
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise MlaPrefixRestoreEvidenceError(
            f"{path} schema_version must be {SCHEMA_VERSION}"
        )

    model_id = _model_id(artifact)
    if model_substring and model_substring.lower() not in model_id.lower():
        raise MlaPrefixRestoreEvidenceError(
            f"{path} model_id {model_id!r} does not include {model_substring!r}"
        )

    config = artifact.get("config")
    if not isinstance(config, dict):
        raise MlaPrefixRestoreEvidenceError(f"{path} lacks config")
    if config.get("mode") != "warm_extend":
        raise MlaPrefixRestoreEvidenceError(f"{path} must be warm_extend mode")
    if config.get("pad_to_block_size") != 16:
        raise MlaPrefixRestoreEvidenceError(
            f"{path} must use pad_to_block_size=16 for MLA snapshot evidence"
        )

    aggregate = artifact.get("aggregate")
    if not isinstance(aggregate, dict):
        raise MlaPrefixRestoreEvidenceError(f"{path} lacks aggregate")
    prompts_total = int(aggregate.get("prompts_total", 0))
    if prompts_total < min_prompts:
        raise MlaPrefixRestoreEvidenceError(
            f"{path} has only {prompts_total} prompts; need at least {min_prompts}"
        )
    if aggregate.get("verdict") != "PASS":
        raise MlaPrefixRestoreEvidenceError(f"{path} aggregate verdict must be PASS")
    if int(aggregate.get("prompts_matching_exactly", -1)) != prompts_total:
        raise MlaPrefixRestoreEvidenceError(
            f"{path} must match every prompt token-exactly"
        )

    per_prompt = artifact.get("per_prompt")
    if not isinstance(per_prompt, list) or len(per_prompt) != prompts_total:
        raise MlaPrefixRestoreEvidenceError(
            f"{path} per_prompt length must match prompts_total"
        )
    for row in per_prompt:
        if not isinstance(row, dict) or row.get("tokens_match") is not True:
            raise MlaPrefixRestoreEvidenceError(
                f"{path} contains a non-matching prompt row"
            )

    if require_default_path:
        if _flag_set(artifact, "AX_ALLOW_MLA_PREFIX_RESTORE") or _flag_truthy(
            artifact, "AX_ALLOW_MLA_PREFIX_RESTORE"
        ):
            raise MlaPrefixRestoreEvidenceError(
                f"{path} must not rely on legacy AX_ALLOW_MLA_PREFIX_RESTORE"
            )
        if _flag_truthy(artifact, "AX_DISABLE_MLA_PREFIX_RESTORE"):
            raise MlaPrefixRestoreEvidenceError(
                f"{path} must not disable MLA prefix restore"
            )
        if _flag_set(artifact, "AX_MLX_MLA_PREFILL_CHUNK"):
            raise MlaPrefixRestoreEvidenceError(
                f"{path} must use the default MLA prefill chunk, not an env override"
            )

    warm_hit_count = _warm_telemetry_sum(artifact, "ax_mlx_prefix_cache_hits")
    warm_reused_tokens = _warm_telemetry_sum(
        artifact, "ax_mlx_prefix_cache_reused_tokens"
    )
    if warm_hit_count <= 0 or warm_reused_tokens <= 0:
        raise MlaPrefixRestoreEvidenceError(
            f"{path} must prove a physical warm prefix-cache hit with reused tokens"
        )

    blocked_layout = _warm_telemetry_sum(
        artifact, "ax_mlx_prefix_cache_blocked_unsupported_layout"
    )
    blocked_policy = _warm_telemetry_sum(
        artifact, "ax_mlx_prefix_cache_blocked_policy_disabled"
    )
    if blocked_layout > 0 or blocked_policy > 0:
        raise MlaPrefixRestoreEvidenceError(
            f"{path} has blocked warm prefix-cache rows; layout={blocked_layout}, "
            f"policy={blocked_policy}"
        )

    return EvidenceSummary(
        path=path,
        model_id=model_id,
        prompts_total=prompts_total,
        warm_hit_count=warm_hit_count,
        warm_reused_tokens=warm_reused_tokens,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        type=Path,
        dest="artifacts",
        help="Evidence artifact to validate. Defaults to the curated GLM 4.7 artifact.",
    )
    parser.add_argument("--min-prompts", type=int, default=5)
    parser.add_argument("--model-substring", default="glm")
    parser.add_argument(
        "--allow-env-override",
        action="store_true",
        help="Allow legacy restore or MLA chunk env overrides. Intended for diagnostics only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    artifacts = args.artifacts or list(DEFAULT_ARTIFACTS)
    try:
        summaries = [
            validate_artifact(
                artifact,
                min_prompts=args.min_prompts,
                require_default_path=not args.allow_env_override,
                model_substring=args.model_substring,
            )
            for artifact in artifacts
        ]
    except MlaPrefixRestoreEvidenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for summary in summaries:
        print(
            "ok: "
            f"{summary.path} model={summary.model_id} prompts={summary.prompts_total} "
            f"warm_hits={summary.warm_hit_count} reused_tokens={summary.warm_reused_tokens}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
