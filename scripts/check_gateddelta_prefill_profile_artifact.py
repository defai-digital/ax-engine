#!/usr/bin/env python3
"""Validate evidence-first GatedDelta prefill profile artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOP_LEVEL_SCHEMA_VERSION = "ax.mlx_inference_stack.v2"
PROFILE_SCHEMA_VERSION = "ax.gateddelta_prefill_profile.v1"
PREFLIGHT_SCHEMA_VERSION = "ax.gateddelta_prefill_model_preflight.v1"
REQUIRED_PROMPT_TOKENS = [512, 2048, 8192, 32768]
PROFILE_ENV = "AX_MLX_LINEAR_ATTENTION_PROFILE=1"
PROMPT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_MODEL_FAMILIES = {"qwen3_5", "qwen3_next"}
REQUIRED_LINEAR_ATTENTION_FIELDS = (
    "num_value_heads",
    "num_key_heads",
    "key_head_dim",
    "value_head_dim",
    "conv_kernel_dim",
)

LINEAR_ATTENTION_PROFILE_KEYS = [
    "ax_mlx_linear_attention_profile_enabled",
    "ax_mlx_linear_attention_profile_layers",
    "ax_mlx_linear_attention_profile_tokens",
    "ax_mlx_linear_attention_profile_projection_wall_us",
    "ax_mlx_linear_attention_profile_conv_wall_us",
    "ax_mlx_linear_attention_profile_qk_norm_wall_us",
    "ax_mlx_linear_attention_profile_recurrent_wall_us",
    "ax_mlx_linear_attention_profile_output_wall_us",
]


class GatedDeltaPrefillProfileArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProfileShape:
    prompt_tokens: int
    generation_tokens: int


@dataclass(frozen=True)
class ProfileRow:
    artifact_path: Path
    engine: str
    shape: ProfileShape
    prompt_hash: str
    payload: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path} is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise GatedDeltaPrefillProfileArtifactError(f"{path} must contain a JSON object")
    return payload


def require_mapping(payload: dict[str, Any], key: str, *, owner: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise GatedDeltaPrefillProfileArtifactError(f"{owner} lacks object field {key!r}")
    return value


def require_non_empty_str(payload: dict[str, Any], key: str, *, owner: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GatedDeltaPrefillProfileArtifactError(
            f"{owner} lacks non-empty string field {key!r}"
        )
    return value


def require_positive_int(payload: dict[str, Any], key: str, *, owner: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value <= 0:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{owner} lacks positive integer field {key!r}"
        )
    return value


def require_non_negative_int(payload: dict[str, Any], key: str, *, owner: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value < 0:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{owner} lacks non-negative integer field {key!r}"
        )
    return value


def require_positive_float(payload: dict[str, Any], key: str, *, owner: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{owner} lacks positive numeric field {key!r}"
        )
    return float(value)


def require_metric_median(payload: dict[str, Any], key: str, *, owner: str) -> float:
    metric = payload.get(key)
    if not isinstance(metric, dict):
        raise GatedDeltaPrefillProfileArtifactError(f"{owner} lacks metric object {key!r}")
    return require_positive_float(metric, "median", owner=f"{owner}.{key}")


def require_prompt_hash(row: dict[str, Any], *, owner: str) -> str:
    prompt_hash = require_non_empty_str(row, "prompt_token_ids_sha256", owner=owner)
    if PROMPT_HASH_RE.fullmatch(prompt_hash) is None:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{owner} has invalid prompt_token_ids_sha256"
        )
    return prompt_hash


def validate_top_level(path: Path, artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != TOP_LEVEL_SCHEMA_VERSION:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path} has schema_version={artifact.get('schema_version')!r}, "
            f"expected {TOP_LEVEL_SCHEMA_VERSION}"
        )

    model_config = require_mapping(artifact, "model_config", owner=str(path))
    if model_config.get("linear_attention_enabled") is not True:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path}.model_config.linear_attention_enabled must be true"
        )

    if artifact.get("ax_linear_attention_profile") is not True:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path}.ax_linear_attention_profile must be true"
        )

    if artifact.get("prompt_tokens") != REQUIRED_PROMPT_TOKENS:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path}.prompt_tokens must be {REQUIRED_PROMPT_TOKENS}"
        )

    profile = require_mapping(artifact, "gateddelta_prefill_profile", owner=str(path))
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path}.gateddelta_prefill_profile.schema_version must be "
            f"{PROFILE_SCHEMA_VERSION}"
        )
    if profile.get("prompt_tokens") != REQUIRED_PROMPT_TOKENS:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path}.gateddelta_prefill_profile.prompt_tokens must be "
            f"{REQUIRED_PROMPT_TOKENS}"
        )
    if profile.get("required_prompt_tokens") != REQUIRED_PROMPT_TOKENS:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path}.gateddelta_prefill_profile.required_prompt_tokens must be "
            f"{REQUIRED_PROMPT_TOKENS}"
        )
    if profile.get("runtime_profile_env") != PROFILE_ENV:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path}.gateddelta_prefill_profile.runtime_profile_env must be {PROFILE_ENV}"
        )
    if profile.get("direct_ax_row_required") is not True:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path}.gateddelta_prefill_profile.direct_ax_row_required must be true"
        )
    if profile.get("ngram_policy_allowed") is not False:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path}.gateddelta_prefill_profile.ngram_policy_allowed must be false"
        )
    if profile.get("kv_compression_allowed") is not False:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path}.gateddelta_prefill_profile.kv_compression_allowed must be false"
        )
    validate_model_preflight(path, profile)


def validate_model_preflight(path: Path, profile: dict[str, Any]) -> None:
    preflight = require_mapping(
        profile,
        "model_preflight",
        owner=f"{path}.gateddelta_prefill_profile",
    )
    owner = f"{path}.gateddelta_prefill_profile.model_preflight"
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{owner}.schema_version must be {PREFLIGHT_SCHEMA_VERSION}"
        )
    if preflight.get("status") != "passed":
        raise GatedDeltaPrefillProfileArtifactError(f"{owner}.status must be passed")
    if preflight.get("checker") != "scripts/check_gateddelta_prefill_model.py":
        raise GatedDeltaPrefillProfileArtifactError(
            f"{owner}.checker must be scripts/check_gateddelta_prefill_model.py"
        )
    if preflight.get("model_family") not in SUPPORTED_MODEL_FAMILIES:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{owner}.model_family must be one of {sorted(SUPPORTED_MODEL_FAMILIES)}"
        )

    linear_attention = require_mapping(preflight, "linear_attention", owner=owner)
    for field in REQUIRED_LINEAR_ATTENTION_FIELDS:
        require_positive_int(
            linear_attention,
            field,
            owner=f"{owner}.linear_attention",
        )
    key_head_dim = linear_attention["key_head_dim"]
    if key_head_dim % 32 != 0:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{owner}.linear_attention.key_head_dim must be divisible by 32"
        )


def validate_linear_attention_profile(
    profile: dict[str, Any],
    *,
    owner: str,
    prompt_tokens: int,
) -> None:
    for key in LINEAR_ATTENTION_PROFILE_KEYS:
        require_non_negative_int(profile, key, owner=owner)

    if profile["ax_mlx_linear_attention_profile_enabled"] != 1:
        raise GatedDeltaPrefillProfileArtifactError(f"{owner}.enabled must be 1")
    if profile["ax_mlx_linear_attention_profile_layers"] <= 0:
        raise GatedDeltaPrefillProfileArtifactError(f"{owner}.layers must be positive")
    if profile["ax_mlx_linear_attention_profile_tokens"] < prompt_tokens:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{owner}.tokens must cover at least the prompt length"
        )
    if profile["ax_mlx_linear_attention_profile_recurrent_wall_us"] <= 0:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{owner}.recurrent_wall_us must be positive"
        )


def parse_row(path: Path, row: dict[str, Any], index: int) -> ProfileRow:
    owner = f"{path}.results[{index}]"
    engine = require_non_empty_str(row, "engine", owner=owner)
    if engine not in {"mlx_lm", "ax_engine_mlx", "mlx_swift_lm"}:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{owner} has unsupported engine {engine!r} for GatedDelta prefill profile"
        )

    prompt_tokens = require_positive_int(row, "prompt_tokens", owner=owner)
    generation_tokens = require_positive_int(row, "generation_tokens", owner=owner)
    prompt_hash = require_prompt_hash(row, owner=owner)
    require_metric_median(row, "prefill_tok_s", owner=owner)

    if engine == "mlx_lm":
        baseline = require_mapping(row, "baseline", owner=owner)
        if baseline.get("role") != "primary_reference":
            raise GatedDeltaPrefillProfileArtifactError(
                f"{owner} mlx_lm row lacks primary reference role"
            )
    elif engine == "ax_engine_mlx":
        if row.get("ax_decode_policy") != "direct_no_ngram_acceleration":
            raise GatedDeltaPrefillProfileArtifactError(
                f"{owner} ax_engine_mlx row must use direct_no_ngram_acceleration"
            )
        if row.get("ax_decode_claim_status") != "direct_same_policy_baseline":
            raise GatedDeltaPrefillProfileArtifactError(
                f"{owner} ax_engine_mlx row must be a direct same-policy baseline"
            )
        if "experimental_mlx_kv_compression" in row or "kv_compression_telemetry" in row:
            raise GatedDeltaPrefillProfileArtifactError(
                f"{owner} must not include KV compression evidence"
            )
        mlx_telemetry = require_mapping(row, "ax_mlx_telemetry", owner=owner)
        require_positive_int(
            mlx_telemetry,
            "ax_mlx_prefill_wall_us",
            owner=f"{owner}.ax_mlx_telemetry",
        )
        baseline = require_mapping(row, "baseline", owner=owner)
        if baseline.get("engine") != "mlx_lm":
            raise GatedDeltaPrefillProfileArtifactError(
                f"{owner}.baseline.engine must be mlx_lm"
            )
        require_positive_float(
            baseline,
            "prefill_ratio_to_mlx_lm",
            owner=f"{owner}.baseline",
        )
        profile = require_mapping(row, "ax_mlx_linear_attention_profile", owner=owner)
        validate_linear_attention_profile(
            profile,
            owner=f"{owner}.ax_mlx_linear_attention_profile",
            prompt_tokens=prompt_tokens,
        )

    return ProfileRow(
        artifact_path=path,
        engine=engine,
        shape=ProfileShape(
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
        ),
        prompt_hash=prompt_hash,
        payload=row,
    )


def validate_shape_groups(path: Path, rows: list[ProfileRow]) -> list[str]:
    by_shape: dict[ProfileShape, dict[str, ProfileRow]] = {}
    prompt_hashes: dict[ProfileShape, set[str]] = {}
    for row in rows:
        engines = by_shape.setdefault(row.shape, {})
        if row.engine in engines:
            raise GatedDeltaPrefillProfileArtifactError(
                f"{path} has duplicate {row.engine} row for "
                f"prompt_tokens={row.shape.prompt_tokens} "
                f"generation_tokens={row.shape.generation_tokens}"
            )
        engines[row.engine] = row
        prompt_hashes.setdefault(row.shape, set()).add(row.prompt_hash)

    expected_shapes = {
        ProfileShape(
            prompt_tokens=prompt_tokens,
            generation_tokens=next(iter(by_shape)).generation_tokens
            if by_shape
            else 0,
        )
        for prompt_tokens in REQUIRED_PROMPT_TOKENS
    }
    if not by_shape:
        raise GatedDeltaPrefillProfileArtifactError(f"{path} lacks result rows")
    if set(by_shape) != expected_shapes:
        raise GatedDeltaPrefillProfileArtifactError(
            f"{path} must contain exactly the prompt matrix {REQUIRED_PROMPT_TOKENS} "
            "for one generation-token shape"
        )

    checked = []
    for shape, engines in sorted(by_shape.items(), key=lambda item: item[0].prompt_tokens):
        missing = {"mlx_lm", "ax_engine_mlx"} - set(engines)
        if missing:
            raise GatedDeltaPrefillProfileArtifactError(
                f"{path} prompt_tokens={shape.prompt_tokens} "
                f"generation_tokens={shape.generation_tokens} lacks engines: {sorted(missing)}"
            )
        if len(prompt_hashes[shape]) != 1:
            raise GatedDeltaPrefillProfileArtifactError(
                f"{path} prompt_tokens={shape.prompt_tokens} "
                "does not reuse one prompt hash across engines"
            )
        checked.append(
            f"prompt_tokens={shape.prompt_tokens}:generation={shape.generation_tokens}"
        )
    return checked


def validate_gateddelta_prefill_profile_artifact(path: Path) -> list[str]:
    artifact = load_json(path)
    validate_top_level(path, artifact)

    raw_results = artifact.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise GatedDeltaPrefillProfileArtifactError(f"{path} lacks non-empty results list")
    rows = [
        parse_row(path, row, index)
        for index, row in enumerate(raw_results)
        if isinstance(row, dict)
    ]
    if len(rows) != len(raw_results):
        raise GatedDeltaPrefillProfileArtifactError(f"{path} results must all be JSON objects")
    return validate_shape_groups(path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate AX GatedDelta prefill profile artifact contracts."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checked = 0
    try:
        for artifact in args.artifacts:
            checked += len(validate_gateddelta_prefill_profile_artifact(artifact))
    except GatedDeltaPrefillProfileArtifactError as error:
        print(f"GatedDelta prefill profile artifact check failed: {error}", file=sys.stderr)
        return 1
    print(f"GatedDelta prefill profile artifact check passed: {checked} shape groups validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
