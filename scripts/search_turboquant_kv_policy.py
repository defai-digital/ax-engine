#!/usr/bin/env python3
"""Enumerate TurboQuant KV policy candidates into a diagnostic artifact.

This is the first Quantum OP / offline-policy-search harness. It performs a
deterministic grid enumeration only; it does not run benchmarks, models, or
runtime experiments. Candidate rows are marked `measurement_status=not_run` so
the artifact can seed later measurement work without becoming a performance
claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

BUILDER_PATH = Path(__file__).with_name("build_offline_policy_search_artifact.py")
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_offline_policy_search_artifact",
    BUILDER_PATH,
)
assert BUILDER_SPEC and BUILDER_SPEC.loader
builder = importlib.util.module_from_spec(BUILDER_SPEC)
sys.modules[BUILDER_SPEC.name] = builder
BUILDER_SPEC.loader.exec_module(builder)

CHECKER_PATH = Path(__file__).with_name("check_offline_policy_search_artifact.py")
CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_offline_policy_search_artifact",
    CHECKER_PATH,
)
assert CHECKER_SPEC and CHECKER_SPEC.loader
checker = importlib.util.module_from_spec(CHECKER_SPEC)
sys.modules[CHECKER_SPEC.name] = checker
CHECKER_SPEC.loader.exec_module(checker)

DEFAULT_KV_PRESETS = ["disabled", "TurboQuantK8V4"]
DEFAULT_HOT_WINDOW_TOKENS = [128, 256, 512, 1024]
DEFAULT_ELIGIBLE_LAYER_MASKS = ["full_attention_only"]
DEFAULT_FALLBACK_POLICIES = ["fail_closed", "fallback_with_accounting"]
DEFAULT_QUALITY_PROFILES = ["reference_k8v4"]
DEFAULT_OUTPUT_ROOT = Path("benchmarks/results/offline-policy-search")
DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
SAFE_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


class TurboQuantPolicySearchError(RuntimeError):
    """Raised when the candidate grid cannot form a valid artifact."""


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return items


def parse_int_csv(value: str) -> list[int]:
    parsed: list[int] = []
    for item in parse_csv(value):
        try:
            number = int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{item!r} is not an integer") from error
        if number < 0:
            raise argparse.ArgumentTypeError("hot window tokens must be non-negative")
        parsed.append(number)
    return parsed


def require_unique_values(values: list[Any], field: str) -> None:
    seen: set[Any] = set()
    for index, value in enumerate(values):
        if value in seen:
            raise TurboQuantPolicySearchError(
                f"{field}[{index}] duplicates an earlier value: {value!r}"
            )
        seen.add(value)


def validate_string_values(values: list[str], field: str) -> None:
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise TurboQuantPolicySearchError(f"{field}[{index}] must be a non-empty string")


def validate_hot_window_tokens(values: list[int]) -> None:
    for index, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TurboQuantPolicySearchError(
                f"hot_window_tokens[{index}] must be an integer"
            )
        if value < 0:
            raise TurboQuantPolicySearchError(
                f"hot_window_tokens[{index}] must be non-negative"
            )


def validate_search_space(
    *,
    kv_presets: list[str],
    hot_window_tokens: list[int],
    eligible_layer_masks: list[str],
    fallback_policies: list[str],
    quality_profiles: list[str],
) -> None:
    dimensions: dict[str, list[Any]] = {
        "kv_presets": kv_presets,
        "hot_window_tokens": hot_window_tokens,
        "eligible_layer_masks": eligible_layer_masks,
        "fallback_policies": fallback_policies,
        "quality_profiles": quality_profiles,
    }
    for field, values in dimensions.items():
        if not values:
            raise TurboQuantPolicySearchError(f"{field} must not be empty")
        require_unique_values(values, field)
    validate_string_values(kv_presets, "kv_presets")
    validate_hot_window_tokens(hot_window_tokens)
    validate_string_values(eligible_layer_masks, "eligible_layer_masks")
    validate_string_values(fallback_policies, "fallback_policies")
    validate_string_values(quality_profiles, "quality_profiles")


def safe_path_component(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise TurboQuantPolicySearchError(f"{field} must be a non-empty string")
    safe = SAFE_PATH_COMPONENT_RE.sub("-", text).strip("-._")
    if not safe:
        raise TurboQuantPolicySearchError(f"{field} must contain a safe path component")
    return safe


def artifact_date(created_at: Any) -> str:
    match = DATE_PREFIX_RE.match(str(created_at).strip())
    if not match:
        raise TurboQuantPolicySearchError(
            "artifact created_at must start with YYYY-MM-DD when --output is omitted"
        )
    return match.group(1)


def artifact_output_path(
    *,
    explicit_output: Path | None,
    output_root: Path,
    artifact: dict[str, Any],
) -> Path:
    if explicit_output is not None:
        return explicit_output
    target = safe_path_component(artifact.get("target"), "target")
    model = safe_path_component(
        builder.require_mapping(artifact.get("model"), "artifact.model").get("id"),
        "model.id",
    )
    date = artifact_date(artifact.get("created_at"))
    return output_root / date / f"{target}-{model}.json"


def policy_id(policy: dict[str, Any]) -> str:
    preset = str(policy["kv_preset"]).lower().replace("turboquant", "tq")
    return (
        f"{preset}-hot{policy['hot_window_tokens']}-"
        f"{policy['eligible_layer_mask']}-{policy['fallback_policy']}-"
        f"{policy['quality_profile']}"
    )


def candidate_row(
    *,
    policy: dict[str, Any],
    baseline: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    prompt_tokens = baseline.get("prompt_tokens")
    generation_tokens = baseline.get("generation_tokens")
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
        raise TurboQuantPolicySearchError("baseline.prompt_tokens must be an integer")
    if not isinstance(generation_tokens, int) or isinstance(generation_tokens, bool):
        raise TurboQuantPolicySearchError("baseline.generation_tokens must be an integer")
    return {
        "policy_id": policy_id(policy),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "seed": seed,
        "policy": dict(policy),
        "route": {
            "selected_backend": "mlx",
            "support_tier": "repo_owned_runtime",
        },
        "metrics": {
            "measurement_status": "not_run",
            "measurement_reason": "grid enumeration only",
        },
        "route_metadata": {
            "candidate_source": "turboquant_kv_policy_grid",
            "kv_preset": policy["kv_preset"],
            "hot_window_tokens": policy["hot_window_tokens"],
            "eligible_layer_mask": policy["eligible_layer_mask"],
            "fallback_policy": policy["fallback_policy"],
            "quality_profile": policy["quality_profile"],
        },
        "fallback_count": 0,
        "fallback_tokens": 0,
        "quality_gate_passed": False,
        "deterministic_replay_passed": False,
    }


def enumerate_policies(
    *,
    kv_presets: list[str],
    hot_window_tokens: list[int],
    eligible_layer_masks: list[str],
    fallback_policies: list[str],
    quality_profiles: list[str],
) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for preset, hot_window, layer_mask, fallback_policy, quality_profile in itertools.product(
        kv_presets,
        hot_window_tokens,
        eligible_layer_masks,
        fallback_policies,
        quality_profiles,
    ):
        policies.append(
            {
                "kv_preset": preset,
                "hot_window_tokens": hot_window,
                "eligible_layer_mask": layer_mask,
                "fallback_policy": fallback_policy,
                "quality_profile": quality_profile,
            }
        )
    return policies


def build_search_artifact(
    *,
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    kv_presets: list[str],
    hot_window_tokens: list[int],
    eligible_layer_masks: list[str],
    fallback_policies: list[str],
    quality_profiles: list[str],
    seed: int,
    repo: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_search_space(
        kv_presets=kv_presets,
        hot_window_tokens=hot_window_tokens,
        eligible_layer_masks=eligible_layer_masks,
        fallback_policies=fallback_policies,
        quality_profiles=quality_profiles,
    )
    policies = enumerate_policies(
        kv_presets=kv_presets,
        hot_window_tokens=hot_window_tokens,
        eligible_layer_masks=eligible_layer_masks,
        fallback_policies=fallback_policies,
        quality_profiles=quality_profiles,
    )
    if not policies:
        raise TurboQuantPolicySearchError("policy grid produced no candidates")
    candidates = [
        candidate_row(policy=policy, baseline=baseline, seed=seed)
        for policy in policies
    ]
    enriched_metadata = dict(metadata)
    enriched_metadata.update(
        {
            "target": "turboquant_kv_policy",
            "status": "diagnostic_only",
            "algorithm": "grid",
            "seed": seed,
            "max_candidates": len(candidates),
            "space": {
                "kv_preset": kv_presets,
                "hot_window_tokens": hot_window_tokens,
                "eligible_layer_mask": eligible_layer_masks,
                "fallback_policy": fallback_policies,
                "quality_profile": quality_profiles,
            },
        }
    )
    return builder.build_offline_policy_search_artifact(
        metadata=enriched_metadata,
        baseline=baseline,
        candidates=candidates,
        decision_classification="diagnostic_only",
        decision_reason=(
            "TurboQuant KV policy grid enumerated for later measurement; "
            "candidate metrics are not performance evidence"
        ),
        best_policy_id=str(candidates[0]["policy_id"]),
        repo=repo,
        seed=seed,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Exact artifact path. Overrides --output-root when set.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root for the canonical artifact path when --output is omitted "
            "(default: benchmarks/results/offline-policy-search)."
        ),
    )
    parser.add_argument("--kv-presets", type=parse_csv, default=DEFAULT_KV_PRESETS)
    parser.add_argument(
        "--hot-window-tokens",
        type=parse_int_csv,
        default=DEFAULT_HOT_WINDOW_TOKENS,
    )
    parser.add_argument(
        "--eligible-layer-masks",
        type=parse_csv,
        default=DEFAULT_ELIGIBLE_LAYER_MASKS,
    )
    parser.add_argument(
        "--fallback-policies",
        type=parse_csv,
        default=DEFAULT_FALLBACK_POLICIES,
    )
    parser.add_argument(
        "--quality-profiles",
        type=parse_csv,
        default=DEFAULT_QUALITY_PROFILES,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-git-repo-metadata", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metadata = builder.load_json(args.metadata)
        baseline = builder.load_json(args.baseline)
        repo = None
        if not args.skip_git_repo_metadata:
            repo = builder.default_repo_metadata(allow_dirty=args.allow_dirty)
        artifact = build_search_artifact(
            metadata=metadata,
            baseline=baseline,
            kv_presets=args.kv_presets,
            hot_window_tokens=args.hot_window_tokens,
            eligible_layer_masks=args.eligible_layer_masks,
            fallback_policies=args.fallback_policies,
            quality_profiles=args.quality_profiles,
            seed=args.seed,
            repo=repo,
        )
        output = artifact_output_path(
            explicit_output=args.output,
            output_root=args.output_root,
            artifact=artifact,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        checker.validate_offline_policy_search_artifact(output)
    except (
        TurboQuantPolicySearchError,
        builder.OfflinePolicySearchBuildError,
        checker.OfflinePolicySearchArtifactError,
    ) as error:
        print(f"turboquant kv policy search failed: {error}", file=sys.stderr)
        return 1
    print(f"wrote TurboQuant KV policy search artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
