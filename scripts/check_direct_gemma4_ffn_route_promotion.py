#!/usr/bin/env python3
"""Gate direct Gemma4 post-attention FFN route promotion decisions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ax.mlx_inference_stack.v2"
ROUTE_SCHEMA_VERSION = "ax.mlx_direct_cpp_gemma4_post_attn_ffn.v1"
BASELINE_ENGINE = "ax_engine_mlx"
CANDIDATE_ENGINE = "ax_engine_mlx_direct_gemma4_ffn"
BASELINE_POLICY = "direct_no_ngram_acceleration"
NOT_PROMOTED = "not_promoted"
PROMOTION_CANDIDATE = "promotion_candidate"
DEFAULT_REQUIRED_PROMPTS = (128, 512, 2048)
DEFAULT_SHORT_GUARD_PROMPTS = (128, 512)
DEFAULT_LONG_PROMPT = 2048
DEFAULT_MIN_LONG_PREFILL_RATIO = 1.10
DEFAULT_MIN_NO_REGRESSION_RATIO = 0.97
DEFAULT_MODEL_FRAGMENTS = ("gemma-4-e2b", "4bit")


class DirectGemma4FfnRoutePromotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class RouteComparison:
    prompt_tokens: int
    generation_tokens: int
    prefill_ratio: float
    decode_ratio: float
    route_classification: str
    route_attempts: int
    route_hits: int


@dataclass(frozen=True)
class DirectGemma4FfnRoutePromotionDecision:
    decision: str
    comparison_count: int
    long_prompt_prefill_ratio: float
    min_short_prefill_ratio: float
    min_decode_ratio: float
    summary: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DirectGemma4FfnRoutePromotionError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be an array")
    return value


def _integer(value: Any, field: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{field} must be an integer",
    )
    return value


def _string(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value, f"{field} must be a non-empty string")
    return value


def _positive_integer(value: Any, field: str) -> int:
    number = _integer(value, field)
    _require(number > 0, f"{field} must be positive")
    return number


def _number(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be numeric",
    )
    return float(value)


def _metric_median(row: dict[str, Any], key: str, owner: str) -> float:
    metric = row.get(key)
    if isinstance(metric, dict):
        value = _number(metric.get("median"), f"{owner}.{key}.median")
    else:
        value = _number(metric, f"{owner}.{key}")
    _require(value > 0.0, f"{owner}.{key} must be positive")
    return value


def _route_summary(row: dict[str, Any]) -> dict[str, Any] | None:
    route = row.get("ax_mlx_direct_cpp_gemma4_post_attn_ffn")
    if route is None:
        return None
    return _mapping(route, "ax_mlx_direct_cpp_gemma4_post_attn_ffn")


def _row_key(row: dict[str, Any], owner: str) -> tuple[int, int]:
    return (
        _positive_integer(row.get("prompt_tokens"), f"{owner}.prompt_tokens"),
        _positive_integer(row.get("generation_tokens"), f"{owner}.generation_tokens"),
    )


def _is_ax_direct_row(row: dict[str, Any]) -> bool:
    return row.get("engine") in {BASELINE_ENGINE, CANDIDATE_ENGINE} and row.get(
        "ax_decode_policy"
    ) == BASELINE_POLICY


def _validate_clean_git_artifact(doc: dict[str, Any], path: Path) -> None:
    build = _mapping(doc.get("build"), f"{path}.build")
    _string(build.get("commit"), f"{path}.build.commit")
    _require(
        build.get("git_tracked_dirty") is False,
        f"{path}: build.git_tracked_dirty must be false for promotion evidence",
    )


def _load_artifact(
    path: Path,
    model_fragments: tuple[str, ...],
    *,
    require_clean_git: bool,
) -> list[dict[str, Any]]:
    try:
        doc = json.loads(path.read_text())
    except OSError as error:
        raise DirectGemma4FfnRoutePromotionError(f"{path}: failed to read artifact: {error}") from error
    except json.JSONDecodeError as error:
        raise DirectGemma4FfnRoutePromotionError(f"{path}: invalid JSON: {error}") from error
    _require(doc.get("schema_version") == SCHEMA_VERSION, f"{path}: schema_version must be {SCHEMA_VERSION}")
    if require_clean_git:
        _validate_clean_git_artifact(doc, path)
    model_text = " ".join(
        str(doc.get(key, ""))
        for key in ("model", "model_repo_id", "model_dir", "model_dir_source")
    ).lower()
    for fragment in model_fragments:
        _require(
            fragment.lower() in model_text,
            f"{path}: model metadata must include {fragment!r}",
        )
    rows = _array(doc.get("results"), f"{path}.results")
    return [_mapping(row, f"{path}.results[{index}]") for index, row in enumerate(rows)]


def collect_route_comparisons(
    artifacts: list[Path],
    *,
    required_prompts: tuple[int, ...],
    allowed_route_classifications: tuple[str, ...],
    model_fragments: tuple[str, ...],
    require_clean_git: bool,
) -> list[RouteComparison]:
    baselines: dict[tuple[int, int], dict[str, Any]] = {}
    candidates: dict[tuple[int, int], dict[str, Any]] = {}

    for path in artifacts:
        for row in _load_artifact(
            path,
            model_fragments,
            require_clean_git=require_clean_git,
        ):
            if not _is_ax_direct_row(row):
                continue
            owner = f"{path}.row"
            key = _row_key(row, owner)
            route = _route_summary(row)
            attempts = int(route.get("attempts", 0)) if route else 0
            engine = row.get("engine")
            if engine == CANDIDATE_ENGINE:
                _require(
                    attempts > 0,
                    f"candidate row for prompt={key[0]} generation={key[1]} "
                    "must carry route attempts > 0",
                )
                target = candidates
            else:
                _require(
                    attempts == 0,
                    f"baseline row for prompt={key[0]} generation={key[1]} "
                    "must not carry enabled route attempts",
                )
                target = baselines
            _require(
                key not in target,
                f"duplicate {'candidate' if engine == CANDIDATE_ENGINE else 'baseline'} "
                f"row for prompt={key[0]} generation={key[1]}",
            )
            target[key] = row

    required = set(required_prompts)
    observed_prompts = {prompt for prompt, _ in baselines} | {prompt for prompt, _ in candidates}
    missing = sorted(required - observed_prompts)
    _require(not missing, f"missing required prompt(s): {missing}")

    comparisons: list[RouteComparison] = []
    for key, candidate in sorted(candidates.items()):
        prompt, generation = key
        if prompt not in required:
            continue
        baseline = baselines.get(key)
        _require(
            baseline is not None,
            f"missing baseline row for prompt={prompt} generation={generation}",
        )
        route = _mapping(
            candidate.get("ax_mlx_direct_cpp_gemma4_post_attn_ffn"),
            "candidate.ax_mlx_direct_cpp_gemma4_post_attn_ffn",
        )
        _require(
            route.get("schema_version") == ROUTE_SCHEMA_VERSION,
            f"candidate route schema_version must be {ROUTE_SCHEMA_VERSION}",
        )
        classification = str(route.get("classification", ""))
        _require(
            classification in allowed_route_classifications,
            "candidate route classification must be one of "
            f"{sorted(allowed_route_classifications)}; got {classification!r}",
        )
        attempts = _positive_integer(route.get("attempts"), "candidate.route.attempts")
        hits = _integer(route.get("hits"), "candidate.route.hits")
        _require(hits >= 0, "candidate.route.hits must be non-negative")
        _require(hits <= attempts, "candidate.route.hits must be <= attempts")
        baseline_prefill = _metric_median(baseline, "prefill_tok_s", "baseline")
        candidate_prefill = _metric_median(candidate, "prefill_tok_s", "candidate")
        baseline_decode = _metric_median(baseline, "decode_tok_s", "baseline")
        candidate_decode = _metric_median(candidate, "decode_tok_s", "candidate")
        comparisons.append(
            RouteComparison(
                prompt_tokens=prompt,
                generation_tokens=generation,
                prefill_ratio=candidate_prefill / baseline_prefill,
                decode_ratio=candidate_decode / baseline_decode,
                route_classification=classification,
                route_attempts=attempts,
                route_hits=hits,
            )
        )

    covered = {comparison.prompt_tokens for comparison in comparisons}
    missing_candidate = sorted(required - covered)
    _require(
        not missing_candidate,
        f"missing candidate row(s) for required prompt(s): {missing_candidate}",
    )
    return comparisons


def decide_direct_gemma4_ffn_route_promotion(
    artifacts: list[Path],
    *,
    required_prompts: tuple[int, ...] = DEFAULT_REQUIRED_PROMPTS,
    short_guard_prompts: tuple[int, ...] = DEFAULT_SHORT_GUARD_PROMPTS,
    long_prompt: int = DEFAULT_LONG_PROMPT,
    min_long_prefill_ratio: float = DEFAULT_MIN_LONG_PREFILL_RATIO,
    min_no_regression_ratio: float = DEFAULT_MIN_NO_REGRESSION_RATIO,
    allowed_route_classifications: tuple[str, ...] = ("all_hits",),
    model_fragments: tuple[str, ...] = DEFAULT_MODEL_FRAGMENTS,
    require_clean_git: bool = True,
) -> DirectGemma4FfnRoutePromotionDecision:
    if min_long_prefill_ratio <= 1.0:
        raise DirectGemma4FfnRoutePromotionError("--min-long-prefill-ratio must be > 1.0")
    if not (0.0 < min_no_regression_ratio <= 1.0):
        raise DirectGemma4FfnRoutePromotionError("--min-no-regression-ratio must be in (0, 1]")
    _require(long_prompt in required_prompts, "--long-prompt must be one of --required-prompt")
    comparisons = collect_route_comparisons(
        artifacts,
        required_prompts=required_prompts,
        allowed_route_classifications=allowed_route_classifications,
        model_fragments=model_fragments,
        require_clean_git=require_clean_git,
    )
    by_prompt = {comparison.prompt_tokens: comparison for comparison in comparisons}
    long_prefill_ratio = by_prompt[long_prompt].prefill_ratio
    short_prefill_ratios = [
        by_prompt[prompt].prefill_ratio
        for prompt in short_guard_prompts
        if prompt in by_prompt
    ]
    _require(short_prefill_ratios, "short guard prompts must overlap collected comparisons")
    min_short_prefill_ratio = min(short_prefill_ratios)
    min_decode_ratio = min(comparison.decode_ratio for comparison in comparisons)
    promoted = (
        long_prefill_ratio >= min_long_prefill_ratio
        and min_short_prefill_ratio >= min_no_regression_ratio
        and min_decode_ratio >= min_no_regression_ratio
    )
    summary = "; ".join(
        f"p{comparison.prompt_tokens}: prefill={comparison.prefill_ratio:.4f}x "
        f"decode={comparison.decode_ratio:.4f}x "
        f"route={comparison.route_classification} "
        f"hits={comparison.route_hits}/{comparison.route_attempts}"
        for comparison in comparisons
    )
    return DirectGemma4FfnRoutePromotionDecision(
        decision=PROMOTION_CANDIDATE if promoted else NOT_PROMOTED,
        comparison_count=len(comparisons),
        long_prompt_prefill_ratio=long_prefill_ratio,
        min_short_prefill_ratio=min_short_prefill_ratio,
        min_decode_ratio=min_decode_ratio,
        summary=summary,
    )


def check_direct_gemma4_ffn_route_promotion(
    artifacts: list[Path],
    *,
    expect_decision: str,
    required_prompts: tuple[int, ...] = DEFAULT_REQUIRED_PROMPTS,
    short_guard_prompts: tuple[int, ...] = DEFAULT_SHORT_GUARD_PROMPTS,
    long_prompt: int = DEFAULT_LONG_PROMPT,
    min_long_prefill_ratio: float = DEFAULT_MIN_LONG_PREFILL_RATIO,
    min_no_regression_ratio: float = DEFAULT_MIN_NO_REGRESSION_RATIO,
    allowed_route_classifications: tuple[str, ...] = ("all_hits",),
    model_fragments: tuple[str, ...] = DEFAULT_MODEL_FRAGMENTS,
    require_clean_git: bool = True,
) -> DirectGemma4FfnRoutePromotionDecision:
    decision = decide_direct_gemma4_ffn_route_promotion(
        artifacts,
        required_prompts=required_prompts,
        short_guard_prompts=short_guard_prompts,
        long_prompt=long_prompt,
        min_long_prefill_ratio=min_long_prefill_ratio,
        min_no_regression_ratio=min_no_regression_ratio,
        allowed_route_classifications=allowed_route_classifications,
        model_fragments=model_fragments,
        require_clean_git=require_clean_git,
    )
    if decision.decision != expect_decision:
        if expect_decision == NOT_PROMOTED:
            raise DirectGemma4FfnRoutePromotionError(
                "direct Gemma4 FFN route evidence is now a promotion candidate; "
                "update the PRD decision and runtime default deliberately"
            )
        raise DirectGemma4FfnRoutePromotionError(
            "direct Gemma4 FFN route evidence is not a promotion candidate: "
            f"{decision.summary}"
        )
    return decision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=Path, required=True)
    parser.add_argument(
        "--expect-decision",
        choices=[NOT_PROMOTED, PROMOTION_CANDIDATE],
        default=PROMOTION_CANDIDATE,
    )
    parser.add_argument("--required-prompt", action="append", type=int, default=None)
    parser.add_argument("--short-guard-prompt", action="append", type=int, default=None)
    parser.add_argument("--long-prompt", type=int, default=DEFAULT_LONG_PROMPT)
    parser.add_argument("--min-long-prefill-ratio", type=float, default=DEFAULT_MIN_LONG_PREFILL_RATIO)
    parser.add_argument("--min-no-regression-ratio", type=float, default=DEFAULT_MIN_NO_REGRESSION_RATIO)
    parser.add_argument(
        "--allow-route-classification",
        action="append",
        default=None,
        help="Accepted candidate route classification. Defaults to all_hits only.",
    )
    parser.add_argument(
        "--model-fragment",
        action="append",
        default=None,
        help="Required case-insensitive model metadata substring.",
    )
    parser.add_argument(
        "--allow-dirty-git-artifact",
        action="store_true",
        help=(
            "Allow artifacts from a dirty tracked worktree. Use only for local "
            "exploration; promotion evidence is clean-git by default."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    required_prompts = tuple(args.required_prompt or DEFAULT_REQUIRED_PROMPTS)
    short_guard_prompts = tuple(args.short_guard_prompt or DEFAULT_SHORT_GUARD_PROMPTS)
    allowed_route_classifications = tuple(args.allow_route_classification or ("all_hits",))
    model_fragments = tuple(args.model_fragment or DEFAULT_MODEL_FRAGMENTS)
    try:
        decision = check_direct_gemma4_ffn_route_promotion(
            args.artifact,
            expect_decision=args.expect_decision,
            required_prompts=required_prompts,
            short_guard_prompts=short_guard_prompts,
            long_prompt=args.long_prompt,
            min_long_prefill_ratio=args.min_long_prefill_ratio,
            min_no_regression_ratio=args.min_no_regression_ratio,
            allowed_route_classifications=allowed_route_classifications,
            model_fragments=model_fragments,
            require_clean_git=not args.allow_dirty_git_artifact,
        )
    except DirectGemma4FfnRoutePromotionError as error:
        print(f"Direct Gemma4 FFN route promotion check failed: {error}", file=sys.stderr)
        return 1
    print(
        "Direct Gemma4 FFN route promotion check passed: "
        f"decision={decision.decision}; "
        f"comparisons={decision.comparison_count}; "
        f"long_prefill={decision.long_prompt_prefill_ratio:.4f}x; "
        f"short_prefill_min={decision.min_short_prefill_ratio:.4f}x; "
        f"decode_min={decision.min_decode_ratio:.4f}x; "
        f"{decision.summary}"
    )
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
