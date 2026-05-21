#!/usr/bin/env python3
"""Gate Qwen linear-attention post-input route promotion decisions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).parent.parent
DEFAULT_ARTIFACTS = (
    ROOT_DIR
    / "benchmarks/results/mlx-inference/2026-05-21-qwen35-post-input-ab/"
    / "qwen3_6-35b-a3b-4bit.json",
)
SCHEMA_VERSION = "ax.mlx_inference_stack.v2"
INPUT_ROUTE_SCHEMA_VERSION = "ax.mlx_direct_cpp_linear_attention_inputs.v1"
POST_INPUT_ROUTE_SCHEMA_VERSION = "ax.mlx_direct_cpp_linear_attention_post_input.v1"
BASELINE_ENGINE = "ax_engine_mlx_direct_linear_attention_inputs"
CANDIDATE_ENGINE = "ax_engine_mlx_direct_linear_attention_post_input"
BASELINE_POLICY = "direct_no_ngram_acceleration"
NOT_PROMOTED = "not_promoted"
PROMOTION_CANDIDATE = "promotion_candidate"
DEFAULT_REQUIRED_PROMPTS = (128, 512, 2048)
DEFAULT_REQUIRED_GENERATION_TOKENS = 128
DEFAULT_MIN_DECODE_RATIO = 1.01
DEFAULT_MODEL_FRAGMENTS = ("qwen3.6-35b-a3b", "4bit")


class QwenPostInputRoutePromotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class RouteComparison:
    prompt_tokens: int
    generation_tokens: int
    decode_ratio: float
    baseline_decode_tok_s: float
    candidate_decode_tok_s: float
    post_input_hits: int
    post_input_attempts: int


@dataclass(frozen=True)
class QwenPostInputRoutePromotionDecision:
    decision: str
    comparison_count: int
    min_decode_ratio: float
    summary: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QwenPostInputRoutePromotionError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be an array")
    return value


def _number(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be numeric",
    )
    return float(value)


def _positive_integer(value: Any, field: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{field} must be a positive integer",
    )
    return int(value)


def _metric_median(row: dict[str, Any], key: str, owner: str) -> float:
    metric = _mapping(row.get(key), f"{owner}.{key}")
    value = _number(metric.get("median"), f"{owner}.{key}.median")
    _require(value > 0.0, f"{owner}.{key}.median must be positive")
    return value


def _row_key(row: dict[str, Any], owner: str) -> tuple[int, int]:
    return (
        _positive_integer(row.get("prompt_tokens"), f"{owner}.prompt_tokens"),
        _positive_integer(row.get("generation_tokens"), f"{owner}.generation_tokens"),
    )


def _validate_clean_git_artifact(doc: dict[str, Any], path: Path) -> None:
    build = _mapping(doc.get("build"), f"{path}.build")
    _require(
        isinstance(build.get("commit"), str) and build["commit"],
        f"{path}: build.commit must be present",
    )
    _require(
        build.get("git_tracked_dirty") is False,
        f"{path}: build.git_tracked_dirty must be false for promotion evidence",
    )


def _validate_route(
    route: dict[str, Any],
    *,
    schema_version: str,
    owner: str,
) -> tuple[int, int]:
    _require(
        route.get("schema_version") == schema_version,
        f"{owner}.schema_version must be {schema_version}",
    )
    _require(
        route.get("classification") == "all_hits",
        f"{owner}.classification must be all_hits",
    )
    attempts = _positive_integer(route.get("attempts"), f"{owner}.attempts")
    hits = _positive_integer(route.get("hits"), f"{owner}.hits")
    _require(hits == attempts, f"{owner}.hits must equal attempts")
    _require(int(route.get("fallbacks", -1)) == 0, f"{owner}.fallbacks must be zero")
    _require(
        int(route.get("profile_blocked", -1)) == 0,
        f"{owner}.profile_blocked must be zero",
    )
    return hits, attempts


def _load_artifact(
    path: Path,
    model_fragments: tuple[str, ...],
    *,
    require_clean_git: bool,
) -> list[dict[str, Any]]:
    try:
        doc = json.loads(path.read_text())
    except OSError as error:
        raise QwenPostInputRoutePromotionError(f"{path}: failed to read: {error}") from error
    except json.JSONDecodeError as error:
        raise QwenPostInputRoutePromotionError(f"{path}: invalid JSON: {error}") from error
    _require(doc.get("schema_version") == SCHEMA_VERSION, f"{path}: schema_version must be {SCHEMA_VERSION}")
    _require(
        doc.get("ax_direct_linear_attention_post_input_route_compare") is True,
        f"{path}: must be produced with --ax-compare-direct-linear-attention-post-input-route",
    )
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
    required_generation_tokens: int,
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
            if row.get("ax_decode_policy") != BASELINE_POLICY:
                continue
            engine = row.get("engine")
            if engine not in {BASELINE_ENGINE, CANDIDATE_ENGINE}:
                continue
            key = _row_key(row, f"{path}.row")
            target = candidates if engine == CANDIDATE_ENGINE else baselines
            _require(key not in target, f"duplicate row for {engine} shape {key}")
            target[key] = row

    required = {(prompt, required_generation_tokens) for prompt in required_prompts}
    missing = sorted(required - (set(baselines) & set(candidates)))
    _require(not missing, f"missing paired prompt/generation shape(s): {missing}")

    comparisons: list[RouteComparison] = []
    for key in sorted(required):
        prompt, generation = key
        baseline = baselines[key]
        candidate = candidates[key]
        base_input = _mapping(
            baseline.get("ax_mlx_direct_cpp_linear_attention_inputs"),
            "baseline.ax_mlx_direct_cpp_linear_attention_inputs",
        )
        _validate_route(
            base_input,
            schema_version=INPUT_ROUTE_SCHEMA_VERSION,
            owner="baseline.input_route",
        )
        _require(
            "ax_mlx_direct_cpp_linear_attention_post_input" not in baseline,
            "baseline row must not carry post-input route summary",
        )
        cand_input = _mapping(
            candidate.get("ax_mlx_direct_cpp_linear_attention_inputs"),
            "candidate.ax_mlx_direct_cpp_linear_attention_inputs",
        )
        _validate_route(
            cand_input,
            schema_version=INPUT_ROUTE_SCHEMA_VERSION,
            owner="candidate.input_route",
        )
        cand_post = _mapping(
            candidate.get("ax_mlx_direct_cpp_linear_attention_post_input"),
            "candidate.ax_mlx_direct_cpp_linear_attention_post_input",
        )
        post_hits, post_attempts = _validate_route(
            cand_post,
            schema_version=POST_INPUT_ROUTE_SCHEMA_VERSION,
            owner="candidate.post_input_route",
        )
        baseline_decode = _metric_median(baseline, "decode_tok_s", "baseline")
        candidate_decode = _metric_median(candidate, "decode_tok_s", "candidate")
        comparisons.append(
            RouteComparison(
                prompt_tokens=prompt,
                generation_tokens=generation,
                decode_ratio=candidate_decode / baseline_decode,
                baseline_decode_tok_s=baseline_decode,
                candidate_decode_tok_s=candidate_decode,
                post_input_hits=post_hits,
                post_input_attempts=post_attempts,
            )
        )
    return comparisons


def summarize_comparisons(comparisons: list[RouteComparison]) -> str:
    return "; ".join(
        (
            f"prompt={item.prompt_tokens} gen={item.generation_tokens}: "
            f"decode_ratio={item.decode_ratio:.5f} "
            f"({item.baseline_decode_tok_s:.3f}->{item.candidate_decode_tok_s:.3f}), "
            f"post_input={item.post_input_hits}/{item.post_input_attempts}"
        )
        for item in comparisons
    )


def decide_qwen_post_input_route_promotion(
    artifacts: list[Path],
    *,
    min_decode_ratio: float = DEFAULT_MIN_DECODE_RATIO,
    required_prompts: tuple[int, ...] = DEFAULT_REQUIRED_PROMPTS,
    required_generation_tokens: int = DEFAULT_REQUIRED_GENERATION_TOKENS,
    model_fragments: tuple[str, ...] = DEFAULT_MODEL_FRAGMENTS,
    require_clean_git: bool = True,
) -> QwenPostInputRoutePromotionDecision:
    _require(min_decode_ratio > 1.0, "--min-decode-ratio must be > 1.0")
    comparisons = collect_route_comparisons(
        artifacts,
        required_prompts=required_prompts,
        required_generation_tokens=required_generation_tokens,
        model_fragments=model_fragments,
        require_clean_git=require_clean_git,
    )
    observed_min_decode_ratio = min(item.decode_ratio for item in comparisons)
    decision = (
        PROMOTION_CANDIDATE
        if observed_min_decode_ratio >= min_decode_ratio
        else NOT_PROMOTED
    )
    return QwenPostInputRoutePromotionDecision(
        decision=decision,
        comparison_count=len(comparisons),
        min_decode_ratio=observed_min_decode_ratio,
        summary=summarize_comparisons(comparisons),
    )


def check_qwen_post_input_route_promotion(
    artifacts: list[Path],
    *,
    expect_decision: str,
    min_decode_ratio: float = DEFAULT_MIN_DECODE_RATIO,
    required_prompts: tuple[int, ...] = DEFAULT_REQUIRED_PROMPTS,
    required_generation_tokens: int = DEFAULT_REQUIRED_GENERATION_TOKENS,
    model_fragments: tuple[str, ...] = DEFAULT_MODEL_FRAGMENTS,
    require_clean_git: bool = True,
) -> QwenPostInputRoutePromotionDecision:
    decision = decide_qwen_post_input_route_promotion(
        artifacts,
        min_decode_ratio=min_decode_ratio,
        required_prompts=required_prompts,
        required_generation_tokens=required_generation_tokens,
        model_fragments=model_fragments,
        require_clean_git=require_clean_git,
    )
    if decision.decision != expect_decision:
        if expect_decision == NOT_PROMOTED:
            raise QwenPostInputRoutePromotionError(
                "Qwen post-input route evidence is now a promotion candidate; "
                "update the PRD decision and runtime default deliberately"
            )
        raise QwenPostInputRoutePromotionError(
            "Qwen post-input route evidence is not a promotion candidate: "
            f"{decision.summary}"
        )
    return decision


def _parse_prompt_list(value: str) -> tuple[int, ...]:
    prompts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    _require(bool(prompts), "--required-prompts must not be empty")
    return prompts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=Path, dest="artifacts")
    parser.add_argument(
        "--expect-decision",
        choices=[NOT_PROMOTED, PROMOTION_CANDIDATE],
        default=NOT_PROMOTED,
    )
    parser.add_argument("--min-decode-ratio", type=float, default=DEFAULT_MIN_DECODE_RATIO)
    parser.add_argument(
        "--required-prompts",
        default=",".join(str(item) for item in DEFAULT_REQUIRED_PROMPTS),
    )
    parser.add_argument(
        "--required-generation-tokens",
        type=int,
        default=DEFAULT_REQUIRED_GENERATION_TOKENS,
    )
    parser.add_argument(
        "--allow-dirty-git",
        action="store_true",
        help="Allow exploratory artifacts with build.git_tracked_dirty=true.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = args.artifacts if args.artifacts else list(DEFAULT_ARTIFACTS)
    try:
        decision = check_qwen_post_input_route_promotion(
            artifacts,
            expect_decision=args.expect_decision,
            min_decode_ratio=args.min_decode_ratio,
            required_prompts=_parse_prompt_list(args.required_prompts),
            required_generation_tokens=args.required_generation_tokens,
            require_clean_git=not args.allow_dirty_git,
        )
    except QwenPostInputRoutePromotionError as error:
        print(f"Qwen post-input route promotion check failed: {error}", file=sys.stderr)
        return 1
    print(
        "Qwen post-input route promotion check passed: "
        f"decision={decision.decision}; "
        f"comparisons={decision.comparison_count}; "
        f"min_decode_ratio={decision.min_decode_ratio:.5f}; "
        f"{decision.summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
