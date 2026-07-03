#!/usr/bin/env python3
"""Gate linear-attention projection-pack promotion decisions."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
FORWARD_CHECKER_PATH = SCRIPT_DIR / "check_mlx_forward_profile_artifact.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mlx_forward_profile_artifact", FORWARD_CHECKER_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
forward_checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = forward_checker
MODULE_SPEC.loader.exec_module(forward_checker)

DEFAULT_ARTIFACTS = (
    ROOT_DIR
    / "benchmarks/results/inference/mlx-inference/2026-05-14-qwen36-linear-pack-ab/"
    / "qwen3_6-35b-a3b-8bit-linear-pack-ab.json",
)
NOT_PROMOTED = "not_promoted"
PROMOTION_CANDIDATE = "promotion_candidate"


class LinearAttentionPackPromotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class LinearAttentionPackPromotionDecision:
    decision: str
    comparison_count: int
    candidate_win_count: int
    candidate_win_prompt_count: int
    candidate_win_shape_count: int
    non_win_count: int
    summary: str


def decide_linear_attention_pack_promotion(
    artifacts: list[Path],
    *,
    min_promotion_candidate_win_prompts: int = 2,
    min_promotion_candidate_win_shapes: int = 2,
) -> LinearAttentionPackPromotionDecision:
    if min_promotion_candidate_win_prompts < 1:
        raise LinearAttentionPackPromotionError(
            "--min-promotion-candidate-win-prompts must be positive"
        )
    if min_promotion_candidate_win_shapes < 1:
        raise LinearAttentionPackPromotionError(
            "--min-promotion-candidate-win-shapes must be positive"
        )
    try:
        checked = forward_checker.check_mlx_forward_profile_artifacts(
            artifacts,
            require_pack_comparison=True,
        )
    except forward_checker.MlxForwardProfileArtifactError as error:
        raise LinearAttentionPackPromotionError(str(error)) from error

    comparisons = checked.pack_comparisons
    if not comparisons:
        raise LinearAttentionPackPromotionError(
            "linear-attention pack promotion requires at least one comparison"
        )
    non_win_count = sum(
        1 for comparison in comparisons if comparison.verdict != "candidate win"
    )
    has_required_coverage = (
        checked.pack_candidate_win_prompt_count
        >= min_promotion_candidate_win_prompts
        and checked.pack_candidate_win_shape_count >= min_promotion_candidate_win_shapes
    )
    decision = (
        PROMOTION_CANDIDATE
        if non_win_count == 0 and has_required_coverage
        else NOT_PROMOTED
    )
    return LinearAttentionPackPromotionDecision(
        decision=decision,
        comparison_count=len(comparisons),
        candidate_win_count=checked.pack_candidate_win_count,
        candidate_win_prompt_count=checked.pack_candidate_win_prompt_count,
        candidate_win_shape_count=checked.pack_candidate_win_shape_count,
        non_win_count=non_win_count,
        summary=forward_checker.summarize_pack_comparisons(comparisons),
    )


def check_linear_attention_pack_promotion(
    artifacts: list[Path],
    *,
    expect_decision: str,
    min_promotion_candidate_win_prompts: int = 2,
    min_promotion_candidate_win_shapes: int = 2,
) -> LinearAttentionPackPromotionDecision:
    decision = decide_linear_attention_pack_promotion(
        artifacts,
        min_promotion_candidate_win_prompts=min_promotion_candidate_win_prompts,
        min_promotion_candidate_win_shapes=min_promotion_candidate_win_shapes,
    )
    if decision.decision != expect_decision:
        if expect_decision == NOT_PROMOTED:
            raise LinearAttentionPackPromotionError(
                "linear-attention pack evidence is now a promotion candidate; "
                "update the PRD decision and runtime default deliberately"
            )
        raise LinearAttentionPackPromotionError(
            "linear-attention pack evidence is not a promotion candidate: "
            f"{decision.summary}"
        )
    return decision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        type=Path,
        dest="artifacts",
        help=(
            "Linear-attention split/packed A/B artifact. Defaults to the curated "
            "Qwen3.6 35B A3B 8-bit artifact."
        ),
    )
    parser.add_argument(
        "--expect-decision",
        choices=[NOT_PROMOTED, PROMOTION_CANDIDATE],
        default=NOT_PROMOTED,
        help="Expected promotion decision for the provided evidence.",
    )
    parser.add_argument(
        "--min-promotion-candidate-win-prompts",
        type=int,
        default=2,
        help="Prompt-length coverage required before evidence can promote packing.",
    )
    parser.add_argument(
        "--min-promotion-candidate-win-shapes",
        type=int,
        default=2,
        help="Shape coverage required before evidence can promote packing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = args.artifacts if args.artifacts else list(DEFAULT_ARTIFACTS)
    try:
        decision = check_linear_attention_pack_promotion(
            artifacts,
            expect_decision=args.expect_decision,
            min_promotion_candidate_win_prompts=(
                args.min_promotion_candidate_win_prompts
            ),
            min_promotion_candidate_win_shapes=args.min_promotion_candidate_win_shapes,
        )
    except LinearAttentionPackPromotionError as error:
        print(f"Linear-attention pack promotion check failed: {error}", file=sys.stderr)
        return 1
    print(
        "Linear-attention pack promotion check passed: "
        f"decision={decision.decision}; "
        f"{decision.candidate_win_count}/{decision.comparison_count} candidate wins; "
        f"{decision.candidate_win_prompt_count} candidate-win prompt length(s); "
        f"{decision.candidate_win_shape_count} candidate-win shape(s); "
        f"{decision.non_win_count} non-win comparison(s); "
        f"{decision.summary}"
    )
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
