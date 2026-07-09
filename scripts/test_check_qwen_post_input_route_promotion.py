#!/usr/bin/env python3
"""Unit tests for Qwen post-input route promotion checks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_qwen_post_input_route_promotion.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_qwen_post_input_route_promotion", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def metric(median: float) -> dict[str, float]:
    return {"mean": median, "median": median, "min": median, "max": median}


def input_route() -> dict[str, object]:
    return {
        "schema_version": checker.INPUT_ROUTE_SCHEMA_VERSION,
        "classification": "all_hits",
        "attempts": 150,
        "hits": 150,
        "fallbacks": 0,
        "profile_blocked": 0,
        "hit_rate_micros": 1_000_000,
    }


def post_route(
    *,
    classification: str = "all_hits",
    attempts: int = 150,
    hits: int = 150,
) -> dict[str, object]:
    return {
        "schema_version": checker.POST_INPUT_ROUTE_SCHEMA_VERSION,
        "classification": classification,
        "attempts": attempts,
        "hits": hits,
        "fallbacks": attempts - hits,
        "profile_blocked": 0,
        "hit_rate_micros": round(hits * 1_000_000 / attempts),
    }


def row(
    *,
    prompt_tokens: int,
    enabled: bool,
    decode_tok_s: float,
    generation_tokens: int = checker.DEFAULT_REQUIRED_GENERATION_TOKENS,
    post_classification: str = "all_hits",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "engine": checker.CANDIDATE_ENGINE if enabled else checker.BASELINE_ENGINE,
        "ax_decode_policy": checker.BASELINE_POLICY,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "decode_tok_s": metric(decode_tok_s),
        checker.INPUT_ROUTE_FLAG: True,
        checker.POST_INPUT_ROUTE_FLAG: bool(enabled),
        "ax_mlx_direct_cpp_linear_attention_inputs": input_route(),
    }
    if enabled:
        payload["ax_mlx_direct_cpp_linear_attention_post_input"] = post_route(
            classification=post_classification,
            hits=150 if post_classification == "all_hits" else 140,
        )
    return payload


def artifact(
    *,
    decode_ratio: float = 1.02,
    post_classification: str = "all_hits",
    git_tracked_dirty: bool = False,
    compare_flag: bool = True,
    generation_tokens: int = checker.DEFAULT_REQUIRED_GENERATION_TOKENS,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for prompt in checker.DEFAULT_REQUIRED_PROMPTS:
        results.append(
            row(
                prompt_tokens=prompt,
                enabled=False,
                generation_tokens=generation_tokens,
                decode_tok_s=100.0,
            )
        )
        results.append(
            row(
                prompt_tokens=prompt,
                enabled=True,
                generation_tokens=generation_tokens,
                decode_tok_s=100.0 * decode_ratio,
                post_classification=post_classification,
            )
        )
    return {
        "schema_version": checker.SCHEMA_VERSION,
        "ax_direct_linear_attention_post_input_route_compare": compare_flag,
        "build": {
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "git_tracked_dirty": git_tracked_dirty,
        },
        "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "model_repo_id": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "results": results,
    }


class QwenPostInputRoutePromotionTests(unittest.TestCase):
    def write_fixture(self, payload: dict[str, object]) -> Path:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = Path(root.name) / "qwen-post-input-ab.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def test_default_cli_skips_without_current_artifacts(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("[skip]", completed.stdout)

    def test_promotion_candidate_passes_decode_threshold(self) -> None:
        path = self.write_fixture(artifact(decode_ratio=1.02))

        decision = checker.check_qwen_post_input_route_promotion(
            [path],
            expect_decision=checker.PROMOTION_CANDIDATE,
        )

        self.assertEqual(decision.decision, checker.PROMOTION_CANDIDATE)
        self.assertGreaterEqual(decision.min_decode_ratio, 1.01)

    def test_promotion_candidate_rejects_sub_threshold_gain(self) -> None:
        path = self.write_fixture(artifact(decode_ratio=1.005))

        with self.assertRaisesRegex(
            checker.QwenPostInputRoutePromotionError,
            "not a promotion candidate",
        ):
            checker.check_qwen_post_input_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_post_input_route_must_be_all_hits(self) -> None:
        path = self.write_fixture(artifact(post_classification="mixed_hit_fallback"))

        with self.assertRaisesRegex(
            checker.QwenPostInputRoutePromotionError,
            "classification",
        ):
            checker.check_qwen_post_input_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_baseline_must_not_carry_post_input_summary(self) -> None:
        payload = artifact()
        baseline = payload["results"][0]  # type: ignore[index]
        baseline["ax_mlx_direct_cpp_linear_attention_post_input"] = post_route()  # type: ignore[index]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.QwenPostInputRoutePromotionError,
            "baseline row must not carry post-input route summary",
        ):
            checker.check_qwen_post_input_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_baseline_route_flags_must_match_engine_role(self) -> None:
        payload = artifact()
        baseline = payload["results"][0]  # type: ignore[index]
        baseline[checker.POST_INPUT_ROUTE_FLAG] = True  # type: ignore[index]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.QwenPostInputRoutePromotionError,
            "baseline.ax_direct_linear_attention_post_input_route must be False",
        ):
            checker.check_qwen_post_input_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_candidate_route_flags_must_match_engine_role(self) -> None:
        payload = artifact()
        candidate = payload["results"][1]  # type: ignore[index]
        candidate[checker.POST_INPUT_ROUTE_FLAG] = False  # type: ignore[index]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.QwenPostInputRoutePromotionError,
            "candidate.ax_direct_linear_attention_post_input_route must be True",
        ):
            checker.check_qwen_post_input_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_compare_flag_is_required(self) -> None:
        path = self.write_fixture(artifact(compare_flag=False))

        with self.assertRaisesRegex(
            checker.QwenPostInputRoutePromotionError,
            "ax-compare-direct-linear-attention-post-input-route",
        ):
            checker.check_qwen_post_input_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_dirty_git_artifact_rejected_by_default(self) -> None:
        path = self.write_fixture(artifact(git_tracked_dirty=True))

        with self.assertRaisesRegex(
            checker.QwenPostInputRoutePromotionError,
            "git_tracked_dirty",
        ):
            checker.check_qwen_post_input_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_required_generation_tokens_default_to_128(self) -> None:
        path = self.write_fixture(artifact(generation_tokens=64))

        with self.assertRaisesRegex(
            checker.QwenPostInputRoutePromotionError,
            "missing paired prompt/generation shape",
        ):
            checker.check_qwen_post_input_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_cli_reports_not_promoted(self) -> None:
        path = self.write_fixture(artifact(decode_ratio=1.005))

        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--artifact", str(path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("decision=not_promoted", completed.stdout)
        self.assertIn("min_decode_ratio=1.00500", completed.stdout)


if __name__ == "__main__":
    unittest.main()
