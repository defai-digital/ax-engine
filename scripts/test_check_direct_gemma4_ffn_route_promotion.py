#!/usr/bin/env python3
"""Unit tests for direct Gemma4 post-attention FFN route promotion checks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_direct_gemma4_ffn_route_promotion.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_direct_gemma4_ffn_route_promotion", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def metric(median: float) -> dict[str, float]:
    return {"mean": median, "median": median, "min": median, "max": median}


def route(
    *,
    classification: str = "all_hits",
    attempts: int = 96,
    hits: int = 96,
) -> dict[str, object]:
    return {
        "schema_version": checker.ROUTE_SCHEMA_VERSION,
        "classification": classification,
        "attempts": attempts,
        "hits": hits,
        "fallbacks": attempts - hits,
        "profile_blocked": 0,
        "hit_rate_micros": round(hits * 1_000_000 / attempts),
    }


def ax_row(
    *,
    prompt_tokens: int,
    prefill_tok_s: float,
    decode_tok_s: float,
    enabled: bool,
    classification: str = "all_hits",
) -> dict[str, object]:
    row: dict[str, object] = {
        "engine": checker.CANDIDATE_ENGINE if enabled else checker.BASELINE_ENGINE,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": 128,
        "ax_decode_policy": checker.BASELINE_POLICY,
        "prefill_tok_s": metric(prefill_tok_s),
        "decode_tok_s": metric(decode_tok_s),
    }
    if enabled:
        row["ax_mlx_direct_cpp_gemma4_post_attn_ffn"] = route(
            classification=classification,
            hits=96 if classification == "all_hits" else 80,
        )
    return row


def artifact(
    *,
    long_prefill_ratio: float = 1.12,
    short_prefill_ratio: float = 1.0,
    decode_ratio: float = 1.0,
    classification: str = "all_hits",
    git_tracked_dirty: bool = False,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for prompt in (128, 512, 2048):
        baseline_prefill = 1000.0
        candidate_prefill = baseline_prefill * (
            long_prefill_ratio if prompt == 2048 else short_prefill_ratio
        )
        baseline_decode = 100.0
        candidate_decode = baseline_decode * decode_ratio
        results.append(
            ax_row(
                prompt_tokens=prompt,
                prefill_tok_s=baseline_prefill,
                decode_tok_s=baseline_decode,
                enabled=False,
            )
        )
        results.append(
            ax_row(
                prompt_tokens=prompt,
                prefill_tok_s=candidate_prefill,
                decode_tok_s=candidate_decode,
                enabled=True,
                classification=classification,
            )
        )
    return {
        "schema_version": checker.SCHEMA_VERSION,
        "build": {
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "git_tracked_dirty": git_tracked_dirty,
        },
        "model": "/tmp/models/gemma-4-e2b-it-4bit",
        "results": results,
    }


class DirectGemma4FfnRoutePromotionTests(unittest.TestCase):
    def write_fixture(self, payload: dict[str, object]) -> Path:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = Path(root.name) / "gemma4-ffn-route-ab.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def test_promotion_candidate_passes_thresholds(self) -> None:
        path = self.write_fixture(artifact())

        decision = checker.check_direct_gemma4_ffn_route_promotion(
            [path],
            expect_decision=checker.PROMOTION_CANDIDATE,
        )

        self.assertEqual(decision.decision, checker.PROMOTION_CANDIDATE)
        self.assertEqual(decision.comparison_count, 3)
        self.assertGreaterEqual(decision.long_prompt_prefill_ratio, 1.10)
        self.assertGreaterEqual(decision.min_decode_ratio, 0.97)

    def test_promotion_candidate_rejects_long_prompt_under_threshold(self) -> None:
        path = self.write_fixture(artifact(long_prefill_ratio=1.05))

        with self.assertRaisesRegex(
            checker.DirectGemma4FfnRoutePromotionError,
            "not a promotion candidate",
        ):
            checker.check_direct_gemma4_ffn_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_promotion_candidate_rejects_short_prompt_regression(self) -> None:
        path = self.write_fixture(artifact(short_prefill_ratio=0.95))

        decision = checker.check_direct_gemma4_ffn_route_promotion(
            [path],
            expect_decision=checker.NOT_PROMOTED,
        )
        self.assertEqual(decision.decision, checker.NOT_PROMOTED)
        self.assertLess(decision.min_short_prefill_ratio, 0.97)

    def test_promotion_candidate_rejects_decode_regression(self) -> None:
        path = self.write_fixture(artifact(decode_ratio=0.96))

        with self.assertRaisesRegex(
            checker.DirectGemma4FfnRoutePromotionError,
            "not a promotion candidate",
        ):
            checker.check_direct_gemma4_ffn_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_route_classification_must_be_all_hits_by_default(self) -> None:
        path = self.write_fixture(artifact(classification="mixed_hit_fallback"))

        with self.assertRaisesRegex(
            checker.DirectGemma4FfnRoutePromotionError,
            "classification",
        ):
            checker.check_direct_gemma4_ffn_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_candidate_row_must_use_candidate_engine_key(self) -> None:
        payload = artifact()
        candidate = payload["results"][1]  # type: ignore[index]
        candidate["engine"] = checker.BASELINE_ENGINE  # type: ignore[index]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.DirectGemma4FfnRoutePromotionError,
            "baseline row .* must not carry enabled route attempts",
        ):
            checker.check_direct_gemma4_ffn_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_candidate_engine_row_must_carry_route_attempts(self) -> None:
        payload = artifact()
        candidate = payload["results"][1]  # type: ignore[index]
        candidate.pop("ax_mlx_direct_cpp_gemma4_post_attn_ffn")  # type: ignore[union-attr]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.DirectGemma4FfnRoutePromotionError,
            "candidate row .* must carry route attempts > 0",
        ):
            checker.check_direct_gemma4_ffn_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_promotion_candidate_rejects_dirty_git_artifact_by_default(self) -> None:
        path = self.write_fixture(artifact(git_tracked_dirty=True))

        with self.assertRaisesRegex(
            checker.DirectGemma4FfnRoutePromotionError,
            "git_tracked_dirty",
        ):
            checker.check_direct_gemma4_ffn_route_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_dirty_git_artifact_can_be_allowed_for_exploration(self) -> None:
        path = self.write_fixture(artifact(git_tracked_dirty=True))

        decision = checker.check_direct_gemma4_ffn_route_promotion(
            [path],
            expect_decision=checker.PROMOTION_CANDIDATE,
            require_clean_git=False,
        )

        self.assertEqual(decision.decision, checker.PROMOTION_CANDIDATE)

    def test_cli_reports_promotion_candidate(self) -> None:
        path = self.write_fixture(artifact())

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--artifact",
                str(path),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("decision=promotion_candidate", completed.stdout)
        self.assertIn("long_prefill=", completed.stdout)

    def test_cli_rejects_unexpected_promotion_candidate(self) -> None:
        path = self.write_fixture(artifact())

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--artifact",
                str(path),
                "--expect-decision",
                checker.NOT_PROMOTED,
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("promotion candidate", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_cli_can_allow_dirty_git_artifact_for_exploration(self) -> None:
        path = self.write_fixture(artifact(git_tracked_dirty=True))

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--artifact",
                str(path),
                "--allow-dirty-git-artifact",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("decision=promotion_candidate", completed.stdout)


if __name__ == "__main__":
    unittest.main()
