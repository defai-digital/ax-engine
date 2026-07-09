#!/usr/bin/env python3
"""Unit tests for the direct model I/O smoke matrix."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_direct_model_io.py")
MODULE_SPEC = importlib.util.spec_from_file_location("check_direct_model_io", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
check_direct_model_io = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = check_direct_model_io
MODULE_SPEC.loader.exec_module(check_direct_model_io)


class DirectModelIoMatrixTests(unittest.TestCase):
    def test_direct_model_cases_follow_supported_direct_llm_families(self) -> None:
        slugs = {case.slug for case in check_direct_model_io.MODEL_CASES}

        self.assertIn("gemma-4-e2b-it-4bit", slugs)
        self.assertIn("gemma-4-e4b-it-4bit", slugs)
        self.assertIn("gemma-4-26b-a4b-it-4bit", slugs)
        self.assertIn("gemma-4-31b-it-4bit", slugs)
        self.assertIn("qwen3-4b-4bit", slugs)
        self.assertIn("qwen3-5-9b-mlx-4bit", slugs)
        self.assertIn("qwen3-6-27b-4bit", slugs)
        self.assertIn("qwen3-6-35b-a3b-4bit", slugs)
        self.assertIn("qwen3-coder-next-6bit", slugs)
        self.assertIn("glm-4-7-flash-4bit", slugs)

    def test_glm_is_a_direct_model_io_case(self) -> None:
        slugs = {case.slug for case in check_direct_model_io.MODEL_CASES}
        model_ids = {case.model_id for case in check_direct_model_io.MODEL_CASES}

        self.assertIn("glm-4-7-flash-4bit", slugs)
        self.assertIn("mlx-community/GLM-4.7-Flash-4bit", model_ids)

    def test_near_tie_equivalence_accepts_topk_token_inside_margin(self) -> None:
        ref = {"top_logprobs": [[[10, -0.50], [20, -0.62], [30, -3.0]]]}

        equivalence = check_direct_model_io.near_tie_prefix_equivalence(
            ref,
            [10, 11],
            [20, 99],
            prefix=0,
            required=2,
            margin=0.25,
        )

        self.assertIsNotNone(equivalence)
        assert equivalence is not None
        self.assertEqual(equivalence["ax_token"], 20)
        self.assertEqual(equivalence["ax_rank"], 2)
        self.assertAlmostEqual(equivalence["logprob_gap"], 0.12)

    def test_near_tie_equivalence_rejects_token_outside_margin(self) -> None:
        ref = {"top_logprobs": [[[10, -0.50], [20, -0.90]]]}

        equivalence = check_direct_model_io.near_tie_prefix_equivalence(
            ref,
            [10],
            [20],
            prefix=0,
            required=1,
            margin=0.25,
        )

        self.assertIsNone(equivalence)

    def test_near_tie_equivalence_rejects_token_absent_from_reference_topk(self) -> None:
        ref = {"top_logprobs": [[[10, -0.50], [20, -0.62]]]}

        equivalence = check_direct_model_io.near_tie_prefix_equivalence(
            ref,
            [10],
            [99],
            prefix=0,
            required=1,
            margin=0.25,
        )

        self.assertIsNone(equivalence)

    def test_aggregate_status_preserves_near_tie_as_distinct_quality_state(self) -> None:
        self.assertEqual(
            check_direct_model_io.aggregate_status(
                [
                    check_direct_model_io.PASS_STATUS,
                    check_direct_model_io.NEAR_TIE_PASS_STATUS,
                ]
            ),
            check_direct_model_io.NEAR_TIE_PASS_STATUS,
        )
        self.assertEqual(
            check_direct_model_io.aggregate_status(
                [
                    check_direct_model_io.NEAR_TIE_PASS_STATUS,
                    check_direct_model_io.FAIL_STATUS,
                ]
            ),
            check_direct_model_io.FAIL_STATUS,
        )
        self.assertEqual(
            check_direct_model_io.aggregate_status(
                [
                    check_direct_model_io.FAIL_STATUS,
                    check_direct_model_io.ERROR_STATUS,
                ]
            ),
            check_direct_model_io.ERROR_STATUS,
        )

    def test_count_statuses_reports_exact_and_near_tie_separately(self) -> None:
        counts = check_direct_model_io.count_statuses(
            [
                {"status": check_direct_model_io.PASS_STATUS},
                {"status": check_direct_model_io.NEAR_TIE_PASS_STATUS},
                {"status": check_direct_model_io.FAIL_STATUS},
            ]
        )

        self.assertEqual(counts["pass"], 1)
        self.assertEqual(counts["near_tie_pass"], 1)
        self.assertEqual(counts["fail"], 1)
        self.assertEqual(counts["accepted"], 2)


if __name__ == "__main__":
    unittest.main()
