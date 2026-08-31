#!/usr/bin/env python3
"""Unit tests for verify_rotation_equivalence.py.

No companion test file previously existed; scope limited to the specific
bug fixed here rather than full coverage of the script.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("verify_rotation_equivalence.py")
MODULE_SPEC = importlib.util.spec_from_file_location("verify_rotation_equivalence", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


class ComparePairTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        cmp = mod.compare_pair([1, 2, 3], [1, 2, 3])
        self.assertTrue(cmp["tokens_match"])
        self.assertEqual(cmp["shared_prefix_len"], 3)


class AggregateComparisonsTests(unittest.TestCase):
    def test_degenerate_tested_output_is_not_invisible_to_the_ratio(self) -> None:
        # Regression test: the denominator used to be tested_token_count
        # alone. A prompt whose tested (rotation-enabled) decode degenerates
        # to an empty output contributed (shared=0, tested=0) — invisible to
        # the aggregate ratio instead of counting as a near-total mismatch.
        per_prompt = [mod.compare_pair([1, 2, 3, 4, 5], []) for _ in range(1)]
        # 4 perfect 64-token matches plus 1 fully-degenerate prompt.
        per_prompt = [mod.compare_pair(list(range(64)), list(range(64))) for _ in range(4)]
        per_prompt.append(mod.compare_pair([1, 2, 3, 4, 5], []))

        result = mod.aggregate_comparisons(per_prompt)

        # Old (buggy) denominator: tested_total = 64*4 + 0 = 256, ratio = 1.0.
        self.assertEqual(result["tested_total"], 64 * 4 + 5)
        self.assertAlmostEqual(result["shared_prefix_ratio"], (64 * 4) / (64 * 4 + 5))
        self.assertLess(result["shared_prefix_ratio"], 1.0)

    def test_all_prompts_matching_is_a_ratio_of_one(self) -> None:
        per_prompt = [mod.compare_pair(list(range(10)), list(range(10))) for _ in range(3)]
        result = mod.aggregate_comparisons(per_prompt)
        self.assertEqual(result["shared_prefix_ratio"], 1.0)
        self.assertEqual(result["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()
