#!/usr/bin/env python3
"""Tests for the speculative suite benchmark helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("bench_speculative_suite.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_speculative_suite", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
bench = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bench_speculative_suite"] = bench
MODULE_SPEC.loader.exec_module(bench)


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "\n".join(message["content"] for message in messages)

    def encode(self, text):
        return [ord(char) % 251 for char in text]


class SpeculativeSuiteTests(unittest.TestCase):
    def test_measurement_gate_rejects_short_and_impossible_runs(self) -> None:
        short = bench._classify_measurement(128, bench.MIN_DECODE_TIME / 2)
        self.assertIsNone(short["tps"])
        self.assertIn("decode_time", short["rejected_reason"])

        too_fast = bench._classify_measurement(1024, 1.0)
        self.assertIsNone(too_fast["tps"])
        self.assertIn("tps", too_fast["rejected_reason"])

        valid = bench._classify_measurement(100, 1.0)
        self.assertEqual(valid["tps"], 100.0)
        self.assertIsNone(valid["rejected_reason"])

    def test_median_uses_only_valid_runs(self) -> None:
        runs = [
            {"tps": None},
            {"tps": 10.0},
            {"tps": 30.0},
        ]
        self.assertEqual(bench._median_valid_tps(runs), 20.0)
        self.assertIsNone(bench._median_valid_tps([{"tps": None}]))

    def test_drafters_use_pending_token_without_double_feeding_current(self) -> None:
        history = [1, 2, 3, 4, 1, 2]

        lightning = bench._LightningDrafter(ngram_size=3, num_draft_tokens=4)
        lightning.feed_many(history)
        self.assertEqual(lightning.predict_next(3), [4, 1, 2])

        suffix = bench._RapidMLXSuffixDrafter(
            max_draft_tokens=4,
            max_suffix_len=3,
            min_confidence=0.3,
        )
        suffix.feed_many(history)
        self.assertEqual(suffix.predict_next(3), [4, 1, 2])

    def test_build_cases_defaults_to_rapid_workloads(self) -> None:
        cases = bench.build_cases(
            prompt_mode="rapid",
            tokenizer=FakeTokenizer(),
            vocab_size=1000,
            prompt_lengths=[128],
        )
        self.assertEqual([case["id"] for case in cases], ["chat", "json_array", "tool_loop", "code_edit"])
        self.assertTrue(all(case["prompt_tokens"] > 0 for case in cases))

    def test_random_cases_keep_requested_prompt_lengths(self) -> None:
        cases = bench.build_cases(
            prompt_mode="random",
            tokenizer=FakeTokenizer(),
            vocab_size=1000,
            prompt_lengths=[4, 8],
        )
        self.assertEqual([case["prompt_tokens"] for case in cases], [4, 8])
        self.assertEqual([len(case["prompt_ids"]) for case in cases], [4, 8])


if __name__ == "__main__":
    unittest.main()
