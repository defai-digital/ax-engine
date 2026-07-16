#!/usr/bin/env python3
"""Unit tests for QA question-bank sampling and exact-answer checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

from checkers import check_exact_answer, run_all_checks  # noqa: E402
from prompts import (  # noqa: E402
    DEFAULT_SAMPLE_SIZE,
    QUESTION_BANK,
    QaPrompt,
    bank_size,
    sample_prompts,
    validate_bank,
)


class QaSamplingTests(unittest.TestCase):
    def test_bank_is_larger_than_default_sample(self) -> None:
        self.assertGreater(bank_size(), DEFAULT_SAMPLE_SIZE)
        self.assertGreaterEqual(bank_size(), 48)

    def test_sample_is_deterministic_with_seed(self) -> None:
        a, seed_a = sample_prompts(n=12, seed=42)
        b, seed_b = sample_prompts(n=12, seed=42)
        self.assertEqual(seed_a, 42)
        self.assertEqual(seed_b, 42)
        self.assertEqual([p.id for p in a], [p.id for p in b])

    def test_different_seeds_usually_differ(self) -> None:
        a, _ = sample_prompts(n=12, seed=1)
        b, _ = sample_prompts(n=12, seed=2)
        self.assertNotEqual([p.id for p in a], [p.id for p in b])

    def test_stratified_covers_multiple_categories(self) -> None:
        selected, _ = sample_prompts(n=12, seed=7, stratified=True)
        cats = {p.category for p in selected}
        self.assertGreaterEqual(len(cats), 6)

    def test_explicit_ids(self) -> None:
        selected, _ = sample_prompts(
            prompt_ids=["math_percent_discount", "knowledge_capital_france"]
        )
        self.assertEqual(
            [p.id for p in selected],
            ["math_percent_discount", "knowledge_capital_france"],
        )

    def test_unique_prompt_ids_in_bank(self) -> None:
        ids = [p.id for p in QUESTION_BANK]
        self.assertEqual(len(ids), len(set(ids)), "duplicate prompt ids in bank")

    def test_bank_validates_clean(self) -> None:
        errors = validate_bank()
        self.assertEqual(errors, [], msg="\n".join(errors))


class ExactAnswerTests(unittest.TestCase):
    def test_exact_answer_passes_substring_for_long(self) -> None:
        prompt = QaPrompt(
            id="t",
            category="math",
            system=None,
            user="?",
            exact_answer="42",
        )
        result = check_exact_answer("The answer is 42.", prompt)
        self.assertTrue(result.passed)

    def test_exact_answer_normalizes_nbsp(self) -> None:
        """gpt-oss-20b emitted NBSP between tokens for instruction_three_colors."""
        prompt = QaPrompt(
            id="t",
            category="instruction",
            system=None,
            user="?",
            exact_answer="red green blue",
            min_length=1,
        )
        self.assertTrue(check_exact_answer("red green\xa0blue", prompt).passed)

    def test_short_token_does_not_match_inside_words(self) -> None:
        prompt = QaPrompt(
            id="t",
            category="knowledge",
            system=None,
            user="?",
            exact_answer="no",
            exact_match="token",
        )
        self.assertFalse(check_exact_answer("I notice something.", prompt).passed)
        self.assertTrue(check_exact_answer("No, we cannot conclude that.", prompt).passed)

    def test_full_match_rejects_prose(self) -> None:
        prompt = QaPrompt(
            id="t",
            category="knowledge",
            system=None,
            user="?",
            exact_answer="a",
            exact_match="full",
        )
        self.assertFalse(check_exact_answer("The tag is a", prompt).passed)
        self.assertTrue(check_exact_answer("a", prompt).passed)

    def test_case_sensitive_match(self) -> None:
        prompt = QaPrompt(
            id="t",
            category="instruction",
            system=None,
            user="?",
            exact_answer="YES",
            exact_match="case",
        )
        self.assertFalse(check_exact_answer("yes", prompt).passed)
        self.assertTrue(check_exact_answer("YES", prompt).passed)

    def test_exact_answer_aliases(self) -> None:
        prompt = QaPrompt(
            id="t",
            category="science",
            system=None,
            user="?",
            exact_answer="10",
            exact_answer_aliases=["9.8", "9.81"],
        )
        self.assertTrue(check_exact_answer("about 9.81 m/s^2", prompt).passed)

    def test_exact_answer_fails_when_missing(self) -> None:
        prompt = QaPrompt(
            id="t",
            category="knowledge",
            system=None,
            user="?",
            exact_answer="Paris",
        )
        self.assertFalse(check_exact_answer("London", prompt).passed)

    def test_keywords_are_soft_and_do_not_fail_hard_pass(self) -> None:
        prompt = QaPrompt(
            id="t",
            category="creative",
            system=None,
            user="?",
            keywords=["robot", "music", "sound"],
            min_length=10,
        )
        report = run_all_checks(
            "A sunny afternoon by the lake with calm weather nearby.", prompt
        )
        soft = {c.name: c for c in report.checks if not c.hard}
        self.assertIn("keywords", soft)
        self.assertFalse(soft["keywords"].passed)
        self.assertTrue(report.auto_pass, "hard pass should ignore soft keyword misses")

    def test_run_all_checks_includes_exact_answer(self) -> None:
        prompt = QaPrompt(
            id="t",
            category="math",
            system=None,
            user="?",
            exact_answer="8",
            min_length=1,
        )
        report = run_all_checks("The final answer is 8.", prompt)
        names = {c.name for c in report.checks}
        self.assertIn("exact_answer", names)
        self.assertTrue(report.auto_pass)


if __name__ == "__main__":
    unittest.main()
