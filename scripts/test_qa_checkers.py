#!/usr/bin/env python3
"""Unit tests for QA quality checkers."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

from checkers import check_unicode_replacement, run_all_checks
from prompts import QaPrompt


class QaCheckerTests(unittest.TestCase):
    def test_unicode_replacement_checker_passes_clean_text(self) -> None:
        result = check_unicode_replacement("Photosynthesis turns light into food.")

        self.assertTrue(result.passed)
        self.assertEqual(result.detail, "none")

    def test_unicode_replacement_checker_fails_corrupt_text(self) -> None:
        result = check_unicode_replacement("### ��� The Recipe for Plant Food")

        self.assertFalse(result.passed)
        self.assertIn("3 replacement chars", result.detail)

    def test_run_all_checks_fails_visible_replacement_chars(self) -> None:
        prompt = QaPrompt(
            id="science_photosynthesis",
            category="science",
            system=None,
            user="Explain photosynthesis in simple terms.",
            min_length=20,
        )

        report = run_all_checks(
            "Photosynthesis turns sunlight, water, and carbon dioxide into food. �",
            prompt,
        )

        self.assertFalse(report.auto_pass)
        checks = {check.name: check for check in report.checks}
        self.assertIn("unicode_replacement", checks)
        self.assertFalse(checks["unicode_replacement"].passed)


if __name__ == "__main__":
    unittest.main()
