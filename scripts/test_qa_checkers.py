#!/usr/bin/env python3
"""Unit tests for QA quality checkers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

from checkers import (  # noqa: E402
    check_regex,
    check_unicode_replacement,
    run_all_checks,
)
from prompts import QaPrompt  # noqa: E402
from reporter import build_results_payload, generate_json_report  # noqa: E402


class QaCheckerTests(unittest.TestCase):
    def test_unicode_replacement_checker_passes_clean_text(self) -> None:
        result = check_unicode_replacement("Photosynthesis turns light into food.")

        self.assertTrue(result.passed)
        self.assertEqual(result.detail, "none")
        self.assertTrue(result.hard)

    def test_unicode_replacement_checker_fails_corrupt_text(self) -> None:
        result = check_unicode_replacement("### \ufffd\ufffd\ufffd The Recipe for Plant Food")

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
            "Photosynthesis turns sunlight, water, and carbon dioxide into food. \ufffd",
            prompt,
        )

        self.assertFalse(report.auto_pass)
        checks = {check.name: check for check in report.checks}
        self.assertIn("unicode_replacement", checks)
        self.assertFalse(checks["unicode_replacement"].passed)

    def test_regex_requires_all_patterns(self) -> None:
        prompt = QaPrompt(
            id="t",
            category="code",
            system=None,
            user="?",
            regex_patterns=[r"SELECT", r"GROUP\s+BY", r"ORDER\s+BY"],
            min_length=1,
        )
        partial = check_regex("SELECT * FROM t GROUP BY id", prompt)
        self.assertFalse(partial.passed)
        full = check_regex("SELECT * FROM t GROUP BY id ORDER BY amount", prompt)
        self.assertTrue(full.passed)


class JsonReportTests(unittest.TestCase):
    def test_json_report_schema(self) -> None:
        from checkers import QualityReport, CheckResult
        from client import QaResponse

        report = QualityReport(
            prompt_id="math_x",
            checks=[
                CheckResult("length", True, "ok", 1.0, hard=True),
                CheckResult("keywords", False, "missing", 0.0, hard=False),
            ],
            auto_pass=True,
            output_preview="42",
        )
        results = [
            {
                "prompt_id": "math_x",
                "category": "math",
                "mode": "direct",
                "stream": False,
                "response": QaResponse(text="42", finish_reason="stop", elapsed_ms=12.0),
                "report": report,
            }
        ]
        meta = {
            "title": "t",
            "version": "v",
            "commit": "abc",
            "seed": 1,
            "bank_size": 10,
            "sample_size": 1,
            "sampled_ids": ["math_x"],
            "model": "m",
            "base_url": "http://127.0.0.1:1",
            "mode_label": "direct",
        }
        payload = build_results_payload(results, meta)
        self.assertEqual(payload["schema_version"], 1)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["totals"]["hard_passed"], 1)
        self.assertEqual(payload["totals"]["soft_failed_checks"], 1)
        raw = generate_json_report(results, meta)
        parsed = json.loads(raw)
        self.assertEqual(parsed["results"][0]["prompt_id"], "math_x")

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.json"
            path.write_text(raw)
            self.assertTrue(path.stat().st_size > 50)


if __name__ == "__main__":
    unittest.main()
