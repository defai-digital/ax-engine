#!/usr/bin/env python3
"""Unit tests for qa/generate_summary.py.

No companion test file previously existed; scope limited to the specific
bug fixed here rather than full coverage of the script.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "qa" / "generate_summary.py"
MODULE_SPEC = importlib.util.spec_from_file_location("qa_generate_summary", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


def _report(model_id: str, mode: str, passed: int, failed: int) -> dict:
    return {
        "model_id": model_id,
        "mode": mode,
        "timestamp": "20260101-000000",
        "pass_rate": f"{passed / (passed + failed) * 100:.1f}%",
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "filename": f"qa-{model_id}-{mode}-20260101-000000.html",
    }


class GenerateSummaryParityLabelTests(unittest.TestCase):
    def test_matching_pass_count_is_not_claimed_as_identical(self) -> None:
        # Regression test: parse_report_info only has aggregate pass/fail
        # counts (scraped from each report's HTML), not per-prompt-id
        # results. Two runs with the same pass count can still have failed
        # completely disjoint sets of prompts, so labeling them "Identical"
        # falsely implies behavioral parity the data can't support.
        reports = [
            _report("model-a", "direct", passed=18, failed=2),
            _report("model-a", "ngram", passed=18, failed=2),
        ]

        html = mod.generate_summary_html(reports)

        self.assertNotIn("Identical", html)
        self.assertIn("Same pass count", html)

    def test_mismatched_pass_count_is_labeled_different(self) -> None:
        reports = [
            _report("model-a", "direct", passed=18, failed=2),
            _report("model-a", "ngram", passed=15, failed=5),
        ]

        html = mod.generate_summary_html(reports)

        self.assertNotIn("Differs", html)
        self.assertIn("Different pass count", html)


if __name__ == "__main__":
    unittest.main()
