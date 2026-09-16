#!/usr/bin/env python3
"""Unit tests for check_qwen38_primary_claims.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("check_qwen38_primary_claims.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_qwen38_primary_claims", MODULE_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)

STATUS = checker.STATUS_SENTENCE
FLASH_NEXT_STATUS = checker.FLASH_NEXT_STATUS_SENTENCE
BODY = (
    f"{STATUS}\n"
    f"{FLASH_NEXT_STATUS}\n"
    "ax-engine serve qwen3.8-27b:axq\n"
    "qwen3.8-flash-next:axq\n"
    "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP "
    "3e290738e96972307c6aeb9934ab170ca0eae1c1\n"
)


class CheckQwen38PrimaryClaimsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def seed_required(self) -> None:
        for relative in (*checker.STATUS_FILES, *checker.FLASH_NEXT_STATUS_FILES):
            self.write(relative, BODY)

    def test_clean_docs_pass(self) -> None:
        self.seed_required()
        checker.check_qwen38_primary_claims(self.root)

    def test_missing_status_sentence_fails(self) -> None:
        self.seed_required()
        self.write("README.md", "ax-engine serve qwen3.8-27b:axq\n")
        with self.assertRaisesRegex(
            checker.PrimaryClaimError, "canonical Qwen 3.8 27B status sentence"
        ):
            checker.check_qwen38_primary_claims(self.root)

    def test_host_alias_in_prose_fails(self) -> None:
        self.seed_required()
        self.write("docs/PERFORMANCE.md", BODY + "Host `df-macbookpro-m5`.\n")
        with self.assertRaisesRegex(
            checker.PrimaryClaimError, r"docs/PERFORMANCE\.md:.*SSH host alias"
        ):
            checker.check_qwen38_primary_claims(self.root)

    def test_historical_benchmark_path_is_allowed(self) -> None:
        self.seed_required()
        self.write(
            "docs/PERFORMANCE.md",
            BODY
            + "[campaign](../benchmarks/results/mtp-axq-peer/"
            "2026-08-31-df-macbookpro-m5/)\n",
        )
        checker.check_qwen38_primary_claims(self.root)

    def test_unearned_tier2_claim_fails(self) -> None:
        self.seed_required()
        self.write(
            "docs/FAQ.md",
            "Qwen 3.8 27B is MTP Tier 2 certified.\n",
        )
        with self.assertRaisesRegex(
            checker.PrimaryClaimError, "unearned Qwen 3.8 MTP Tier 2"
        ):
            checker.check_qwen38_primary_claims(self.root)

    def test_missing_flash_next_status_sentence_fails(self) -> None:
        self.seed_required()
        self.write(
            "docs/model-certifications/qwen3.8-flash-next.md",
            "qwen3.8-flash-next:axq\n",
        )
        with self.assertRaisesRegex(
            checker.PrimaryClaimError, "canonical Qwen 3.8 Flash Next status sentence"
        ):
            checker.check_qwen38_primary_claims(self.root)


if __name__ == "__main__":
    unittest.main()
