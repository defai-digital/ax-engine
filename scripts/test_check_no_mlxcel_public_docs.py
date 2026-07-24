#!/usr/bin/env python3
"""Unit tests for check_no_mlxcel_public_docs.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("check_no_mlxcel_public_docs.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_no_mlxcel_public_docs", MODULE_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


class NoMlxcelPublicDocsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_clean_docs_pass(self) -> None:
        self.write("docs/PERFORMANCE.md", "AX Engine performance\n")
        checker.check_no_mlxcel_public_docs(self.root)

    def test_content_reference_fails_case_insensitively(self) -> None:
        self.write("docs/performance/status.md", "Compare against MLXCEL.\n")
        with self.assertRaisesRegex(
            checker.PublicDocsReferenceError,
            r"docs/performance/status\.md:1",
        ):
            checker.check_no_mlxcel_public_docs(self.root)

    def test_filename_reference_fails(self) -> None:
        self.write("docs/performance/mlxcel-status.md", "Internal results\n")
        with self.assertRaisesRegex(
            checker.PublicDocsReferenceError,
            r"docs/performance/mlxcel-status\.md: filename",
        ):
            checker.check_no_mlxcel_public_docs(self.root)

    def test_non_markdown_and_non_docs_files_are_out_of_scope(self) -> None:
        self.write("docs/performance/mlxcel-results.json", '{"peer": "mlxcel"}\n')
        self.write(".internal/reports/mlxcel-status.md", "MLXcel evidence\n")
        checker.check_no_mlxcel_public_docs(self.root)


if __name__ == "__main__":
    unittest.main()
