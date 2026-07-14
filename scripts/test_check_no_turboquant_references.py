#!/usr/bin/env python3
"""Unit tests for check_no_turboquant_references.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("check_no_turboquant_references.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_no_turboquant_references", MODULE_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


class NoTurboquantReferencesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_clean_tree_passes(self) -> None:
        self.write("docs/KV-CACHE.md", "durable tiered prefix cache\n")
        checker.check_no_turboquant_references(self.root)

    def test_content_hit_outside_allowlist_fails(self) -> None:
        self.write("docs/KV-CACHE.md", "enable TurboQuant fused decode\n")
        with self.assertRaisesRegex(
            checker.NoTurboquantReferencesError, r"docs/KV-CACHE\.md:1"
        ):
            checker.check_no_turboquant_references(self.root)

    def test_path_name_hit_outside_allowlist_fails(self) -> None:
        self.write("scripts/build_turboquant_thing.py", "print('x')\n")
        with self.assertRaisesRegex(
            checker.NoTurboquantReferencesError,
            r"scripts/build_turboquant_thing\.py: path name",
        ):
            checker.check_no_turboquant_references(self.root)

    def test_matching_is_case_insensitive(self) -> None:
        self.write("README.md", "TURBOQUANT was fast\n")
        with self.assertRaisesRegex(
            checker.NoTurboquantReferencesError, r"README\.md:1"
        ):
            checker.check_no_turboquant_references(self.root)

    def test_allowlisted_locations_are_ignored(self) -> None:
        self.write(
            "benchmarks/results/profiling/turboquant/README.md",
            "historical TurboQuant artifact\n",
        )
        self.write("docs/designs/turboquant-fused-decode.md", "TurboQuant design\n")
        self.write(".internal/notes.md", "turboquant planning\n")
        self.write("CHANGELOG.md", "Retired the TurboQuant runtime path (ADR-002).\n")
        checker.check_no_turboquant_references(self.root)

    def test_checker_and_test_are_allowlisted(self) -> None:
        self.assertTrue(
            checker.is_allowed("scripts/check_no_turboquant_references.py")
        )
        self.assertTrue(
            checker.is_allowed("scripts/test_check_no_turboquant_references.py")
        )

    def test_binary_content_does_not_crash(self) -> None:
        path = self.root / "assets" / "blob.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00\xffTurboQuant\x00")
        with self.assertRaisesRegex(
            checker.NoTurboquantReferencesError, r"assets/blob\.bin:1"
        ):
            checker.check_no_turboquant_references(self.root)


if __name__ == "__main__":
    unittest.main()
