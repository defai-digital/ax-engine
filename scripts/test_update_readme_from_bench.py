#!/usr/bin/env python3
"""Unit tests for scripts.update_readme_from_bench."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("update_readme_from_bench.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "update_readme_from_bench", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
updater = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = updater
MODULE_SPEC.loader.exec_module(updater)


class UpdateReadmeFromBenchTests(unittest.TestCase):
    def test_prefill_cell_bolds_faster_ax_rows_only(self) -> None:
        self.assertEqual(
            updater.cell_prefill(120.0, 100.0),
            "**120.0 (+20.0%)**",
        )
        self.assertEqual(
            updater.cell_prefill(80.0, 100.0),
            "80.0 (-20.0%)",
        )


if __name__ == "__main__":
    unittest.main()
