#!/usr/bin/env python3
"""Unit tests for scripts.bench_ax_only_sweep."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("bench_ax_only_sweep.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_ax_only_sweep", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
sweep = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bench_ax_only_sweep"] = sweep
MODULE_SPEC.loader.exec_module(sweep)


class BenchAxOnlySweepTests(unittest.TestCase):
    def test_filter_manifest_rows_returns_all_rows_without_filter(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}]

        self.assertEqual(sweep.filter_manifest_rows(rows, None), rows)

    def test_filter_manifest_rows_keeps_manifest_order(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}, {"slug": "c"}]

        selected = sweep.filter_manifest_rows(rows, ["c", "a"])

        self.assertEqual([row["slug"] for row in selected], ["a", "c"])

    def test_filter_manifest_rows_rejects_unknown_filter_slug(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}]

        with self.assertRaisesRegex(
            sweep.AxOnlySweepError,
            "--rows-filter references unknown slug",
        ):
            sweep.filter_manifest_rows(rows, ["a", "missing"])

    def test_filter_manifest_rows_rejects_duplicate_manifest_slug(self) -> None:
        rows = [{"slug": "a"}, {"slug": "a"}]

        with self.assertRaisesRegex(
            sweep.AxOnlySweepError,
            "manifest contains duplicate slug",
        ):
            sweep.filter_manifest_rows(rows, None)

    def test_filter_manifest_rows_rejects_duplicate_filter_slug(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}]

        with self.assertRaisesRegex(
            sweep.AxOnlySweepError,
            "--rows-filter contains duplicate slug",
        ):
            sweep.filter_manifest_rows(rows, ["a", "a"])

    def test_filter_manifest_rows_rejects_empty_filter(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}]

        with self.assertRaisesRegex(sweep.AxOnlySweepError, "requires at least one"):
            sweep.filter_manifest_rows(rows, [])

    def test_filter_manifest_rows_rejects_empty_manifest(self) -> None:
        with self.assertRaisesRegex(sweep.AxOnlySweepError, "manifest contains no rows"):
            sweep.filter_manifest_rows([], None)

    def test_filter_manifest_rows_rejects_missing_row_slug(self) -> None:
        with self.assertRaisesRegex(sweep.AxOnlySweepError, "non-empty slug"):
            sweep.filter_manifest_rows([{"readme_model": "Gemma"}], None)

    def test_filter_manifest_rows_rejects_non_object_row(self) -> None:
        with self.assertRaisesRegex(sweep.AxOnlySweepError, "row must be an object"):
            sweep.filter_manifest_rows(["not-a-row"], None)

    def test_main_rejects_unknown_filter_before_creating_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"rows": [{"slug": "a"}]}) + "\n")
            out_dir = root / "out"

            argv = [
                "bench_ax_only_sweep.py",
                "--manifest",
                str(manifest),
                "--output-root",
                str(out_dir),
                "--reuse-reference-root",
                str(root / "reference"),
                "--rows-filter",
                "missing",
            ]
            with patch.object(sys, "argv", argv), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    sweep.main()

            self.assertEqual(caught.exception.code, 1)
            self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
