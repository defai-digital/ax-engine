#!/usr/bin/env python3
"""Unit tests for README performance chart helpers."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

CHART_SCRIPT_PATH = Path(__file__).with_name("render_readme_performance_charts.py")
CHART_MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_readme_performance_charts", CHART_SCRIPT_PATH
)
assert CHART_MODULE_SPEC and CHART_MODULE_SPEC.loader
charts = importlib.util.module_from_spec(CHART_MODULE_SPEC)
sys.modules[CHART_MODULE_SPEC.name] = charts
CHART_MODULE_SPEC.loader.exec_module(charts)

MTP_SCRIPT_PATH = Path(__file__).with_name("bench_mtp_6bit_ax_refresh.py")
MTP_MODULE_SPEC = importlib.util.spec_from_file_location(
    "bench_mtp_6bit_ax_refresh", MTP_SCRIPT_PATH
)
assert MTP_MODULE_SPEC and MTP_MODULE_SPEC.loader
mtp_refresh = importlib.util.module_from_spec(MTP_MODULE_SPEC)
sys.modules[MTP_MODULE_SPEC.name] = mtp_refresh
MTP_MODULE_SPEC.loader.exec_module(mtp_refresh)


class ReadmePerformanceChartTests(unittest.TestCase):
    def test_mtp_6bit_summary_accepts_speculative_results_tree(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            summary_path = (
                root
                / "benchmarks/results/speculative/mtp-6bit/local-run/summary.json"
            )
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text('{"rows": []}\n')
            readme = root / "README.md"
            readme.write_text(
                "[summary](benchmarks/results/speculative/mtp-6bit/local-run/summary.json)\n"
            )

            self.assertEqual(charts.find_mtp_6bit_summary(readme), summary_path)

    def test_mtp_6bit_refresh_defaults_to_speculative_results_tree(self) -> None:
        self.assertTrue(
            mtp_refresh.DEFAULT_OUTPUT_BASE.as_posix().endswith(
                "/benchmarks/results/speculative/mtp-6bit"
            )
        )

    def test_embedding_scale_charts_use_embedding_results_tree(self) -> None:
        self.assertIn(
            "benchmarks/results/embedding/embedding-scale/",
            charts.EMBEDDING_SCALE_REFERENCE_ARTIFACT.as_posix(),
        )
        self.assertIn(
            "benchmarks/results/embedding/embedding-scale/",
            charts.EMBEDDING_SCALE_AX_ARTIFACT.as_posix(),
        )
        self.assertIn(
            "benchmarks/results/embedding/embedding-scale/",
            charts.EMBEDDINGGEMMA_SCALE_REFERENCE_ARTIFACT.as_posix(),
        )
        self.assertIn(
            "benchmarks/results/embedding/embedding-scale/",
            charts.EMBEDDINGGEMMA_SCALE_AX_ARTIFACT.as_posix(),
        )


if __name__ == "__main__":
    unittest.main()
