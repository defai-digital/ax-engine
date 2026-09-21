#!/usr/bin/env python3
"""Tests for scripts/render_tiel_peer_chart.py.

Guards that the committed chart stays byte-identical to the published
four-machine tables, so the README figure cannot drift from the report.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_chart():
    path = REPO_ROOT / "scripts" / "render_tiel_peer_chart.py"
    spec = importlib.util.spec_from_file_location("render_tiel_peer_chart", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


chart = _load_chart()


class TielPeerChartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cells = chart.parse_completion_table(chart.SOURCE_DOC.read_text(encoding="utf-8"))

    def test_reads_all_sixteen_cells(self) -> None:
        self.assertEqual(len(self.cells), 16)
        self.assertEqual({cell.machine for cell in self.cells}, set(chart.MACHINE_LABELS))

    def test_published_cell_matches_report(self) -> None:
        m5_tiel_lru = next(
            cell
            for cell in self.cells
            if cell.machine.startswith("M5")
            and cell.model == "Tiel"
            and cell.workload == "python-lru"
        )
        self.assertEqual(m5_tiel_lru.ax_tok_s, 194.88)
        self.assertEqual(m5_tiel_lru.mtplx_tok_s, 177.44)
        self.assertEqual(m5_tiel_lru.delta_pct, 9.8)

    def test_render_is_deterministic(self) -> None:
        first = chart.render_svg(self.cells)
        second = chart.render_svg(self.cells)
        self.assertEqual(first, second)
        # One bar per cell plus one legend swatch per engine.
        self.assertEqual(first.count(chart.AX_COLOR), len(self.cells) + 1)
        self.assertEqual(first.count(chart.MTPLX_COLOR), len(self.cells) + 1)

    def test_committed_svg_matches_source_tables(self) -> None:
        expected = chart.render_svg(self.cells)
        committed = Path(chart.OUTPUT_SVG).read_text(encoding="utf-8")
        self.assertEqual(
            committed,
            expected,
            "docs/assets/perf-tiel-vs-mtplx-2026-09-20.svg is stale; "
            "run scripts/render_tiel_peer_chart.py",
        )


if __name__ == "__main__":
    unittest.main()
