#!/usr/bin/env python3
"""Tests for scripts/render_m5_peer_2026_09_21.py.

Guards that the committed chart stays identical to the checked-in campaign
summary.json artifacts, that unsupported lanes are omitted (never drawn as
zero), and that rendering is deterministic.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_chart():
    path = REPO_ROOT / "scripts" / "render_m5_peer_2026_09_21.py"
    spec = importlib.util.spec_from_file_location("render_m5_peer_2026_09_21", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


chart = _load_chart()


class M5PeerChartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dense, self.moe, self.footnotes = chart.collect_groups()

    def _bars(self, group):
        return dict(group[2])

    def test_dense_panel_has_four_groups(self) -> None:
        self.assertEqual(len(self.dense), 4)
        self.assertEqual(len(self.moe), 2)

    def test_published_medians_match_artifacts(self) -> None:
        flappy = self._bars(self.dense[0])
        self.assertEqual(flappy["ax_engine"], 76.04)
        self.assertEqual(flappy["mtplx"], 73.07)
        self.assertEqual(flappy["omlx"], 37.71)
        self.assertEqual(flappy["mlx_lm"], 27.92)
        moe_pack = self._bars(self.moe[0])
        self.assertEqual(moe_pack["ax_engine"], 239.52)
        self.assertEqual(moe_pack["mtplx"], 127.87)
        g12 = self._bars(self.dense[3])
        self.assertEqual(g12["ax_engine"], 67.68)
        self.assertEqual(g12["omlx"], 60.05)

    def test_unsupported_lanes_are_omitted_not_zero(self) -> None:
        moe_pack = self._bars(self.moe[0])
        self.assertNotIn("omlx", moe_pack)
        g12 = self._bars(self.dense[3])
        self.assertNotIn("mtplx", g12)
        self.assertNotIn("mlx_lm", g12)
        svg = chart.render_svg()
        self.assertNotIn(">0.00</text>", svg)

    def test_render_is_deterministic(self) -> None:
        self.assertEqual(chart.render_svg(), chart.render_svg())

    def test_committed_svg_matches_render(self) -> None:
        committed = chart.OUTPUT_SVG.read_text(encoding="utf-8")
        self.assertEqual(committed, chart.render_svg())

    def test_footnotes_present(self) -> None:
        svg = chart.render_svg()
        for note in self.footnotes:
            self.assertIn(note.split(" (")[0], svg)


if __name__ == "__main__":
    unittest.main()
