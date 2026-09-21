#!/usr/bin/env python3
"""Tests for scripts/render_mtp_peer_decode_chart.py.

Guards that the committed three-way MTP decode chart (plus the direct-AR
baseline bar) stays byte-identical to the published qualification-SKU
artifact, so the README figure cannot drift.
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO_ROOT / "scripts" / "render_mtp_peer_decode_chart.py"
    spec = importlib.util.spec_from_file_location("render_mtp_peer_decode_chart", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module


mod = _load_module()


class MtpPeerDecodeChartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = json.loads(mod.SUMMARY.read_text(encoding="utf-8"))
        self.peers = mod.load_peers(self.summary)
        self.reference = mod.load_reference(self.summary)

    def test_loads_three_mtp_peers_in_order(self) -> None:
        self.assertEqual([peer.name for peer in self.peers], ["AX Engine", "MTPLX", "OMLX"])

    def test_values_match_artifact(self) -> None:
        by_name = {peer.name: peer for peer in self.peers}
        self.assertEqual(by_name["AX Engine"].decode_tok_s, 31.05)
        self.assertEqual(by_name["MTPLX"].decode_tok_s, 28.16)
        self.assertEqual(by_name["OMLX"].decode_tok_s, 15.02)
        self.assertIn("depth 1", by_name["OMLX"].config)

    def test_reference_is_direct_ar_baseline(self) -> None:
        self.assertIsNotNone(self.reference)
        assert self.reference is not None
        self.assertEqual(self.reference.name, "mlx-lm")
        self.assertEqual(self.reference.decode_tok_s, 12.78)
        self.assertIn("no MTP", self.reference.config)

    def test_render_is_deterministic(self) -> None:
        svg = mod.render_svg(self.peers, self.reference)
        self.assertEqual(svg, mod.render_svg(self.peers, self.reference))
        for color in ("#2eaf5f", "#f2b705", "#3a7bd5", mod.REFERENCE_COLOR):
            self.assertEqual(svg.count(color), 1)
        self.assertIn("31.05", svg)
        self.assertIn("12.78", svg)
        self.assertIn("decode tokens/s", svg)

    def test_load_peers_rejects_missing_engine(self) -> None:
        broken = {"measured": {"ax_engine": {"decode_tok_s_median_20": 31.05}}}
        with self.assertRaises(SystemExit):
            mod.load_peers(broken)

    def test_committed_svg_matches_artifact(self) -> None:
        expected = mod.render_svg(self.peers, self.reference)
        committed = Path(mod.OUTPUT_SVG).read_text(encoding="utf-8")
        self.assertEqual(
            committed,
            expected,
            "docs/assets/perf-mtp-peer-decode-m4-pro-2026-09-17.svg is stale; "
            "run scripts/render_mtp_peer_decode_chart.py",
        )




class MtpPeerDecodeChartValidationTests(unittest.TestCase):
    def _summary(self, value):
        measured = {
            key: {"decode_tok_s_median_20": 50.0, "version": "x"} for key in mod.PEERS
        }
        first = next(iter(mod.PEERS))
        measured[first]["decode_tok_s_median_20"] = value
        return {"measured": measured}

    def test_boolean_is_not_a_decode_median(self) -> None:
        with self.assertRaises(SystemExit):
            mod.load_peers(self._summary(True))

    def test_non_finite_is_not_a_decode_median(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.assertRaises(SystemExit):
                mod.load_peers(self._summary(value))

    def test_reference_rejects_boolean_and_non_finite(self) -> None:
        for value in (True, float("nan")):
            self.assertIsNone(
                mod.load_reference({"measured": {"mlx_lm": {"decode_tok_s_median_20": value}}})
            )

    def test_display_path_outside_repo_does_not_raise(self) -> None:
        self.assertEqual(mod._display_path(Path("/nowhere/out.svg")), "/nowhere/out.svg")

if __name__ == "__main__":
    unittest.main()
