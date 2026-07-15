#!/usr/bin/env python3
"""Tests for Gemma 4 assistant-MTP chart rendering."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("render_gemma4_assistant_mtp_charts.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_gemma4_assistant_mtp_charts", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
charts = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = charts
MODULE_SPEC.loader.exec_module(charts)


class Gemma4AssistantMtpChartTests(unittest.TestCase):
    def render(self, *, lower_is_better: bool) -> str:
        groups = [
            {
                "label": "flappy",
                "values": {
                    charts.ENGINE_DIRECT: [10.0, 10.0, 10.0],
                    charts.ENGINE_MTP: [20.0, 20.0, 20.0],
                },
            },
            {
                "label": "long_code",
                "values": {
                    charts.ENGINE_DIRECT: [30.0, 30.0, 30.0],
                    charts.ENGINE_MTP: [25.0, 25.0, 25.0],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "chart.svg"
            charts.write_box_whisker_svg(
                output,
                title="Gemma chart",
                subtitle="test",
                engines=[charts.ENGINE_DIRECT, charts.ENGINE_MTP],
                unit="tok/s",
                direction_label=(
                    "Lower is better" if lower_is_better else "Higher is better"
                ),
                groups=groups,
                lower_is_better=lower_is_better,
            )
            return output.read_text()

    def test_higher_is_better_marks_each_group_winner_red_without_reference_line(
        self,
    ) -> None:
        svg = self.render(lower_is_better=False)

        self.assertNotIn("stroke-dasharray", svg)
        self.assertNotIn("data-label=", svg)
        self.assertIn('fill="#dc2626" stroke="#ffffff"', svg)
        self.assertIn(">20.0</text>", svg)
        self.assertIn(">30.0</text>", svg)

    def test_lower_is_better_marks_each_group_winner_red(self) -> None:
        svg = self.render(lower_is_better=True)

        self.assertIn("the lowest median value label is red", svg)
        self.assertIn(">10.0</text>", svg)
        self.assertIn(">25.0</text>", svg)


if __name__ == "__main__":
    unittest.main()
