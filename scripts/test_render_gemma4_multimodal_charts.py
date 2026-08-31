#!/usr/bin/env python3
"""Unit tests for render_gemma4_multimodal_charts.py.

No companion test file previously existed; scope limited to the specific
bug fixed here rather than full coverage of the script.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("render_gemma4_multimodal_charts.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_gemma4_multimodal_charts", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


def _ax_row(case_id: str, engine: str, client_wall_ms_median: float) -> dict:
    return {
        "case_id": case_id,
        "status": "measured",
        "engine": engine,
        "layer": "openai_chat_e2e",
        "client_wall_ms": {"median": client_wall_ms_median},
    }


class AxOpenaiChatE2eRowsByCaseTests(unittest.TestCase):
    def test_duplicate_case_id_fails_closed(self) -> None:
        # Regression test: two ax_engine(_mlx) openai_chat_e2e rows sharing
        # the same case_id (e.g. one under the old "ax_engine" name and one
        # under "ax_engine_mlx" during a migration) used to be silently
        # collapsed by a dict comprehension, discarding whichever
        # measurement came first with no error or trace in the rendered
        # chart.
        rows = [
            _ax_row("image_single", "ax_engine", 800.0),
            _ax_row("image_single", "ax_engine_mlx", 1200.0),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate openai_chat_e2e ax row"):
            mod._ax_openai_chat_e2e_rows_by_case(rows)

    def test_unique_case_ids_are_indexed(self) -> None:
        rows = [
            _ax_row("image_single", "ax_engine_mlx", 800.0),
            _ax_row("audio_only", "ax_engine_mlx", 500.0),
        ]
        indexed = mod._ax_openai_chat_e2e_rows_by_case(rows)
        self.assertEqual(set(indexed), {"image_single", "audio_only"})


if __name__ == "__main__":
    unittest.main()
