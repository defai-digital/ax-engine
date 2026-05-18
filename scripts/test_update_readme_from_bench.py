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

    def test_current_qwen36_slug_maps_to_readme_row(self) -> None:
        self.assertEqual(
            updater.SLUG_TO_README["qwen3_6-35b-a3b-4bit"],
            ("Qwen 3.6 35B A3B", "4-bit"),
        )

    def test_decode_direct_only_update_preserves_existing_ngram_column(self) -> None:
        lines = [
            "### Decode throughput (tok/s) — generation=128 tokens, temp=0",
            "",
            "| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax direct baseline | ax default n-gram |",
            "|---|---|---:| ---: |---:|---:|---:|",
            "| Gemma 4 E2B | 4-bit | 128 | 157.5 | 211.2 | 183.1 (-13.3%) | **580.1 (+174.6%)** |",
            "|         |         | 512 | 154.5 | 206.0 | 178.3 (-13.5%) | **576.1 (+179.7%)** |",
            "",
            "### Time to first token",
        ]
        vals = {
            ("mlx_lm", 128): {"decode": 200.0},
            ("ax_engine_mlx", 128): {"decode": 180.0},
            ("mlx_lm", 512): {"decode": 190.0},
            ("ax_engine_mlx", 512): {"decode": 195.0},
        }

        changed = updater.update_decode_rows(lines, "Gemma 4 E2B", "4-bit", vals)

        self.assertEqual(changed, 2)
        self.assertIn("| Gemma 4 E2B | 4-bit | 128 | 157.5 | 200.0 | 180.0 (-10.0%) | **580.1 (+174.6%)** |", lines)
        self.assertIn("|  |  | 512 | 154.5 | 190.0 | **195.0 (+2.6%)** | **576.1 (+179.7%)** |", lines)


if __name__ == "__main__":
    unittest.main()
