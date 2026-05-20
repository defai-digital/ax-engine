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
        self.assertEqual(
            updater.cell_decode_ngram(120.0, 100.0),
            "**120.0 (+20.0%)**",
        )
        self.assertEqual(
            updater.cell_decode_ngram(80.0, 100.0),
            "80.0 (-20.0%)",
        )

    def test_current_qwen36_slug_maps_to_readme_row(self) -> None:
        self.assertEqual(
            updater.SLUG_TO_README["qwen3_6-35b-a3b-4bit"],
            ("Qwen 3.6 35B A3B", "4-bit"),
        )

    def test_extract_bench_values_rejects_duplicate_engine_prompt_rows(self) -> None:
        payload = {
            "results": [
                {
                    "engine": "mlx_lm",
                    "prompt_tokens": 128,
                    "prefill_tok_s": {"median": 100.0},
                    "decode_tok_s": {"median": 10.0},
                    "ttft_ms": 1.0,
                },
                {
                    "engine": "mlx_lm",
                    "prompt_tokens": 128,
                    "prefill_tok_s": {"median": 101.0},
                    "decode_tok_s": {"median": 11.0},
                    "ttft_ms": 1.1,
                },
            ],
        }

        with self.assertRaisesRegex(updater.ReadmeBenchUpdateError, "duplicate benchmark row"):
            updater.extract_bench_values(payload)

    def test_extract_bench_values_preserves_null_medians(self) -> None:
        payload = {
            "results": [
                {
                    "engine": "ax_engine_mlx",
                    "prompt_tokens": 128,
                    "prefill_tok_s": {"median": None},
                    "decode_tok_s": {"median": 148.9},
                    "ttft_ms": {"median": None},
                }
            ],
        }

        values = updater.extract_bench_values(payload)

        self.assertIsNone(values[("ax_engine_mlx", 128)]["prefill"])
        self.assertEqual(values[("ax_engine_mlx", 128)]["decode"], 148.9)
        self.assertIsNone(values[("ax_engine_mlx", 128)]["ttft"])

    def test_prefill_null_ax_value_clears_stale_ax_cell(self) -> None:
        lines = [
            "### Prefill throughput (tok/s) — percentages vs mlx_lm",
            "",
            "| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |",
            "|---|---|---:| ---: |---:|---:|",
            "| Qwen 3.6 35B A3B | 4-bit | 128 | 1,706.9 | 539.4 | **5,000,000.0 (+927000.0%)** |",
            "",
            "### Decode throughput",
        ]
        vals = {
            ("mlx_lm", 128): {"prefill": 539.4},
            ("ax_engine_mlx", 128): {"prefill": None},
        }

        changed = updater.update_prefill_rows(lines, "Qwen 3.6 35B A3B", "4-bit", vals)

        self.assertEqual(changed, 1)
        self.assertIn(
            "| Qwen 3.6 35B A3B | 4-bit | 128 | 1,706.9 | 539.4 | — |",
            lines,
        )

    def test_ttft_null_ax_value_clears_stale_ax_cell(self) -> None:
        lines = [
            "### Time to first token (ms) — generation=128 tokens, temp=0",
            "",
            "| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax engine |",
            "|---|---|---:| ---: |---:|---:|",
            "| Qwen 3.6 35B A3B | 4-bit | 128 | 75.0 | 237.3 | **0.3 (-99.9%)** |",
            "",
            "### Installation",
        ]
        vals = {
            ("mlx_lm", 128): {"ttft": 237.3},
            ("ax_engine_mlx", 128): {"ttft": None},
        }

        changed = updater.update_ttft_rows(lines, "Qwen 3.6 35B A3B", "4-bit", vals)

        self.assertEqual(changed, 1)
        self.assertIn(
            "| Qwen 3.6 35B A3B | 4-bit | 128 | 75.0 | 237.3 | — |",
            lines,
        )

    def test_decode_direct_only_update_preserves_missing_ngram_column(self) -> None:
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
        self.assertIn(
            "| Gemma 4 E2B | 4-bit | 128 | 157.5 | 200.0 | 180.0 (-10.0%) | **580.1 (+174.6%)** |",
            lines,
        )
        self.assertIn(
            "|  |  | 512 | 154.5 | 190.0 | **195.0 (+2.6%)** | **576.1 (+179.7%)** |",
            lines,
        )

    def test_decode_missing_reference_row_leaves_existing_cells_unchanged(self) -> None:
        lines = [
            "### Decode throughput (tok/s) — generation=128 tokens, temp=0",
            "",
            "| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax direct baseline | ax default n-gram |",
            "|---|---|---:| ---: |---:|---:|---:|",
            "| Gemma 4 E2B | 4-bit | 128 | 157.5 | 211.2 | 183.1 (-13.3%) | **580.1 (+174.6%)** |",
            "",
            "### Time to first token",
        ]
        vals = {
            ("ax_engine_mlx", 128): {"decode": 180.0},
            ("ax_engine_mlx_ngram_accel", 128): {"decode": 580.0},
        }

        changed = updater.update_decode_rows(lines, "Gemma 4 E2B", "4-bit", vals)

        self.assertEqual(changed, 0)
        self.assertIn("| Gemma 4 E2B | 4-bit | 128 | 157.5 | 211.2 | 183.1 (-13.3%) | **580.1 (+174.6%)** |", lines)

    def test_decode_null_reference_metric_clears_all_comparison_cells(self) -> None:
        lines = [
            "### Decode throughput (tok/s) — generation=128 tokens, temp=0",
            "",
            "| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax direct baseline | ax default n-gram |",
            "|---|---|---:| ---: |---:|---:|---:|",
            "| Gemma 4 E2B | 4-bit | 128 | 157.5 | 211.2 | 183.1 (-13.3%) | **580.1 (+174.6%)** |",
            "",
            "### Time to first token",
        ]
        vals = {
            ("mlx_lm", 128): {"decode": None},
            ("ax_engine_mlx", 128): {"decode": 180.0},
            ("ax_engine_mlx_ngram_accel", 128): {"decode": 580.0},
        }

        changed = updater.update_decode_rows(lines, "Gemma 4 E2B", "4-bit", vals)

        self.assertEqual(changed, 1)
        self.assertIn("| Gemma 4 E2B | 4-bit | 128 | 157.5 | — | — | — |", lines)

    def test_prompt_specific_overlay_leaves_unmentioned_prompt_rows_unchanged(self) -> None:
        lines = [
            "### Decode throughput (tok/s) — generation=128 tokens, temp=0",
            "",
            "| Model | MLX quantization | Prompt tok | llama.cpp Metal* | mlx_lm | ax direct baseline | ax default n-gram |",
            "|---|---|---:| ---: |---:|---:|---:|",
            "| Qwen 3.6 27B | 4-bit | 128 | 26.0 | 34.0 | 33.0 (-2.9%) | 32.7 (-3.9%) |",
            "|  |  | 512 | 26.0 | 33.9 | 33.0 (-2.6%) | 32.6 (-3.9%) |",
            "|  |  | 2048 | 18.8 | 50.0 | 20.0 (-60.0%) | 20.0 (-60.0%) |",
            "",
            "### Time to first token",
        ]
        vals = {
            ("mlx_lm", 2048): {"decode": 33.4},
            ("ax_engine_mlx", 2048): {"decode": 31.6},
            ("ax_engine_mlx_ngram_accel", 2048): {"decode": 31.1},
        }

        changed = updater.update_decode_rows(lines, "Qwen 3.6 27B", "4-bit", vals)

        self.assertEqual(changed, 1)
        self.assertIn("| Qwen 3.6 27B | 4-bit | 128 | 26.0 | 34.0 | 33.0 (-2.9%) | 32.7 (-3.9%) |", lines)
        self.assertIn("|  |  | 512 | 26.0 | 33.9 | 33.0 (-2.6%) | 32.6 (-3.9%) |", lines)
        self.assertIn("|  |  | 2048 | 18.8 | 33.4 | 31.6 (-5.4%) | 31.1 (-6.9%) |", lines)


if __name__ == "__main__":
    unittest.main()
