#!/usr/bin/env python3
"""Unit tests for the fair embedding benchmark contract."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).parent


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = load_script("bench_embedding_fair", "bench_embedding_fair.py")


class FairEmbeddingBenchmarkTests(unittest.TestCase):
    def test_parse_model_spec_accepts_label_equals_path(self) -> None:
        spec = bench.parse_model_spec("qwen=/tmp/model")
        self.assertEqual(spec.label, "qwen")
        self.assertEqual(spec.path, Path("/tmp/model"))

    def test_parse_csv_ints_rejects_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            bench.parse_csv_ints("1,0,8", name="batch-sizes")

    def test_model_vocab_size_reads_nested_text_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"text_config": {"vocab_size": 1234}}) + "\n"
            )
            self.assertEqual(bench.model_vocab_size(root), 1234)

    def test_synthetic_batch_is_bounded_and_sized(self) -> None:
        rows = bench.synthetic_batch(length=4, batch_size=3, vocab_size=10)
        self.assertEqual([len(row) for row in rows], [4, 4, 4])
        self.assertTrue(all(1 <= token < 10 for row in rows for token in row))
        self.assertNotEqual(rows[0], rows[1])

    def test_build_workloads_expands_short_and_fixed_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"vocab_size": 100}) + "\n")
            with patch.object(
                bench,
                "tokenize_short_queries",
                return_value=[[1, 2], [3, 4, 5]],
            ):
                workloads = bench.build_workloads(
                    root,
                    batch_sizes=[1, 2],
                    fixed_lengths=[4],
                    include_short_query=True,
                )
        self.assertEqual(
            [workload.name for workload in workloads],
            ["short_query_b1", "short_query_b2", "fixed_4_b1", "fixed_4_b2"],
        )
        self.assertEqual(workloads[1].token_counts, [2, 3])
        self.assertEqual(workloads[3].token_counts, [4, 4])

    def test_compare_results_reports_token_and_item_delta(self) -> None:
        comparison = bench.compare_results(
            {
                "mlx_lm": {
                    "median_tokens_per_sec": 100.0,
                    "median_items_per_sec": 10.0,
                },
                "ax_engine_py": {
                    "median_tokens_per_sec": 125.0,
                    "median_items_per_sec": 9.0,
                },
            }
        )
        self.assertAlmostEqual(comparison["ax_vs_mlx_lm_tokens_pct"], 25.0)
        self.assertAlmostEqual(comparison["ax_vs_mlx_lm_items_pct"], -10.0)

    def test_render_summary_includes_contract_and_delta(self) -> None:
        summary = bench.render_summary(
            {
                "output_contract": bench.OUTPUT_CONTRACT,
                "models": [
                    {
                        "model_label": "qwen-test",
                        "rows": [
                            {
                                "workload": "fixed_16_b8",
                                "batch_size": 8,
                                "max_tokens": 16,
                                "results": {
                                    "mlx_lm": {"median_tokens_per_sec": 100.0},
                                    "ax_engine_py": {"median_tokens_per_sec": 125.0},
                                },
                                "comparison": {"ax_vs_mlx_lm_tokens_pct": 25.0},
                            }
                        ],
                    }
                ],
            }
        )
        self.assertIn(bench.OUTPUT_CONTRACT, summary)
        self.assertIn("| qwen-test | fixed_16_b8 | 8 | 16 | 100.0 | 125.0 | +25.0% |", summary)


if __name__ == "__main__":
    unittest.main()
