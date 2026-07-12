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
    def test_default_output_dir_uses_embedding_results_tree(self) -> None:
        args = bench.build_parser().parse_args(["--model", "qwen=/tmp/model"])

        self.assertTrue(
            args.output_dir.as_posix().endswith(
                "/benchmarks/results/embedding/embedding-fair"
            )
        )

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
                    "median_ms_per_item": 10.0,
                    "median_ms_per_batch": 10.0,
                },
                "ax_engine_py": {
                    "median_tokens_per_sec": 125.0,
                    "median_items_per_sec": 9.0,
                    "median_ms_per_item": 8.0,
                    "median_ms_per_batch": 8.0,
                },
            }
        )
        self.assertAlmostEqual(comparison["ax_vs_reference_tokens_pct"], 25.0)
        self.assertAlmostEqual(comparison["ax_vs_reference_items_pct"], -10.0)
        self.assertAlmostEqual(comparison["ax_vs_reference_ms_per_item_pct"], -20.0)
        self.assertAlmostEqual(comparison["ax_vs_reference_ms_per_batch_pct"], -20.0)

    def test_classify_mlx_path_distinguishes_homebrew_and_pip(self) -> None:
        self.assertEqual(
            bench.classify_mlx_path("/opt/homebrew/opt/mlx/lib/libmlx.dylib"),
            "homebrew",
        )
        self.assertEqual(
            bench.classify_mlx_path(
                "/Users/x/.venv/lib/python3.14/site-packages/mlx/lib/libmlx.dylib"
            ),
            "pip_or_venv",
        )

    def test_primary_metric_for_workload(self) -> None:
        self.assertEqual(
            bench.primary_metric_for_workload("short_query_b8"),
            "median_ms_per_item",
        )
        self.assertEqual(
            bench.primary_metric_for_workload("fixed_256_b32"),
            "median_tokens_per_sec",
        )

    def test_trial_stats_includes_ms_per_item(self) -> None:
        stats = bench.trial_stats(
            [
                {
                    "ms_per_batch": 20.0,
                    "ms_per_item": 10.0,
                    "tokens_per_sec": 100.0,
                    "items_per_sec": 50.0,
                },
                {
                    "ms_per_batch": 24.0,
                    "ms_per_item": 12.0,
                    "tokens_per_sec": 90.0,
                    "items_per_sec": 45.0,
                },
            ],
            "ax",
        )
        self.assertEqual(stats["median_ms_per_item"], 11.0)
        self.assertIn("median_ms_per_batch", stats)

    def test_interleaved_trials_alternate_engine_order(self) -> None:
        calls = []
        workload = bench.Workload(
            name="fixed_1_b1",
            input_kind="synthetic_token_ids",
            batch_size=1,
            token_ids=[[1]],
        )

        def make_step(name: str):
            def step(_batch):
                calls.append(name)
                return (b"\x00\x00\x00\x00", 1, 1)

            return step

        results = bench.run_trials_interleaved(
            [
                bench.EngineRunner("ref", "reference", make_step("ref")),
                bench.EngineRunner("ax_engine_py", "ax-engine-py", make_step("ax")),
            ],
            workload,
            warmup=1,
            trials=3,
            cooldown=0.0,
        )

        self.assertEqual(
            calls,
            [
                "ref",
                "ax",
                "ref",
                "ax",
                "ax",
                "ref",
                "ref",
                "ax",
            ],
        )
        self.assertEqual(set(results), {"ref", "ax_engine_py"})

    def test_parser_defaults_to_publication_cooldown(self) -> None:
        args = bench.build_parser().parse_args(["--model", "qwen=/tmp/model"])

        self.assertEqual(args.cooldown, 15.0)

    def test_render_summary_includes_contract_and_delta(self) -> None:
        summary = bench.render_summary(
            {
                "output_contract": bench.OUTPUT_CONTRACT,
                "reference": "mlx_lm",
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
                                "comparison": {"ax_vs_reference_tokens_pct": 25.0},
                            }
                        ],
                    }
                ],
            }
        )
        self.assertIn(bench.OUTPUT_CONTRACT, summary)
        self.assertIn(
            "| qwen-test | fixed_16_b8 | 8 | 16 | tok/s | 100.0 | 125.0 | +25.0% |",
            summary,
        )

    def test_render_summary_short_query_headlines_latency(self) -> None:
        summary = bench.render_summary(
            {
                "output_contract": bench.OUTPUT_CONTRACT,
                "reference": "mlx_lm",
                "models": [
                    {
                        "model_label": "qwen-test",
                        "rows": [
                            {
                                "workload": "short_query_b1",
                                "batch_size": 1,
                                "max_tokens": 15,
                                "results": {
                                    "mlx_lm": {
                                        "median_tokens_per_sec": 100.0,
                                        "median_ms_per_item": 12.5,
                                    },
                                    "ax_engine_py": {
                                        "median_tokens_per_sec": 110.0,
                                        "median_ms_per_item": 10.0,
                                    },
                                },
                                "comparison": {
                                    "ax_vs_reference_ms_per_item_pct": -20.0,
                                    "ax_vs_reference_tokens_pct": 10.0,
                                },
                            }
                        ],
                    }
                ],
            }
        )
        self.assertIn("ms/item", summary)
        self.assertIn(
            "| qwen-test | short_query_b1 | 1 | 15 | ms/item | 12.50 | 10.00 | -20.0% |",
            summary,
        )

    def test_render_summary_supports_ax_only(self) -> None:
        summary = bench.render_summary(
            {
                "output_contract": bench.OUTPUT_CONTRACT,
                "ax_only": True,
                "pooling": "mean",
                "models": [
                    {
                        "model_label": "embed-test",
                        "rows": [
                            {
                                "workload": "fixed_64_b8",
                                "batch_size": 8,
                                "max_tokens": 64,
                                "results": {
                                    "ax_engine_py": {
                                        "median_tokens_per_sec": 1234.0,
                                        "median_items_per_sec": 19.3,
                                    },
                                },
                            }
                        ],
                    }
                ],
            }
        )
        self.assertIn("# AX-Only Embedding Benchmark", summary)
        self.assertIn("pooling: `mean`", summary)
        self.assertIn(
            "| embed-test | fixed_64_b8 | 8 | 64 | tok/s | 1,234.0 | 1,234.0 | 19.3 |",
            summary,
        )


if __name__ == "__main__":
    unittest.main()
