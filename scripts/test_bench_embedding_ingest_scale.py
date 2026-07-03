#!/usr/bin/env python3
"""Unit tests for large-scale embedding ingest benchmark contracts."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = load_script("bench_embedding_ingest_scale", "bench_embedding_ingest_scale.py")


class EmbeddingIngestScaleBenchmarkTests(unittest.TestCase):
    def test_default_output_dir_uses_embedding_results_tree(self) -> None:
        args = bench.build_parser().parse_args(["--model", "qwen=/tmp/model"])

        self.assertTrue(
            args.output_dir.as_posix().endswith(
                "/benchmarks/results/embedding/embedding-scale"
            )
        )

    def test_split_batches_keeps_partial_tail(self) -> None:
        corpus = [[1], [2], [3], [4], [5]]

        self.assertEqual(
            bench.split_batches(corpus, 2),
            [[[1], [2]], [[3], [4]], [[5]]],
        )

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(bench.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(bench.percentile([10.0, 20.0], 0.95), 19.5)

    def test_build_chunk_corpus_is_sized(self) -> None:
        corpus = bench.build_chunk_corpus(total_chunks=3, chunk_tokens=4, vocab_size=20)

        self.assertEqual([len(row) for row in corpus], [4, 4, 4])
        self.assertTrue(all(1 <= token < 20 for row in corpus for token in row))

    def test_interleaved_trials_alternate_engine_order(self) -> None:
        calls = []
        workload = bench.ScaleWorkload(
            name="scale_1x1_b1",
            chunk_tokens=1,
            batch_size=1,
            total_chunks=1,
            batches=[[[1]]],
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

    def test_render_summary_includes_scale_metrics(self) -> None:
        summary = bench.render_summary(
            {
                "output_contract": bench.OUTPUT_CONTRACT,
                "total_chunks": 128,
                "reference": "mlx_lm",
                "models": [
                    {
                        "model_label": "qwen-test",
                        "rows": [
                            {
                                "chunk_tokens": 256,
                                "batch_size": 32,
                                "batches_per_trial": 4,
                                "results": {
                                    "mlx_lm": {"median_tokens_per_sec": 1000.0},
                                    "ax_engine_py": {
                                        "median_tokens_per_sec": 1250.0,
                                        "median_chunks_per_sec": 4.8,
                                        "median_batch_p95_ms": 11.2,
                                    },
                                },
                                "comparison": {"ax_vs_reference_tokens_pct": 25.0},
                            }
                        ],
                    }
                ],
            }
        )

        self.assertIn("# Embedding Ingest Scale Benchmark", summary)
        self.assertIn("| qwen-test | 256 | 32 | 4 | 1,000.0 | 1,250.0 | +25.0%", summary)
        self.assertIn("4.8", summary)
        self.assertIn("11.2", summary)


if __name__ == "__main__":
    unittest.main()
