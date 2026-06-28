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
