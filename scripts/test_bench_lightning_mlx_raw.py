#!/usr/bin/env python3
"""Tests for the lightning-mlx raw benchmark wrapper."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("bench_lightning_mlx_raw.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_lightning_mlx_raw", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
bench = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bench_lightning_mlx_raw"] = bench
MODULE_SPEC.loader.exec_module(bench)


class LightningMlxRawBenchTests(unittest.TestCase):
    def test_mtp_command_matches_readme_raw_decode_settings(self) -> None:
        cmd = bench.build_bench_command(
            python=Path("python"),
            lightning_source=Path("lightning"),
            model="qwen3.6-27b",
            mode="mtp",
            num_prompts=3,
            max_tokens=512,
        )

        self.assertEqual(cmd[:4], ["python", "-m", "vllm_mlx.cli", "bench"])
        self.assertIn("qwen3.6-27b", cmd)
        self.assertIn("--disable-prefix-cache", cmd)
        self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "1")
        self.assertEqual(cmd[cmd.index("--prefill-batch-size") + 1], "1")
        self.assertEqual(cmd[cmd.index("--completion-batch-size") + 1], "1")
        self.assertEqual(cmd[cmd.index("--prefill-step-size") + 1], "8192")
        self.assertEqual(cmd[cmd.index("--mtp-num-draft-tokens") + 1], "3")
        self.assertIn("--mtp-optimistic", cmd)
        self.assertNotIn("--enable-ngram", cmd)

    def test_ngram_command_adds_source_preset_flags(self) -> None:
        cmd = bench.build_bench_command(
            python=Path("python"),
            lightning_source=Path("lightning"),
            model="qwen3.6-35b",
            mode="mtp-ngram",
            num_prompts=3,
            max_tokens=512,
        )

        self.assertIn("--enable-ngram", cmd)
        self.assertEqual(cmd[cmd.index("--ngram-num-draft-tokens") + 1], "6")
        self.assertEqual(cmd[cmd.index("--ngram-min-occurrences") + 1], "2")
        self.assertIn("--ngram-hybrid-verify", cmd)
        self.assertIn("--ngram-everywhere", cmd)
        self.assertIn("--ngram-skip-tool-calls", cmd)
        self.assertEqual(
            cmd[cmd.index("--ngram-auto-disable-mtp-threshold") + 1], "0.85"
        )
        self.assertEqual(
            cmd[cmd.index("--ngram-auto-disable-min-ngram") + 1], "0.50"
        )

    def test_parse_bench_stdout_extracts_lightning_metrics(self) -> None:
        metrics = bench.parse_bench_stdout(
            """
Results:
  Total time: 12.34s
  Prompts: 3
  Prompts/second: 0.24
  Total prompt tokens: 42
  Total completion tokens: 1536
  Total tokens: 1578
  Tokens/second: 124.47
  Throughput: 127.88 tok/s
"""
        )

        self.assertEqual(metrics["total_time_s"], 12.34)
        self.assertEqual(metrics["prompts"], 3.0)
        self.assertEqual(metrics["completion_tokens_per_second"], 124.47)
        self.assertEqual(metrics["throughput_tok_s"], 127.88)


if __name__ == "__main__":
    unittest.main()
