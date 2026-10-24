#!/usr/bin/env python3
"""Unit tests for the OMLX prompt-suite runner's timing helpers."""

from __future__ import annotations

import asyncio
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_omlx_prompt_suites as omlx_bench  # noqa: E402


@dataclass
class _Chunk:
    new_text: str = ""
    completion_tokens: int = 0
    text: str = ""


class _StreamingEngine:
    async def stream_generate(self, tokens, **kwargs):
        yield _Chunk()  # prefill heartbeat: no token yet
        yield _Chunk(new_text="a", completion_tokens=1, text="a")
        yield _Chunk(new_text="b", completion_tokens=2, text="ab")


class _BlockingEngine:
    async def generate(self, tokens, **kwargs):
        return _Chunk(completion_tokens=2, text="ab")


class StreamTrialMetricsTests(unittest.TestCase):
    def test_prefill_and_decode_are_separated(self) -> None:
        metrics = omlx_bench.stream_trial_metrics(10.0, 10.5, 12.5, 201, 300)
        self.assertAlmostEqual(metrics["ttft_s"], 0.5)
        self.assertAlmostEqual(metrics["prefill_tok_s"], 600.0)
        self.assertAlmostEqual(metrics["decode_only_tok_s"], 100.0)  # 200 tokens in 2 s
        self.assertAlmostEqual(metrics["decode_tok_s"], 201 / 2.5)  # whole call, as before
        self.assertAlmostEqual(metrics["elapsed_s"], 2.5)

    def test_a_single_token_has_no_decode_rate(self) -> None:
        metrics = omlx_bench.stream_trial_metrics(0.0, 0.2, 0.2, 1, 100)
        self.assertIsNone(metrics["decode_only_tok_s"])
        self.assertAlmostEqual(metrics["prefill_tok_s"], 500.0)

    def test_an_engine_that_cannot_stream_reports_wall_rate_only(self) -> None:
        metrics = omlx_bench.stream_trial_metrics(0.0, None, 2.0, 100, 300)
        self.assertEqual(metrics["decode_tok_s"], 50.0)
        for key in ("ttft_s", "prefill_tok_s", "decode_only_tok_s"):
            self.assertIsNone(metrics[key])


class GenerateTimedTests(unittest.TestCase):
    def test_streaming_marks_the_first_token_not_the_heartbeat(self) -> None:
        output, first_at = asyncio.run(omlx_bench.generate_timed(_StreamingEngine(), [1, 2]))
        self.assertEqual(output.text, "ab")
        self.assertIsNotNone(first_at)

    def test_blocking_engines_fall_back_to_one_call(self) -> None:
        output, first_at = asyncio.run(omlx_bench.generate_timed(_BlockingEngine(), [1, 2]))
        self.assertEqual(output.completion_tokens, 2)
        self.assertIsNone(first_at)


if __name__ == "__main__":
    unittest.main()
