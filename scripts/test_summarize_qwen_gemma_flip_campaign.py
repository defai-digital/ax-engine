#!/usr/bin/env python3
"""Tests for repeated Qwen/Gemma campaign aggregation."""

from __future__ import annotations

import copy
import unittest

import summarize_qwen_gemma_flip_campaign as summary
import test_check_ax_multimodel_serving_artifact as fixtures


class QwenGemmaFlipCampaignSummaryTests(unittest.TestCase):
    def test_aggregate_scenario_uses_medians_and_passes_green_ratios(self) -> None:
        candidate = []
        baseline = []
        for index in range(3):
            cand = fixtures._artifact()
            base = copy.deepcopy(cand)
            cand["summary"]["output_token_throughput_tok_s"] = 120.0 + index  # type: ignore[index]
            base["summary"]["output_token_throughput_tok_s"] = 100.0 + index  # type: ignore[index]
            cand["summary"]["ttft_ms"] = fixtures._dist(80.0 + index)  # type: ignore[index]
            base["summary"]["ttft_ms"] = fixtures._dist(100.0 + index)  # type: ignore[index]
            cand["interactive_stream_gap_ms"] = fixtures._dist(16.0 + index)  # type: ignore[index]
            base["interactive_stream_gap_ms"] = fixtures._dist(20.0 + index)  # type: ignore[index]
            candidate.append(cand)
            baseline.append(base)

        row = summary.aggregate_scenario(
            scenario_id="s3",
            candidate=candidate,
            baseline=baseline,
            min_throughput_ratio=1.15,
            max_ttft_p95_ratio=0.90,
            max_stream_gap_p95_ratio=0.90,
            max_stream_gap_p95_ms=50.0,
        )

        self.assertTrue(row["passed"])
        self.assertAlmostEqual(row["candidate"]["throughput_median_tok_s"], 121.0)
        self.assertAlmostEqual(row["baseline"]["throughput_median_tok_s"], 101.0)

    def test_aggregate_scenario_reports_not_green_without_hiding_failures(self) -> None:
        candidate = [fixtures._artifact() for _ in range(3)]
        baseline = [fixtures._artifact() for _ in range(3)]
        for cand, base in zip(candidate, baseline, strict=True):
            cand["summary"]["output_token_throughput_tok_s"] = 80.0  # type: ignore[index]
            base["summary"]["output_token_throughput_tok_s"] = 100.0  # type: ignore[index]

        row = summary.aggregate_scenario(
            scenario_id="s0",
            candidate=candidate,
            baseline=baseline,
            min_throughput_ratio=1.15,
            max_ttft_p95_ratio=1.10,
            max_stream_gap_p95_ratio=1.10,
            max_stream_gap_p95_ms=None,
        )

        self.assertFalse(row["passed"])
        self.assertIn("median_throughput_ratio", row["failed_required_gates"])


if __name__ == "__main__":
    unittest.main()
