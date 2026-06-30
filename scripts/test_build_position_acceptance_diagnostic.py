#!/usr/bin/env python3
"""Unit tests for build_position_acceptance_diagnostic.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_position_acceptance_diagnostic as diag


class TestMtpConditionalAcceptance(unittest.TestCase):
    def test_no_truncation_is_exact_hazard(self) -> None:
        # 3 cycles, every cycle drafts 2 tokens; accept_lens = [2, 1, 0].
        #   accepted_by_depth = [#accept>0, #accept>1] = [2, 1]
        #   drafted_by_depth  = [#draft>0,  #draft>1]  = [3, 3]
        rows = diag.mtp_conditional_acceptance([2, 1], [3, 3])
        self.assertEqual(len(rows), 2)
        p1, p2 = rows
        # depth 0: 2/3 exact (prefix == N == drafted, no truncation)
        self.assertTrue(p1["exact"])
        self.assertAlmostEqual(p1["cond_lower"], 2 / 3)
        self.assertAlmostEqual(p1["cond_upper"], 2 / 3)
        # depth 1: conditional 1/2 exact; note unconditional is only 1/3.
        self.assertTrue(p2["exact"])
        self.assertAlmostEqual(p2["cond_lower"], 0.5)
        self.assertAlmostEqual(p2["unconditional"], 1 / 3)

    def test_truncation_widens_bracket_and_brackets_truth(self) -> None:
        # Cycles: A(draft2,acc2), B(draft1,acc1 -- gate truncated), C(draft2,acc0).
        #   accepted_by_depth = [#accept>0, #accept>1] = [2, 1]
        #   drafted_by_depth  = [#draft>0,  #draft>1]  = [3, 2]   (B did not draft depth 1)
        # True conditional at depth 1 = 1.0 (only cycle A reached & accepted depth 1).
        rows = diag.mtp_conditional_acceptance([2, 1], [3, 2])
        p2 = rows[1]
        self.assertFalse(p2["exact"])
        self.assertAlmostEqual(p2["cond_lower"], 0.5)
        self.assertAlmostEqual(p2["cond_upper"], 1.0)
        # The naive DSpark hazard (A[d]/A[d-1] = 1/2) would understate the truth and
        # fall outside the bracket under truncation, so it is not emitted for MTP.
        self.assertNotIn("cond_point", p2)
        # Truth (1.0) lies within the reported bracket.
        self.assertLessEqual(p2["cond_lower"], 1.0)
        self.assertGreaterEqual(p2["cond_upper"], 1.0)

    def test_unreached_depth_is_marked(self) -> None:
        # Nothing accepted past depth 0 -> depth 1 prefix is 0 -> not reached.
        rows = diag.mtp_conditional_acceptance([0, 0], [5, 0])
        self.assertIsNone(rows[1]["cond_lower"])
        self.assertFalse(rows[1]["exact"])

    def test_upper_bound_clamped_to_one(self) -> None:
        rows = diag.mtp_conditional_acceptance([1, 1], [1, 1])
        for r in rows:
            if r["cond_upper"] is not None:
                self.assertLessEqual(r["cond_upper"], 1.0)

    def test_mean_accept_length_identity(self) -> None:
        # sum(accepted_by_depth)/N must equal mean accept_len over cycles.
        # accept_lens [2,1,0] -> mean 1.0; accepted_by_depth [2,1], N=3.
        self.assertAlmostEqual(diag.mean_accept_length_mtp([2, 1], 3), 1.0)
        self.assertIsNone(diag.mean_accept_length_mtp([2, 1], 0))


class TestNgramConditionalAcceptance(unittest.TestCase):
    def test_hazard_from_histogram(self) -> None:
        # histogram[k] = #attempts accepting exactly k. [2,3,5], total 10.
        # S0=10, S1=8, S2=5 -> pos1 = 8/10, pos2 = 5/8.
        rows = diag.ngram_conditional_acceptance([2, 3, 5])
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["cond_point"], 0.8)
        self.assertAlmostEqual(rows[1]["cond_point"], 5 / 8)
        self.assertEqual(rows[0]["kind"], "hazard_lower_bound")

    def test_mean_accept_length_ngram(self) -> None:
        # (0*2 + 1*3 + 2*5)/10 = 1.3
        self.assertAlmostEqual(diag.mean_accept_length_ngram([2, 3, 5]), 1.3)
        self.assertIsNone(diag.mean_accept_length_ngram([0, 0, 0]))

    def test_empty_histogram(self) -> None:
        rows = diag.ngram_conditional_acceptance([0, 0, 0, 0])
        for r in rows:
            self.assertIsNone(r["cond_point"])


class TestExtraction(unittest.TestCase):
    def test_aggregate_sums_across_results(self) -> None:
        obj = {
            "model": "demo",
            "ax_mtp_max_depth": 2,
            "results": [
                {"ngram_acceleration_telemetry": {"ax_mtp_drafted_depth0": 10, "ax_mtp_accepted_depth0": 9}},
                {"ngram_acceleration_telemetry": {"ax_mtp_drafted_depth0": 5, "ax_mtp_accepted_depth0": 4}},
            ],
        }
        context, route = diag.extract_ax_artifact(obj)
        self.assertEqual(context["model"], "demo")
        self.assertEqual(context["result_rows"], 2)
        self.assertEqual(route["ax_mtp_drafted_depth0"], 15)
        self.assertEqual(route["ax_mtp_accepted_depth0"], 13)

    def test_fallback_tree_search(self) -> None:
        obj = {"nested": {"deep": {"ax_mtp_accepted_depth0": 7, "ax_mtp_drafted_depth0": 8}}}
        _, route = diag.extract_ax_artifact(obj)
        self.assertIsNotNone(route)
        self.assertEqual(route["ax_mtp_accepted_depth0"], 7)

    def test_max_depth_warning(self) -> None:
        route = {"ax_mtp_drafted_depth0": 10, "ax_mtp_accepted_depth0": 9}
        report = diag.analyze_route_decisions(route, {"ax_mtp_max_depth": 5})
        self.assertTrue(any("exceeds" in w for w in report["warnings"]))


class TestAnalyzeAndRender(unittest.TestCase):
    def _flappy_like_route(self) -> dict[str, int]:
        # Mirrors the real flappy 31b artifact numbers (depth-1-dominated).
        return {
            "ax_mtp_decode_steps": 2436,
            "ax_mtp_accepted_tokens": 2415,
            "ax_mtp_drafted_depth0": 2436,
            "ax_mtp_drafted_depth1": 7,
            "ax_mtp_drafted_depth2": 5,
            "ax_mtp_accepted_depth0": 2415,
            "ax_mtp_accepted_depth1": 0,
            "ax_mtp_accepted_depth2": 0,
            "ax_mtp_accepted_source_mtp_tokens": 2410,
            "ax_mtp_accepted_source_ngram_tokens": 5,
        }

    def test_realistic_flappy_numbers(self) -> None:
        report = diag.analyze_route_decisions(self._flappy_like_route(), {"model": "gemma-31b"})
        mtp = report["sources"]["mtp"]
        p1 = mtp["positions"][0]
        # Position-1 conditional ~99.1%, exact (prefix==drafted==N at depth 0).
        self.assertTrue(p1["exact"])
        self.assertAlmostEqual(p1["cond_lower"], 2415 / 2436, places=4)
        # Position 2 reached by 2415 prefix-accepted cycles but only 7 drafted it.
        p2 = mtp["positions"][1]
        self.assertEqual(p2["reached_prefix"], 2415)
        self.assertEqual(p2["drafted"], 7)
        self.assertEqual(p2["accepted"], 0)
        # Render must not raise.
        text = diag.render_text("flappy", report)
        self.assertIn("MTP", text)

    def test_end_to_end_file(self) -> None:
        obj = {
            "model": "demo",
            "ax_mtp_max_depth": 2,
            "results": [{"ngram_acceleration_telemetry": self._flappy_like_route()}],
        }
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "artifact.json"
            out = Path(td) / "report.json"
            src.write_text(json.dumps(obj))
            rc = diag.main([str(src), "--output", str(out)])
            self.assertEqual(rc, 0)
            written = json.loads(out.read_text())
            self.assertIn(str(src), written)
            self.assertIn("mtp", written[str(src)]["sources"])

    def test_missing_counters_is_graceful(self) -> None:
        report = diag.analyze_route_decisions({"unrelated": 1}, {})
        self.assertEqual(report["sources"], {})
        text = diag.render_text("empty", report)
        self.assertIn("no speculative-decode counters", text)


class TestRealArtifactSmoke(unittest.TestCase):
    """Parse a real on-disk artifact if present (skips in CI when absent)."""

    REAL = Path(__file__).resolve().parents[1] / (
        "benchmarks/results/gemma4-assistant-mtp/"
        "2026-06-09-gemma4-26b-31b-optimized-scenario/31b-4bit/flappy/mtp-ngram.json"
    )

    def test_real_artifact_parses(self) -> None:
        if not self.REAL.exists():
            self.skipTest(f"real artifact not present: {self.REAL}")
        obj = json.loads(self.REAL.read_text())
        context, route = diag.extract_ax_artifact(obj)
        self.assertIsNotNone(route)
        report = diag.analyze_route_decisions(route, context)
        self.assertIn("mtp", report["sources"])
        # Position-1 conditional acceptance should be a sane probability.
        p1 = report["sources"]["mtp"]["positions"][0]
        self.assertIsNotNone(p1["cond_lower"])
        self.assertGreaterEqual(p1["cond_lower"], 0.0)
        self.assertLessEqual(p1["cond_upper"], 1.0)


if __name__ == "__main__":
    unittest.main()
