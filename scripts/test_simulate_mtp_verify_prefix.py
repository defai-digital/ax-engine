#!/usr/bin/env python3
"""Unit tests for the MTP verify-prefix simulator."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("simulate_mtp_verify_prefix.py")
MODULE_SPEC = importlib.util.spec_from_file_location("simulate_mtp_verify_prefix", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
sim = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = sim
MODULE_SPEC.loader.exec_module(sim)


def record(t_us, generated, used, accepted, probs=(), depth=None, ctx=100, draft_us=0):
    return {
        "t_us": t_us,
        "ctx": ctx,
        "generated": generated,
        "depth": used if depth is None else depth,
        "used": used,
        "accepted": accepted,
        "pure_mtp": True,
        "temperature": 0.0,
        "verify_forward_us": 0,
        "verify_eval_us": 0,
        "draft_us": draft_us,
        "draft_log_probs": [math.log(p) if p > 0 else None for p in probs],
    }


def write_trace(directory: Path, name: str, records) -> Path:
    path = directory / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


class LoadTraceTests(unittest.TestCase):
    def test_walls_and_request_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_trace(
                Path(tmp),
                "t.jsonl",
                [
                    record(1_000, 0, 3, 3),
                    record(41_000, 4, 3, 1),
                    # generated did not increase: a new request starts.
                    record(90_000, 0, 3, 2),
                    record(130_000, 3, 3, 0),
                ],
            )
            cycles = sim.load_trace(path)
        self.assertEqual([c.request for c in cycles], [0, 0, 1, 1])
        self.assertEqual([c.wall_us for c in cycles], [None, 40_000, None, 40_000])

    def test_non_contiguous_records_have_no_wall(self):
        # The previous cycle emitted 2 tokens (accepted 1 + 1) but generated
        # advanced by 3: an untraced direct step ran in between.
        with tempfile.TemporaryDirectory() as tmp:
            path = write_trace(Path(tmp), "t.jsonl", [record(0, 0, 3, 1), record(70_000, 3, 3, 1)])
            cycles = sim.load_trace(path)
        self.assertEqual([c.wall_us for c in cycles], [None, None])

    def test_null_log_prob_excludes_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_trace(Path(tmp), "t.jsonl", [record(1, 0, 2, 0, probs=(0.5, 0.0))])
            cycles = sim.load_trace(path)
        self.assertAlmostEqual(cycles[0].probs[0], 0.5)
        self.assertFalse(cycles[0].probs_finite)
        self.assertEqual(sim.eligible_cycles(cycles), [])


class PolicyTests(unittest.TestCase):
    def test_threshold_keeps_leading_run(self):
        self.assertEqual(sim.threshold_k([0.99, 0.95, 0.5], 0.9), 2)
        self.assertEqual(sim.threshold_k([0.2, 0.99], 0.9), 0)
        self.assertEqual(sim.threshold_k([0.99, 0.99], 0.9), 2)

    def test_truncated_cost_charges_sunk_drafts(self):
        cost = {1: 30.0, 2: 36.0, 3: 40.0}
        self.assertEqual(sim.truncated_cost(3, 3, cost, 2.0), 40.0)
        self.assertEqual(sim.truncated_cost(1, 3, cost, 2.0), 34.0)

    def test_cost_policy_truncates_low_confidence_tail(self):
        calibration = sim.Calibration(edges=sim.DEFAULT_BIN_EDGES, max_position=3)
        cycle_hi = sim.Cycle(0, 0, 0, 0, 3, 3, 3, True, 0.0, 0, (0.99, 0.99, 0.99))
        cycle_lo = sim.Cycle(0, 0, 0, 0, 3, 3, 2, True, 0.0, 0, (0.99, 0.99, 0.1))
        for _ in range(50):
            calibration.observe(cycle_hi)
            calibration.observe(cycle_lo)
        # Steep marginal cost for the third draft: truncating it pays.
        cost = {1: 30.0, 2: 32.0, 3: 60.0}
        self.assertEqual(sim.cost_optimal_k((0.99, 0.99, 0.1), 3, calibration, cost, 0.0), 2)
        # Flat cost: verifying everything is optimal.
        flat = {1: 30.0, 2: 30.5, 3: 30.6}
        self.assertEqual(sim.cost_optimal_k((0.99, 0.99, 0.1), 3, calibration, flat, 0.0), 3)

    def test_calibration_ignores_positions_after_first_rejection(self):
        calibration = sim.Calibration(edges=sim.DEFAULT_BIN_EDGES, max_position=3)
        calibration.observe(sim.Cycle(0, 0, 0, 0, 3, 3, 0, True, 0.0, 0, (0.99, 0.99, 0.99)))
        self.assertEqual(sum(calibration.trials.values()), 1)
        self.assertEqual(sum(calibration.accepts.values()), 0)

    def test_simulate_oracle_bounds_full_and_matches_tokens(self):
        cycles = [
            sim.Cycle(r, 0, 0, i, 3, 3, a, True, 0.0, 0, (0.99, 0.9, 0.4))
            for r in range(4)
            for i, a in enumerate((3, 1, 0, 2))
        ]
        cost = {1: 30.0, 2: 36.0, 3: 40.0}
        results = {r.name: r for r in sim.simulate(cycles, cost, 0.0, [0.9], sim.DEFAULT_BIN_EDGES)}
        full = results["full"]
        oracle = results["oracle"]
        self.assertEqual(full.tokens, oracle.tokens)
        self.assertEqual(full.tokens, sum(a + 1 for a in (3, 1, 0, 2)) * 4)
        self.assertGreaterEqual(oracle.tok_per_s, full.tok_per_s)
        self.assertEqual(full.cycles, 16)

    def test_oracle_never_below_full_with_non_monotone_cost(self):
        cycles = [sim.Cycle(r, 0, 0, 0, 3, 3, 2, True, 0.0, 0, (0.99, 0.9, 0.4)) for r in range(2)]
        cost = {1: 30.0, 2: 90.0, 3: 40.0}
        results = {r.name: r for r in sim.simulate(cycles, cost, 0.0, [], sim.DEFAULT_BIN_EDGES)}
        self.assertGreaterEqual(results["oracle"].tok_per_s, results["full"].tok_per_s)

    def test_single_request_cannot_cross_fit(self):
        cycles = [sim.Cycle(0, 0, 0, i, 3, 3, 1, True, 0.0, 0, (0.9, 0.9, 0.9)) for i in range(3)]
        with self.assertRaises(ValueError):
            sim.simulate(cycles, {1: 1.0, 2: 2.0, 3: 3.0}, 0.0, [], sim.DEFAULT_BIN_EDGES)


class MainTests(unittest.TestCase):
    def test_main_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            calibration = write_trace(
                directory,
                "cal.jsonl",
                [record(i * 40_000, (i % 4) * 3, 3, 2, probs=(0.99, 0.9, 0.5)) for i in range(8)],
            )
            cost_records = []
            t = 0
            for depth, wall in ((1, 30_000), (2, 36_000), (3, 40_000)):
                for i in range(4):
                    t += wall
                    cost_records.append(record(t, i * 2, depth, 1))
                t += 1_000_000
            cost = write_trace(directory, "cost.jsonl", cost_records)
            output = directory / "report.json"
            with unittest.mock.patch("sys.stdout"):
                status = sim.main(
                    [
                        "--calibration-trace",
                        str(calibration),
                        "--cost-trace",
                        str(cost),
                        "--draft-cost-us",
                        "1500",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["cost_us_by_verify_width"]["3"], 40_000.0)
        names = [p["name"] for p in report["policies"]]
        self.assertIn("cost", names)
        self.assertIn("oracle", names)
        self.assertEqual(report["policies"][0]["vs_full"], 1.0)


if __name__ == "__main__":
    unittest.main()
