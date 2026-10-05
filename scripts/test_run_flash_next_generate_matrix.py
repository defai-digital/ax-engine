#!/usr/bin/env python3
"""Weight-free regressions for the Flash Next generate-matrix comparison."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT = Path(__file__).with_name("run_flash_next_generate_matrix.py")
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("flash_generate_matrix", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

PERF = {"time_to_first_token_us": 2_000_000, "model_eval_time_us": 4_000_000, "model_eval_token_count": 127}


def traj(prompt: str, tokens: list[int], verified: float = 0.0, errors: float = 0.0) -> dict:
    return {"prompt": prompt, "output_tokens": tokens, "verified_steps": verified,
            "route_errors": errors}


def cell(length: int, run: int, tokens: list[int], time_us: int, verified: float = 0.0) -> dict:
    perf = {**PERF, "model_eval_time_us": time_us}
    return {"length": length, "run": run, "warmup": run < mod.WARMUPS, "output_tokens": tokens,
            "performance": perf, "verified_steps": verified, "route_errors": 0.0}


def run(mode: str, trajectories: list[dict], cells: list[dict], hashes: dict | None = None) -> dict:
    return {"mode": mode, "trajectories": trajectories, "cells": cells,
            "prompt_id_hashes": hashes or {"a": "1"}}


class GenerateMatrixTests(unittest.TestCase):
    def test_decode_rate_requires_time_and_count(self):
        self.assertAlmostEqual(mod.decode_rate(PERF), 127 / 4.0)
        self.assertIsNone(mod.decode_rate({"model_eval_time_us": 0, "model_eval_token_count": 5}))
        self.assertIsNone(mod.decode_rate({}))

    def test_workload_slices_have_exact_length_and_distinct_prefixes(self):
        long_ids = list(range(10_000))
        slices = [mod.workload_ids(long_ids, 512, r) for r in range(mod.WARMUPS + mod.MEASURED)]
        self.assertTrue(all(len(s) == 512 for s in slices))
        self.assertEqual(len({s[0] for s in slices}), len(slices))
        with self.assertRaises(ValueError):
            mod.workload_ids(long_ids, 9_999, 1)

    def test_summary_reports_identity_divergence_ratio_and_route_errors(self):
        measured = range(mod.WARMUPS, mod.WARMUPS + mod.MEASURED)
        d_cells = [cell(512, r, [1, 2], 4_000_000) for r in range(mod.WARMUPS + mod.MEASURED)]
        m_cells = [cell(512, r, [1, 2] if r != 3 else [1, 9], 2_000_000, verified=5)
                   for r in range(mod.WARMUPS + mod.MEASURED)]
        direct = run("disabled", [traj("p1", [1, 2, 3]), traj("p2", [4, 5, 6])], d_cells)
        mtp = run("required", [traj("p1", [1, 2, 3], 7), traj("p2", [4, 9, 6], 7, errors=1)], m_cells)
        summary = mod.summarize({"disabled": direct, "required": mtp}, lengths=(512,))
        self.assertTrue(summary["prompt_ids_equal_across_modes"])
        self.assertEqual(summary["trajectories"]["identical"], 1)
        self.assertEqual(summary["trajectories"]["divergences"][0]["first_difference_index"], 1)
        self.assertEqual(summary["trajectories"]["mtp_verified_steps"], 14)
        self.assertEqual(summary["cells"]["512"]["measured_runs"], len(measured))
        self.assertAlmostEqual(summary["cells"]["512"]["mtp_over_direct_decode_ratio_median"], 2.0)
        self.assertEqual(summary["cell_token_identity"], {"identical": 2, "compared": 3})
        self.assertEqual(summary["route_errors"], 1)

    def test_summary_flags_prompt_id_mismatch_between_modes(self):
        cells = [cell(512, r, [1], 1_000_000) for r in range(mod.WARMUPS + mod.MEASURED)]
        a = run("disabled", [traj("p", [1])], cells, {"a": "1"})
        b = run("required", [traj("p", [1])], cells, {"a": "2"})
        self.assertFalse(mod.summarize({"disabled": a, "required": b}, lengths=(512,))
                         ["prompt_ids_equal_across_modes"])

    def test_dry_run_cannot_claim_qualification_or_a_reference_baseline(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--dry-run"], capture_output=True,
                             text=True, check=True)
        contract = json.loads(out.stdout)
        self.assertFalse(contract["qualification"])
        self.assertFalse(contract["release_ready"])
        self.assertEqual(contract["reference_baseline"], "none")
        self.assertEqual(contract["lengths"], [512, 2048, 8192])
        self.assertGreater(contract["total_blocks"] * 16, max(contract["lengths"]) + contract["output_tokens"])

    def test_live_run_requires_all_inputs(self):
        done = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("live run requires", done.stderr)


if __name__ == "__main__":
    unittest.main()
