#!/usr/bin/env python3
"""Weight-free regressions for the pinned MLX-VLM Flash Next reference runner."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT = Path(__file__).with_name("run_flash_next_reference.py")
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("flash_reference", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class Clock:
    def __init__(self, ticks):
        self.ticks = iter(ticks)

    def __call__(self):
        return next(self.ticks)


class ReferenceRunnerTests(unittest.TestCase):
    def test_collect_stops_at_eos_without_recording_it(self):
        generated, terminal, times = mod.collect(iter([5, 6, 248044, 7]), 10, mod.EOS_IDS,
                                                 Clock([1.0, 2.0, 3.0]))
        self.assertEqual((generated, terminal, times), ([5, 6], 248044, [1.0, 2.0]))

    def test_collect_stops_at_budget_and_reports_no_terminal(self):
        generated, terminal, times = mod.collect(iter(range(100)), 3, mod.EOS_IDS,
                                                 Clock([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual((generated, terminal, len(times)), ([0, 1, 2], None, 3))

    def test_timing_uses_the_first_token_as_ttft_and_the_rest_as_decode(self):
        record = mod.timing_record(10.0, [12.0, 12.5, 13.0, 14.0], 14.5)
        self.assertEqual(record["ttft_seconds"], 2.0)
        self.assertEqual(record["decode_tokens"], 3)
        self.assertEqual(record["decode_seconds"], 2.0)
        self.assertAlmostEqual(record["decode_tok_s"], 1.5)
        self.assertEqual(record["generated_tokens"], 4)

    def test_timing_without_a_decode_window_has_no_rate(self):
        self.assertNotIn("decode_tok_s", mod.timing_record(0.0, [1.0], 1.0))
        self.assertNotIn("ttft_seconds", mod.timing_record(0.0, [], 1.0))

    def test_window_hashes_cover_every_cell_and_match_the_shared_workload(self):
        hashes = mod.window_hashes(248320, (512, 2048))
        self.assertEqual(len(hashes), 2 * (mod.WARMUPS + mod.MEASURED))
        ids = mod.matrix.workload_ids(248320, 512, 3)
        self.assertEqual(hashes["cell-512-3"], mod.matrix.ids_sha(ids))
        self.assertEqual(len(set(hashes.values())), len(hashes))

    def test_prompt_hashes_must_match_the_ax_matrix(self):
        ours = mod.window_hashes(248320, (512,))
        self.assertEqual(mod.verify_prompt_hashes(dict(ours), ours), ours)
        with self.assertRaisesRegex(ValueError, "cell-512-3"):
            mod.verify_prompt_hashes({**ours, "cell-512-3": "0"}, ours)
        with self.assertRaisesRegex(ValueError, "cell-512-0"):
            mod.verify_prompt_hashes({}, ours)

    def test_matrix_summary_uses_measured_runs_and_reports_short_outputs(self):
        def cell(run, tokens, rate, ttft):
            return {"length": 512, "run": run, "warmup": run < mod.WARMUPS, "generated_tokens": tokens,
                    "decode_tok_s": rate, "ttft_seconds": ttft}
        cells = [cell(0, 128, 99.0, 99.0), cell(1, 128, 99.0, 99.0), cell(2, 128, 10.0, 4.0),
                 cell(3, 106, 30.0, 5.0), cell(4, 128, 20.0, 6.0)]
        summary = mod.summarize_matrix(cells, (512,))["512"]
        self.assertEqual(summary["measured_runs"], 3)
        self.assertEqual(summary["full_length_runs"], 2)
        self.assertEqual(summary["decode_tok_s_median"], 20.0)
        self.assertEqual(summary["ttft_s_median"], 5.0)

    def test_dry_run_labels_the_reference_and_cannot_claim_qualification(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--dry-run"], capture_output=True,
                             text=True, check=True)
        contract = json.loads(out.stdout)
        self.assertFalse(contract["qualification"])
        self.assertFalse(contract["release_ready"])
        self.assertEqual(contract["reference"]["versions"]["mlx-vlm"], "0.7.0rc0")
        self.assertIn("non-primary", contract["reference"]["role"])
        self.assertEqual(contract["lengths"], [512, 2048, 8192])

    def test_live_run_requires_inputs_and_the_ax_matrix_for_the_matrix_phase(self):
        done = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("live run requires", done.stderr)


if __name__ == "__main__":
    unittest.main()
