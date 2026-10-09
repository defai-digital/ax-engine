#!/usr/bin/env python3
"""Weight-free regressions for the Flash Next generate-matrix comparison."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("run_flash_next_generate_matrix.py")
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("flash_generate_matrix", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def timing(decode_seconds: float | None = 4.0, decode_tokens: int = 127, ttft: float = 2.0) -> dict:
    return {"ttft_seconds": ttft, "decode_seconds": decode_seconds, "decode_tokens": decode_tokens}


def traj(prompt: str, tokens: list[int], verified: float = 0.0, errors: float = 0.0) -> dict:
    return {"prompt": prompt, "output_tokens": tokens, "verified_steps": verified,
            "route_errors": errors}


def cell(length: int, run: int, tokens: list[int], decode_seconds: float, verified: float = 0.0) -> dict:
    return {"length": length, "run": run, "warmup": run < mod.WARMUPS, "output_tokens": tokens,
            "timing": timing(decode_seconds), "mtp": {"draft_tokens": 4, "accepted_tokens": 3},
            "verified_steps": verified, "route_errors": 0.0}


def run(mode: str, trajectories: list[dict], cells: list[dict], hashes: dict | None = None) -> dict:
    return {"mode": mode, "trajectories": trajectories, "cells": cells,
            "prompt_id_hashes": hashes or {"a": "1"}}


class GenerateMatrixTests(unittest.TestCase):
    def test_generate_timing_includes_waiting_for_http_headers(self):
        class Stream:
            def __enter__(self):
                return iter([
                    b'event: step\n', b'data: {"delta_tokens": [1]}\n', b'\n',
                    b'event: response\n',
                    b'data: {"response": {"output_tokens": [1]}}\n', b'\n',
                ])

            def __exit__(self, *args):
                pass

        now = [10.0]

        def open_stream(*args, **kwargs):
            now[0] = 12.0  # Two seconds elapse before HTTP headers arrive.
            return Stream()

        with patch.object(mod.urllib.request, "urlopen", side_effect=open_stream), \
                patch.object(mod.time, "monotonic", side_effect=lambda: now[0]):
            response = mod.generate("http://127.0.0.1:31418", [2], max_tokens=1)
        self.assertEqual(response["client_timing"]["ttft_seconds"], 2.0)
        self.assertEqual(response["client_timing"]["total_seconds"], 2.0)

    def test_generate_stream_times_token_events_and_returns_the_terminal_response(self):
        ticks = iter([0.0, 1.0, 1.5, 3.5, 4.0])  # t0, two token events, end
        body = [b'event: request\n', b'data: {"request": {}}\n', b'\n',
                b'event: step\n', b'data: {"delta_tokens": [1]}\n', b'\n',
                b'event: step\n', b'data: {"delta_tokens": [2, 3]}\n', b'\n',
                b'event: step\n', b'data: {"delta_tokens": []}\n', b'\n',
                b'event: response\n', b'data: {"response": {"output_tokens": [1, 2, 3]}}\n', b'\n']
        response, clocked = mod.read_generate_stream(body, clock=lambda: next(ticks))
        self.assertEqual(response, {"output_tokens": [1, 2, 3]})
        self.assertEqual(clocked["ttft_seconds"], 1.0)
        self.assertEqual(clocked["first_event_tokens"], 1)
        self.assertEqual(clocked["decode_tokens"], 2)
        self.assertEqual(clocked["decode_seconds"], 0.5)
        self.assertEqual(mod.decode_rate(clocked), 4.0)

    def test_generate_stream_rejects_missing_response_and_error_events(self):
        with self.assertRaises(ValueError):
            mod.read_generate_stream([b'event: step\n', b'data: {"delta_tokens": [1]}\n'])
        with self.assertRaises(ValueError):
            mod.read_generate_stream([b'event: error\n', b'data: {"error": "boom"}\n'])

    def test_decode_rate_needs_a_window_and_tokens(self):
        self.assertAlmostEqual(mod.decode_rate(timing()), 127 / 4.0)
        self.assertIsNone(mod.decode_rate({"decode_seconds": None, "decode_tokens": 5}))
        self.assertIsNone(mod.decode_rate({"decode_seconds": 2.0, "decode_tokens": 0}))
        self.assertIsNone(mod.decode_rate({}))

    def test_workloads_are_deterministic_uniform_ids_with_distinct_runs(self):
        vocab = 248320
        runs = [mod.workload_ids(vocab, 512, r) for r in range(mod.WARMUPS + mod.MEASURED)]
        self.assertTrue(all(len(ids) == 512 for ids in runs))
        self.assertTrue(all(0 <= token < vocab for ids in runs for token in ids))
        self.assertEqual(runs[0], mod.workload_ids(vocab, 512, 0))
        self.assertEqual(len({tuple(ids[:8]) for ids in runs}), len(runs), "runs must not share a prefix")
        self.assertNotEqual(mod.workload_ids(vocab, 512, 0)[:8], mod.workload_ids(vocab, 2048, 0)[:8])
        self.assertEqual(len(mod.workload_ids(vocab, 8192, 4)), 8192)

    def test_pack_vocab_size_reads_flat_or_nested_config(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text(json.dumps({"vocab_size": 7}))
            self.assertEqual(mod.pack_vocab_size(root), 7)
            (root / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 9}}))
            self.assertEqual(mod.pack_vocab_size(root), 9)
            (root / "config.json").write_text(json.dumps({"vocab_size": "x"}))
            with self.assertRaises(ValueError):
                mod.pack_vocab_size(root)

    def test_summary_reports_identity_divergence_ratio_and_route_errors(self):
        total = mod.WARMUPS + mod.MEASURED
        d_cells = [cell(512, r, [1, 2], 4.0) for r in range(total)]
        m_cells = [cell(512, r, [1, 2] if r != 3 else [1, 9], 2.0, verified=5) for r in range(total)]
        direct = run("disabled", [traj("p1", [1, 2, 3]), traj("p2", [4, 5, 6])], d_cells)
        mtp = run("required", [traj("p1", [1, 2, 3], 7), traj("p2", [4, 9, 6], 7, errors=1)], m_cells)
        summary = mod.summarize({"disabled": direct, "required": mtp}, lengths=(512,))
        self.assertTrue(summary["prompt_ids_equal_across_modes"])
        self.assertEqual(summary["trajectories"]["identical"], 1)
        self.assertEqual(summary["trajectories"]["divergences"][0]["first_difference_index"], 1)
        self.assertEqual(summary["trajectories"]["mtp_verified_steps"], 14)
        self.assertEqual(summary["cells"]["512"]["measured_runs"], mod.MEASURED)
        self.assertAlmostEqual(summary["cells"]["512"]["mtp_over_direct_decode_ratio_median"], 2.0)
        self.assertEqual(summary["cells"]["512"]["mtp_accepted_tokens"], 3 * mod.MEASURED)
        self.assertEqual(summary["cell_token_identity"], {"identical": 2, "compared": 3})
        self.assertEqual(summary["route_errors"], 1)

    def test_summary_names_the_default_challenger_and_rejects_other_shapes(self):
        total = mod.WARMUPS + mod.MEASURED
        d_cells = [cell(512, r, [1, 2], 4.0) for r in range(total)]
        p_cells = [cell(512, r, [1, 2], 5.0) for r in range(total)]
        direct = run("disabled", [traj("p1", [1, 2, 3])], d_cells)
        default = run("default", [traj("p1", [1, 2, 3])], p_cells)
        summary = mod.summarize({"disabled": direct, "default": default}, lengths=(512,))
        self.assertEqual(summary["challenger_mode"], "default")
        self.assertAlmostEqual(summary["cells"]["512"]["default_over_direct_decode_ratio_median"], 0.8)
        self.assertEqual(summary["trajectories"]["default_verified_steps"], 0)
        self.assertNotIn("mtp_over_direct_decode_ratio_median", summary["cells"]["512"])
        with self.assertRaises(ValueError):
            mod.summarize({"disabled": direct}, lengths=(512,))
        with self.assertRaises(ValueError):
            mod.summarize({"disabled": direct, "default": default, "required": default}, lengths=(512,))

    def test_modes_option_requires_the_baseline_and_one_challenger(self):
        for modes in (["required"], ["disabled", "required", "default"], ["default", "required"]):
            done = subprocess.run([sys.executable, str(SCRIPT), "--dry-run", "--modes", *modes],
                                  capture_output=True, text=True)
            self.assertNotEqual(done.returncode, 0, modes)
        done = subprocess.run([sys.executable, str(SCRIPT), "--dry-run", "--modes", "disabled", "default"],
                              capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(done.stdout)["modes"], ["disabled", "default"])

    def test_summary_flags_prompt_id_mismatch_between_modes(self):
        cells = [cell(512, r, [1], 1.0) for r in range(mod.WARMUPS + mod.MEASURED)]
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
        self.assertGreater(contract["total_blocks"] * 16,
                           max(contract["lengths"]) + contract["output_tokens"])

    def test_best_repeat_keeps_the_lowest_latency_process_and_identity_is_checked(self):
        def process(ttft, tokens=(1, 2)):
            cells = [{"length": 512, "run": r, "warmup": r < mod.WARMUPS, "output_tokens": list(tokens),
                      "timing": {"ttft_seconds": ttft + (10 if r < mod.WARMUPS else 0)}}
                     for r in range(mod.WARMUPS + mod.MEASURED)]
            return {"trajectories": [{"output_tokens": [9]}], "cells": cells}
        slow, fast = process(1.8), process(1.2)
        self.assertIs(mod.best_repeat([slow, fast], (512,)), fast)
        self.assertIs(mod.best_repeat([fast, slow], (512,)), fast)
        self.assertEqual(mod.repeat_ttft_medians({"disabled": [slow, fast]}, (512,)),
                         {"disabled": [{"512": 1.8}, {"512": 1.2}]})
        same = mod.repeat_token_identity({"disabled": [slow, fast]})["disabled"]
        self.assertEqual(same, {"compared": 6, "identical": 6})
        drift = mod.repeat_token_identity({"disabled": [slow, process(1.2, (1, 3))]})["disabled"]
        self.assertEqual(drift, {"compared": 6, "identical": 1})

    def test_repeats_must_be_positive(self):
        done = subprocess.run([sys.executable, str(SCRIPT), "--dry-run", "--repeats", "0"],
                              capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        ok = subprocess.run([sys.executable, str(SCRIPT), "--dry-run", "--repeats", "3"],
                            capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(ok.stdout)["repeats_per_arm"], 3)

    def test_warm_pass_can_be_skipped_only_by_an_explicit_flag(self):
        done = subprocess.run([sys.executable, str(SCRIPT), "--dry-run", "--no-warm-pass"],
                              capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(done.stdout)["lengths"], [512, 2048, 8192])
        text = SCRIPT.read_text()
        self.assertIn("warm_pass", text)
        self.assertIn('label="warm"', text)

    def test_batched_verifier_is_explicit_scoped_to_the_mtp_arm_and_recorded(self):
        def dry(*extra):
            return subprocess.run([sys.executable, str(SCRIPT), "--dry-run", *extra],
                                  capture_output=True, text=True)
        self.assertEqual(json.loads(dry().stdout)["mtp_verifier"], "canonical")
        batched = dry("--mtp-verifier", "batched")
        self.assertEqual(json.loads(batched.stdout)["mtp_verifier"], "batched")
        block = dry("--mtp-verifier", "block")
        self.assertEqual(json.loads(block.stdout)["mtp_verifier"], "block")
        self.assertNotEqual(dry("--mtp-verifier", "legacy").returncode, 0)
        self.assertNotEqual(dry("--mtp-verifier", "batched", "--modes", "disabled", "default").returncode, 0)

    def test_ax_server_env_is_recorded_and_validated(self):
        def dry(*extra):
            return subprocess.run([sys.executable, str(SCRIPT), "--dry-run", *extra],
                                  capture_output=True, text=True)
        self.assertEqual(json.loads(dry().stdout)["ax_server_env"], {})
        done = dry("--ax-server-env", "AX_MLX_FLASH_NEXT_STICKY_FALLBACK=1")
        self.assertEqual(json.loads(done.stdout)["ax_server_env"],
                         {"AX_MLX_FLASH_NEXT_STICKY_FALLBACK": "1"})
        self.assertNotEqual(dry("--ax-server-env", "NOEQUALSSIGN").returncode, 0)
        self.assertNotEqual(dry("--ax-server-env", "=1").returncode, 0)

    def test_server_env_scrubs_inherited_overrides_but_keeps_an_explicit_one(self):
        native = mod.native
        import os
        previous = os.environ.get(mod.VERIFIER_ENV)
        os.environ[mod.VERIFIER_ENV] = "batched"
        try:
            self.assertNotIn(mod.VERIFIER_ENV, native.server_env())
            self.assertEqual(native.server_env({mod.VERIFIER_ENV: "batched"})[mod.VERIFIER_ENV], "batched")
        finally:
            if previous is None:
                os.environ.pop(mod.VERIFIER_ENV)
            else:
                os.environ[mod.VERIFIER_ENV] = previous

    def test_reported_schedule_must_match_the_requested_verifier(self):
        key = mod.VERIFIER_SCHEDULE_KEY
        response = {"output_tokens": [1], "prompt_tokens": [2], "client_timing": {},
                    "route": {"crossover_decisions": {key: 2}}}
        row = mod.slim(response, {}, {})
        self.assertEqual(row["verifier_schedule"], 2)
        mod.check_verifier_schedule(row, "required", "batched")
        # The direct arm never selects the batched verifier, so it reports canonical.
        mod.check_verifier_schedule({"verifier_schedule": 1}, "disabled", "batched")
        mod.check_verifier_schedule({"verifier_schedule": 1}, "required", "canonical")
        for bad_row, mode, verifier in (({"verifier_schedule": 1}, "required", "batched"),
                                        ({"verifier_schedule": 2}, "required", "canonical"),
                                        ({"verifier_schedule": 2}, "disabled", "batched"),
                                        ({}, "required", "canonical")):
            with self.assertRaises(RuntimeError):
                mod.check_verifier_schedule(bad_row, mode, verifier)
        self.assertIsNone(mod.slim({"output_tokens": [], "prompt_tokens": [], "client_timing": {}},
                                   {}, {})["verifier_schedule"])

    def test_live_run_requires_all_inputs(self):
        done = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("live run requires", done.stderr)


if __name__ == "__main__":
    unittest.main()
