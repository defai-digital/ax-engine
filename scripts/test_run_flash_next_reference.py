#!/usr/bin/env python3
"""Weight-free regressions for the pinned MLX-VLM Flash Next reference runner."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

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

    def test_fixed_token_mode_keeps_eos_as_an_ordinary_token(self):
        generated, terminal, times = mod.collect(iter([5, 248044, 7, 8]), 4, set(), Clock([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual((generated, terminal, len(times)), ([5, 248044, 7, 8], None, 4))

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
        self.assertNotIn("memory_trace", contract)

    def test_diagnostic_is_bounded_to_qa_and_cannot_claim_qualification(self):
        command = [sys.executable, str(SCRIPT), "--dry-run", "--diagnostic"]
        contract = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual((contract["phases"], contract["qa_items"]), (["qa"], 4))
        self.assertTrue(contract["diagnostic_only"])
        self.assertFalse(contract["release_ready"])
        self.assertEqual(contract["memory_trace"], mod.DIAGNOSTIC_TRACE)
        for options in (["--phases", "matrix"], ["--lengths", "512"],
                        ["--limit-items", "5"], ["--limit-items", "0"]):
            with self.subTest(options=options):
                result = subprocess.run(command + options, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("diagnostic", result.stderr)

    def test_ordinary_mode_still_rejects_m2(self):
        args = [str(SCRIPT), "--model-dir", "model", "--inventory", "inventory.json",
                "--output", "result"]
        with (patch.object(sys, "argv", args),
              patch.object(mod.platform, "system", return_value="Darwin"),
              patch.object(mod.subprocess, "check_output", side_effect=["206158430208", "Apple M2 Ultra"]),
              patch.object(mod.importlib.metadata, "version") as version,
              patch("sys.stderr"),
              self.assertRaises(SystemExit) as rejected):
            mod.main()
        self.assertEqual(rejected.exception.code, 2)
        version.assert_not_called()

    def test_diagnostic_admits_only_the_known_m2_studio(self):
        args = [str(SCRIPT), "--model-dir", "model", "--inventory", "inventory.json",
                "--output", "result", "--diagnostic"]
        with (patch.object(sys, "argv", args), patch.object(mod.platform, "system", return_value="Darwin"),
              patch.object(mod.subprocess, "check_output") as sysctl,
              patch.object(mod.importlib.metadata, "version", side_effect=RuntimeError("runtime check")) as version,
              patch("sys.stderr")):
            sysctl.side_effect = ["206158430208", "Apple M2 Ultra", "unrecognized"]
            with self.assertRaises(SystemExit):
                mod.main()
            version.assert_not_called()
            sysctl.side_effect = ["206158430208", "Apple M2 Ultra", "Mac14,14"]
            with self.assertRaisesRegex(RuntimeError, "runtime check"):
                mod.main()
            version.assert_called_once()

    def test_materialization_observer_keeps_original_batches(self):
        model = SimpleNamespace(parameters=lambda: [(str(i), i) for i in range(5)])
        runtime = Mock()
        observer = Mock()
        with patch.dict(sys.modules, {"mlx.utils": SimpleNamespace(tree_flatten=lambda arrays: arrays)}):
            self.assertEqual(mod.materialize(model, runtime, batch=2, observe=observer), 5)
        self.assertEqual([call.args for call in runtime.eval.call_args_list], [(0, 1), (2, 3), (4,)])
        self.assertEqual([call.kwargs["arrays_materialized"] for call in observer.call_args_list], [2, 4, 5])

    def test_first_token_observer_keeps_original_mlx_operations(self):
        def run(observer):
            runtime = Mock()
            model = MagicMock()
            runtime.argmax.return_value.item.return_value = 17
            stream = mod.token_stream(model, runtime, [1, 2, 3, 4], chunk=2, observe=observer)
            first = next(stream)
            stream.close()
            return first, [call[0] for call in runtime.mock_calls]

        observer = Mock()
        self.assertEqual(run(observer), run(None))
        self.assertEqual([call.args[0] for call in observer.call_args_list],
                         ["prefill-chunk-complete", "prefill-chunk-complete",
                          "first-token-start", "first-token-ready"])
        self.assertEqual([call.kwargs["prefill_tokens"] for call in observer.call_args_list[:2]], [2, 3])

    def test_pressure_stop_retains_the_checkpoint_and_separate_counters(self):
        runtime = Mock()
        runtime.get_active_memory.return_value = 40
        runtime.get_cache_memory.return_value = 7
        runtime.get_peak_memory.return_value = 65
        with TemporaryDirectory() as folder:
            trace = Path(folder) / mod.DIAGNOSTIC_TRACE
            with (patch.object(mod, "diagnostic_pressure", return_value=4),
                  self.assertRaisesRegex(RuntimeError, "diagnostic stopped at weights-ready")):
                mod.record_startup_memory(trace, runtime, "weights-ready", arrays_materialized=2326)
            row = json.loads(trace.read_text())
        self.assertEqual(row["phase"], "weights-ready")
        self.assertEqual(row["arrays_materialized"], 2326)
        self.assertEqual((row["mlx_active_bytes"], row["mlx_cache_bytes"], row["mlx_peak_bytes"]),
                         (40, 7, 65))
        self.assertEqual(row["system_pressure_level"], 4)
        self.assertIn("stop_reason", row)
        self.assertEqual([call[0] for call in runtime.mock_calls],
                         ["get_active_memory", "get_cache_memory", "get_peak_memory"])

    def test_live_run_requires_inputs_and_the_ax_matrix_for_the_matrix_phase(self):
        done = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("live run requires", done.stderr)

    def test_missing_prepared_qa_input_fails_before_reference_runtime_setup(self):
        items = (mod.qa.FROZEN / "full-qa-items.json").read_bytes()
        with TemporaryDirectory() as folder:
            frozen = Path(folder)
            (frozen / "full-qa-items.json").write_bytes(items)
            args = [str(SCRIPT), "--model-dir", folder, "--inventory", "inventory.json",
                    "--output", "result", "--phases", "qa"]
            with (patch.object(sys, "argv", args), patch.object(mod.qa, "FROZEN", frozen),
                  patch.object(mod.platform, "system", return_value="Darwin"),
                  patch.object(mod.subprocess, "check_output", side_effect=["137438953472", "Apple M5 Max"]),
                  patch.object(mod.native, "is_target_host", return_value=True),
                  patch.object(mod.importlib.metadata, "version") as version):
                with self.assertRaisesRegex(FileNotFoundError, "full-qa-inputs.json"):
                    mod.main()
            version.assert_not_called()


if __name__ == "__main__":
    unittest.main()
