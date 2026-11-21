"""Regression tests for the paired native serving artifact reader."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CompareServingRunsTests(unittest.TestCase):
    def test_pairs_full_outputs_and_binds_traces_by_request_id(self):
        script = Path(__file__).with_name("compare_ax_serving_runs.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def artifact(name, request_id, output, *, fixed=True, budget=2, tpot=4, valid_stream=True):
                path = root / f"{name}.json"
                tokens = json.dumps(output, separators=(",", ":")).encode()
                path.write_text(json.dumps({
                    "schema_version": "ax.serving_benchmark.v1",
                    "target": {"input_kind": "tokens", "model_id": "qwen"},
                    "sampling": {"temperature": 0, "top_p": 1, "top_k": 0, "ignore_eos": fixed},
                    "corpus": {"sha256": "same"}, "load": {"measured_requests": 1}, "summary": {},
                    "observations": [{
                        "phase": "measured", "prompt_id": "code", "category": "code", "ok": True,
                        "input_tokens": 1, "prompt_token_ids": [1], "request_id": request_id,
                        "output_tokens": len(output), "output_token_ids": output,
                        "max_output_tokens": budget, "client_tpot_ms": tpot,
                        "output_identity": {"kind": "token_ids", "count": len(output),
                                            "sha256": hashlib.sha256(tokens).hexdigest() if valid_stream else "missing"},
                    }],
                }))
                return path

            def compare(left, right, *extra):
                return subprocess.run([
                    sys.executable, script, "--reference", "solo", "--arm", f"solo={left}",
                    "--arm", f"packed={right}", *extra,
                ], capture_output=True, text=True, check=False)

            solo = artifact("solo", 8, [4, 5])
            packed = artifact("packed", 9, [4, 6], tpot=2)
            trace = root / "server.log"
            trace.write_text(
                "dflash2_round request_id=9 generated=1 emitted=1 cohort_lanes=2 "
                "keep=1 window=2 proposed=1 accepted=0 recomputed=0\n"
                "dflash2_margin request_id=9 position=1 token=6 left_id=5 right_id=6 "
                "left_bits=1065353216 right_bits=1073741824\n"
            )
            report = json.loads(compare(solo, packed, "--trace", f"{packed}={trace}").stdout)
            result = report["comparisons"]["packed"]
            self.assertEqual(result["exact"], 0)
            self.assertEqual(result["paired_decode_time_ratio_median"], 0.5)
            self.assertIsNone(result["identical_output_decode_time_ratio_median"])
            self.assertEqual(result["pairs"][0]["candidate_differences_from_first_reference"], [
                {"index": 1, "reference_token": 5, "candidate_token": 6},
            ])
            rounds = report["diagnostic_traces"][str(packed)]["code"]
            self.assertEqual(rounds["cohort_rounds"], 1)
            self.assertEqual(rounds["output_tokens_outside_traced_rounds"], 1)
            self.assertEqual(rounds["logit_probes"][0]["left_minus_right"], -1.0)

            repeat = artifact("repeat", 12, [4, 5])
            report = json.loads(compare(solo, packed, "--arm", f"packed={repeat}").stdout)
            self.assertFalse(report["comparisons"]["packed"]["pairs"][0]["candidate_repeatable"])

            short = artifact("short", 10, [4])
            self.assertIn("unexpected output length", compare(solo, short).stderr)
            wrong = artifact("wrong", 11, [4, 5], budget=3, fixed=False)
            self.assertIn("workload differs", compare(solo, wrong).stderr)
            incomplete = artifact("incomplete", 12, [4, 5], valid_stream=False)
            self.assertIn("stream token identity differs", compare(solo, incomplete).stderr)
            self.assertIn("no DFlash2 rounds", compare(solo, packed, "--trace", f"{solo}={trace}").stderr)
