"""Offline controls for Qwen 3.8 27B paired replay validation."""

import copy
import json
import unittest

from scripts import verify_qwen38_27b_mtp_margin_replay as replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "case_id": "sample",
            "prompt_tokens": [50, 51],
            "max_output_tokens": 3,
            "ignore_eos": True,
            "direct_tokens": [9, 10, 11],
            "mtp_tokens": [9, 20, 21],
        }
        self.report = {
            "case_id": self.case["case_id"],
            "prompt_tokens": self.case["prompt_tokens"],
            "max_output_tokens": self.case["max_output_tokens"],
            "ignore_eos": self.case["ignore_eos"],
            "manifest_sha256": replay.MANIFEST_SHA256,
            "direct": {"tokens": self.case["direct_tokens"]},
            "mtp": {"tokens": self.case["mtp_tokens"]},
            "bound_to_recorded_outputs": True,
            "complete_aligned_margins": True,
            "first_unpaired_index": None,
            "differences": [],
        }
        for index, source in ((1, "verifier"), (2, "rollback_replay")):
            self.report["differences"].append({
                "index": index,
                "same_prior_tokens": index == 1,
                "direct_token": self.case["direct_tokens"][index],
                "mtp_token": self.case["mtp_tokens"][index],
                "direct_row": {"Ok": {
                    "source": "target", "emitted_logit": 6.0, "rival_logit": 5.875,
                    "paired_gap": 0.125, "runner_up_gap": 0.125,
                }},
                "mtp_row": {"Ok": {
                    "source": source, "emitted_logit": 5.0, "rival_logit": 5.0,
                    "paired_gap": 0.0, "runner_up_gap": 0.0,
                }},
                "max_paired_logit_gap": 0.125,
            })

    def verify(self, report):
        return replay.verify(
            json.dumps([self.case]).encode(), json.dumps(report).encode(), rust_log=False,
        )

    def test_report_preserves_first_split_and_row_provenance(self):
        summary = self.verify(self.report)
        self.assertEqual(summary["cases_verified"], 1)
        self.assertEqual(summary["aligned_differences_scored"], 2)
        self.assertEqual(summary["cases"][0]["first_differing_index"], 1)
        self.assertEqual(summary["cases"][0]["first_difference_same_prefix_gap"], 0.125)
        self.assertEqual(summary["cases"][0]["row_sources"], {
            "direct:target": 2, "mtp:verifier": 1, "mtp:rollback_replay": 1,
        })

    def test_failed_native_run_cannot_prove_a_replay(self):
        case = json.dumps([self.case]).encode()
        line = f"test runner::tests::replay ... {json.dumps(self.report)}\nok\n"
        with self.assertRaisesRegex(ValueError, "missing successful Rust test result"):
            replay.verify(case, line.encode(), rust_log=True)
        result = replay.verify(
            case, (line + "test result: ok. 1 passed; 0 failed;").encode(), rust_log=True,
        )
        self.assertEqual(result["cases_verified"], 1)

    def test_rejects_missing_divergence_and_recomputed_gap_mismatch(self):
        report = copy.deepcopy(self.report)
        report["differences"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete aligned divergence"):
            self.verify(report)
        report = copy.deepcopy(self.report)
        report["differences"][0]["mtp_row"]["Ok"]["paired_gap"] = 0.5
        with self.assertRaisesRegex(ValueError, "invalid paired logit margin"):
            self.verify(report)

    def test_rejects_different_output_and_non_target_direct_rows(self):
        report = copy.deepcopy(self.report)
        report["direct"]["tokens"][-1] += 1
        with self.assertRaisesRegex(ValueError, "does not match the measured"):
            self.verify(report)
        report = copy.deepcopy(self.report)
        report["differences"][0]["direct_row"]["Ok"]["source"] = "rollback_replay"
        with self.assertRaisesRegex(ValueError, "invalid logit row source"):
            self.verify(report)
