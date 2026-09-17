"""Regression controls for incomplete or misattributed release evidence."""

import copy
import json

import tempfile
import unittest
from pathlib import Path

from curate_flash_next_evidence import curate, summarize


def throughput():
    return {
        "qualification": False, "binary_sha256": "a" * 64, "completed": True,
        "config": {"packs": ["4bit"], "prompt_tokens": [512],
                   "routes": ["ax_direct"], "measurement_repetitions": 2,
                   "generation_tokens": 128},
        "cells": [{"cell_id": "4bit/512/ax_direct", "samples": [
            {"warmup": False, "done": True, "generated_tokens": 128},
            {"warmup": False, "done": True, "generated_tokens": 128},
        ]}],
    }


class CurateFlashNextEvidenceTest(unittest.TestCase):
    def test_empty_or_duplicate_axes_cannot_produce_vacuous_pass(self):
        for axis in ("packs", "prompt_tokens", "routes"):
            for duplicate in (False, True):
                with self.subTest(axis=axis, duplicate=duplicate):
                    raw = throughput()
                    raw["config"][axis] = raw["config"][axis] * 2 if duplicate else []
                    raw["cells"] = []
                    with self.assertRaisesRegex(ValueError, axis):
                        summarize("throughput", raw)

    def test_invalid_counts_and_prompt_sizes_are_rejected(self):
        for field in ("measurement_repetitions", "generation_tokens", "warmup_repetitions"):
            bad_values = [-1, True, 1.5] + ([0] if field != "warmup_repetitions" else [])
            for value in bad_values:
                with self.subTest(field=field, value=value):
                    raw = throughput()
                    raw["config"][field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        summarize("throughput", raw)
        for value in (0, True, 512.0):
            raw = throughput()
            raw["config"]["prompt_tokens"] = [value]
            with self.assertRaisesRegex(ValueError, "prompt_tokens"):
                summarize("throughput", raw)

    def test_unlabeled_extra_sample_cannot_disappear(self):
        for warmup in (None, "false", 0):
            raw = throughput()
            raw["cells"][0]["samples"].append({"warmup": warmup, "generated_tokens": 0})
            result = summarize("throughput", raw)
            self.assertFalse(result["passed"])
            self.assertIn("unlabeled_sample", result["matrix"][0]["reasons"])
            self.assertEqual(result["matrix"][0]["total_samples"], 3)
            self.assertEqual(result["matrix"][0]["unlabeled_samples"], 1)

    def test_cell_route_cannot_spoof_reference_completion_exception(self):
        raw = throughput()
        raw["cells"][0]["route"] = "reference"
        for sample in raw["cells"][0]["samples"]:
            del sample["done"]
        with self.assertRaisesRegex(ValueError, "route does not match"):
            summarize("throughput", raw)
        del raw["cells"][0]["route"]
        self.assertFalse(summarize("throughput", raw)["passed"])

    def test_failed_skipped_and_missing_remain_distinct(self):
        raw = throughput()
        raw["config"]["prompt_tokens"] = [512, 2048, 8192]
        failed = raw["cells"][0]
        failed.update(failed=True, error="[METAL] GPU Timeout Error")
        raw["cells"].append({"cell_id": "4bit/2048/ax_direct", "skipped": True,
                             "skip_reason": "configured wall budget", "samples": []})
        result = summarize("throughput", raw)
        rows = {row["cell_id"]: row for row in result["matrix"]}
        self.assertEqual(rows["4bit/512/ax_direct"]["status"], "failed")
        self.assertEqual(rows["4bit/512/ax_direct"]["recorded_error"], failed["error"])
        self.assertEqual(rows["4bit/2048/ax_direct"]["status"], "skipped")
        self.assertEqual(rows["4bit/2048/ax_direct"]["recorded_skip_reason"], "configured wall budget")
        self.assertEqual(rows["4bit/8192/ax_direct"]["status"], "missing")
        self.assertEqual(result["status_counts"]["failed"], 1)
        self.assertFalse(result["passed"])

    def test_warmup_output_is_not_the_measured_output_contract(self):
        raw = throughput()
        raw["config"]["warmup_repetitions"] = 1
        raw["cells"][0]["samples"].append({"warmup": True, "generated_tokens": 1})
        self.assertTrue(summarize("throughput", raw)["passed"])

    def test_boolean_output_count_cannot_pass(self):
        raw = throughput()
        raw["config"]["generation_tokens"] = 1
        for sample in raw["cells"][0]["samples"]:
            sample["generated_tokens"] = True
        self.assertFalse(summarize("throughput", raw)["passed"])

    def test_unknown_cell_and_identity_mismatch_are_rejected(self):
        raw = throughput()
        raw["cells"].append({"cell_id": "4bit/999/ax_direct"})
        with self.assertRaisesRegex(ValueError, "unexpected throughput cell"):
            summarize("throughput", raw)
        for key, value in (("pack", "6bit"), ("prompt_tokens", 2048),
                           ("prompt_tokens", 512.0)):
            raw = throughput()
            raw["cells"][0][key] = value
            with self.assertRaisesRegex(ValueError, "does not match"):
                summarize("throughput", raw)

    def test_failed_flag_outranks_skip_and_curation_preserves_raw(self):
        raw = throughput()
        raw["cells"][0].update(failed=True, skipped=True, error="stream ended",
                                skip_reason="configured skip")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps(raw))
            before = path.read_bytes()
            result = curate("throughput", path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(result["evidence"], raw)
            row = result["summary"]["matrix"][0]
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["recorded_error"], "stream ended")
            self.assertEqual(row["recorded_skip_reason"], "configured skip")
            self.assertNotIn("timeout", str(row))
            self.assertFalse(result["release_ready"])
            self.assertFalse(result["qualification"])
            self.assertEqual(len(result["curator_sha256"]), 64)

    def test_recorded_campaign_keeps_failures_and_missing_coverage(self):
        path = Path(__file__).resolve().parents[1] / "benchmarks/results/flash-next-throughput-ab-m2-20260916.json"
        raw = json.loads(path.read_text())["evidence"]
        before = copy.deepcopy(raw)
        result = summarize("throughput", raw)
        self.assertEqual(raw, before)
        self.assertEqual(result["status_counts"], {
            "complete": 11, "failed": 2, "skipped": 0,
            "incomplete_output_or_samples": 0, "missing": 5,
        })
        self.assertFalse(result["coverage_complete"])
        self.assertFalse(result["passed"])

    def test_early_eos_does_not_complete_fixed_decode(self):
        raw = throughput()
        assert summarize("throughput", raw)["passed"]
        raw["cells"][0]["samples"][1]["generated_tokens"] = 18
        assert not summarize("throughput", raw)["passed"]


    def test_missing_cell_cannot_be_hidden_by_completed_flag(self):
        raw = throughput()
        raw["config"]["packs"].append("6bit")
        result = summarize("throughput", raw)
        assert not result["completed"]
        assert result["matrix"][1]["status"] == "missing"


    def test_duplicate_cells_rejected(self):
        raw = throughput()
        raw["cells"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize("throughput", raw)


    def test_warmups_cannot_replace_measured_samples(self):
        raw = throughput()
        raw["cells"][0]["samples"][1]["warmup"] = True
        assert not summarize("throughput", raw)["passed"]


    def test_failed_cell_and_missing_warmups_cannot_pass(self):
        raw = throughput()
        raw["cells"][0]["failed"] = True
        assert not summarize("throughput", raw)["passed"]
        del raw["cells"][0]["failed"]
        raw["config"]["warmup_repetitions"] = 2
        assert not summarize("throughput", raw)["passed"]


    def test_current_harness_cannot_overwrite_recorded_identity(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_path = Path(tmp.name)
        raw = throughput()
        raw["harness_sha256"] = "b" * 64
        raw["path"] = "/Users/private/model"
        path = tmp_path / "raw.json"
        path.write_text(json.dumps(raw))
        harness = tmp_path / "harness.py"
        harness.write_text("# changed after run\n")
        result = curate("throughput", path, harness)
        assert result["harness_provenance"]["verified"] is False
        assert result["evidence"]["harness_sha256"] == "b" * 64
        assert "/Users/" not in json.dumps(result)
        assert result["release_ready"] is False


    def test_fabricated_payload_rejected(self):
        raw = throughput()
        raw["fabricated"] = True
        with self.assertRaisesRegex(ValueError, "fabricated"):
            summarize("throughput", raw)


    def test_missing_mtp_cells_are_explicit(self):
        raw = {"qualification": False, "binary_sha256": "a" * 64,
               "completed": True, "packs": {}}
        result = summarize("mtp", raw)
        assert len(result["matrix"]) == 12
        assert not result["passed"]


    def test_completed_mtp_matrix_keeps_failed_verdict(self):
        control = {"identity_until_first_tie": True, "within_tolerance": True}
        pack = {"state": control, "runner": control,
                "tie_prompt": {"state": control, "runner": control}}
        raw = {"qualification": False, "binary_sha256": "a" * 64,
               "completed": True,
               "packs": {name: copy.deepcopy(pack) for name in ("2bit", "4bit", "6bit")}}
        raw["packs"]["4bit"]["tie_prompt"]["state"] = {
            "failed": True, "failure_log_tail": "logit bound exceeded"}
        result = summarize("mtp", raw)
        assert result["completed"]
        assert not result["passed"]
        assert sum(row["status"] == "failed" for row in result["matrix"]) == 1


    def test_qa_coverage_must_match(self):
        section = {"completed": True, "cases": [{"id": "one", "text": "answer"}]}
        raw = {"qualification": False, "binary_sha256": "a" * 64,
               "modes": {"disabled": copy.deepcopy(section), "required": copy.deepcopy(section)},
               "reference": copy.deepcopy(section)}
        raw["reference"]["cases"][0]["id"] = "different"
        with self.assertRaisesRegex(ValueError, "coverage"):
            summarize("qa", raw)


    def test_oracle_short_requests_do_not_inflate_scored_denominator(self):
        def side(permuted, accepted):
            return {"head_permuted": permuted, "proposed": 10, "accepted": accepted,
                    "greedy_identity": True, "requests": [
                        {"proposed": 10, "accepted": accepted, "too_short": False},
                        {"proposed": 1, "accepted": 0, "too_short": True},
                    ]}
        raw = {"qualification": False, "binary_sha256": "a" * 64, "completed": True,
               "real_head": side(False, 8), "permuted_head": side(True, 0),
               "thresholds": {"real_min": 0.7, "permuted_max": 0.1},
               "falsification_ok": True}
        result = summarize("head", raw)
        assert result["real_acceptance"] == 0.8
        assert result["scored_requests"]["real_head"] == 1
        raw["real_head"]["proposed"] = 9
        with self.assertRaisesRegex(ValueError, "proposal totals"):
            summarize("head", raw)


    def test_reference_samples_need_no_sse_done_marker(self):
        raw = throughput()
        raw["config"]["routes"] = ["reference"]
        cell = raw["cells"][0]
        cell["cell_id"] = "4bit/512/reference"
        cell["route"] = "reference"
        for sample in cell["samples"]:
            del sample["done"]
        assert summarize("throughput", raw)["completed"]

    def test_http_success_does_not_hide_failed_shutdown(self):
        raw = {"qualification": False, "binary_sha256": "a" * 64,
               "completed": True, "cells": [
                   {"pack": pack, "mode": mode, "passed": True, "exit_code": 0,
                    "pack_metadata_unchanged": True, "requests": [{}, {}]}
                   for pack in ("2bit", "4bit", "6bit")
                   for mode in ("disabled", "required")
               ]}
        assert summarize("http", raw)["passed"]
        raw["cells"][0]["exit_code"] = -15
        result = summarize("http", raw)
        assert result["completed"]
        assert not result["passed"]
        raw["cells"][0]["exit_code"] = 0
        raw["cells"][0]["direct_identity"] = False
        result = summarize("http", raw)
        assert not result["passed"]
        assert "direct_identity" in result["matrix"][0]["failed_checks"]


if __name__ == "__main__":
    unittest.main()
