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


if __name__ == "__main__":
    unittest.main()
