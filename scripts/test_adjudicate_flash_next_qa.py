#!/usr/bin/env python3
"""Mutation-control tests for the Flash Next QA adjudication layer."""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import adjudicate_flash_next_qa as mod  # noqa: E402

VALIDATION_PATH = ROOT / "benchmarks/results/flash-next-installed-full-qa-4bit-c101-m2-20260917.json"
ADJUDICATION_PATH = (
    ROOT / "benchmarks/results/flash-next-installed-full-qa-4bit-c101-m2-20260917.adjudication.json"
)
FROZEN_ITEMS_PATH = (
    ROOT
    / "benchmarks/results/qualification/2026-09-19-flash-next-mxfp4-installed-qa/full-qa-items.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class AdjudicateFlashNextQaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.validation = _load(VALIDATION_PATH)
        self.adjudication = _load(ADJUDICATION_PATH)
        self.frozen_items_sha = mod._sha256(FROZEN_ITEMS_PATH)

    def verdict(self, validation=None, adjudication=None):
        return mod.adjudicated_verdict(
            validation if validation is not None else self.validation,
            adjudication if adjudication is not None else self.adjudication,
            frozen_items_sha256=self.frozen_items_sha,
        )

    def test_tracked_record_closes_with_retained_failures(self) -> None:
        result = self.verdict()
        self.assertEqual(result["verdict"], "passed_with_retained")
        self.assertEqual(result["problems"], [])
        self.assertEqual(
            result["retained_failures"],
            ["format_csv_pair", "knowledge_water_formula", "science_gravity_earth"],
        )
        self.assertFalse(result["qualification"])
        self.assertFalse(result["release_ready"])

    def test_uncovered_failure_is_rejected(self) -> None:
        adjudication = copy.deepcopy(self.adjudication)
        adjudication["cases"] = [
            case for case in adjudication["cases"] if case["id"] != "science_gravity_earth"
        ]
        result = self.verdict(adjudication=adjudication)
        self.assertEqual(result["verdict"], "failed")
        self.assertTrue(
            any("science_gravity_earth" in p for p in result["problems"]),
            result["problems"],
        )

    def test_runtime_cause_cannot_be_retained(self) -> None:
        adjudication = copy.deepcopy(self.adjudication)
        for case in adjudication["cases"]:
            case["runtime_cause_established"] = True
        result = self.verdict(adjudication=adjudication)
        self.assertEqual(result["verdict"], "failed")
        self.assertTrue(any("runtime cause" in p for p in result["problems"]))

    def test_sha_binding_mismatch_is_rejected(self) -> None:
        adjudication = copy.deepcopy(self.adjudication)
        adjudication["source_qa"]["sha256"] = "0" * 64
        result = self.verdict(adjudication=adjudication)
        self.assertEqual(result["verdict"], "failed")
        self.assertTrue(any("source_qa" in p for p in result["problems"]))

        adjudication = copy.deepcopy(self.adjudication)
        adjudication["source_items"]["sha256"] = "0" * 64
        result = self.verdict(adjudication=adjudication)
        self.assertTrue(any("source_items" in p for p in result["problems"]))

    def test_adjudicating_a_passing_id_is_rejected(self) -> None:
        adjudication = copy.deepcopy(self.adjudication)
        adjudication["cases"].append(
            {
                "id": "instruction_sort_numbers",
                "disposition": "did not fail",
                "runtime_cause_established": False,
            }
        )
        result = self.verdict(adjudication=adjudication)
        self.assertEqual(result["verdict"], "failed")
        self.assertTrue(any("did not fail" in p for p in result["problems"]))

    def test_identity_flags_and_accounting_are_enforced(self) -> None:
        for key in ("text_identity", "checker_identity"):
            validation = copy.deepcopy(self.validation)
            validation[key] = False
            self.assertEqual(self.verdict(validation=validation)["verdict"], "failed")

        validation = copy.deepcopy(self.validation)
        validation["modes"]["disabled"]["hard_pass"] = 103
        result = self.verdict(validation=validation)
        self.assertEqual(result["verdict"], "failed")
        self.assertTrue(any("accounting" in p for p in result["problems"]))

    def test_threshold_or_promotion_changes_are_rejected(self) -> None:
        adjudication = copy.deepcopy(self.adjudication)
        adjudication["threshold_changed"] = True
        self.assertEqual(self.verdict(adjudication=adjudication)["verdict"], "failed")

        adjudication = copy.deepcopy(self.adjudication)
        adjudication["release_ready"] = True
        result = self.verdict(adjudication=adjudication)
        self.assertEqual(result["verdict"], "failed")

    def test_missing_disposition_is_rejected(self) -> None:
        adjudication = copy.deepcopy(self.adjudication)
        adjudication["cases"][0]["disposition"] = ""
        result = self.verdict(adjudication=adjudication)
        self.assertEqual(result["verdict"], "failed")
        self.assertTrue(any("disposition" in p for p in result["problems"]))

    def test_absent_and_nonboolean_flags_are_rejected(self) -> None:
        for record, keys, expected in (
            ("validation", ("completed", "text_identity", "checker_identity"), True),
            ("validation", ("qualification", "release_ready"), False),
            ("adjudication", ("threshold_changed", "qualification", "release_ready"), False),
        ):
            for key in keys:
                for value in (None, "false", "true", 0, 1, not expected):
                    with self.subTest(record=record, key=key, value=value):
                        v, a = copy.deepcopy(self.validation), copy.deepcopy(self.adjudication)
                        target = v if record == "validation" else a
                        if value is None:
                            target.pop(key)
                        else:
                            target[key] = value
                        self.assertEqual(self.verdict(v, a)["verdict"], "failed")

    def test_modes_must_be_complete_and_named(self) -> None:
        for modes in (None, [], {}, {"disabled": self.validation["modes"]["disabled"]},
                      {"direct": self.validation["modes"]["disabled"],
                       "required": self.validation["modes"]["required"]},
                      {**self.validation["modes"], "extra": {}}):
            with self.subTest(modes=modes):
                validation = copy.deepcopy(self.validation)
                validation["modes"] = modes
                self.assertEqual(self.verdict(validation=validation)["verdict"], "failed")

    def test_missing_both_hashes_cannot_bind(self) -> None:
        for value in (None, "", "short", "z" * 64, 0, [], {}):
            with self.subTest(value=value):
                v, a = copy.deepcopy(self.validation), copy.deepcopy(self.adjudication)
                v["raw_sha256"] = value
                a["source_qa"]["sha256"] = value
                self.assertEqual(self.verdict(v, a)["verdict"], "failed")
                a = copy.deepcopy(self.adjudication)
                a["source_items"]["sha256"] = value
                result = mod.adjudicated_verdict(v, a, frozen_items_sha256=value)
                self.assertEqual(result["verdict"], "failed")
        self.assertEqual(mod.adjudicated_verdict(self.validation, self.adjudication)["verdict"], "failed")

    def test_malformed_records_fail_without_crashing(self) -> None:
        for value in (None, [], 0, "record"):
            self.assertEqual(mod.adjudicated_verdict(value, self.adjudication)["verdict"], "failed")
            self.assertEqual(mod.adjudicated_verdict(self.validation, value)["verdict"], "failed")
            for key in ("source_qa", "source_items", "cases"):
                a = copy.deepcopy(self.adjudication)
                a[key] = value
                self.assertEqual(self.verdict(adjudication=a)["verdict"], "failed")

    def test_duplicate_failures_and_adjudications_are_rejected(self) -> None:
        v = copy.deepcopy(self.validation)
        mode = v["modes"]["disabled"]
        mode["failed_ids"].append(mode["failed_ids"][0])
        mode["hard_pass"] -= 1  # Counts still close; repeated IDs do not.
        self.assertEqual(self.verdict(validation=v)["verdict"], "failed")
        a = copy.deepcopy(self.adjudication)
        a["cases"].append(copy.deepcopy(a["cases"][0]))
        self.assertEqual(self.verdict(adjudication=a)["verdict"], "failed")

    def test_counts_require_nonnegative_integers(self) -> None:
        for value in (True, False, -1, 1.0, "105", None):
            for key in ("cases_per_mode", "normal_stops"):
                v = copy.deepcopy(self.validation)
                v[key] = value
                self.assertEqual(self.verdict(validation=v)["verdict"], "failed")
            v = copy.deepcopy(self.validation)
            v["modes"]["disabled"]["hard_pass"] = value
            self.assertEqual(self.verdict(validation=v)["verdict"], "failed")

    def test_failed_ids_and_dispositions_require_nonempty_strings(self) -> None:
        for value in (None, False, 7, [], {}, "", " "):
            v = copy.deepcopy(self.validation)
            v["modes"]["disabled"]["failed_ids"][0] = value
            self.assertEqual(self.verdict(validation=v)["verdict"], "failed")
            for key in ("id", "disposition"):
                a = copy.deepcopy(self.adjudication)
                a["cases"][0][key] = value
                self.assertEqual(self.verdict(adjudication=a)["verdict"], "failed")
        a = copy.deepcopy(self.adjudication)
        del a["cases"][0]["runtime_cause_established"]
        self.assertEqual(self.verdict(adjudication=a)["verdict"], "failed")
        for value in (0, "false", [], None):
            a["cases"][0]["runtime_cause_established"] = value
            self.assertEqual(self.verdict(adjudication=a)["verdict"], "failed")

    def test_all_pass_record_still_requires_bindings(self) -> None:
        v, a = copy.deepcopy(self.validation), copy.deepcopy(self.adjudication)
        for mode in v["modes"].values():
            mode.update(hard_pass=v["cases_per_mode"], failed_ids=[])
        a["cases"] = []
        self.assertEqual(self.verdict(v, a)["verdict"], "passed")
        del v["raw_sha256"]
        del a["source_qa"]
        self.assertEqual(self.verdict(v, a)["verdict"], "failed")

    def test_schema_and_failed_id_array_are_required(self) -> None:
        for record in ("validation", "adjudication"):
            v, a = copy.deepcopy(self.validation), copy.deepcopy(self.adjudication)
            del (v if record == "validation" else a)["schema"]
            self.assertEqual(self.verdict(v, a)["verdict"], "failed")
        v = copy.deepcopy(self.validation)
        del v["modes"]["required"]["failed_ids"]
        self.assertEqual(self.verdict(validation=v)["verdict"], "failed")

    def test_cli_rejects_malformed_json_and_incomplete_records(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ax-adjudication-") as directory:
            root = Path(directory)
            v_path, a_path = root / "validation.json", root / "adjudication.json"
            a_path.write_text(json.dumps(self.adjudication), encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts/adjudicate_flash_next_qa.py"),
                       "--validation", str(v_path), "--adjudication", str(a_path),
                       "--frozen-items-sha", self.frozen_items_sha, "--json"]
            v_path.write_text("{", encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)
            v_path.write_text(json.dumps(self.validation), encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["verdict"], "passed_with_retained")
            v = copy.deepcopy(self.validation)
            del v["modes"]["required"]
            v_path.write_text(json.dumps(v), encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(result.stdout)["verdict"], "failed")

    def test_mode_failures_must_agree_with_checker_identity(self) -> None:
        v = copy.deepcopy(self.validation)
        mode = v["modes"]["required"]
        mode["failed_ids"].pop()
        mode["hard_pass"] += 1
        result = self.verdict(validation=v)
        self.assertEqual(result["verdict"], "failed")
        self.assertTrue(any("checker_identity" in p for p in result["problems"]))


if __name__ == "__main__":
    unittest.main()
