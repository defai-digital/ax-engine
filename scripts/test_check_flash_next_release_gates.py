#!/usr/bin/env python3
"""Weight-free regressions for the Flash Next release-gate evaluator."""
from __future__ import annotations

import copy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("check_flash_next_release_gates.py")
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("flash_release_gates", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
import qualify_qwen38_flash_next as qualify  # noqa: E402

TH = json.loads(mod.THRESHOLDS.read_text())
GIB_KIB = 1024 * 1024
FAILURES = ["format_csv_pair", "format_status_colon", "instruction_alphabet_first", "knowledge_water_formula"]
HARDWARE = {"form_factor": "MacBook Pro", "soc": "Apple M5 Max", "memory_bytes": 128 * 1024**3,
            "os": "27.2.1"}
CELL_HASHES = {f"cell-{length}-{run}": f"{length * 10 + run:064x}" for length in (512, 2048, 8192)
               for run in range(5)}
TRAJECTORY_HASHES = {f"traj-{index:02d}": f"{index:064x}" for index in range(16)}
COMMON = {"server_sha256": "a" * 64, "inventory_sha256": "b" * 64, "hardware": HARDWARE,
          "qualification": False, "release_ready": False,
          "host_quiet": {"cpu_idle_percent": 95.0}, "host_idle_after": 93.0}
COHORT_PATH = mod.qa_cohort.FROZEN / "full-qa-items.json"
COHORT_IDS = [item["id"] for item in mod.qa_cohort.load_items(COHORT_PATH)]
SIGNED_REFERENCE = (SCRIPT.parent.parent / "benchmarks/results/qualification"
                    / "2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/reference/result.json")
REFERENCE_QA = json.loads(SIGNED_REFERENCE.read_text())["qa"]


def ax_cells(decode_seconds=7.0, ttft=(4.0, 16.0, 70.0), tokens=128):
    cells = []
    for length, first in zip((512, 2048, 8192), ttft):
        for run in range(5):
            cells.append({"length": length, "run": run, "warmup": run < 2, "verified_steps": 0.0,
                          "status": "finished",
                          "output_tokens": [0] * tokens,
                          "timing": {"ttft_seconds": first, "decode_seconds": decode_seconds,
                                     "first_event_tokens": 1, "decode_tokens": 127}})
    return cells


def ref_cells(decode_seconds=7.0, ttft=(4.0, 16.0, 70.0)):
    cells = []
    for length, first in zip((512, 2048, 8192), ttft):
        for run in range(5):
            cells.append({"length": length, "run": run, "warmup": run < 2, "generated_tokens": 128,
                          "generated_ids": [0] * 128,
                          "ttft_seconds": first, "decode_seconds": decode_seconds, "decode_tokens": 127})
    return cells


def good():
    arm = {"completed": 105, "passed": 101, "quality_failures": list(FAILURES), "normal_stops": True,
           "verified_steps": 0.0, "route_errors": 0.0, "ready_seconds": 33.0,
           "peak_rss_kib": 80 * GIB_KIB, "long_context_elapsed_seconds": 270.0}
    qa = {**COMMON, "items": len(COHORT_IDS), "items_sha256": mod.sha256_file(COHORT_PATH),
          "grades": {mode: {item_id: {"passed": item_id not in FAILURES} for item_id in COHORT_IDS}
                     for mode in mod.AX_ARMS},
          "summary": {"modes": {"disabled": copy.deepcopy(arm), "default": copy.deepcopy(arm)},
                                "text_identity_vs_disabled": {"default": {
                                    "matching_text_pairs": 105, "differing_text_ids": []}}}}
    lifecycle = {**COMMON, "lifecycle_passed": True, "runs": [
        {"mode": mode, "ready_seconds": 33.0,
         "actions": [{"mode": mode, "action": action,
                      "drained": dict.fromkeys(sorted(mod.LIFECYCLE_COUNTERS), 0.0)}
                     for action in mod.LIFECYCLE_ACTIONS], "final_verified_steps": 0.0,
         "exit_code_after_stop": 0} for mode in ("disabled", "default")]}
    matrix = {**COMMON, "repeats_per_arm": 2,
              "repeat_token_identity": {mode: {"compared": 31, "identical": 31}
                                        for mode in ("disabled", "default")},
              "runs": {mode: {"mode": mode, "ready_seconds": 33.0,
                              "trajectories": [
                                  {"index": index, "verified_steps": 0.0,
                                   "status": "finished", "output_tokens": [0]}
                                  for index in range(16)],
                              "prompt_id_hashes": {**TRAJECTORY_HASHES, **CELL_HASHES},
                              "cells": ax_cells()} for mode in ("disabled", "default")}}
    reference = {**{key: value for key, value in COMMON.items() if key != "server_sha256"},
                 "reference": {"runtime": "mlx-vlm", "commit": mod.REFERENCE_COMMIT,
                               "versions": dict(mod.REFERENCE_VERSIONS),
                               "language_sha256": mod.REFERENCE_LANGUAGE_SHA256},
                 "completed": True, "phase": "complete", "runtime_files_bound": 1750,
                 "versions": dict(mod.REFERENCE_VERSIONS), "language_sha256": mod.REFERENCE_LANGUAGE_SHA256,
                 **mod.REFERENCE_EXECUTION_SETTINGS,
                 "qa": copy.deepcopy(REFERENCE_QA),
                 "matrix": {"cells": ref_cells(), "prompt_id_hashes": dict(CELL_HASHES)}}
    delivery = {**COMMON, "completed": True, "fresh_cache": True, "revision": TH["target"]["pack_revision"],
                "download": {"exit_code": 0, "elapsed_seconds": 900.0}, "previous_attempts": [],
                "verification": {"members_verified": 49, "bytes_verified": 132261877478,
                                 "all_sizes_and_sha256_match": True, "native_manifest_runtime_ready": True},
                "native_support": {"passed": True, "modes": ["disabled", "required", "default"],
                                   "page_cache": "not controlled",
                                   "readiness_seconds": {"disabled": 120.0, "required": 70.0,
                                                         "default": 40.0}}}
    return {"delivery": delivery, "qa": qa, "lifecycle": lifecycle, "matrix": matrix, "reference": reference}


def raw_repeats(results):
    return {mode: [copy.deepcopy(run), copy.deepcopy(run)]
            for mode, run in results["matrix"]["runs"].items()}


def raw_delivery_controls(delivery):
    support = delivery["native_support"]
    controls = {key: delivery.get(key) for key in ("server_sha256", "inventory_sha256", "revision")}
    controls.update(functional_controls_passed=support["passed"], modes=support["modes"],
                    readiness_seconds=support["readiness_seconds"], page_cache=support.get("page_cache"))
    raw = json.dumps(controls).encode()
    support["result_sha256"] = sha256(raw).hexdigest()
    return raw


def verdict(results, repeats=None):
    if repeats is None:
        repeats = raw_repeats(results)
    controls = raw_delivery_controls(results["delivery"])
    return mod.evaluate(results, TH, qualify.contract(), repeats, controls)


def failed(results):
    return {g["gate"] for g in verdict(results)["gates"] if not g["passed"]}


class ReleaseGateTests(unittest.TestCase):
    def test_complete_good_evidence_passes_every_gate(self):
        result = verdict(good())
        self.assertTrue(result["release_ready"], [g for g in result["gates"] if not g["passed"]])
        self.assertEqual({g["gate"] for g in result["gates"]},
                         {"delivery", "integrity", "quality", "lifecycle", "memory", "readiness",
                          "long_lookup", "reference_relative", "mtp_policy"})

    def test_a_failure_the_reference_passes_fails_quality(self):
        r = good()
        item_id = "math_gcd_18_24"
        r["qa"]["grades"]["default"][item_id]["passed"] = False
        default = r["qa"]["summary"]["modes"]["default"]
        default["quality_failures"].append(item_id)
        default["passed"] -= 1
        result = next(g for g in verdict(r)["gates"] if g["gate"] == "quality")
        self.assertFalse(result["passed"])
        self.assertEqual(result["ax_only_failures"], {"disabled": [], "default": [item_id]})

    def test_default_text_that_differs_from_direct_fails_quality(self):
        r = good()
        r["qa"]["summary"]["text_identity_vs_disabled"]["default"]["differing_text_ids"] = ["x"]
        self.assertEqual(failed(r), {"quality"})

    def test_an_incomplete_reference_fails_quality_and_the_matrix_gate(self):
        r = good()
        r["reference"].update(completed=False, phase="qa", qa={"rows": [{}]})
        del r["reference"]["runtime_files_bound"]
        del r["reference"]["matrix"]
        result = {g["gate"]: g for g in verdict(r)["gates"]}
        self.assertEqual({name for name, gate in result.items() if not gate["passed"]},
                         {"quality", "reference_relative"})
        self.assertIsNone(result["quality"]["reference_failures"])
        self.assertEqual(result["quality"]["ax_only_failures"], {"disabled": None, "default": None})
        self.assertNotIn("failures the reference passes", result["quality"]["detail"])
        self.assertNotIn("prompt token IDs differ", result["reference_relative"]["detail"])

    def test_altered_reference_provenance_fails_integrity(self):
        for label, mutate in (
            ("missing description", lambda ref: ref.pop("reference")),
            ("other commit", lambda ref: ref["reference"].update(commit="0" * 40)),
            ("other declared package", lambda ref: ref["reference"]["versions"].pop("tokenizers")),
            ("other installed package", lambda ref: ref["versions"].update(mlx="0.32.4")),
            ("other language hash", lambda ref: ref.update(language_sha256="0" * 64)),
            ("other prefill chunk", lambda ref: ref.update(prefill_chunk=512)),
            ("other memory cap", lambda ref: ref.update(memory_limit_bytes=128 * mod.GIB)),
            ("other cache cap", lambda ref: ref.update(cache_limit_bytes=0)),
            ("recheck not complete", lambda ref: ref.update(phase="posthash")),
            ("missing recheck", lambda ref: ref.pop("runtime_files_bound")),
            ("invalid recheck count", lambda ref: ref.update(runtime_files_bound=True)),
        ):
            with self.subTest(label=label):
                r = good()
                mutate(r["reference"])
                self.assertEqual(failed(r), {"integrity"})

    def test_comparison_requires_matching_macos_versions(self):
        for name in ("qa", "matrix", "reference"):
            for version in (None, " ", 27, "27.0.1"):
                with self.subTest(receipt=name, version=version):
                    r = good()
                    r[name]["hardware"] = {**HARDWARE, "os": version}
                    self.assertEqual(failed(r), {"integrity"})

    def test_a_complete_reference_without_grades_cannot_pass_quality(self):
        for missing in (None, "", {"case": False}, [12]):
            with self.subTest(failure_list=missing):
                r = good()
                r["reference"]["qa"]["failures"] = missing
                self.assertEqual(failed(r), {"quality"})

    def test_reference_rows_and_grades_must_match_the_frozen_replay(self):
        for label, mutate in (
            ("duplicate case", lambda qa: qa["rows"].__setitem__(1, qa["rows"][0])),
            ("wrong prompt length", lambda qa: qa["rows"][0].update(prompt_tokens=1)),
            ("changed answer", lambda qa: qa["rows"][0].update(text="")),
            ("stale grade", lambda qa: qa["grades"][COHORT_IDS[0]].update(passed=False)),
            ("numeric grade", lambda qa: qa["grades"][COHORT_IDS[0]].update(passed=1)),
            ("missing grades", lambda qa: qa.pop("grades")),
        ):
            with self.subTest(case=label):
                r = good()
                mutate(r["reference"]["qa"])
                self.assertEqual(failed(r), {"quality"})

    def test_native_summary_requires_frozen_per_item_evidence(self):
        for label, mutate in (
            ("different cohort", lambda qa: qa.update(items_sha256="0" * 64)),
            ("malformed grades", lambda qa: qa.update(grades=[True])),
            ("missing case", lambda qa: qa["grades"]["disabled"].pop(COHORT_IDS[0])),
            ("unreported failure", lambda qa: qa["grades"]["default"][COHORT_IDS[0]].update(passed=False)),
            ("missing text pair", lambda qa: qa["summary"]["text_identity_vs_disabled"]["default"]
             .update(matching_text_pairs=104)),
        ):
            with self.subTest(case=label):
                r = good()
                mutate(r["qa"])
                self.assertEqual(failed(r), {"quality"})

    def test_default_mode_activating_the_drafter_fails_the_policy_gate(self):
        for mutate in (lambda r: r["qa"]["summary"]["modes"]["default"].update(verified_steps=3.0),
                       lambda r: r["lifecycle"]["runs"][1].update(final_verified_steps=1.0),
                       lambda r: r["lifecycle"]["runs"][1].pop("final_verified_steps"),
                       lambda r: r["matrix"]["runs"]["default"]["cells"][0].update(verified_steps=2.0)):
            r = good()
            mutate(r)
            self.assertIn("mtp_policy", failed(r))

    def test_memory_and_long_lookup_bounds(self):
        r = good()
        r["qa"]["summary"]["modes"]["disabled"]["peak_rss_kib"] = 97 * GIB_KIB
        self.assertEqual(failed(r), {"memory"})
        r = good()
        r["qa"]["summary"]["modes"]["default"]["long_context_elapsed_seconds"] = 401.0
        self.assertEqual(failed(r), {"long_lookup"})
        r = good()
        r["qa"]["summary"]["modes"]["default"]["long_context_elapsed_seconds"] = None
        self.assertEqual(failed(r), {"long_lookup"})

    def test_invalid_memory_and_lookup_measurements_fail_without_nonfinite_verdict_values(self):
        for field, gate in (("peak_rss_kib", "memory"), ("long_context_elapsed_seconds", "long_lookup")):
            for invalid in (float("nan"), float("inf"), -1, 0, True, "31", None):
                with self.subTest(field=field, invalid=invalid):
                    r = good()
                    r["qa"]["summary"]["modes"]["default"][field] = invalid
                    self.assertEqual(failed(r), {gate})
                    json.dumps(verdict(r), allow_nan=False)

    def test_first_load_and_warm_readiness_use_their_own_bounds(self):
        r = good()
        r["delivery"]["native_support"]["readiness_seconds"]["disabled"] = 301.0
        self.assertEqual(failed(r), {"readiness"})
        r = good()
        r["delivery"]["native_support"]["readiness_seconds"]["required"] = 121.0
        self.assertEqual(failed(r), {"readiness"})
        r = good()
        r["lifecycle"]["runs"][0]["ready_seconds"] = 121.0
        self.assertEqual(failed(r), {"readiness"})
        r = good()
        r["delivery"]["native_support"]["readiness_seconds"] = {}
        self.assertEqual(failed(r), {"readiness"})

    def test_readiness_requires_complete_finite_records_and_uses_delivery_order(self):
        r = good()
        r["delivery"]["native_support"]["readiness_seconds"].pop("required")
        self.assertEqual(failed(r), {"readiness"})
        r = good()
        r["delivery"]["native_support"]["readiness_seconds"] = {
            "required": 121.0, "disabled": 120.0, "default": 40.0,
        }
        self.assertEqual(failed(r), {"readiness"})
        for invalid in (float("nan"), -1, True, None):
            r = good()
            r["delivery"]["native_support"]["readiness_seconds"]["default"] = invalid
            self.assertEqual(failed(r), {"readiness"})
            json.dumps(verdict(r), allow_nan=False)
            r = good()
            r["qa"]["summary"]["modes"]["default"]["ready_seconds"] = invalid
            self.assertEqual(failed(r), {"readiness"})
        r = good()
        r["matrix"]["runs"]["disabled"].pop("ready_seconds")
        self.assertEqual(failed(r), {"readiness"})
        for invalid_modes in (1, {"disabled": True, "default": True, "required": True}):
            r = good()
            r["delivery"]["native_support"]["modes"] = invalid_modes
            self.assertEqual(failed(r), {"delivery", "readiness", "integrity", "mtp_policy"})
        r = good()
        r["delivery"]["native_support"]["modes"] = ["disabled", "default", "default"]
        self.assertEqual(failed(r), {"delivery", "readiness"})

    def test_reordering_delivery_modes_cannot_reclassify_warm_readiness(self):
        r = good()
        support = r["delivery"]["native_support"]
        support["readiness_seconds"]["default"] = 157.292
        self.assertEqual(failed(r), {"readiness"})
        support["modes"] = ["default", "disabled", "required"]
        result = verdict(r)
        self.assertEqual({g["gate"] for g in result["gates"] if not g["passed"]},
                         {"delivery", "readiness"})
        readiness = next(g for g in result["gates"] if g["gate"] == "readiness")
        self.assertIn("delivered default readiness 157.292s exceeds 120s", readiness["detail"])

    def test_delivery_summary_cannot_replace_raw_readiness(self):
        r = good()
        support = r["delivery"]["native_support"]
        support["readiness_seconds"]["default"] = 157.292
        raw = raw_delivery_controls(r["delivery"])
        support["readiness_seconds"]["default"] = 100.0
        result = mod.evaluate(r, TH, qualify.contract(), raw_repeats(r), raw)
        errors = {g["gate"]: g["detail"] for g in result["gates"] if not g["passed"]}
        self.assertEqual(set(errors), {"integrity"})
        self.assertIn("raw native controls differ from the delivery summary", errors["integrity"])

    def test_raw_delivery_controls_require_original_bytes_and_runtime_identity(self):
        r = good()
        delivery = r["delivery"]
        raw = raw_delivery_controls(delivery)
        with self.subTest(mutation="missing"):
            result = mod.evaluate(r, TH, qualify.contract(), raw_repeats(r), None)
            errors = {g["gate"]: g["detail"] for g in result["gates"] if not g["passed"]}
            self.assertEqual(errors, {"integrity": "delivery: missing raw native controls"})
        with self.subTest(mutation="duplicate field"):
            duplicate = b'{"revision":"one","revision":"two"}'
            self.assertEqual(mod.delivery_control_problems(delivery, duplicate),
                             ["delivery: invalid raw native controls"])
        with self.subTest(mutation="different bytes"):
            self.assertIn("SHA-256 differs", mod.delivery_control_problems(delivery, raw + b"\n")[0])
        with self.subTest(mutation="missing projected fields"):
            self.assertEqual(mod.delivery_control_problems(delivery, b"{}"),
                             ["delivery: invalid raw native control summary"])
        for key in ("server_sha256", "inventory_sha256", "revision"):
            with self.subTest(mutation=key):
                controls = json.loads(raw)
                controls[key] = "changed"
                other = json.dumps(controls).encode()
                delivery["native_support"]["result_sha256"] = sha256(other).hexdigest()
                self.assertEqual(mod.delivery_control_problems(delivery, other),
                                 ["delivery: raw native controls differ in server, inventory or revision"])
        with self.subTest(mutation="non-object delivery summary"):
            delivery["native_support"] = ["invalid"]
            result = mod.evaluate(r, TH, qualify.contract(), raw_repeats(r), raw)
            self.assertFalse(result["release_ready"])
            integrity = next(g for g in result["gates"] if g["gate"] == "integrity")
            self.assertIn("raw native controls differ from the delivery summary", integrity["detail"])

    def test_reference_relative_bounds_apply_to_both_ax_arms_and_every_length(self):
        r = good()
        r["matrix"]["runs"]["default"]["cells"] = ax_cells(decode_seconds=7.0 / 0.89)
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        r["matrix"]["runs"]["disabled"]["cells"] = ax_cells(ttft=(4.0, 16.0, 70.0 * 1.16))
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        r["matrix"]["runs"]["disabled"]["cells"] = ax_cells(decode_seconds=7.0 / 0.9001, ttft=(4.0, 16.0, 80.0))
        self.assertEqual(failed(r), set(), "values just inside both bounds must pass")

    def test_reference_relative_rejects_nonfinite_derived_timing(self):
        r = good()
        r["matrix"]["runs"]["default"]["cells"] = ax_cells(decode_seconds=0.1)
        for row in r["reference"]["matrix"]["cells"]:
            row["decode_seconds"] = 1e308
        self.assertEqual(failed(r), {"reference_relative"})
        json.dumps(verdict(r), allow_nan=False)

        r = good()
        cells = r["matrix"]["runs"]["default"]["cells"]
        cells.pop(4)
        for row in cells[2:4]:
            row["timing"].update(decode_seconds=1e-308)
        self.assertEqual(failed(r), {"integrity", "reference_relative"})
        json.dumps(verdict(r), allow_nan=False)

    def test_the_reference_must_use_the_same_token_ids_as_both_ax_arms(self):
        r = good()
        r["reference"]["matrix"]["prompt_id_hashes"]["cell-2048-3"] = "other"
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        r["matrix"]["runs"]["default"]["prompt_id_hashes"]["cell-512-0"] = "other"
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        r["matrix"]["runs"]["disabled"]["prompt_id_hashes"] = dict(TRAJECTORY_HASHES)
        self.assertEqual(failed(r), {"reference_relative"})

    def test_fixed_matrix_rejects_missing_cells_repeated_runs_and_wrong_warmup_labels(self):
        r = good()
        for arm in (*r["matrix"]["runs"].values(), r["reference"]["matrix"]):
            arm["cells"] = [row for row in arm["cells"] if row["length"] != 8192]
            arm["prompt_id_hashes"] = {key: digest for key, digest in arm["prompt_id_hashes"].items()
                                       if not key.startswith("cell-8192-")}
        self.assertEqual(failed(r), {"integrity", "reference_relative"})
        r = good()
        cells = r["matrix"]["runs"]["default"]["cells"]
        cells[3] = copy.deepcopy(cells[2])
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        r["reference"]["matrix"]["cells"][0]["warmup"] = False
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        for arm in (*r["matrix"]["runs"].values(), r["reference"]["matrix"]):
            arm["prompt_id_hashes"].pop("cell-512-0")
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        for arm in (*r["matrix"]["runs"].values(), r["reference"]["matrix"]):
            arm["prompt_id_hashes"]["cell-512-0"] = None
        self.assertEqual(failed(r), {"reference_relative"})
        for invalid_run in (False, [], {}):
            r = good()
            r["reference"]["matrix"]["cells"][0]["run"] = invalid_run
            self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        r["reference"]["matrix"]["cells"][0]["warmup"] = 1
        self.assertEqual(failed(r), {"reference_relative"})

    def test_every_measured_row_needs_valid_timing(self):
        r = good()
        r["matrix"]["runs"]["default"]["cells"][2]["timing"].pop("decode_seconds")
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        r["reference"]["matrix"]["cells"][2]["ttft_seconds"] = float("nan")
        self.assertEqual(failed(r), {"reference_relative"})

    def test_decode_window_must_match_the_emitted_output(self):
        r = good()
        for row in r["matrix"]["runs"]["default"]["cells"]:
            row["timing"].update(decode_tokens=254)
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        r["reference"]["matrix"]["cells"][2]["decode_tokens"] = 126
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        r["reference"]["matrix"]["cells"][2]["generated_ids"].pop()
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        for row in r["matrix"]["runs"]["default"]["cells"]:
            row["timing"].update(first_event_tokens=2, decode_tokens=126)
        streamed = r["matrix"]["runs"]["default"]["cells"][2]
        streamed.update(streamed.pop("timing"))
        self.assertEqual(failed(r), set())

    def test_a_short_measured_output_is_reported_not_averaged_away(self):
        r = good()
        r["matrix"]["runs"]["default"]["cells"][3]["output_tokens"] = [0] * 106
        result = verdict(r)
        gate = next(g for g in result["gates"] if g["gate"] == "reference_relative")
        self.assertFalse(gate["passed"])
        self.assertIn("short or invalid measured outputs", gate["detail"])
        r = good()
        r["reference"]["matrix"]["cells"][4]["generated_tokens"] = 100
        self.assertEqual(failed(r), {"reference_relative"})

    def test_delivery_must_be_fresh_complete_and_pinned(self):
        for mutate in (lambda d: d.update(fresh_cache=False),
                       lambda d: d.update(completed=False),
                       lambda d: d["download"].update(exit_code=1),
                       lambda d: d.update(revision="main"),
                       lambda d: d["verification"].update(members_verified=48),
                       lambda d: d["verification"].update(all_sizes_and_sha256_match=False),
                       lambda d: d["verification"].update(native_manifest_runtime_ready=False),
                       lambda d: d["native_support"].update(modes=["disabled", "required"]),
                       lambda d: d["native_support"].update(passed=False)):
            r = good()
            mutate(r["delivery"])
            self.assertEqual(failed(r), {"delivery", "readiness", "mtp_policy"} if
                             "default" not in r["delivery"]["native_support"]["modes"] else {"delivery"})

    def test_a_resumed_delivery_passes_but_reports_its_failed_attempts(self):
        r = good()
        r["delivery"]["previous_attempts"] = [{"exit_code": 1, "elapsed_seconds": 450.0}]
        result = verdict(r)
        gate = next(g for g in result["gates"] if g["gate"] == "delivery")
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failed_attempts"], 1)
        self.assertIn("1 failed attempt", gate["detail"])

    def test_integrity_requires_one_binary_one_inventory_and_the_target_sku(self):
        r = good()
        r["qa"]["server_sha256"] = "c" * 64
        self.assertEqual(failed(r), {"integrity"})
        r = good()
        r["matrix"]["inventory_sha256"] = "c" * 64
        self.assertEqual(failed(r), {"integrity"})
        r = good()
        r["lifecycle"]["hardware"] = {**HARDWARE, "form_factor": "unrecognized (x)"}
        self.assertEqual(failed(r), {"integrity"})
        r = good()
        r["qa"]["release_ready"] = True
        self.assertEqual(failed(r), {"integrity"})
        r = good()
        r["reference"]["versions"] = {"mlx-vlm": "0.8"}
        self.assertEqual(failed(r), {"integrity"})

    def test_every_required_receipt_must_have_its_digests(self):
        for name in mod.RESULT_FILES:
            keys = ("inventory_sha256",) if name == "reference" else ("inventory_sha256", "server_sha256")
            for key in keys:
                for value in (None, "", "a" * 63, "g" * 64, ["a" * 64]):
                    with self.subTest(name=name, key=key, value=value):
                        r = good()
                        if value is None:
                            r[name].pop(key)
                        else:
                            r[name][key] = value
                        self.assertEqual(failed(r), {"integrity"})

    def test_timing_results_must_record_an_idle_host_before_and_after(self):
        for name in ("qa", "lifecycle", "matrix", "reference"):
            r = good()
            r[name]["host_quiet"] = {"cpu_idle_percent": 80.0}
            self.assertEqual(failed(r), {"integrity"}, name)
            r = good()
            r[name]["host_idle_after"] = 50.0
            self.assertEqual(failed(r), {"integrity"}, name)
            r = good()
            del r[name]["host_quiet"]
            self.assertEqual(failed(r), {"integrity"}, name)

    def test_timing_idle_samples_must_be_real_percentages(self):
        for value in (float("nan"), float("inf"), 101.0, True, "95", None):
            with self.subTest(value=value):
                r = good()
                r["matrix"]["host_idle_after"] = value
                self.assertEqual(failed(r), {"integrity"})
                r = good()
                r["lifecycle"]["host_quiet"] = {"cpu_idle_percent": value}
                self.assertEqual(failed(r), {"integrity"})

    def test_repeats_of_one_arm_must_be_token_identical(self):
        r = good()
        r["matrix"]["repeat_token_identity"]["default"]["identical"] = 30
        self.assertEqual(failed(r), {"integrity"})
        for mutate in (lambda m: m.pop("repeat_token_identity"),
                       lambda m: m["repeat_token_identity"].pop("default"),
                       lambda m: m["repeat_token_identity"]["default"].update(compared=0, identical=0),
                       lambda m: m.update(repeats_per_arm=1),
                       lambda m: m["runs"]["default"]["trajectories"].pop(),
                       lambda m: m["runs"]["default"]["trajectories"][1].update(index=0),
                       lambda m: m["runs"]["default"]["prompt_id_hashes"].pop("traj-01")):
            r = good()
            mutate(r["matrix"])
            self.assertEqual(failed(r), {"integrity"})

    def test_raw_unselected_repeat_must_match_tokens_and_workload(self):
        for label, mutate, reason in (
            ("changed measured token", lambda run: run["cells"][2]["output_tokens"].__setitem__(7, 41),
             "do not have identical output tokens"),
            ("changed warmup token", lambda run: run["cells"][0]["output_tokens"].__setitem__(7, 41),
             "do not have identical output tokens"),
            ("changed trajectory", lambda run: run["trajectories"][0]["output_tokens"].__setitem__(0, 41),
             "do not have identical output tokens"),
            ("incomplete", lambda run: run["cells"][2].update(status="running"),
             "lack matching complete workloads"),
            ("changed prompt", lambda run: run["prompt_id_hashes"].update({"cell-512-2": "f" * 64}),
             "lack matching complete workloads"),
            ("bad timing", lambda run: run["cells"][2]["timing"].update(ttft_seconds=True),
             "lack matching complete workloads"),
            ("unexpected MTP", lambda run: run["cells"][2].update(verified_steps=1),
             "require zero verified MTP steps"),
            ("missing verifier count", lambda run: run["cells"][2].pop("verified_steps"),
             "require zero verified MTP steps"),
        ):
            with self.subTest(label=label):
                r = good()
                repeats = raw_repeats(r)
                mutate(repeats["default"][1])
                result = verdict(r, repeats)
                self.assertFalse(result["release_ready"])
                integrity = next(g for g in result["gates"] if g["gate"] == "integrity")
                self.assertIn(reason, integrity["detail"])

    def test_selected_run_must_match_fastest_raw_process(self):
        r = good()
        repeats = raw_repeats(r)
        for row in repeats["default"][1]["cells"]:
            row["timing"]["ttft_seconds"] /= 2
        self.assertFalse(verdict(r, repeats)["release_ready"])
        r["matrix"]["runs"]["default"] = repeats["default"][1]
        self.assertTrue(verdict(r, repeats)["release_ready"])
        self.assertFalse(verdict(r, {})["release_ready"])

    def test_selected_repeat_matches_types_in_timing_evidence(self):
        r = good()
        repeats = raw_repeats(r)
        repeats["default"][0]["cells"][2]["timing"]["first_event_tokens"] = True
        result = verdict(r, repeats)
        integrity = next(g for g in result["gates"] if g["gate"] == "integrity")
        self.assertIn("selected run differs", integrity["detail"])

    def test_signed_evidence_replays_with_raw_delivery_and_matrix_digests(self):
        root = SIGNED_REFERENCE.parent.parent
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "verdict.json"
            subprocess.run([sys.executable, str(SCRIPT), "--evidence-dir", str(root),
                            "--output", str(output)], check=True, capture_output=True, text=True)
            result = json.loads(output.read_bytes())
            self.assertTrue(result["release_ready"])
            self.assertEqual(result["evidence_sha256"]["matrix_raw"],
                             mod.sha256_file(root / mod.MATRIX_REPEATS_FILE))
            self.assertEqual(result["evidence_sha256"]["delivery_controls"],
                             mod.sha256_file(root / mod.DELIVERY_CONTROLS_FILE))

    def test_cli_records_failed_integrity_for_missing_or_malformed_raw_repeats(self):
        archive = SIGNED_REFERENCE.parent.parent
        original = (archive / mod.MATRIX_REPEATS_FILE).read_bytes()
        duplicate = original.replace(b'"output_tokens": [', b'"output_tokens": [], "output_tokens": [', 1)
        self.assertNotEqual(duplicate, original)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in mod.RESULT_FILES.values():
                path = root / name
                path.parent.mkdir()
                path.symlink_to(archive / name)
            controls = root / mod.DELIVERY_CONTROLS_FILE
            controls.parent.mkdir()
            controls.symlink_to(archive / mod.DELIVERY_CONTROLS_FILE)
            raw = root / mod.MATRIX_REPEATS_FILE
            for label, data in (("missing", None), ("duplicate", duplicate),
                                ("deeply nested", b"[" * 4_000 + b"0" + b"]" * 4_000)):
                with self.subTest(case=label):
                    if data is not None:
                        raw.write_bytes(data)
                    output = root / "verdict.json"
                    done = subprocess.run([sys.executable, str(SCRIPT), "--evidence-dir", str(root),
                                           "--output", str(output)], capture_output=True, text=True)
                    self.assertEqual(done.returncode, 1, done.stderr)
                    result = json.loads(output.read_bytes())
                    self.assertEqual(
                        {g["gate"] for g in result["gates"] if not g["passed"]}, {"integrity"}
                    )

    def test_lifecycle_needs_seven_drained_actions_in_each_required_mode(self):
        r = good()
        r["lifecycle"]["runs"][1]["actions"] = r["lifecycle"]["runs"][1]["actions"][:6]
        self.assertEqual(failed(r), {"lifecycle"})
        r = good()
        r["lifecycle"]["runs"][0]["actions"][2]["drained"]["ax_engine_jobs_in_flight"] = 1.0
        self.assertEqual(failed(r), {"lifecycle"})

    def test_lifecycle_requires_recorded_action_identities_and_all_drain_counters(self):
        for mutate in (
            lambda run: run.update(actions=None),
            lambda run: run["actions"].__setitem__(6, None),
            lambda run: run["actions"].__setitem__(6, run["actions"][0]),
            lambda run: run["actions"][6].update(mode="default"),
            lambda run: run["actions"][6].update(drained={}),
            lambda run: run["actions"][6]["drained"].pop("ax_engine_generation_buffered_stream_events"),
            lambda run: run["actions"][6]["drained"].update(ax_engine_jobs_in_flight=False),
            lambda run: run.update(exit_code_after_stop=False),
            lambda run: run.update(exit_code_after_stop=1),
        ):
            r = good()
            mutate(r["lifecycle"]["runs"][0])
            self.assertEqual(failed(r), {"lifecycle"})
        r = good()
        r["lifecycle"]["runs"].append(copy.deepcopy(r["lifecycle"]["runs"][0]))
        self.assertEqual(failed(r), {"lifecycle"})
        r = good()
        r["lifecycle"]["lifecycle_passed"] = 1
        self.assertEqual(failed(r), {"lifecycle"})

    def test_the_contract_must_keep_mtp_s_p_d_not_assessed(self):
        bad = copy.deepcopy(qualify.contract())
        bad["mtp_certification"]["MTP-P"] = "passed"
        r = good()
        result = mod.evaluate(r, TH, bad, raw_repeats(r), raw_delivery_controls(r["delivery"]))
        self.assertEqual({g["gate"] for g in result["gates"] if not g["passed"]}, {"mtp_policy"})

    def test_frozen_thresholds_match_the_documented_scope(self):
        self.assertEqual(TH["target"]["pack_revision"], qualify.PACK_REVISION)
        self.assertEqual(TH["target"]["pack_repo"], qualify.PRIMARY_REPO)
        self.assertEqual(TH["mtp"]["gate_state"], qualify.contract()["mtp_certification"])
        self.assertEqual(TH["release_scope"], qualify.RELEASE_SCOPE)

    def test_cli_rejects_thresholds_for_another_release_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            thresholds = root / "thresholds.json"
            thresholds.write_text(json.dumps({**TH, "release_scope": "other scope"}))
            report = root / "verdict.json"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--evidence-dir", str(root),
                 "--thresholds", str(thresholds), "--output", str(report)],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("release_scope differs", result.stderr)
            self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
