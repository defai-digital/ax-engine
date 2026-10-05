#!/usr/bin/env python3
"""Weight-free regressions for the Flash Next release-gate evaluator."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
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
HARDWARE = {"form_factor": "Mac Studio", "soc": "Apple M2 Ultra", "memory_bytes": 192 * 1024**3}
CELL_HASHES = {f"cell-{length}-{run}": f"h{length}{run}" for length in (512, 2048, 8192)
               for run in range(5)}
COMMON = {"server_sha256": "s", "inventory_sha256": "i", "hardware": HARDWARE,
          "qualification": False, "release_ready": False}


def ax_cells(decode_seconds=7.0, ttft=(4.0, 16.0, 70.0), tokens=128):
    cells = []
    for length, first in zip((512, 2048, 8192), ttft):
        for run in range(5):
            cells.append({"length": length, "run": run, "warmup": run < 2, "verified_steps": 0.0,
                          "output_tokens": [0] * tokens,
                          "timing": {"ttft_seconds": first, "decode_seconds": decode_seconds,
                                     "decode_tokens": 127}})
    return cells


def ref_cells(decode_seconds=7.0, ttft=(4.0, 16.0, 70.0)):
    cells = []
    for length, first in zip((512, 2048, 8192), ttft):
        for run in range(5):
            cells.append({"length": length, "run": run, "warmup": run < 2, "generated_tokens": 128,
                          "ttft_seconds": first, "decode_seconds": decode_seconds, "decode_tokens": 127})
    return cells


def good():
    arm = {"completed": 105, "passed": 101, "quality_failures": list(FAILURES), "normal_stops": True,
           "verified_steps": 0.0, "route_errors": 0.0, "ready_seconds": 33.0,
           "peak_rss_kib": 80 * GIB_KIB, "long_context_elapsed_seconds": 270.0}
    qa = {**COMMON, "summary": {"modes": {"disabled": dict(arm), "default": dict(arm)},
                                "text_identity_vs_disabled": {"default": {
                                    "matching_text_pairs": 105, "differing_text_ids": []}}}}
    action = {"drained": {"ax_engine_jobs_in_flight": 0.0}}
    lifecycle = {**COMMON, "lifecycle_passed": True, "runs": [
        {"mode": mode, "ready_seconds": 33.0, "actions": [action] * 7, "final_verified_steps": 0.0,
         "exit_code_after_stop": 0} for mode in ("disabled", "default")]}
    matrix = {**COMMON, "runs": {m: {"ready_seconds": 33.0, "trajectories": [{"verified_steps": 0.0}],
                                     "prompt_id_hashes": {"traj-00": "t", **CELL_HASHES},
                                     "cells": ax_cells()} for m in ("disabled", "default")}}
    reference = {**COMMON, "completed": True, "versions": {"mlx-vlm": "0.7.0rc0"},
                 "qa": {"rows": [0] * 105, "failures": list(FAILURES)},
                 "matrix": {"cells": ref_cells(), "prompt_id_hashes": dict(CELL_HASHES)}}
    delivery = {**COMMON, "completed": True, "fresh_cache": True, "revision": TH["target"]["pack_revision"],
                "download": {"exit_code": 0, "elapsed_seconds": 900.0},
                "verification": {"members_verified": 49, "bytes_verified": 132261877478,
                                 "all_sizes_and_sha256_match": True, "native_manifest_runtime_ready": True},
                "native_support": {"passed": True, "modes": ["disabled", "required", "default"],
                                   "readiness_seconds": {"disabled": 120.0, "required": 70.0,
                                                         "default": 40.0}}}
    return {"delivery": delivery, "qa": qa, "lifecycle": lifecycle, "matrix": matrix, "reference": reference}


def verdict(results):
    return mod.evaluate(results, TH, qualify.contract())


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
        r["qa"]["summary"]["modes"]["default"]["quality_failures"].append("math_gcd_18_24")
        self.assertEqual(failed(r), {"quality"})

    def test_default_text_that_differs_from_direct_fails_quality(self):
        r = good()
        r["qa"]["summary"]["text_identity_vs_disabled"]["default"]["differing_text_ids"] = ["x"]
        self.assertEqual(failed(r), {"quality"})

    def test_an_incomplete_reference_fails_quality_and_the_matrix_gate(self):
        r = good()
        r["reference"]["completed"] = False
        self.assertEqual(failed(r), {"quality", "reference_relative"})

    def test_default_mode_activating_the_drafter_fails_the_policy_gate(self):
        for mutate in (lambda r: r["qa"]["summary"]["modes"]["default"].update(verified_steps=3.0),
                       lambda r: r["lifecycle"]["runs"][1].update(final_verified_steps=1.0),
                       lambda r: r["lifecycle"]["runs"][1].pop("final_verified_steps"),
                       lambda r: r["matrix"]["runs"]["default"]["cells"][0].update(verified_steps=2.0)):
            r = good()
            mutate(r)
            self.assertEqual(failed(r), {"mtp_policy"})

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

    def test_the_reference_must_use_the_same_token_ids_as_both_ax_arms(self):
        r = good()
        r["reference"]["matrix"]["prompt_id_hashes"]["cell-2048-3"] = "other"
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        r["matrix"]["runs"]["default"]["prompt_id_hashes"]["cell-512-0"] = "other"
        self.assertEqual(failed(r), {"reference_relative"})
        r = good()
        r["matrix"]["runs"]["disabled"]["prompt_id_hashes"] = {"traj-00": "t"}
        self.assertEqual(failed(r), {"reference_relative"})

    def test_a_short_measured_output_is_reported_not_averaged_away(self):
        r = good()
        r["matrix"]["runs"]["default"]["cells"][3]["output_tokens"] = [0] * 106
        result = verdict(r)
        gate = next(g for g in result["gates"] if g["gate"] == "reference_relative")
        self.assertFalse(gate["passed"])
        self.assertIn("short measured outputs", gate["detail"])
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
            self.assertEqual(failed(r), {"delivery", "mtp_policy"} if
                             "default" not in r["delivery"]["native_support"]["modes"] else {"delivery"})

    def test_integrity_requires_one_binary_one_inventory_and_the_target_sku(self):
        r = good()
        r["qa"]["server_sha256"] = "other"
        self.assertEqual(failed(r), {"integrity"})
        r = good()
        r["matrix"]["inventory_sha256"] = "other"
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

    def test_lifecycle_needs_seven_drained_actions_in_each_required_mode(self):
        r = good()
        r["lifecycle"]["runs"][1]["actions"] = r["lifecycle"]["runs"][1]["actions"][:6]
        self.assertEqual(failed(r), {"lifecycle"})
        r = good()
        r["lifecycle"]["runs"][0]["actions"][2] = {"drained": {"ax_engine_jobs_in_flight": 1.0}}
        self.assertEqual(failed(r), {"lifecycle"})
        r = good()
        r["lifecycle"]["runs"][0]["exit_code_after_stop"] = 1
        self.assertEqual(failed(r), {"lifecycle"})

    def test_the_contract_must_keep_mtp_s_p_d_not_assessed(self):
        bad = copy.deepcopy(qualify.contract())
        bad["mtp_certification"]["MTP-P"] = "passed"
        result = mod.evaluate(good(), TH, bad)
        self.assertEqual({g["gate"] for g in result["gates"] if not g["passed"]}, {"mtp_policy"})

    def test_frozen_thresholds_match_the_documented_scope(self):
        self.assertEqual(TH["target"]["pack_revision"], qualify.PACK_REVISION)
        self.assertEqual(TH["target"]["pack_repo"], qualify.PRIMARY_REPO)
        self.assertEqual(TH["mtp"]["gate_state"], qualify.contract()["mtp_certification"])
        self.assertIn("MTP", TH["release_scope"])


if __name__ == "__main__":
    unittest.main()
