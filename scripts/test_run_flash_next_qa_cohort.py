#!/usr/bin/env python3
"""Weight-free regressions for the Flash Next QA cohort replay."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("run_flash_next_qa_cohort.py")
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("flash_qa_cohort", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def row(item_id: str, text: str = "ok", verified: float = 0.0, errors: float = 0.0,
        finish: str = "stop", elapsed: float = 1.0) -> dict:
    delta = {mod.native.VERIFIED: verified, **{name: 0.0 for name in mod.native.ERRORS}}
    delta[mod.native.ERRORS[0]] = errors
    return {"id": item_id, "text": text, "finish_reason": finish, "elapsed_seconds": elapsed,
            "activation_delta": delta}


class QaCohortTests(unittest.TestCase):
    def test_frozen_cohort_loads_and_budgets_follow_the_original_caps(self):
        items = mod.load_items(mod.FROZEN / "full-qa-items.json")
        self.assertEqual(len(items), 105)
        self.assertEqual(mod.budget_for(items[0]), 256)
        self.assertEqual(mod.budget_for(items[-1]), 64)
        body = mod.request_body(items[0])
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["messages"][-1]["content"], items[0]["user"])

    def test_cohort_rejects_wrong_length_or_missing_long_item(self):
        items = json.loads((mod.FROZEN / "full-qa-items.json").read_text())
        for bad in (items[:-1], items[1:] + items[:1]):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "items.json"
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    mod.load_items(path)

    def test_frozen_grader_separates_pass_and_fail(self):
        items = mod.load_items(mod.FROZEN / "full-qa-items.json")
        sort_item = next(i for i in items if i["id"] == "instruction_sort_numbers")
        graded = mod.grade([sort_item], {sort_item["id"]: "2,2,7,9"})
        self.assertTrue(graded[sort_item["id"]]["passed"])
        graded = mod.grade([sort_item], {sort_item["id"]: "9,7,2,2"})
        self.assertFalse(graded[sort_item["id"]]["passed"])

    def test_summary_counts_failures_pairs_route_errors_and_stops(self):
        items = [{"id": "a"}, {"id": mod.LONG_ID}]
        runs = {
            "disabled": {"ready_seconds": 5.0, "peak_rss_kib": 10,
                         "rows": [row("a", "x"), row(mod.LONG_ID, "1734", elapsed=7.5)]},
            "required": {"ready_seconds": 6.0, "peak_rss_kib": 20,
                         "rows": [row("a", "y", verified=3), row(mod.LONG_ID, "1734", verified=2,
                                                                  errors=1, finish="length")]},
        }
        grades = {"disabled": {"a": {"passed": True}, mod.LONG_ID: {"passed": True}},
                  "required": {"a": {"passed": False}, mod.LONG_ID: {"passed": True}}}
        summary = mod.summarize(items, runs, grades)
        self.assertEqual(summary["requests"], 4)
        self.assertEqual(summary["matching_text_pairs"], 1)
        self.assertEqual(summary["differing_text_ids"], ["a"])
        disabled, required = summary["modes"]["disabled"], summary["modes"]["required"]
        self.assertEqual(disabled["quality_failures"], [])
        self.assertTrue(disabled["normal_stops"])
        self.assertEqual(disabled["long_context_elapsed_seconds"], 7.5)
        self.assertEqual(required["quality_failures"], ["a"])
        self.assertFalse(required["normal_stops"])
        self.assertEqual(required["verified_steps"], 5)
        self.assertEqual(required["route_errors"], 1)

    def test_summary_compares_every_mode_with_the_disabled_baseline(self):
        items = [{"id": "a"}]
        runs = {m: {"ready_seconds": 1.0, "peak_rss_kib": 1, "rows": [row("a", text)]}
                for m, text in (("disabled", "x"), ("default", "x"), ("required", "y"))}
        grades = {m: {"a": {"passed": True}} for m in runs}
        identity = mod.summarize(items, runs, grades)["text_identity_vs_disabled"]
        self.assertEqual(identity["default"], {"matching_text_pairs": 1, "differing_text_ids": []})
        self.assertEqual(identity["required"]["differing_text_ids"], ["a"])
        pair = mod.summarize(items, {k: runs[k] for k in ("disabled", "default")},
                             {k: grades[k] for k in ("disabled", "default")})
        self.assertNotIn("matching_text_pairs", pair)
        self.assertEqual(pair["text_identity_vs_disabled"]["default"]["matching_text_pairs"], 1)

    def test_modes_option_requires_the_disabled_baseline(self):
        done = subprocess.run([sys.executable, str(SCRIPT), "--dry-run", "--modes", "default"],
                              capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        done = subprocess.run([sys.executable, str(SCRIPT), "--dry-run", "--modes", "disabled", "default"],
                              capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(done.stdout)["modes"], ["disabled", "default"])

    def test_rss_sampler_reports_a_positive_peak_for_a_live_process(self):
        sampler = mod.RssPeak(os.getpid(), interval=0.05)
        sampler.start()
        peak = sampler.finish()
        self.assertGreater(peak, 0)

    def test_dry_run_cannot_claim_qualification(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--dry-run"], capture_output=True,
                             text=True, check=True)
        contract = json.loads(out.stdout)
        self.assertFalse(contract["qualification"])
        self.assertFalse(contract["release_ready"])
        self.assertEqual(contract["items"], 105)
        # The pool must hold the 29,774-token lookup plus its output budget.
        self.assertGreater(contract["total_blocks"] * contract["block_size_tokens"],
                           29774 + contract["long_budget"])
        self.assertEqual(contract["modes"], ["disabled", "required"])

    def test_live_run_requires_all_inputs(self):
        done = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("live run requires", done.stderr)


if __name__ == "__main__":
    unittest.main()
