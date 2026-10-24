#!/usr/bin/env python3
"""Unit tests for the hybrid prefix-reuse probe helpers (no server needed)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("probe_hybrid_prefix_reuse.py")
MODULE_SPEC = importlib.util.spec_from_file_location("probe_hybrid_prefix_reuse", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
probe = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = probe
MODULE_SPEC.loader.exec_module(probe)


class TokenRunTests(unittest.TestCase):
    def test_token_run_is_deterministic_and_bounded(self):
        first = probe.token_run(64, seed=1, start=1000, span=50)
        self.assertEqual(first, probe.token_run(64, seed=1, start=1000, span=50))
        self.assertNotEqual(first, probe.token_run(64, seed=2, start=1000, span=50))
        self.assertTrue(all(1000 <= token < 1050 for token in first))


class BuildRequestsTests(unittest.TestCase):
    def test_request_shapes_share_the_expected_prefixes(self):
        requests = dict(probe.build_requests(prefix_tokens=100, suffix_tokens=10, middle_tokens=7))
        self.assertEqual(list(requests), ["cold", "sibling", "followup", "repeat"])
        cold, sibling, followup, repeat = (
            requests["cold"],
            requests["sibling"],
            requests["followup"],
            requests["repeat"],
        )
        self.assertEqual(len(cold), 110)
        self.assertEqual(len(followup), 100 + 10 + 7 + 10)
        # Sibling shares exactly the 100-token prefix with cold and then diverges.
        self.assertEqual(sibling[:100], cold[:100])
        self.assertNotEqual(sibling[100:], cold[100:])
        # Follow-up extends the cold prompt; repeat is identical to the sibling.
        self.assertEqual(followup[:110], cold)
        self.assertEqual(repeat, sibling)


class FindRouteTests(unittest.TestCase):
    def test_finds_nested_crossover_decisions(self):
        response = {"response": {"route": {"crossover_decisions": {"a": 1, "b": "2"}}}}
        self.assertEqual(probe.find_route(response), {"a": 1, "b": 2})

    def test_missing_route_is_empty(self):
        self.assertEqual(probe.find_route({"tokens": [1, 2, 3]}), {})
        self.assertEqual(probe.find_route({"items": [{"x": 1}]}), {})


class FindOutputTokensTests(unittest.TestCase):
    def test_reads_top_level_output_tokens(self):
        self.assertEqual(probe.find_output_tokens({"output_tokens": [3, "4"]}), [3, 4])

    def test_missing_output_tokens_is_empty(self):
        self.assertEqual(probe.find_output_tokens({"output_tokens": None}), [])
        self.assertEqual(probe.find_output_tokens({}), [])


class PositiveIntTests(unittest.TestCase):
    def test_rejects_non_positive(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            probe.positive_int("0")
        self.assertEqual(probe.positive_int("7"), 7)


if __name__ == "__main__":
    unittest.main()
