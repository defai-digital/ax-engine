#!/usr/bin/env python3
"""Unit tests for multi-turn KV evidence provenance helpers."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("profile_kv_multiturn_chat_evidence.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "profile_kv_multiturn_chat_evidence", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
profile = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = profile
MODULE_SPEC.loader.exec_module(profile)


class KvMultiturnEvidenceProvenanceTests(unittest.TestCase):
    def test_truthy_env_parser_matches_fastpath_contract(self) -> None:
        for value in ["1", "true", "TRUE", " yes "]:
            with self.subTest(value=value):
                self.assertTrue(profile.parse_truthy_env(value))
        for value in [None, "", "0", "false", "no", "enabled", "256"]:
            with self.subTest(value=value):
                self.assertFalse(profile.parse_truthy_env(value))

    def test_collect_environment_flags_records_prefix_cache_policy_values(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            flags = profile.collect_environment_flags()
        self.assertEqual(set(flags), set(profile.PROVENANCE_ENV_FLAGS))
        self.assertFalse(flags["AX_MLX_PREFIX_CACHE_MAX_BYTES"]["set"])
        self.assertIsNone(flags["AX_MLX_PREFIX_CACHE_MAX_BYTES"]["value"])
        self.assertFalse(flags["AX_MLX_PREFIX_CACHE_MAX_BYTES"]["truthy"])

        with patch.dict(
            os.environ,
            {
                "AX_ALLOW_MLA_PREFIX_RESTORE": "yes",
                "AX_MLX_PREFIX_CACHE_MAX_BYTES": "268435456",
                "AX_MLX_PREFIX_CACHE_MAX_ENTRIES": "2",
                "AX_NO_SPEC": "0",
            },
            clear=True,
        ):
            flags = profile.collect_environment_flags()
        self.assertTrue(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["truthy"])
        self.assertEqual(flags["AX_MLX_PREFIX_CACHE_MAX_BYTES"]["value"], "268435456")
        self.assertTrue(flags["AX_MLX_PREFIX_CACHE_MAX_BYTES"]["set"])
        self.assertEqual(flags["AX_MLX_PREFIX_CACHE_MAX_ENTRIES"]["value"], "2")
        self.assertFalse(flags["AX_NO_SPEC"]["truthy"])


if __name__ == "__main__":
    unittest.main()
