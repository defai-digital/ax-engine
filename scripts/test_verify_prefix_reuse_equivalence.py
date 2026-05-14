#!/usr/bin/env python3
"""Unit tests for prefix-reuse equivalence harness provenance helpers."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("verify_prefix_reuse_equivalence.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "verify_prefix_reuse_equivalence", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
verify_prefix = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = verify_prefix
MODULE_SPEC.loader.exec_module(verify_prefix)


class PrefixReuseEquivalenceProvenanceTests(unittest.TestCase):
    def test_truthy_env_parser_matches_fastpath_contract(self) -> None:
        for value in ["1", "true", "TRUE", " yes "]:
            with self.subTest(value=value):
                self.assertTrue(verify_prefix.parse_truthy_env(value))
        for value in [None, "", "0", "false", "no", "enabled"]:
            with self.subTest(value=value):
                self.assertFalse(verify_prefix.parse_truthy_env(value))

    def test_collect_environment_flags_records_unset_and_truthy_values(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            flags = verify_prefix.collect_environment_flags()
        self.assertEqual(
            set(flags),
            set(verify_prefix.PROVENANCE_ENV_FLAGS),
        )
        self.assertFalse(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["set"])
        self.assertIsNone(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["value"])
        self.assertFalse(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["truthy"])
        self.assertFalse(flags["AX_MLX_MLA_PREFILL_CHUNK"]["set"])
        self.assertIsNone(flags["AX_MLX_MLA_PREFILL_CHUNK"]["value"])
        self.assertIsNone(flags["AX_MLX_MLA_PREFILL_CHUNK"]["truthy"])

        with patch.dict(
            os.environ,
            {
                "AX_ALLOW_MLA_PREFIX_RESTORE": " yes ",
                "AX_MLX_MLA_PREFILL_CHUNK": "16",
                "AX_NO_SPEC": "0",
            },
            clear=True,
        ):
            flags = verify_prefix.collect_environment_flags()
        self.assertTrue(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["set"])
        self.assertEqual(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["value"], " yes ")
        self.assertTrue(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["truthy"])
        self.assertTrue(flags["AX_MLX_MLA_PREFILL_CHUNK"]["set"])
        self.assertEqual(flags["AX_MLX_MLA_PREFILL_CHUNK"]["value"], "16")
        self.assertIsNone(flags["AX_MLX_MLA_PREFILL_CHUNK"]["truthy"])
        self.assertTrue(flags["AX_NO_SPEC"]["set"])
        self.assertEqual(flags["AX_NO_SPEC"]["value"], "0")
        self.assertFalse(flags["AX_NO_SPEC"]["truthy"])


if __name__ == "__main__":
    unittest.main()
