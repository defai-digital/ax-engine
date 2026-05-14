#!/usr/bin/env python3
"""Unit tests for multi-turn KV evidence provenance helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("profile_kv_multiturn_chat_evidence.py")
REPO_ROOT = SCRIPT_PATH.resolve().parents[1]
GLM_MLA_FIXED_ARTIFACT = (
    REPO_ROOT
    / "benchmarks/results/kv-long-context/"
    / "glm47-flash-4bit-multiturn-mla-fixed-2026-05-14.json"
)
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
        self.assertIsNone(flags["AX_MLX_PREFIX_CACHE_MAX_BYTES"]["truthy"])
        self.assertFalse(flags["AX_MLX_MLA_PREFILL_CHUNK"]["set"])
        self.assertIsNone(flags["AX_MLX_MLA_PREFILL_CHUNK"]["value"])
        self.assertIsNone(flags["AX_MLX_MLA_PREFILL_CHUNK"]["truthy"])

        with patch.dict(
            os.environ,
            {
                "AX_ALLOW_MLA_PREFIX_RESTORE": "yes",
                "AX_DISABLE_MLA_PREFIX_RESTORE": "0",
                "AX_MLX_MLA_PREFILL_CHUNK": "16",
                "AX_MLX_PREFIX_CACHE_MAX_BYTES": "268435456",
                "AX_MLX_PREFIX_CACHE_MAX_ENTRIES": "2",
                "AX_NO_SPEC": "0",
            },
            clear=True,
        ):
            flags = profile.collect_environment_flags()
        self.assertTrue(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["truthy"])
        self.assertFalse(flags["AX_DISABLE_MLA_PREFIX_RESTORE"]["truthy"])
        self.assertEqual(flags["AX_MLX_PREFIX_CACHE_MAX_BYTES"]["value"], "268435456")
        self.assertTrue(flags["AX_MLX_PREFIX_CACHE_MAX_BYTES"]["set"])
        self.assertIsNone(flags["AX_MLX_PREFIX_CACHE_MAX_BYTES"]["truthy"])
        self.assertEqual(flags["AX_MLX_MLA_PREFILL_CHUNK"]["value"], "16")
        self.assertIsNone(flags["AX_MLX_MLA_PREFILL_CHUNK"]["truthy"])
        self.assertEqual(flags["AX_MLX_PREFIX_CACHE_MAX_ENTRIES"]["value"], "2")
        self.assertIsNone(flags["AX_MLX_PREFIX_CACHE_MAX_ENTRIES"]["truthy"])
        self.assertFalse(flags["AX_NO_SPEC"]["truthy"])

    def test_glm_mla_fixed_multiturn_artifact_has_default_path_hits(self) -> None:
        artifact = json.loads(GLM_MLA_FIXED_ARTIFACT.read_text())

        self.assertEqual(artifact["schema_version"], profile.SCHEMA_VERSION)
        self.assertEqual(artifact["args"]["model_id"], "glm47-flash-4bit")
        flags = artifact["environment_flags"]
        self.assertFalse(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["set"])
        self.assertFalse(flags["AX_DISABLE_MLA_PREFIX_RESTORE"]["truthy"])
        self.assertFalse(flags["AX_MLX_MLA_PREFILL_CHUNK"]["set"])
        self.assertEqual(artifact["summary"]["turns"], 10)
        self.assertEqual(artifact["summary"]["ax_mlx_prefix_cache_hits_total"], 10)
        self.assertEqual(artifact["summary"]["ax_mlx_prefix_cache_misses_total"], 0)
        self.assertLess(artifact["summary"]["ttft_growth_ratio"], 1.0)
        self.assertEqual(
            artifact["summary"]["verdict_hint"],
            "phase_c_skip__existing_infra_captures_win",
        )


if __name__ == "__main__":
    unittest.main()
