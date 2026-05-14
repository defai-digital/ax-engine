#!/usr/bin/env python3
"""Unit tests for prefix-reuse equivalence harness provenance helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("verify_prefix_reuse_equivalence.py")
REPO_ROOT = SCRIPT_PATH.resolve().parents[1]
GLM_CHUNK16_ARTIFACT = (
    REPO_ROOT
    / "benchmarks/results/prefix-reuse-equivalence/"
    / "glm47-warm-extend-chunk16-provenance-2026-05-14.json"
)
GLM_DEFAULT_MLA_CHUNK16_ARTIFACT = (
    REPO_ROOT
    / "benchmarks/results/prefix-reuse-equivalence/"
    / "glm47-warm-extend-default-mla-chunk16-2026-05-14.json"
)
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
        self.assertFalse(flags["AX_DISABLE_MLA_PREFIX_RESTORE"]["set"])
        self.assertIsNone(flags["AX_DISABLE_MLA_PREFIX_RESTORE"]["value"])
        self.assertFalse(flags["AX_DISABLE_MLA_PREFIX_RESTORE"]["truthy"])
        self.assertFalse(flags["AX_MLX_MLA_PREFILL_CHUNK"]["set"])
        self.assertIsNone(flags["AX_MLX_MLA_PREFILL_CHUNK"]["value"])
        self.assertIsNone(flags["AX_MLX_MLA_PREFILL_CHUNK"]["truthy"])

        with patch.dict(
            os.environ,
            {
                "AX_ALLOW_MLA_PREFIX_RESTORE": " yes ",
                "AX_DISABLE_MLA_PREFIX_RESTORE": "0",
                "AX_MLX_MLA_PREFILL_CHUNK": "16",
                "AX_NO_SPEC": "0",
            },
            clear=True,
        ):
            flags = verify_prefix.collect_environment_flags()
        self.assertTrue(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["set"])
        self.assertEqual(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["value"], " yes ")
        self.assertTrue(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["truthy"])
        self.assertTrue(flags["AX_DISABLE_MLA_PREFIX_RESTORE"]["set"])
        self.assertEqual(flags["AX_DISABLE_MLA_PREFIX_RESTORE"]["value"], "0")
        self.assertFalse(flags["AX_DISABLE_MLA_PREFIX_RESTORE"]["truthy"])
        self.assertTrue(flags["AX_MLX_MLA_PREFILL_CHUNK"]["set"])
        self.assertEqual(flags["AX_MLX_MLA_PREFILL_CHUNK"]["value"], "16")
        self.assertIsNone(flags["AX_MLX_MLA_PREFILL_CHUNK"]["truthy"])
        self.assertTrue(flags["AX_NO_SPEC"]["set"])
        self.assertEqual(flags["AX_NO_SPEC"]["value"], "0")
        self.assertFalse(flags["AX_NO_SPEC"]["truthy"])

    def test_glm_chunk16_investigation_artifact_has_provenance_and_real_hit(self) -> None:
        artifact = json.loads(GLM_CHUNK16_ARTIFACT.read_text())

        self.assertEqual(artifact["schema_version"], verify_prefix.SCHEMA_VERSION)
        self.assertEqual(artifact["config"]["mode"], "warm_extend")
        self.assertEqual(artifact["config"]["pad_to_block_size"], 16)
        self.assertEqual(
            artifact["aggregate"],
            {
                "prompts_matching_exactly": 5,
                "prompts_total": 5,
                "verdict": "PASS",
            },
        )
        flags = artifact["environment_flags"]
        self.assertTrue(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["truthy"])
        self.assertEqual(flags["AX_MLX_MLA_PREFILL_CHUNK"]["value"], "16")
        self.assertIsNone(flags["AX_MLX_MLA_PREFILL_CHUNK"]["truthy"])
        self.assertTrue(
            any(
                row["warm_telemetry"]["ax_mlx_prefix_cache_hits"] > 0
                for row in artifact["per_prompt"]
            )
        )

    def test_glm_default_mla_chunk16_artifact_has_no_opt_in_restore_flag(self) -> None:
        artifact = json.loads(GLM_DEFAULT_MLA_CHUNK16_ARTIFACT.read_text())

        self.assertEqual(artifact["schema_version"], verify_prefix.SCHEMA_VERSION)
        self.assertEqual(artifact["config"]["mode"], "warm_extend")
        self.assertEqual(artifact["config"]["pad_to_block_size"], 16)
        self.assertEqual(
            artifact["aggregate"],
            {
                "prompts_matching_exactly": 5,
                "prompts_total": 5,
                "verdict": "PASS",
            },
        )
        flags = artifact["environment_flags"]
        self.assertFalse(flags["AX_ALLOW_MLA_PREFIX_RESTORE"]["set"])
        self.assertFalse(flags["AX_DISABLE_MLA_PREFIX_RESTORE"]["truthy"])
        self.assertFalse(flags["AX_MLX_MLA_PREFILL_CHUNK"]["set"])
        self.assertIsNone(flags["AX_MLX_MLA_PREFILL_CHUNK"]["truthy"])
        self.assertTrue(
            any(
                row["warm_telemetry"]["ax_mlx_prefix_cache_hits"] > 0
                for row in artifact["per_prompt"]
            )
        )


if __name__ == "__main__":
    unittest.main()
