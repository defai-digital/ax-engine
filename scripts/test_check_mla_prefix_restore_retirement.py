#!/usr/bin/env python3
"""Unit tests for MLA prefix-restore kill-switch retirement checks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_mla_prefix_restore_retirement.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mla_prefix_restore_retirement", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def artifact(*, model_id: str, mode: str) -> dict[str, object]:
    return {
        "schema_version": checker.evidence_checker.SCHEMA_VERSION,
        "model": {
            "model_id": model_id,
            "artifacts_dir": f".internal/models/{model_id}",
        },
        "environment_flags": {
            "AX_ALLOW_MLA_PREFIX_RESTORE": {
                "set": False,
                "truthy": False,
                "value": None,
            },
            "AX_DISABLE_MLA_PREFIX_RESTORE": {
                "set": False,
                "truthy": False,
                "value": None,
            },
            "AX_MLX_MLA_PREFILL_CHUNK": {
                "set": False,
                "truthy": None,
                "value": None,
            },
        },
        "config": {
            "mode": mode,
            "pad_to_block_size": 16,
            "prompt_count": 5,
        },
        "aggregate": {
            "prompts_total": 5,
            "prompts_matching_exactly": 5,
            "verdict": "PASS",
        },
        "per_prompt": [
            {
                "id": f"p{i}",
                "tokens_match": True,
                "warm_telemetry": {
                    "ax_mlx_prefix_cache_hits": 1 if i == 0 else 0,
                    "ax_mlx_prefix_cache_reused_tokens": 16 if i == 0 else 0,
                    "ax_mlx_prefix_cache_blocked_unsupported_layout": 0,
                    "ax_mlx_prefix_cache_blocked_policy_disabled": 0,
                },
            }
            for i in range(5)
        ],
    }


class MlaPrefixRestoreRetirementTests(unittest.TestCase):
    def write_fixture(self, payload: dict[str, object]) -> Path:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = Path(root.name) / "prefix-restore-evidence.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def write_matrix_fixture(self) -> list[Path]:
        return [
            self.write_fixture(artifact(model_id="glm47-flash", mode="warm_extend")),
            self.write_fixture(artifact(model_id="glm47-flash", mode="warm_repeat")),
            self.write_fixture(artifact(model_id="deepseek-v3", mode="warm_extend")),
            self.write_fixture(artifact(model_id="deepseek-v3", mode="warm_repeat")),
        ]

    def test_default_cli_skips_without_current_artifacts(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("[skip]", completed.stdout)

    def test_full_glm_deepseek_matrix_supports_retirement(self) -> None:
        decision = checker.check_mla_prefix_restore_retirement(
            self.write_matrix_fixture(),
            expect_decision=checker.RETIRE_SWITCH,
            required_families=["glm", "deepseek"],
            required_modes=["warm_extend", "warm_repeat"],
        )

        self.assertEqual(decision.decision, checker.RETIRE_SWITCH)
        self.assertEqual(decision.missing_requirements, [])

    def test_keep_guardrail_fails_when_evidence_now_supports_retirement(self) -> None:
        with self.assertRaisesRegex(
            checker.MlaPrefixRestoreRetirementError,
            "retiring the kill switch",
        ):
            checker.check_mla_prefix_restore_retirement(
                self.write_matrix_fixture(),
                expect_decision=checker.KEEP_GUARDRAIL,
                required_families=["glm", "deepseek"],
                required_modes=["warm_extend", "warm_repeat"],
            )

    def test_retirement_rejects_missing_family_coverage(self) -> None:
        paths = [
            self.write_fixture(artifact(model_id="glm47-flash", mode="warm_extend")),
            self.write_fixture(artifact(model_id="glm47-flash", mode="warm_repeat")),
        ]

        with self.assertRaisesRegex(
            checker.MlaPrefixRestoreRetirementError,
            "missing deepseek:warm_extend",
        ):
            checker.check_mla_prefix_restore_retirement(
                paths,
                expect_decision=checker.RETIRE_SWITCH,
                required_families=["glm", "deepseek"],
                required_modes=["warm_extend", "warm_repeat"],
            )

    def test_rejects_env_override_evidence(self) -> None:
        payload = artifact(model_id="glm47-flash", mode="warm_extend")
        flags = payload["environment_flags"]
        assert isinstance(flags, dict)
        flags["AX_MLX_MLA_PREFILL_CHUNK"] = {
            "set": True,
            "truthy": None,
            "value": "32",
        }
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.MlaPrefixRestoreRetirementError,
            "default MLA prefill chunk",
        ):
            checker.check_mla_prefix_restore_retirement(
                [path],
                expect_decision=checker.KEEP_GUARDRAIL,
                required_families=["glm"],
                required_modes=["warm_extend"],
            )

    def test_cli_reports_keep_guardrail(self) -> None:
        path = self.write_fixture(artifact(model_id="glm47-flash", mode="warm_extend"))

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--artifact",
                str(path),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("decision=keep_guardrail", completed.stdout)
        self.assertIn("missing=glm:warm_repeat", completed.stdout)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
