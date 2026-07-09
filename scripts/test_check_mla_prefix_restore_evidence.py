import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_mla_prefix_restore_evidence.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mla_prefix_restore_evidence", SCRIPT_PATH
)
assert MODULE_SPEC is not None
checker = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def write_artifact(path: Path, *, mutate=None) -> None:
    artifact = {
        "schema_version": checker.SCHEMA_VERSION,
        "model": {
            "model_id": "glm47-flash",
            "artifacts_dir": ".internal/models/GLM-4.7-Flash-4bit",
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
            "mode": "warm_extend",
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
    if mutate is not None:
        mutate(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


class MlaPrefixRestoreEvidenceTests(unittest.TestCase):
    def test_accepts_default_path_warm_extend_artifact_with_real_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            write_artifact(path)

            summary = checker.validate_artifact(
                path,
                min_prompts=5,
                require_default_path=True,
                model_substring="glm",
            )

            self.assertEqual(summary.model_id, "glm47-flash")
            self.assertEqual(summary.mode, "warm_extend")
            self.assertEqual(summary.warm_hit_count, 1)
            self.assertEqual(summary.warm_reused_tokens, 16)

    def test_rejects_env_override_on_default_path_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"

            def mutate(artifact):
                artifact["environment_flags"]["AX_MLX_MLA_PREFILL_CHUNK"] = {
                    "set": True,
                    "truthy": None,
                    "value": "32",
                }

            write_artifact(path, mutate=mutate)

            with self.assertRaisesRegex(
                checker.MlaPrefixRestoreEvidenceError,
                "default MLA prefill chunk",
            ):
                checker.validate_artifact(
                    path,
                    min_prompts=5,
                    require_default_path=True,
                    model_substring="glm",
                )

    def test_rejects_artifact_without_real_warm_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"

            def mutate(artifact):
                for row in artifact["per_prompt"]:
                    row["warm_telemetry"]["ax_mlx_prefix_cache_hits"] = 0
                    row["warm_telemetry"]["ax_mlx_prefix_cache_reused_tokens"] = 0

            write_artifact(path, mutate=mutate)

            with self.assertRaisesRegex(
                checker.MlaPrefixRestoreEvidenceError,
                "physical warm prefix-cache hit",
            ):
                checker.validate_artifact(
                    path,
                    min_prompts=5,
                    require_default_path=True,
                    model_substring="glm",
                )

    def test_rejects_non_matching_prompt_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"

            def mutate(artifact):
                artifact["aggregate"]["prompts_matching_exactly"] = 4
                artifact["per_prompt"][3]["tokens_match"] = False

            write_artifact(path, mutate=mutate)

            with self.assertRaisesRegex(
                checker.MlaPrefixRestoreEvidenceError,
                "match every prompt",
            ):
                checker.validate_artifact(
                    path,
                    min_prompts=5,
                    require_default_path=True,
                    model_substring="glm",
                )

    def test_accepts_warm_repeat_when_expected_mode_allows_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"

            def mutate(artifact):
                artifact["config"]["mode"] = "warm_repeat"

            write_artifact(path, mutate=mutate)

            summary = checker.validate_artifact(
                path,
                min_prompts=5,
                require_default_path=True,
                model_substring="glm",
                expected_mode=None,
            )

            self.assertEqual(summary.mode, "warm_repeat")

    def test_default_gate_still_requires_warm_extend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"

            def mutate(artifact):
                artifact["config"]["mode"] = "warm_repeat"

            write_artifact(path, mutate=mutate)

            with self.assertRaisesRegex(
                checker.MlaPrefixRestoreEvidenceError,
                "warm_extend",
            ):
                checker.validate_artifact(
                    path,
                    min_prompts=5,
                    require_default_path=True,
                    model_substring="glm",
                )


if __name__ == "__main__":
    unittest.main()
