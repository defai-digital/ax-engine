#!/usr/bin/env python3
"""Unit tests for TurboQuant KV policy search artifact enumeration."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("search_turboquant_kv_policy.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "search_turboquant_kv_policy", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
search = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = search
MODULE_SPEC.loader.exec_module(search)

CHECKER_PATH = Path(__file__).with_name("check_offline_policy_search_artifact.py")
CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_offline_policy_search_artifact", CHECKER_PATH
)
assert CHECKER_SPEC and CHECKER_SPEC.loader
checker = importlib.util.module_from_spec(CHECKER_SPEC)
sys.modules[CHECKER_SPEC.name] = checker
CHECKER_SPEC.loader.exec_module(checker)


def metadata() -> dict[str, object]:
    return {
        "created_at": "2026-05-14T00:00:00Z",
        "repo": {
            "commit": "abc1234",
            "dirty": False,
        },
        "model": {
            "id": "gemma-4-e2b-it-4bit",
            "family": "gemma4",
            "artifacts_dir": ".internal/models/gemma-4-e2b-it-4bit",
            "manifest_digest": "sha256:test",
        },
        "route": {
            "selected_backend": "mlx",
            "support_tier": "repo_owned_runtime",
        },
    }


def baseline() -> dict[str, object]:
    return {
        "policy_id": "disabled",
        "route": {
            "selected_backend": "mlx",
            "support_tier": "repo_owned_runtime",
        },
        "prompt_tokens": 8192,
        "generation_tokens": 256,
        "metrics": {
            "decode_tok_s": 100.0,
            "ttft_ms": 120.0,
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


class TurboQuantKvPolicySearchTests(unittest.TestCase):
    def test_enumerates_policy_grid_deterministically(self) -> None:
        policies = search.enumerate_policies(
            kv_presets=["disabled", "TurboQuantK8V4"],
            hot_window_tokens=[128, 256],
            eligible_layer_masks=["full_attention_only"],
            fallback_policies=["fail_closed", "fallback_with_accounting"],
            quality_profiles=["reference_k8v4"],
        )

        self.assertEqual(len(policies), 8)
        self.assertEqual(
            [search.policy_id(policy) for policy in policies[:3]],
            [
                "disabled-hot128-full_attention_only-fail_closed-reference_k8v4",
                "disabled-hot128-full_attention_only-fallback_with_accounting-reference_k8v4",
                "disabled-hot256-full_attention_only-fail_closed-reference_k8v4",
            ],
        )

    def test_quality_profiles_produce_unique_policy_ids(self) -> None:
        policies = search.enumerate_policies(
            kv_presets=["TurboQuantK8V4"],
            hot_window_tokens=[256],
            eligible_layer_masks=["full_attention_only"],
            fallback_policies=["fail_closed"],
            quality_profiles=["reference_k8v4", "strict_k8v4"],
        )

        policy_ids = [search.policy_id(policy) for policy in policies]

        self.assertEqual(
            policy_ids,
            [
                "tqk8v4-hot256-full_attention_only-fail_closed-reference_k8v4",
                "tqk8v4-hot256-full_attention_only-fail_closed-strict_k8v4",
            ],
        )
        self.assertEqual(len(policy_ids), len(set(policy_ids)))

    def test_builds_valid_diagnostic_artifact_without_performance_claim(self) -> None:
        artifact = search.build_search_artifact(
            metadata=metadata(),
            baseline=baseline(),
            kv_presets=["disabled", "TurboQuantK8V4"],
            hot_window_tokens=[256],
            eligible_layer_masks=["full_attention_only"],
            fallback_policies=["fail_closed"],
            quality_profiles=["reference_k8v4"],
            seed=7,
            repo={"commit": "abc1234", "dirty": False},
        )

        self.assertEqual(artifact["target"], "turboquant_kv_policy")
        self.assertEqual(artifact["status"], "diagnostic_only")
        self.assertEqual(artifact["search"]["budget"]["max_candidates"], 2)
        self.assertEqual(artifact["decision"]["classification"], "diagnostic_only")
        self.assertEqual(
            artifact["best_candidate"]["policy_id"],
            "disabled-hot256-full_attention_only-fail_closed-reference_k8v4",
        )
        for row in artifact["candidates"]:
            self.assertEqual(row["metrics"]["measurement_status"], "not_run")
            self.assertFalse(row["quality_gate_passed"])
            self.assertFalse(row["deterministic_replay_passed"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            write_json(path, artifact)
            self.assertEqual(
                checker.validate_offline_policy_search_artifact(path),
                "diagnostic_only",
            )

    def test_cli_writes_valid_diagnostic_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "metadata.json"
            baseline_path = root / "baseline.json"
            output_path = root / "artifact.json"
            write_json(metadata_path, metadata())
            write_json(baseline_path, baseline())

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--metadata",
                    str(metadata_path),
                    "--baseline",
                    str(baseline_path),
                    "--output",
                    str(output_path),
                    "--kv-presets",
                    "disabled,TurboQuantK8V4",
                    "--hot-window-tokens",
                    "128,256",
                    "--fallback-policies",
                    "fail_closed",
                    "--skip-git-repo-metadata",
                ],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                checker.validate_offline_policy_search_artifact(output_path),
                "diagnostic_only",
            )
            payload = json.loads(output_path.read_text())
            self.assertEqual(len(payload["candidates"]), 4)

    def test_missing_baseline_token_shape_fails_closed(self) -> None:
        broken_baseline = baseline()
        broken_baseline["prompt_tokens"] = "8192"

        with self.assertRaisesRegex(
            search.TurboQuantPolicySearchError,
            "baseline.prompt_tokens must be an integer",
        ):
            search.build_search_artifact(
                metadata=metadata(),
                baseline=broken_baseline,
                kv_presets=["TurboQuantK8V4"],
                hot_window_tokens=[256],
                eligible_layer_masks=["full_attention_only"],
                fallback_policies=["fail_closed"],
                quality_profiles=["reference_k8v4"],
                seed=42,
                repo={"commit": "abc1234", "dirty": False},
            )

    def test_duplicate_dimension_values_fail_before_artifact_build(self) -> None:
        with self.assertRaisesRegex(
            search.TurboQuantPolicySearchError,
            "kv_presets\\[1\\] duplicates",
        ):
            search.build_search_artifact(
                metadata=metadata(),
                baseline=baseline(),
                kv_presets=["TurboQuantK8V4", "TurboQuantK8V4"],
                hot_window_tokens=[256],
                eligible_layer_masks=["full_attention_only"],
                fallback_policies=["fail_closed"],
                quality_profiles=["reference_k8v4"],
                seed=42,
                repo={"commit": "abc1234", "dirty": False},
            )

    def test_invalid_direct_search_space_values_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            search.TurboQuantPolicySearchError,
            "hot_window_tokens\\[0\\] must be non-negative",
        ):
            search.build_search_artifact(
                metadata=metadata(),
                baseline=baseline(),
                kv_presets=["TurboQuantK8V4"],
                hot_window_tokens=[-1],
                eligible_layer_masks=["full_attention_only"],
                fallback_policies=["fail_closed"],
                quality_profiles=["reference_k8v4"],
                seed=42,
                repo={"commit": "abc1234", "dirty": False},
            )

        with self.assertRaisesRegex(
            search.TurboQuantPolicySearchError,
            "quality_profiles\\[0\\] must be a non-empty string",
        ):
            search.build_search_artifact(
                metadata=metadata(),
                baseline=baseline(),
                kv_presets=["TurboQuantK8V4"],
                hot_window_tokens=[256],
                eligible_layer_masks=["full_attention_only"],
                fallback_policies=["fail_closed"],
                quality_profiles=[""],
                seed=42,
                repo={"commit": "abc1234", "dirty": False},
            )

    def test_cli_duplicate_dimension_values_fail_without_writing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "metadata.json"
            baseline_path = root / "baseline.json"
            output_path = root / "artifact.json"
            write_json(metadata_path, metadata())
            write_json(baseline_path, baseline())

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--metadata",
                    str(metadata_path),
                    "--baseline",
                    str(baseline_path),
                    "--output",
                    str(output_path),
                    "--kv-presets",
                    "TurboQuantK8V4,TurboQuantK8V4",
                    "--skip-git-repo-metadata",
                ],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("duplicates", result.stderr)
            self.assertFalse(output_path.exists())

    def test_rejects_negative_hot_window_tokens(self) -> None:
        with self.assertRaisesRegex(
            search.argparse.ArgumentTypeError,
            "non-negative",
        ):
            search.parse_int_csv("128,-1")


if __name__ == "__main__":
    unittest.main()
