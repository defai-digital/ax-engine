#!/usr/bin/env python3
"""Unit tests for GatedDelta prefill profile artifact checks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_gateddelta_prefill_profile_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_gateddelta_prefill_profile_artifact", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def metric(median: float) -> dict[str, float]:
    return {
        "mean": median,
        "median": median,
        "min": median * 0.9,
        "max": median * 1.1,
    }


def prompt_hash(prompt_tokens: int) -> str:
    return f"{prompt_tokens:064x}"[-64:]


def mlx_lm_row(prompt_tokens: int, generation_tokens: int) -> dict[str, object]:
    return {
        "engine": "mlx_lm",
        "method": "mlx_lm.benchmark",
        "timing_scope": "upstream_mlx_lm_response_stats",
        "prompt_contract": "mlx_lm_random_tokens_seed_0",
        "prompt_token_ids_sha256": prompt_hash(prompt_tokens),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prefill_tok_s": metric(2000.0),
        "decode_tok_s": metric(100.0),
        "baseline": {
            "engine": "mlx_lm",
            "method": "mlx_lm.benchmark",
            "role": "primary_reference",
        },
    }


def ax_row(prompt_tokens: int, generation_tokens: int) -> dict[str, object]:
    return {
        "engine": "ax_engine_mlx",
        "method": "server_sse_runner_time_us",
        "timing_scope": "ax_engine_runner_time_us",
        "ax_decode_policy": "direct_no_ngram_acceleration",
        "ax_decode_claim_status": "direct_same_policy_baseline",
        "prompt_contract": "mlx_lm_random_tokens_seed_0",
        "prompt_token_ids_sha256": prompt_hash(prompt_tokens),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prefill_tok_s": metric(1800.0),
        "decode_tok_s": metric(90.0),
        "ax_mlx_telemetry": {
            "ax_mlx_prefill_steps": 1,
            "ax_mlx_prefill_wall_us": prompt_tokens * 10,
        },
        "ax_mlx_linear_attention_profile": {
            "ax_mlx_linear_attention_profile_enabled": 1,
            "ax_mlx_linear_attention_profile_layers": 28,
            "ax_mlx_linear_attention_profile_tokens": prompt_tokens * 28,
            "ax_mlx_linear_attention_profile_projection_wall_us": 100,
            "ax_mlx_linear_attention_profile_conv_wall_us": 80,
            "ax_mlx_linear_attention_profile_qk_norm_wall_us": 30,
            "ax_mlx_linear_attention_profile_recurrent_wall_us": 90,
            "ax_mlx_linear_attention_profile_output_wall_us": 20,
        },
        "baseline": {
            "engine": "mlx_lm",
            "method": "mlx_lm.benchmark",
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
            "prefill_tok_s": 2000.0,
            "decode_tok_s": 100.0,
            "prefill_ratio_to_mlx_lm": 0.9,
            "decode_ratio_to_mlx_lm": 0.9,
        },
    }


def mlx_swift_lm_row(prompt_tokens: int, generation_tokens: int) -> dict[str, object]:
    return {
        "engine": "mlx_swift_lm",
        "method": "mlx-swift-lm BenchmarkHelpers/MLXLMCommon adapter",
        "prompt_token_ids_sha256": prompt_hash(prompt_tokens),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prefill_tok_s": metric(1900.0),
        "decode_tok_s": metric(95.0),
    }


def valid_artifact() -> dict[str, object]:
    generation_tokens = 128
    results: list[dict[str, object]] = []
    for prompt_tokens in checker.REQUIRED_PROMPT_TOKENS:
        results.append(mlx_lm_row(prompt_tokens, generation_tokens))
        results.append(ax_row(prompt_tokens, generation_tokens))
    return {
        "schema_version": checker.TOP_LEVEL_SCHEMA_VERSION,
        "model": "mlx-community/Qwen3.5-9B-4bit",
        "model_config": {
            "model_family": "qwen3_next",
            "linear_attention_enabled": True,
        },
        "prompt_tokens": checker.REQUIRED_PROMPT_TOKENS,
        "generation_tokens": generation_tokens,
        "ax_linear_attention_profile": True,
        "gateddelta_prefill_profile": {
            "schema_version": checker.PROFILE_SCHEMA_VERSION,
            "purpose": "evidence_first_gateddelta_long_prompt_prefill_profile",
            "linear_attention_required": True,
            "direct_ax_row_required": True,
            "ngram_policy_allowed": False,
            "kv_compression_allowed": False,
            "prompt_tokens": checker.REQUIRED_PROMPT_TOKENS,
            "required_prompt_tokens": checker.REQUIRED_PROMPT_TOKENS,
            "runtime_profile_env": checker.PROFILE_ENV,
            "model_preflight": {
                "schema_version": checker.PREFLIGHT_SCHEMA_VERSION,
                "status": "passed",
                "checker": "scripts/check_gateddelta_prefill_model.py",
                "model_family": "qwen3_next",
                "model_type": "qwen3_5",
                "linear_attention": {
                    "num_value_heads": 4,
                    "num_key_heads": 4,
                    "key_head_dim": 64,
                    "value_head_dim": 128,
                    "conv_kernel_dim": 4,
                },
            },
        },
        "results": results,
    }


class GatedDeltaPrefillProfileArtifactTests(unittest.TestCase):
    def write_fixture(self, artifact: dict[str, object]) -> Path:
        root = Path(tempfile.mkdtemp())
        path = root / "gateddelta-prefill-profile.json"
        path.write_text(json.dumps(artifact, indent=2) + "\n")
        self.addCleanup(lambda: root.rmdir())
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_valid_artifact_passes(self) -> None:
        path = self.write_fixture(valid_artifact())

        checked = checker.validate_gateddelta_prefill_profile_artifact(path)

        self.assertEqual(
            checked,
            [
                "prompt_tokens=512:generation=128",
                "prompt_tokens=2048:generation=128",
                "prompt_tokens=8192:generation=128",
                "prompt_tokens=32768:generation=128",
            ],
        )

    def test_optional_mlx_swift_lm_rows_are_allowed(self) -> None:
        artifact = valid_artifact()
        for prompt_tokens in checker.REQUIRED_PROMPT_TOKENS:
            artifact["results"].append(mlx_swift_lm_row(prompt_tokens, 128))
        path = self.write_fixture(artifact)

        checked = checker.validate_gateddelta_prefill_profile_artifact(path)

        self.assertEqual(len(checked), 4)

    def test_missing_runtime_profile_fails(self) -> None:
        artifact = valid_artifact()
        artifact["ax_linear_attention_profile"] = False
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(
            checker.GatedDeltaPrefillProfileArtifactError,
            "ax_linear_attention_profile",
        ):
            checker.validate_gateddelta_prefill_profile_artifact(path)

    def test_missing_model_preflight_fails(self) -> None:
        artifact = valid_artifact()
        del artifact["gateddelta_prefill_profile"]["model_preflight"]
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(
            checker.GatedDeltaPrefillProfileArtifactError,
            "model_preflight",
        ):
            checker.validate_gateddelta_prefill_profile_artifact(path)

    def test_bad_preflight_schema_fails(self) -> None:
        artifact = valid_artifact()
        artifact["gateddelta_prefill_profile"]["model_preflight"][
            "schema_version"
        ] = "old"
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(
            checker.GatedDeltaPrefillProfileArtifactError,
            "schema_version",
        ):
            checker.validate_gateddelta_prefill_profile_artifact(path)

    def test_bad_preflight_head_dim_fails(self) -> None:
        artifact = valid_artifact()
        artifact["gateddelta_prefill_profile"]["model_preflight"]["linear_attention"][
            "key_head_dim"
        ] = 48
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(
            checker.GatedDeltaPrefillProfileArtifactError,
            "divisible by 32",
        ):
            checker.validate_gateddelta_prefill_profile_artifact(path)

    def test_ngram_row_fails(self) -> None:
        artifact = valid_artifact()
        artifact["results"].append(
            {
                **ax_row(512, 128),
                "engine": "ax_engine_mlx_ngram_accel",
                "ax_decode_policy": "ngram_acceleration_linear_attention_branch_recompute",
            }
        )
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(
            checker.GatedDeltaPrefillProfileArtifactError,
            "unsupported engine",
        ):
            checker.validate_gateddelta_prefill_profile_artifact(path)

    def test_missing_recurrent_counter_fails(self) -> None:
        artifact = valid_artifact()
        for row in artifact["results"]:
            if row["engine"] == "ax_engine_mlx" and row["prompt_tokens"] == 8192:
                row["ax_mlx_linear_attention_profile"][
                    "ax_mlx_linear_attention_profile_recurrent_wall_us"
                ] = 0
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(
            checker.GatedDeltaPrefillProfileArtifactError,
            "recurrent_wall_us",
        ):
            checker.validate_gateddelta_prefill_profile_artifact(path)

    def test_prompt_matrix_mismatch_fails(self) -> None:
        artifact = valid_artifact()
        artifact["results"] = [
            row for row in artifact["results"] if row["prompt_tokens"] != 32768
        ]
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(
            checker.GatedDeltaPrefillProfileArtifactError,
            "prompt matrix",
        ):
            checker.validate_gateddelta_prefill_profile_artifact(path)

    def test_cli_validates_artifact_file(self) -> None:
        path = self.write_fixture(valid_artifact())

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("4 shape groups validated", result.stdout)


if __name__ == "__main__":
    unittest.main()
