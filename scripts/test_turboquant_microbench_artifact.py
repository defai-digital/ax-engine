#!/usr/bin/env python3
"""Unit tests for TurboQuant fused microbenchmark artifact validation."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_turboquant_microbench_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location("check_turboquant_microbench_artifact", SCRIPT_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
checker = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(checker)


def microbench_artifact() -> dict:
    return {
        "schema_version": "ax.turboquant_fused_decode_microbench.v1",
        "decode_path": "fused_compressed_decode",
        "kernel": "turboquant_fused_cold_decode_k8v4",
        "preset": "k8v4",
        "key_bits": 8,
        "value_bits": 4,
        "config": {
            "cold_tokens": [512, 8192],
            "hot_tokens": 128,
            "variants": ["dim_parallel", "two_stage_scores"],
        },
        "rows": [
            {
                "cold_tokens": 512,
                "n_kv_heads": 1,
                "head_dim": 128,
                "cpu_reference_wall_us": 700,
                "full_precision_cold_kv_bytes": 524288,
                "compressed_buffer_bytes": 122880,
                "estimated_cold_saved_bytes": 407552,
                "kernel_variants": [
                    variant("dim_parallel", median_us=4700),
                    variant("two_stage_scores", median_us=270),
                ],
            },
            {
                "cold_tokens": 8192,
                "n_kv_heads": 1,
                "head_dim": 128,
                "cpu_reference_wall_us": 11252,
                "full_precision_cold_kv_bytes": 8388608,
                "compressed_buffer_bytes": 1966080,
                "estimated_cold_saved_bytes": 6520832,
                "hot_tail_merge": {
                    "hot_tokens": 128,
                    "contract": "shared_logsumexp_partition_merge",
                    "host_wall_us": {
                        "median": 42,
                        "min": 40,
                        "max": 44,
                        "samples": [40, 42, 44],
                    },
                    "quality": {
                        "max_abs_diff": 1e-7,
                        "mean_abs_diff": 1e-8,
                        "min_cosine_similarity": 0.99999994,
                    },
                },
                "kernel_variants": [
                    variant("dim_parallel", median_us=78363),
                    variant("two_stage_scores", median_us=1041),
                ],
            },
        ],
    }


def variant(
    name: str,
    *,
    median_us: float,
    max_abs_diff: float = 1e-7,
    min_cosine_similarity: float = 0.99999994,
) -> dict:
    return {
        "name": name,
        "metal_wall_us": {
            "median": median_us,
            "min": median_us,
            "max": median_us,
            "samples": [median_us],
        },
        "token_heads_per_second": 1_000_000,
        "estimated_read_gib_s": 1.0,
        "quality": {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": max_abs_diff / 2.0,
            "min_cosine_similarity": min_cosine_similarity,
        },
    }


class TurboQuantMicrobenchArtifactTests(unittest.TestCase):
    def test_valid_microbench_artifact_passes(self) -> None:
        checker.validate_artifact(microbench_artifact())

    def test_missing_long_context_row_fails_closed(self) -> None:
        artifact = microbench_artifact()
        artifact["rows"] = [artifact["rows"][0]]

        with self.assertRaisesRegex(
            checker.MicrobenchArtifactValidationError,
            "cold_tokens >= 8192",
        ):
            checker.validate_artifact(artifact)

    def test_missing_two_stage_scores_variant_fails_closed(self) -> None:
        artifact = microbench_artifact()
        artifact["rows"][1]["kernel_variants"] = [variant("dim_parallel", median_us=78363)]

        with self.assertRaisesRegex(
            checker.MicrobenchArtifactValidationError,
            "two_stage_scores",
        ):
            checker.validate_artifact(artifact)

    def test_quality_regression_fails_closed(self) -> None:
        artifact = microbench_artifact()
        artifact["rows"][1]["kernel_variants"][1] = variant(
            "two_stage_scores",
            median_us=1041,
            max_abs_diff=0.01,
        )

        with self.assertRaisesRegex(
            checker.MicrobenchArtifactValidationError,
            "max_abs_diff",
        ):
            checker.validate_artifact(artifact)

    def test_hot_tail_merge_regression_fails_closed(self) -> None:
        artifact = microbench_artifact()
        artifact["rows"][1]["hot_tail_merge"]["quality"]["max_abs_diff"] = 0.01

        with self.assertRaisesRegex(
            checker.MicrobenchArtifactValidationError,
            "hot-tail merge max_abs_diff",
        ):
            checker.validate_artifact(artifact)

    def test_hot_tail_merge_config_mismatch_fails_closed(self) -> None:
        artifact = microbench_artifact()
        artifact["config"]["hot_tokens"] = 64

        with self.assertRaisesRegex(
            checker.MicrobenchArtifactValidationError,
            "config.hot_tokens",
        ):
            checker.validate_artifact(artifact)

    def test_hot_tail_merge_wall_time_shape_fails_closed(self) -> None:
        artifact = microbench_artifact()
        artifact["rows"][1]["hot_tail_merge"]["host_wall_us"]["samples"] = []

        with self.assertRaisesRegex(
            checker.MicrobenchArtifactValidationError,
            "host_wall_us.samples",
        ):
            checker.validate_artifact(artifact)

    def test_cpu_speedup_regression_fails_closed(self) -> None:
        artifact = microbench_artifact()
        artifact["rows"][1]["kernel_variants"][1] = variant("two_stage_scores", median_us=12000)

        with self.assertRaisesRegex(
            checker.MicrobenchArtifactValidationError,
            "speedup versus CPU",
        ):
            checker.validate_artifact(artifact)

    def test_cli_validates_artifact_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "microbench.json"
            artifact_path.write_text(json.dumps(microbench_artifact()))

            completed = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), str(artifact_path)],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ok:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
