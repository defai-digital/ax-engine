#!/usr/bin/env python3
"""Unit tests for direct-MLX hotpath probe artifact validation."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_direct_mlx_hotpath_probe_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_direct_mlx_hotpath_probe_artifact",
    SCRIPT_PATH,
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
checker = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(checker)


def direct_mlx_artifact(candidate: str = "gelu_approx_mul") -> dict:
    if candidate == "gelu_approx_mul":
        portable_name = "portable_gelu_approx_mul"
        direct_name = "direct_cpp_gelu_approx_mul"
        shape = [8, 16]
    elif candidate == "gelu_approx_mul_matmul":
        portable_name = "portable_gelu_approx_mul_matmul"
        direct_name = "direct_cpp_gelu_approx_mul_matmul"
        shape = [8, 4]
    elif candidate == "gelu_approx_quantized_ffn":
        portable_name = "portable_gelu_approx_quantized_ffn"
        direct_name = "direct_cpp_gelu_approx_quantized_ffn"
        shape = [8, 4]
    elif candidate == "qk_norm_rope":
        portable_name = "portable_qk_norm_rope"
        direct_name = "direct_cpp_qk_norm_rope"
        # cols=16 splits into n_heads=4 * head_dim=4; output is BHSD.
        shape = [1, 4, 8, 4]
    else:
        raise AssertionError(f"unknown candidate {candidate}")
    return {
        "schema": "ax.microbench.v1",
        "surface": "direct-mlx-hotpath",
        "command": f"target/debug/direct-mlx-hotpath-probe --candidate {candidate}",
        "git": {"commit": "abc123", "dirty": True},
        "host": {"os": "macos", "arch": "aarch64"},
        "config": {
            "candidate": candidate,
            "rows": 8,
            "cols": 16,
            "down_cols": 4,
            "head_dim": 4,
            "n_heads": 4,
            "dtype": "float32",
            "group_size": 64,
            "bits": 4,
            "warmup": 1,
            "iterations": 3,
        },
        "correctness": {
            "passed": True,
            "max_abs_error": 0.0,
            "tolerance": 1e-6,
            "shape": shape,
        },
        "measurements": [
            timing(portable_name, median=400.0, op_count_median=10),
            timing(direct_name, median=200.0, op_count_median=1),
            {
                "name": "direct_cpp_speedup_ratio",
                "unit": "ratio",
                "samples": 1,
                "mean": 2.0,
                "median": 2.0,
                "min": 2.0,
                "max": 2.0,
            },
        ],
    }


def timing(name: str, *, median: float, op_count_median: int) -> dict:
    return {
        "name": name,
        "unit": "microseconds",
        "samples": 3,
        "mean": median,
        "median": median,
        "min": median * 0.9,
        "max": median * 1.1,
        "op_count_median": op_count_median,
    }


class DirectMlxHotpathProbeArtifactTests(unittest.TestCase):
    def test_valid_artifact_passes(self) -> None:
        checker.validate_artifact(direct_mlx_artifact())

    def test_valid_activation_down_artifact_passes(self) -> None:
        checker.validate_artifact(direct_mlx_artifact("gelu_approx_mul_matmul"))

    def test_valid_quantized_ffn_artifact_passes(self) -> None:
        checker.validate_artifact(direct_mlx_artifact("gelu_approx_quantized_ffn"))

    def test_valid_qk_norm_rope_artifact_passes(self) -> None:
        checker.validate_artifact(direct_mlx_artifact("qk_norm_rope"))

    def test_qk_norm_rope_rejects_head_dim_n_heads_cols_mismatch(self) -> None:
        artifact = direct_mlx_artifact("qk_norm_rope")
        # cols=16 no longer matches n_heads * head_dim once n_heads is bumped.
        artifact["config"]["n_heads"] = 5

        with self.assertRaisesRegex(
            checker.DirectMlxHotpathProbeArtifactError,
            r"n_heads \* config\.head_dim must equal config\.cols",
        ):
            checker.validate_artifact(artifact)

    def test_qk_norm_rope_rejects_2d_shape(self) -> None:
        artifact = direct_mlx_artifact("qk_norm_rope")
        # The shape contract for qk_norm_rope is BHSD ([1, n_heads, rows,
        # head_dim]). Falling back to the matmul-style [rows, output_cols]
        # shape must fail closed.
        artifact["correctness"]["shape"] = [8, 16]

        with self.assertRaisesRegex(checker.DirectMlxHotpathProbeArtifactError, "shape"):
            checker.validate_artifact(artifact)

    def test_quantized_ffn_rejects_unsupported_group_size(self) -> None:
        artifact = direct_mlx_artifact("gelu_approx_quantized_ffn")
        artifact["config"]["group_size"] = 16

        with self.assertRaisesRegex(checker.DirectMlxHotpathProbeArtifactError, "group_size"):
            checker.validate_artifact(artifact)

    def test_wrong_schema_fails_closed(self) -> None:
        artifact = direct_mlx_artifact()
        artifact["schema"] = "other"

        with self.assertRaisesRegex(checker.DirectMlxHotpathProbeArtifactError, "schema"):
            checker.validate_artifact(artifact)

    def test_correctness_failure_fails_closed(self) -> None:
        artifact = direct_mlx_artifact()
        artifact["correctness"]["passed"] = False

        with self.assertRaisesRegex(checker.DirectMlxHotpathProbeArtifactError, "correctness"):
            checker.validate_artifact(artifact)

    def test_shape_mismatch_fails_closed(self) -> None:
        artifact = direct_mlx_artifact("gelu_approx_mul_matmul")
        artifact["correctness"]["shape"] = [8, 15]

        with self.assertRaisesRegex(checker.DirectMlxHotpathProbeArtifactError, "shape"):
            checker.validate_artifact(artifact)

    def test_missing_direct_measurement_fails_closed(self) -> None:
        artifact = direct_mlx_artifact()
        artifact["measurements"] = [artifact["measurements"][0], artifact["measurements"][2]]

        with self.assertRaisesRegex(checker.DirectMlxHotpathProbeArtifactError, "direct_cpp"):
            checker.validate_artifact(artifact)

    def test_direct_op_count_must_be_lower_than_portable(self) -> None:
        artifact = direct_mlx_artifact()
        artifact["measurements"][1]["op_count_median"] = 10

        with self.assertRaisesRegex(checker.DirectMlxHotpathProbeArtifactError, "op_count_median"):
            checker.validate_artifact(artifact)

    def test_optional_speedup_gate_fails_closed(self) -> None:
        artifact = direct_mlx_artifact()

        with self.assertRaisesRegex(checker.DirectMlxHotpathProbeArtifactError, "median"):
            checker.validate_artifact(artifact, min_speedup=2.05)

    def test_speedup_claim_must_match_timing_medians(self) -> None:
        artifact = direct_mlx_artifact()
        artifact["measurements"][2]["median"] = 9.0
        artifact["measurements"][2]["mean"] = 9.0
        artifact["measurements"][2]["min"] = 9.0
        artifact["measurements"][2]["max"] = 9.0

        with self.assertRaisesRegex(
            checker.DirectMlxHotpathProbeArtifactError, "portable/direct"
        ):
            checker.validate_artifact(artifact)

    def test_speedup_row_stats_must_match_derived_ratio(self) -> None:
        artifact = direct_mlx_artifact()
        artifact["measurements"][2]["mean"] = 9.0

        with self.assertRaisesRegex(
            checker.DirectMlxHotpathProbeArtifactError, "direct_cpp_speedup_ratio.mean"
        ):
            checker.validate_artifact(artifact)

    def test_speedup_row_is_single_derived_sample(self) -> None:
        artifact = direct_mlx_artifact()
        artifact["measurements"][2]["samples"] = 3

        with self.assertRaisesRegex(
            checker.DirectMlxHotpathProbeArtifactError, "derived scalar"
        ):
            checker.validate_artifact(artifact)

    def test_cli_validates_artifact_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "direct-mlx-hotpath.json"
            artifact_path.write_text(json.dumps(direct_mlx_artifact()))

            completed = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), str(artifact_path), "--min-speedup", "1.05"],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ok:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
