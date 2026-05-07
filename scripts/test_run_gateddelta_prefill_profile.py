#!/usr/bin/env python3
"""Unit tests for the GatedDelta prefill profile wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run-gateddelta-prefill-profile.sh"


def write_model(root: Path, *, model_family: str = "gemma4") -> Path:
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_family}))
    (model_dir / "model-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ax.native_model_manifest.v1",
                "model_family": model_family,
                "linear_attention": {
                    "num_value_heads": 4,
                    "num_key_heads": 4,
                    "key_head_dim": 64,
                    "value_head_dim": 128,
                    "conv_kernel_dim": 4,
                },
            }
        )
    )
    return model_dir


class GatedDeltaPrefillProfileRunnerTests(unittest.TestCase):
    def test_dry_run_prints_plan_without_requiring_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            output_root = root / "out"

            result = subprocess.run(
                [
                    "bash",
                    str(RUNNER),
                    "--model-dir",
                    str(model_dir),
                    "--output-root",
                    str(output_root),
                    "--run-label",
                    "dry-run",
                    "--dry-run",
                ],
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "PYTHON_BIN": sys.executable},
            )

            artifact = output_root / "dry-run/gateddelta-prefill-profile.json"
            output_dir_exists = (output_root / "dry-run").exists()
            artifact_exists = artifact.exists()

        self.assertEqual(result.returncode, 0)
        self.assertIn("Preflight:", result.stdout)
        self.assertIn("Command:", result.stdout)
        self.assertIn("--gateddelta-prefill-profile", result.stdout)
        self.assertNotIn("model-manifest.json is missing", result.stdout)
        self.assertFalse(output_dir_exists)
        self.assertFalse(artifact_exists)

    def test_non_dry_run_fails_model_preflight_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = write_model(root, model_family="gemma4")
            output_root = root / "out"

            result = subprocess.run(
                [
                    "bash",
                    str(RUNNER),
                    "--model-dir",
                    str(model_dir),
                    "--output-root",
                    str(output_root),
                    "--run-label",
                    "invalid-model",
                ],
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "PYTHON_BIN": sys.executable},
            )

            artifact = output_root / "invalid-model/gateddelta-prefill-profile.json"
            artifact_exists = artifact.exists()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Preflight:", result.stdout)
        self.assertIn("model_family must be", result.stdout)
        self.assertNotIn("cargo build", result.stdout)
        self.assertNotIn("Compiling", result.stderr)
        self.assertFalse(artifact_exists)


if __name__ == "__main__":
    unittest.main()
