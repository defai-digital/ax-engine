#!/usr/bin/env python3
"""Unit tests for MLX artifact shell wrapper defaults."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PREFILL_RUNNER = SCRIPT_DIR / "run-mlx-prefill-scaling-artifact.sh"
P2_RUNNER = SCRIPT_DIR / "run-mlx-p2-latency-artifacts.sh"


def write_model(root: Path) -> Path:
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen"}))
    return model_dir


class MlxArtifactWrapperTests(unittest.TestCase):
    def run_wrapper(self, runner: Path) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = write_model(root)
            output_root = root / "out"

            return subprocess.run(
                [
                    "bash",
                    str(runner),
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

    def test_prefill_scaling_wrapper_defaults_to_cooldown_15(self) -> None:
        result = self.run_wrapper(PREFILL_RUNNER)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--repetitions 5", result.stdout)
        self.assertIn("--cooldown 15", result.stdout)

    def test_p2_latency_wrapper_defaults_to_cooldown_15(self) -> None:
        result = self.run_wrapper(P2_RUNNER)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--repetitions 5", result.stdout)
        self.assertIn("--cooldown 15", result.stdout)


if __name__ == "__main__":
    unittest.main()
