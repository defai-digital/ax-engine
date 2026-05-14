#!/usr/bin/env python3
"""Unit tests for the TurboQuant quality artifact runner wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run-turboquant-quality-artifact.sh"


def write_model(root: Path, *, head_dim: int = 128) -> Path:
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "model-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ax.native_model_manifest.v1",
                "model_family": "gemma4",
                "global_head_dim": head_dim,
            }
        )
    )
    return model_dir


def run_runner(*args: str, model_dir: Path, output_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model-dir",
            str(model_dir),
            "--output-root",
            str(output_root),
            *args,
        ],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHON_BIN": sys.executable},
    )


class TurboQuantQualityArtifactRunnerTests(unittest.TestCase):
    def test_dry_run_defaults_match_promotion_readiness_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = write_model(root)
            output_root = root / "out"

            result = run_runner(
                "--run-label",
                "dry-run",
                "--dry-run",
                model_dir=model_dir,
                output_root=output_root,
            )

            run_dirs = list(output_root.glob("*-dry-run"))
            quality_gate_exists = any((path / "quality-gate.json").exists() for path in run_dirs)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dry_run=true", result.stdout)
        self.assertIn("--repetitions 3", result.stdout)
        self.assertIn("--cooldown 3", result.stdout)
        self.assertIn("--generation-tokens 256", result.stdout)
        self.assertIn("--prompt-tokens 8192", result.stdout)
        self.assertEqual(len(run_dirs), 1)
        self.assertFalse(quality_gate_exists)

    def test_dry_run_fails_fast_for_non_promotable_shapes(self) -> None:
        cases = [
            (
                ["--context-tokens", "4096"],
                "--context-tokens must be an integer >= 8192",
            ),
            (
                ["--generation-tokens", "64"],
                "--generation-tokens must be an integer >= 128",
            ),
            (
                ["--repetitions", "1"],
                "--repetitions must be an integer >= 2",
            ),
            (
                ["--cooldown", "0"],
                "--cooldown must be > 0",
            ),
            (
                ["--cooldown", "none"],
                "--cooldown must be a positive number",
            ),
            (
                ["--head-dim", "64"],
                "--head-dim must be 128, 256, or 512",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = write_model(root)
            output_root = root / "out"

            for args, expected_error in cases:
                with self.subTest(args=args):
                    result = run_runner(
                        *args,
                        "--dry-run",
                        model_dir=model_dir,
                        output_root=output_root,
                    )

                    self.assertEqual(result.returncode, 2)
                    self.assertIn(expected_error, result.stderr)
                    self.assertNotIn("Building release server binary", result.stdout)


if __name__ == "__main__":
    unittest.main()
