#!/usr/bin/env python3
"""Unit tests for check_quantization_quality_gate.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_quantization_quality_gate.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_quantization_quality_gate", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["check_quantization_quality_gate"] = mod
MODULE_SPEC.loader.exec_module(mod)


def _write_artifact(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


class CheckQuantizationQualityGateTests(unittest.TestCase):
    def test_approve_good_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "quality.json"
            _write_artifact(
                artifact,
                {
                    "schema_version": "ax.quantization_quality_gate.v1",
                    "quality_metrics": {
                        "cosine_similarity": 0.999,
                        "mean_abs_diff": 0.01,
                        "max_abs_diff": 0.05,
                    },
                    "speed_metrics": {"decode_tok_s": 40.0},
                    "baseline_decode_tok_s": 35.0,
                },
            )
            result = mod.validate_artifact(artifact)

        self.assertEqual(result["decision"], "approve")
        self.assertEqual(result["total_failures"], 0)

    def test_reject_low_cosine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "quality.json"
            _write_artifact(
                artifact,
                {
                    "schema_version": "ax.quantization_quality_gate.v1",
                    "quality_metrics": {"cosine_similarity": 0.990},
                    "speed_metrics": {},
                },
            )
            result = mod.validate_artifact(artifact)

        self.assertEqual(result["decision"], "reject")
        self.assertEqual(len(result["quality_failures"]), 1)
        self.assertIn("cosine_similarity", result["quality_failures"][0])

    def test_reject_high_mean_abs_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "quality.json"
            _write_artifact(
                artifact,
                {
                    "schema_version": "ax.quantization_quality_gate.v1",
                    "quality_metrics": {"mean_abs_diff": 0.10},
                    "speed_metrics": {},
                },
            )
            result = mod.validate_artifact(artifact)

        self.assertEqual(result["decision"], "reject")
        self.assertIn("mean_abs_diff", result["quality_failures"][0])

    def test_reject_slow_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "quality.json"
            _write_artifact(
                artifact,
                {
                    "schema_version": "ax.quantization_quality_gate.v1",
                    "quality_metrics": {"cosine_similarity": 0.999},
                    "speed_metrics": {"decode_tok_s": 25.0},
                    "baseline_decode_tok_s": 35.0,
                },
            )
            result = mod.validate_artifact(artifact)

        self.assertEqual(result["decision"], "reject")
        self.assertEqual(len(result["speed_failures"]), 1)

    def test_reject_missing_artifact(self) -> None:
        result = mod.validate_artifact(Path("/nonexistent/artifact.json"))
        self.assertEqual(result["decision"], "reject")
        self.assertEqual(result["reason"], "artifact_not_found")

    def test_reject_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "quality.json"
            _write_artifact(artifact, {"schema_version": "wrong.version"})
            result = mod.validate_artifact(artifact)

        self.assertEqual(result["decision"], "reject")
        self.assertIn("schema_version mismatch", result["reason"])

    def test_no_baseline_skips_speed_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "quality.json"
            _write_artifact(
                artifact,
                {
                    "schema_version": "ax.quantization_quality_gate.v1",
                    "quality_metrics": {"cosine_similarity": 0.999},
                    "speed_metrics": {"decode_tok_s": 10.0},
                },
            )
            result = mod.validate_artifact(artifact)

        self.assertEqual(result["decision"], "approve")
        self.assertEqual(len(result["speed_failures"]), 0)


if __name__ == "__main__":
    unittest.main()
