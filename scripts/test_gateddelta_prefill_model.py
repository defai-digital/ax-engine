#!/usr/bin/env python3
"""Unit tests for GatedDelta prefill model preflight."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_gateddelta_prefill_model.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_gateddelta_prefill_model", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def write_model(
    root: Path,
    *,
    model_family: str | None = "qwen3_5",
    linear_attention: dict[str, object] | None = None,
) -> Path:
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    manifest: dict[str, object] = {
        "schema_version": "ax.native_model_manifest.v1",
        "model_family": model_family,
        "linear_attention": linear_attention
        if linear_attention is not None
        else {
            "num_value_heads": 4,
            "num_key_heads": 4,
            "key_head_dim": 64,
            "value_head_dim": 128,
            "conv_kernel_dim": 4,
        },
    }
    (model_dir / "model-manifest.json").write_text(json.dumps(manifest))
    return model_dir


class GatedDeltaPrefillModelTests(unittest.TestCase):
    def test_valid_qwen35_manifest_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = write_model(Path(tmp))

            metadata = checker.validate_gateddelta_prefill_model(model_dir)

        self.assertEqual(metadata["model_family"], "qwen3_5")
        self.assertEqual(metadata["schema_version"], checker.PREFLIGHT_SCHEMA_VERSION)
        self.assertEqual(metadata["linear_attention"]["key_head_dim"], 64)

    def test_valid_qwen_next_manifest_passes_with_explicit_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = write_model(
                Path(tmp),
                model_family="qwen3_next",
                linear_attention={
                    "full_attention_interval": 4,
                    "num_value_heads": 4,
                    "num_key_heads": 4,
                    "key_head_dim": 64,
                    "value_head_dim": 128,
                    "conv_kernel_dim": 4,
                },
            )

            metadata = checker.validate_gateddelta_prefill_model(model_dir)

        self.assertEqual(metadata["model_family"], "qwen3_next")
        self.assertEqual(metadata["linear_attention"]["full_attention_interval"], 4)

    def test_missing_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))

            with self.assertRaisesRegex(
                checker.GatedDeltaPrefillModelError,
                "model-manifest.json is missing",
            ):
                checker.validate_gateddelta_prefill_model(model_dir)

    def test_non_qwen_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = write_model(Path(tmp), model_family="gemma4")

            with self.assertRaisesRegex(
                checker.GatedDeltaPrefillModelError,
                "model_family must be",
            ):
                checker.validate_gateddelta_prefill_model(model_dir)

    def test_missing_linear_attention_field_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = write_model(
                Path(tmp),
                linear_attention={
                    "num_value_heads": 4,
                    "num_key_heads": 4,
                    "key_head_dim": 64,
                    "conv_kernel_dim": 4,
                },
            )

            with self.assertRaisesRegex(
                checker.GatedDeltaPrefillModelError,
                "value_head_dim",
            ):
                checker.validate_gateddelta_prefill_model(model_dir)

    def test_key_head_dim_alignment_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = write_model(
                Path(tmp),
                linear_attention={
                    "num_value_heads": 4,
                    "num_key_heads": 4,
                    "key_head_dim": 48,
                    "value_head_dim": 128,
                    "conv_kernel_dim": 4,
                },
            )

            with self.assertRaisesRegex(
                checker.GatedDeltaPrefillModelError,
                "divisible by 32",
            ):
                checker.validate_gateddelta_prefill_model(model_dir)

    def test_cli_json_outputs_normalized_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = write_model(Path(tmp))

            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), str(model_dir), "--json"],
                check=True,
                text=True,
                capture_output=True,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema_version"], checker.PREFLIGHT_SCHEMA_VERSION)
        self.assertEqual(payload["model_family"], "qwen3_5")


if __name__ == "__main__":
    unittest.main()
