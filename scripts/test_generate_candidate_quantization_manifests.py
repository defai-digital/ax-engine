#!/usr/bin/env python3
"""Unit tests for generate_candidate_quantization_manifests.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("generate_candidate_quantization_manifests.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "generate_candidate_quantization_manifests", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["generate_candidate_quantization_manifests"] = mod
MODULE_SPEC.loader.exec_module(mod)


def _write_model(
    model_dir: Path, *, config: dict | None = None, manifest: dict | None = None
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (model_dir / "config.json").write_text(json.dumps(config))
    if manifest is not None:
        (model_dir / "model-manifest.json").write_text(json.dumps(manifest))


class GenerateCandidateQuantizationManifestsTests(unittest.TestCase):
    def test_generates_all_default_recipes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            _write_model(root, config={"model_type": "qwen3"})

            result = mod.build_candidates(root)

        self.assertEqual(
            result["schema_version"], "ax.candidate_quantization_manifests.v1"
        )
        self.assertEqual(result["candidate_count"], len(mod.CANDIDATE_RECIPES))
        self.assertEqual(len(result["candidates"]), len(mod.CANDIDATE_RECIPES))

    def test_filters_by_recipe_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            _write_model(root, config={"model_type": "qwen3"})

            result = mod.build_candidates(root, recipe_ids=["uniform_3bit_g64"])

        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["recipe_id"], "uniform_3bit_g64")
        self.assertEqual(result["candidates"][0]["quantization"]["bits"], 3)

    def test_candidate_has_quantization_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            _write_model(root, config={"model_type": "gemma4"})

            result = mod.build_candidates(root, recipe_ids=["uniform_3bit_g64"])

        candidate = result["candidates"][0]
        self.assertIn("quantization", candidate)
        self.assertEqual(candidate["quantization"]["bits"], 3)
        self.assertEqual(candidate["quantization"]["group_size"], 64)
        self.assertEqual(candidate["quantization"]["layout"], "uniform")

    def test_mixed_recipe_has_ffn_bits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            _write_model(root, config={"model_type": "gemma4"})

            result = mod.build_candidates(
                root, recipe_ids=["mixed_4bit_attn_3bit_ffn_g64"]
            )

        candidate = result["candidates"][0]
        self.assertEqual(candidate["quantization"]["bits"], 4)
        self.assertEqual(candidate["quantization"]["ffn_bits"], 3)
        self.assertEqual(candidate["quantization"]["layout"], "mixed")

    def test_byte_estimate_with_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            _write_model(
                root,
                config={"model_type": "qwen3"},
                manifest={
                    "schema_version": "ax.native_model_manifest.v1",
                    "model_family": "qwen3",
                    "tensors": [
                        {"role": "attention_q", "length_bytes": 1_000_000},
                        {"role": "ffn_gate", "length_bytes": 2_000_000},
                    ],
                },
            )

            result = mod.build_candidates(root, recipe_ids=["uniform_3bit_g64"])

        candidate = result["candidates"][0]
        self.assertIsNotNone(candidate["byte_estimate"]["estimated_bytes"])
        self.assertGreater(candidate["byte_estimate"]["estimated_bytes"], 0)

    def test_byte_estimate_uses_actual_source_bit_width(self) -> None:
        # Regression test: estimate_candidate_bytes hardcoded a 16-bit
        # (fp16/bf16) source bit-width when backing out the element count
        # from length_bytes, regardless of the base model's real
        # quantization. Almost every real model directory in this repo is
        # already quantized (4-bit/8-bit on disk), so the hardcoded divisor
        # undercounted the element count (and thus every derived byte
        # estimate) whenever the source wasn't genuinely fp16.
        tensors = [
            {"role": "attention_q", "length_bytes": 1_000_000},
            {"role": "ffn_gate", "length_bytes": 2_000_000},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            fp16_root = Path(tmp) / "fp16_model"
            _write_model(
                fp16_root,
                config={"model_type": "qwen3"},
                manifest={
                    "schema_version": "ax.native_model_manifest.v1",
                    "model_family": "qwen3",
                    "tensors": tensors,
                },
            )
            fp16_result = mod.build_candidates(fp16_root, recipe_ids=["uniform_3bit_g64"])

            quantized_root = Path(tmp) / "quantized_model"
            _write_model(
                quantized_root,
                config={"model_type": "qwen3", "quantization": {"bits": 4}},
                manifest={
                    "schema_version": "ax.native_model_manifest.v1",
                    "model_family": "qwen3",
                    "tensors": tensors,
                },
            )
            quantized_result = mod.build_candidates(
                quantized_root, recipe_ids=["uniform_3bit_g64"]
            )

        fp16_bytes = fp16_result["candidates"][0]["byte_estimate"]["estimated_bytes"]
        quantized_bytes = quantized_result["candidates"][0]["byte_estimate"]["estimated_bytes"]
        # The already-4-bit source has 4x the element count for the same
        # length_bytes, so its 3-bit-target byte estimate must be ~4x larger.
        self.assertGreater(quantized_bytes, fp16_bytes * 3)

    def test_no_manifest_no_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            _write_model(root, config={"model_type": "qwen3"})

            result = mod.build_candidates(root, recipe_ids=["uniform_3bit_g64"])

        candidate = result["candidates"][0]
        self.assertIsNone(candidate["byte_estimate"]["estimated_bytes"])

    def test_missing_directory(self) -> None:
        result = mod.build_candidates(Path("/nonexistent/path"))
        self.assertEqual(result["error"], "directory_not_found")

    def test_output_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            _write_model(root, config={"model_type": "qwen3"})
            out_path = Path(tmp) / "out" / "candidates.json"

            import subprocess

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    str(root),
                    "--output",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertTrue(out_path.is_file())
            doc = json.loads(out_path.read_text())
            self.assertEqual(
                doc["schema_version"], "ax.candidate_quantization_manifests.v1"
            )


if __name__ == "__main__":
    unittest.main()
