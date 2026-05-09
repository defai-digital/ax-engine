#!/usr/bin/env python3
"""Unit tests for audit_dense_weights.py classification logic."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("audit_dense_weights.py")
MODULE_SPEC = importlib.util.spec_from_file_location("audit_dense_weights", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
auditor = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(auditor)


class ClassifyTensorTest(unittest.TestCase):
    def _cls(self, t: dict) -> tuple[str, str]:
        return auditor._classify_tensor(t)

    def test_quantized_projection(self):
        t = {"role": "attention_q", "dtype": "u32", "source_quantized": True,
             "quantization": {"mode": "affine", "group_size": 64, "bits": 4}}
        cls, action = self._cls(t)
        self.assertEqual(cls, "intentional_quantized")
        self.assertEqual(action, "none")

    def test_u32_without_source_quantized_is_converter_gap(self):
        t = {"role": "attention_q", "dtype": "u32"}
        cls, action = self._cls(t)
        self.assertEqual(cls, "converter_metadata_gap")
        self.assertEqual(action, "fix_converter")

    def test_norm_role_is_intentional_dense(self):
        for role in ["attention_norm", "ffn_norm", "final_norm", "attention_q_norm"]:
            cls, action = self._cls({"role": role, "dtype": "bf16"})
            self.assertEqual(cls, "intentional_dense", msg=role)
            self.assertEqual(action, "none", msg=role)

    def test_layer_scalar_is_non_gemv(self):
        cls, action = self._cls({"role": "layer_scalar", "dtype": "bf16", "shape": [1]})
        self.assertEqual(cls, "non_gemv")
        self.assertEqual(action, "none")

    def test_large_bf16_projection_no_source_quantized_is_document(self):
        t = {"role": "per_layer_model_projection", "dtype": "bf16", "shape": [8960, 1536]}
        cls, action = self._cls(t)
        self.assertEqual(cls, "intentional_dense")
        self.assertEqual(action, "document")

    def test_1d_projection_is_intentional_dense(self):
        t = {"role": "ffn_gate", "dtype": "bf16", "shape": [6144],
             "source_quantized": False}
        cls, action = self._cls(t)
        self.assertEqual(cls, "intentional_dense")
        self.assertEqual(action, "none")

    def test_explicitly_not_quantized_projection_is_document(self):
        t = {"role": "per_layer_model_projection", "dtype": "bf16",
             "shape": [8960, 1536], "source_quantized": False}
        cls, action = self._cls(t)
        self.assertEqual(cls, "intentional_dense")
        self.assertEqual(action, "document")

    def test_ffn_roles_quantized(self):
        for role in ["ffn_gate", "ffn_up", "ffn_down"]:
            t = {"role": role, "dtype": "u32", "source_quantized": True,
                 "quantization": {"bits": 4}}
            cls, action = self._cls(t)
            self.assertEqual(cls, "intentional_quantized", msg=role)

    def test_quantization_bits_extracted(self):
        t = {"role": "attention_q", "dtype": "u32", "source_quantized": True,
             "quantization": {"bits": 4, "group_size": 64}}
        self.assertEqual(auditor._quantization_bits(t), 4)

    def test_quantization_bits_zero_for_dense(self):
        self.assertEqual(auditor._quantization_bits({"dtype": "bf16"}), 0)

    def test_sidecar_present_true_for_quantized(self):
        t = {"source_quantized": True, "quantization": {"bits": 4}}
        self.assertTrue(auditor._sidecar_present(t))

    def test_sidecar_present_false_for_dense(self):
        self.assertFalse(auditor._sidecar_present({"dtype": "bf16"}))


class BuildAuditTest(unittest.TestCase):
    def _minimal_manifest(self, tensors: list[dict]) -> dict:
        return {
            "schema_version": 1,
            "model_family": "test",
            "layer_count": 1,
            "tensors": tensors,
        }

    def test_build_audit_produces_required_fields(self):
        tensors = [
            {"role": "attention_q", "layer_index": 0, "dtype": "u32",
             "source_quantized": True, "quantization": {"bits": 4},
             "shape": [64, 8]},
            {"role": "attention_norm", "layer_index": 0, "dtype": "bf16",
             "shape": [64]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "model-manifest.json"
            manifest_path.write_text(json.dumps(self._minimal_manifest(tensors)))
            audit = auditor.build_audit(manifest_path, "test-model", "abc1234")

        self.assertEqual(audit["model"], "test-model")
        self.assertEqual(audit["commit"], "abc1234")
        self.assertLen = len
        self.assertEqual(len(audit["tensors"]), 2)

        q_entry = audit["tensors"][0]
        self.assertEqual(q_entry["role"], "attention_q")
        self.assertEqual(q_entry["layer"], 0)
        self.assertEqual(q_entry["source_dtype"], "BF16")
        self.assertEqual(q_entry["quantization_bits"], 4)
        self.assertTrue(q_entry["sidecar_present"])
        self.assertEqual(q_entry["runtime_dtype"], "BF16")
        self.assertEqual(q_entry["classification"], "intentional_quantized")
        self.assertEqual(q_entry["action"], "none")

        n_entry = audit["tensors"][1]
        self.assertEqual(n_entry["classification"], "intentional_dense")
        self.assertEqual(n_entry["quantization_bits"], 0)
        self.assertFalse(n_entry["sidecar_present"])

    def test_main_writes_artifact(self):
        tensors = [
            {"role": "ffn_gate", "layer_index": 0, "dtype": "u32",
             "source_quantized": True, "quantization": {"bits": 4},
             "shape": [512, 64]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_dir = tmp_path / "my-model"
            model_dir.mkdir()
            (model_dir / "model-manifest.json").write_text(
                json.dumps(self._minimal_manifest(tensors))
            )
            output_root = tmp_path / "output"
            rc = auditor.main([
                "--model-dir", str(model_dir),
                "--model-id", "my-model",
                "--output-root", str(output_root),
            ])
            self.assertEqual(rc, 0)
            artifacts = list(output_root.glob("*.json"))
            self.assertEqual(len(artifacts), 1)
            artifact = json.loads(artifacts[0].read_text())
            self.assertEqual(artifact["model"], "my-model")

    def test_main_returns_2_on_converter_gap(self):
        tensors = [
            # u32 without source_quantized → converter_metadata_gap
            {"role": "attention_q", "layer_index": 0, "dtype": "u32",
             "shape": [64, 8]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_dir = tmp_path / "gap-model"
            model_dir.mkdir()
            (model_dir / "model-manifest.json").write_text(
                json.dumps(self._minimal_manifest(tensors))
            )
            rc = auditor.main([
                "--model-dir", str(model_dir),
                "--model-id", "gap-model",
                "--output-root", str(tmp_path / "out"),
            ])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
