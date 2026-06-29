#!/usr/bin/env python3
"""Unit tests for the embedding correctness verifier contract."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).parent


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify = load_script("verify_embedding_models", "verify_embedding_models.py")


class VerifyEmbeddingModelsTests(unittest.TestCase):
    def write_manifest(self, root: Path, model_family: str = "qwen3") -> None:
        (root / "model-manifest.json").write_text(
            json.dumps({"model_family": model_family}) + "\n"
        )

    def test_infer_model_kind_uses_embeddinggemma_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root, model_family="embeddinggemma")
            self.assertEqual(verify.infer_model_kind(root, "auto"), "embeddinggemma")

    def test_infer_model_kind_defaults_non_embeddinggemma_to_qwen3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root, model_family="qwen3")
            self.assertEqual(verify.infer_model_kind(root, "auto"), "qwen3")

    def test_qwen_8b_4bit_uses_quantization_tolerant_threshold(self) -> None:
        model_dir = Path("/models/Qwen3-Embedding-8B-4bit-DWQ/snapshots/sha")
        self.assertEqual(
            verify.default_cosine_threshold(model_dir, "qwen3"),
            verify.QWEN_8B_4BIT_COSINE_THRESHOLD,
        )

    def test_embeddinggemma_contract_uses_reference_single_oracle(self) -> None:
        contract = verify.build_contract(
            Path("/models/embeddinggemma-300m-8bit"),
            "embeddinggemma",
            cosine_threshold=None,
            batch_consistency_threshold=None,
        )
        self.assertEqual(contract.reference, "mlx-embeddings")
        self.assertEqual(contract.pooling, "mean")
        self.assertTrue(contract.reference_single_oracle)

    def test_qwen_contract_uses_mlx_lm_last_pooling(self) -> None:
        contract = verify.build_contract(
            Path("/models/qwen3-embedding-0.6b-8bit"),
            "qwen3",
            cosine_threshold=None,
            batch_consistency_threshold=None,
        )
        self.assertEqual(contract.reference, "mlx-lm")
        self.assertEqual(contract.pooling, "last")
        self.assertTrue(contract.reference_single_oracle)

    def test_print_results_fails_low_batch_consistency(self) -> None:
        import numpy as np

        contract = verify.EmbeddingContract(
            model_kind="embeddinggemma",
            model_id="embeddinggemma",
            reference="mlx-embeddings",
            pooling="mean",
            cosine_threshold=0.999,
            batch_consistency_threshold=0.999,
            reference_single_oracle=True,
        )
        rows = verify.VerificationRows(
            token_ids=[[1, 2, 3]],
            reference_single=np.array([[1.0, 0.0]], dtype=np.float32),
            ax_single=np.array([[1.0, 0.0]], dtype=np.float32),
            ax_batch=np.array([[0.0, 1.0]], dtype=np.float32),
        )
        with patch("builtins.print"):
            self.assertFalse(verify.print_results(contract, rows))

    def test_print_results_passes_matching_single_and_batch(self) -> None:
        import numpy as np

        contract = verify.EmbeddingContract(
            model_kind="qwen3",
            model_id="qwen3",
            reference="mlx-lm",
            pooling="last",
            cosine_threshold=0.999,
            batch_consistency_threshold=0.999,
            reference_single_oracle=True,
        )
        rows = verify.VerificationRows(
            token_ids=[[1, 2, 3]],
            reference_single=np.array([[1.0, 0.0]], dtype=np.float32),
            ax_single=np.array([[1.0, 0.0]], dtype=np.float32),
            ax_batch=np.array([[1.0, 0.0]], dtype=np.float32),
        )
        with patch("builtins.print"):
            self.assertTrue(verify.print_results(contract, rows))


if __name__ == "__main__":
    unittest.main()
