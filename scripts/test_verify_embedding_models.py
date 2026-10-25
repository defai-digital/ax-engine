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

    def test_incidental_8b_inside_snapshot_hash_does_not_trigger_tolerant_threshold(
        self,
    ) -> None:
        # Regression test: a plain substring check for "8b" matched
        # incidentally inside a HuggingFace-cache snapshot hash (both hex
        # digits), silently loosening the cosine-similarity gate for an
        # unrelated 0.6B model whose snapshot hash happens to contain "8b".
        model_dir = Path("/models/Qwen3-Embedding-0.6B-4bit/snapshots/a8b3f9e2c1")
        self.assertEqual(
            verify.default_cosine_threshold(model_dir, "qwen3"),
            verify.DEFAULT_COSINE_THRESHOLD,
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

    def test_gemma2_contract_uses_its_own_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root, "embedding_gemma2")
            self.assertEqual(verify.infer_model_kind(root, "auto"), "embedding_gemma2")
            contract = verify.build_contract(root, "embedding_gemma2", None, None)
            self.assertEqual(contract.reference, "mlx-vlm")
            self.assertEqual(contract.pooling, "mean")

    def test_direct_verifier_exercises_single_api_instead_of_batch_of_one(self) -> None:
        from unittest.mock import Mock
        import numpy as np

        contract = verify.build_contract(Path("/models/gemma"), "embeddinggemma", None, None)
        session = Mock()
        session.embed.return_value = [1.0, 0.0]
        with patch.object(verify, "tokenize_sentences", return_value=[[1, 2, 3]]), \
             patch.object(verify, "compute_reference_single", return_value=np.array([[1.0, 0.0]])), \
             patch.object(verify, "load_ax_session", return_value=session), \
             patch.object(verify, "ax_batch_matrix", return_value=np.array([[1.0, 0.0]])) as batch:
            verify.verify_direct(Path("/models/gemma"), contract)
        session.embed.assert_called_once_with([1, 2, 3], pooling="mean", normalize=True)
        self.assertEqual(batch.call_count, 1)
        session.close.assert_called_once()

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
