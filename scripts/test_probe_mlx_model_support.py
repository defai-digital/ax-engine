#!/usr/bin/env python3
"""Unit tests for the MLX model support probe."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("probe_mlx_model_support.py")
MODULE_SPEC = importlib.util.spec_from_file_location("probe_mlx_model_support", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
probe = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(probe)


def write_model(root: Path, model_type: str, keys: list[str]) -> Path:
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model-00001.safetensors" for key in keys}})
    )
    return model_dir


class MlxModelSupportProbeTests(unittest.TestCase):
    def test_glm_is_implementation_candidate_when_references_and_features_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            mlx_lm = repo / ".internal/reference/mlx-lm/mlx_lm/models"
            swift = repo / ".internal/reference/mlx-swift-lm/Libraries/MLXLLM/Models"
            mlx_lm.mkdir(parents=True)
            swift.mkdir(parents=True)
            (mlx_lm / "glm4_moe_lite.py").write_text(
                "q_a_proj kv_a_proj_with_mqa e_score_correction_bias"
            )
            (swift / "GLM4MOELite.swift").write_text(
                "GLM4MoELiteAttention GLM4MoELiteGate eScoreCorrectionBias"
            )
            model_dir = write_model(
                root,
                "glm4_moe_lite",
                [
                    "model.layers.0.self_attn.q_a_proj.weight",
                    "model.layers.0.self_attn.q_b_proj.weight",
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                    "model.layers.0.self_attn.embed_q.weight",
                    "model.layers.0.self_attn.unembed_out.weight",
                    "model.layers.1.mlp.gate.e_score_correction_bias",
                    "model.layers.1.mlp.experts.gate_up_proj.weight",
                ],
            )

            with patch.object(probe, "REPO_ROOT", repo):
                report = probe.probe_model(model_dir)

        self.assertEqual(report["support_decision"], "implementation_candidate")
        self.assertTrue(report["can_implement_repo_owned_runtime"])
        self.assertEqual(report["reference_support"], "complete_enough_for_ax_port")
        self.assertIn("GLM4MoELite MLA", " ".join(report["blockers"]))

    def test_deepseek_v4_fails_closed_when_partial_reference_drops_required_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            swift = repo / ".internal/reference/SwiftLM/mlx-swift-lm/Libraries/MLXLLM/Models"
            tests = repo / ".internal/reference/SwiftLM/mlx-swift-lm/Tests/MLXLMTests"
            swift.mkdir(parents=True)
            tests.mkdir(parents=True)
            (swift / "DeepseekV4.swift").write_text(
                "DeepseekV4Attention compressor/indexer tid2eid"
            )
            (tests / "DeepseekV4Tests.swift").write_text(
                "Compressor/indexer sub-modules are not yet implemented"
            )
            model_dir = write_model(
                root,
                "deepseek_v4",
                [
                    "model.layers.0.attn.attn_sink",
                    "model.layers.0.attn.wo_a.weight",
                    "model.layers.0.attn.wo_b.weight",
                    "model.layers.0.attn.compressor.wkv.weight",
                    "model.layers.0.attn.indexer.wq_b.weight",
                    "model.layers.0.ffn.gate.tid2eid",
                    "model.layers.1.ffn.gate.e_score_correction_bias",
                ],
            )

            with patch.object(probe, "REPO_ROOT", repo):
                report = probe.probe_model(model_dir)

        self.assertEqual(report["support_decision"], "fail_closed_partial_reference")
        self.assertFalse(report["can_implement_repo_owned_runtime"])
        self.assertEqual(report["reference_support"], "partial_only")
        self.assertTrue(report["checkpoint_features"]["attention_compressor"])
        self.assertIn("tid2eid", " ".join(report["blockers"]))


if __name__ == "__main__":
    unittest.main()
