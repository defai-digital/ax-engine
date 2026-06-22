#!/usr/bin/env python3
"""Tests for prepare_glm_mtp_sidecar.py.

Pure-Python fakes only (no MLX import) so the suite runs in the script hygiene
gate.  Verifies:
  - _rename_key maps all expected model.layers.47.* keys correctly
  - embed_tokens.weight is skipped
  - _stack_per_expert_tensors stacks per-expert tensors into [E, D, in] arrays
  - _should_quantize matches the expected eligibility contract
  - The runtime contract contains the required fields
  - The provenance manifest schema is written correctly
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load(name: str):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = _load("prepare_glm_mtp_sidecar")


# ── fake array stubs ─────────────────────────────────────────────────────────
class FakeArray:
    """Minimal array stub with .shape attribute."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = "bfloat16"


# ── rename key tests ─────────────────────────────────────────────────────────
class RenameKeyTests(unittest.TestCase):
    """_rename_key must map all model.layers.47.* keys to glm_mtp.* correctly."""

    def test_enorm_rename(self) -> None:
        self.assertEqual(
            prepare._rename_key("model.layers.47.enorm.weight"),
            "glm_mtp.enorm.weight",
        )

    def test_hnorm_rename(self) -> None:
        self.assertEqual(
            prepare._rename_key("model.layers.47.hnorm.weight"),
            "glm_mtp.hnorm.weight",
        )

    def test_eh_proj_rename(self) -> None:
        self.assertEqual(
            prepare._rename_key("model.layers.47.eh_proj.weight"),
            "glm_mtp.eh_proj.weight",
        )

    def test_shared_head_head_rename(self) -> None:
        self.assertEqual(
            prepare._rename_key("model.layers.47.shared_head.head.weight"),
            "glm_mtp.shared_head.head.weight",
        )

    def test_shared_head_norm_rename(self) -> None:
        self.assertEqual(
            prepare._rename_key("model.layers.47.shared_head.norm.weight"),
            "glm_mtp.shared_head.norm.weight",
        )

    def test_input_layernorm_rename(self) -> None:
        self.assertEqual(
            prepare._rename_key("model.layers.47.input_layernorm.weight"),
            "glm_mtp.layer.input_layernorm.weight",
        )

    def test_post_attention_layernorm_rename(self) -> None:
        self.assertEqual(
            prepare._rename_key("model.layers.47.post_attention_layernorm.weight"),
            "glm_mtp.layer.post_attention_layernorm.weight",
        )

    def test_self_attn_rename(self) -> None:
        self.assertEqual(
            prepare._rename_key("model.layers.47.self_attn.q_a_proj.weight"),
            "glm_mtp.layer.self_attn.q_a_proj.weight",
        )
        self.assertEqual(
            prepare._rename_key("model.layers.47.self_attn.kv_a_proj.weight"),
            "glm_mtp.layer.self_attn.kv_a_proj.weight",
        )
        self.assertEqual(
            prepare._rename_key("model.layers.47.self_attn.o_proj.weight"),
            "glm_mtp.layer.self_attn.o_proj.weight",
        )

    def test_mlp_rename(self) -> None:
        self.assertEqual(
            prepare._rename_key("model.layers.47.mlp.gate.weight"),
            "glm_mtp.layer.mlp.gate.weight",
        )
        self.assertEqual(
            prepare._rename_key("model.layers.47.mlp.experts.0.gate_proj.weight"),
            "glm_mtp.layer.mlp.experts.0.gate_proj.weight",
        )

    def test_embed_tokens_is_skipped(self) -> None:
        result = prepare._rename_key("model.layers.47.embed_tokens.weight")
        self.assertIsNone(result, "embed_tokens.weight must be skipped (shared with main model)")

    def test_wrong_layer_returns_none(self) -> None:
        self.assertIsNone(prepare._rename_key("model.layers.46.enorm.weight"))
        self.assertIsNone(prepare._rename_key("model.layers.0.self_attn.q_proj.weight"))

    def test_non_layer_key_returns_none(self) -> None:
        self.assertIsNone(prepare._rename_key("model.embed_tokens.weight"))
        self.assertIsNone(prepare._rename_key("lm_head.weight"))


# ── expert stacking tests ─────────────────────────────────────────────────────
class FakeStackable:
    """Fake array that records stacking calls."""

    def __init__(self, tag: str):
        self.shape = (4, 16)   # dummy 2-D shape
        self.tag = tag

    def __repr__(self) -> str:
        return f"FakeStackable({self.tag!r})"


def _make_mx_mock():
    """Return a minimal mlx.core stub that captures stack calls."""
    import types

    mx = types.ModuleType("mlx.core")
    stacked_calls: list[list] = []

    def _stack(arrays, axis=0):
        stacked_calls.append(list(arrays))
        return ("stacked", axis, arrays)

    def _eval(arr):
        pass

    mx.stack = _stack  # type: ignore[attr-defined]
    mx.eval = _eval  # type: ignore[attr-defined]
    stacked_calls_ref = stacked_calls
    mx._stacked_calls = stacked_calls_ref  # type: ignore[attr-defined]
    return mx


class ExpertStackTests(unittest.TestCase):
    """_stack_per_expert_tensors stacks per-expert arrays in expert-index order."""

    def setUp(self) -> None:
        # Patch mlx.core with a fake during these tests.
        import sys

        self._orig = sys.modules.get("mlx.core")
        self._mx = _make_mx_mock()
        sys.modules["mlx.core"] = self._mx

    def tearDown(self) -> None:
        import sys

        if self._orig is None:
            sys.modules.pop("mlx.core", None)
        else:
            sys.modules["mlx.core"] = self._orig

    def test_per_expert_keys_are_stacked_and_removed(self) -> None:
        tensors = {
            "glm_mtp.layer.mlp.experts.0.gate_proj.weight": FakeStackable("g0"),
            "glm_mtp.layer.mlp.experts.1.gate_proj.weight": FakeStackable("g1"),
            "glm_mtp.layer.mlp.experts.0.up_proj.weight": FakeStackable("u0"),
            "glm_mtp.layer.mlp.experts.1.up_proj.weight": FakeStackable("u1"),
            "glm_mtp.layer.mlp.experts.0.down_proj.weight": FakeStackable("d0"),
            "glm_mtp.layer.mlp.experts.1.down_proj.weight": FakeStackable("d1"),
            "glm_mtp.layer.mlp.gate.weight": FakeStackable("router"),
        }
        out = prepare._stack_per_expert_tensors(tensors)
        # Per-expert keys are removed; stacked keys appear.
        self.assertIn("glm_mtp.layer.mlp.gate_proj.weight", out)
        self.assertIn("glm_mtp.layer.mlp.up_proj.weight", out)
        self.assertIn("glm_mtp.layer.mlp.down_proj.weight", out)
        self.assertNotIn("glm_mtp.layer.mlp.experts.0.gate_proj.weight", out)
        # Non-expert passthrough key preserved.
        self.assertIn("glm_mtp.layer.mlp.gate.weight", out)

    def test_no_expert_keys_passthrough(self) -> None:
        tensors = {
            "glm_mtp.enorm.weight": FakeStackable("e"),
            "glm_mtp.layer.input_layernorm.weight": FakeStackable("n"),
        }
        out = prepare._stack_per_expert_tensors(tensors)
        self.assertEqual(set(out), set(tensors))


# ── quantization eligibility tests ──────────────────────────────────────────
class QuantizeEligibilityTests(unittest.TestCase):
    GROUP_SIZE = 64

    def test_2d_projection_eligible(self) -> None:
        self.assertTrue(
            prepare._should_quantize("glm_mtp.eh_proj.weight", (256, 128), self.GROUP_SIZE)
        )
        self.assertTrue(
            prepare._should_quantize("glm_mtp.layer.self_attn.q_a_proj.weight", (64, 64), self.GROUP_SIZE)
        )

    def test_1d_norm_not_eligible(self) -> None:
        self.assertFalse(
            prepare._should_quantize("glm_mtp.enorm.weight", (256,), self.GROUP_SIZE)
        )
        self.assertFalse(
            prepare._should_quantize("glm_mtp.layer.input_layernorm.weight", (256,), self.GROUP_SIZE)
        )

    def test_router_gate_not_eligible(self) -> None:
        self.assertFalse(
            prepare._should_quantize("glm_mtp.layer.mlp.gate.weight", (16, 256), self.GROUP_SIZE)
        )

    def test_3d_expert_stacks_not_eligible(self) -> None:
        self.assertFalse(
            prepare._should_quantize(
                "glm_mtp.layer.mlp.gate_proj.weight", (4, 256, 128), self.GROUP_SIZE
            )
        )

    def test_last_dim_too_small_not_eligible(self) -> None:
        self.assertFalse(
            prepare._should_quantize("glm_mtp.eh_proj.weight", (256, 32), self.GROUP_SIZE)
        )


# ── runtime contract tests ───────────────────────────────────────────────────
class RuntimeContractTests(unittest.TestCase):
    def test_required_fields_present(self) -> None:
        contract = prepare._runtime_contract(
            depth_max=1,
            base_repo="mlx-community/GLM-4.7-Flash-6bit",
            tensor_count=42,
            quant_bits=None,
        )
        self.assertEqual(contract["arch_id"], "glm-4.7-flash-mtp")
        self.assertEqual(contract["mtp_depth_max"], 1)
        self.assertIn("recommended_draft_sampler", contract)
        self.assertIn("verified_on", contract)

    def test_quant_hint_present_when_quantized(self) -> None:
        contract = prepare._runtime_contract(
            depth_max=1,
            base_repo="mlx-community/GLM-4.7-Flash-6bit",
            tensor_count=42,
            quant_bits=4,
        )
        self.assertIn("mtp_sidecar", contract)
        self.assertIn("INT4", contract["mtp_sidecar"])

    def test_no_quant_hint_when_bf16(self) -> None:
        contract = prepare._runtime_contract(
            depth_max=1,
            base_repo="mlx-community/GLM-4.7-Flash-6bit",
            tensor_count=42,
            quant_bits=None,
        )
        self.assertNotIn("mtp_sidecar", contract)


# ── provenance manifest schema tests ─────────────────────────────────────────
class ProvenanceManifestTests(unittest.TestCase):
    def _make_record(self, name: str = "x") -> dict:
        return {
            "path": f"/tmp/{name}",
            "exists": True,
            "size_bytes": 16,
            "sha256": "a" * 64,
        }

    def test_schema_version_and_arch_id(self) -> None:
        runtime = prepare._runtime_contract(
            depth_max=1,
            base_repo="mlx-community/GLM-4.7-Flash-6bit",
            tensor_count=42,
            quant_bits=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text("{}")
            manifest = prepare._provenance_manifest(
                base_model_id="mlx-community/GLM-4.7-Flash-6bit",
                base_snapshot_dir=tmp,
                source_model_id="zai-org/GLM-4.7-Flash",
                shard_recs=[self._make_record("shard")],
                output_dir=tmp,
                mtp_rec=self._make_record("glm_mtp.safetensors"),
                runtime_rec=self._make_record("glm_mtp_runtime.json"),
                quantize=None,
                group_size=64,
                runtime=runtime,
            )
        self.assertEqual(manifest["schema_version"], "ax.glm_mtp_sidecar_provenance.v1")
        self.assertEqual(manifest["arch_id"], "glm-4.7-flash-mtp")
        self.assertIn("base", manifest)
        self.assertIn("source", manifest)
        self.assertIn("output", manifest)
        self.assertIn("transform", manifest)
        self.assertEqual(manifest["source"]["mtp_layer_index"], 47)
        self.assertTrue(manifest["transform"]["embed_tokens_skipped"])
        self.assertTrue(manifest["transform"]["moe_expert_stack"])


if __name__ == "__main__":
    unittest.main()
