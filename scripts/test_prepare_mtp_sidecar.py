#!/usr/bin/env python3
"""Tests for prepare_mtp_sidecar.py.

Pure-Python fakes only (no MLX import) so the suite runs in the script hygiene
gate. The MLX-backed transforms (quantization, safetensors IO) are exercised
manually; here we lock down the layout logic and, crucially, that the generated
provenance manifest passes scripts/check_mtp_sidecar_provenance.py.
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


prepare = _load("prepare_mtp_sidecar")
provenance = _load("check_mtp_sidecar_provenance")


class FakeArray:
    """1-D / 2-D array stub supporting + and .astype (for norm-shift tests)."""

    def __init__(self, values):
        self.values = values
        self.dtype = "fake"
        self.ndim = 1 if not isinstance(values[0], list) else 2

    def __add__(self, other: float):
        if self.ndim == 1:
            return FakeArray([v + other for v in self.values])
        return FakeArray([[v + other for v in row] for row in self.values])

    def astype(self, _dtype):
        return self

    def tolist(self):
        return self.values


class SliceFake:
    """3-D expert stub: records slice ops so we can assert split/rename layout."""

    def __init__(self, shape, tag):
        self.shape = shape
        self.tag = tag

    def __getitem__(self, key):
        return ("sliced", self.tag, key)


def _fake_record(name: str = "x") -> dict:
    return {
        "path": f"/tmp/{name}",
        "exists": True,
        "size_bytes": 16,
        "sha256": "a" * 64,
    }


class TransformTests(unittest.TestCase):
    def test_shift_norm_weights_matches_mlx_plus_one_convention(self) -> None:
        tensors = {
            "mtp.norm.weight": FakeArray([0.0, -0.5]),
            "mtp.layers.0.self_attn.q_norm.weight": FakeArray([0.25, -0.25]),
            "mtp.fc.weight": FakeArray([[2.0]]),
        }
        shifted = prepare._shift_norm_weights(tensors)
        self.assertEqual(shifted["mtp.norm.weight"].tolist(), [1.0, 0.5])
        self.assertEqual(
            shifted["mtp.layers.0.self_attn.q_norm.weight"].tolist(), [1.25, 0.75]
        )
        # 2-D projection weights are left untouched.
        self.assertEqual(shifted["mtp.fc.weight"].tolist(), [[2.0]])

    def test_moe_unpack_splits_gate_up_and_renames_down(self) -> None:
        raw = {
            "mtp.layers.0.mlp.experts.gate_up_proj": SliceFake((2, 6, 4), "gate_up"),
            "mtp.layers.0.mlp.experts.down_proj": SliceFake((2, 4, 3), "down"),
            "mtp.layers.0.mlp.gate.weight": "router",
        }
        out = prepare._normalize_qwen_moe_packed(raw)
        self.assertEqual(
            sorted(out),
            [
                "mtp.layers.0.mlp.down_proj.weight",
                "mtp.layers.0.mlp.gate.weight",
                "mtp.layers.0.mlp.gate_proj.weight",
                "mtp.layers.0.mlp.up_proj.weight",
            ],
        )
        # gate is the first half [:, :3, :], up is the second half [:, 3:, :].
        gate = out["mtp.layers.0.mlp.gate_proj.weight"]
        up = out["mtp.layers.0.mlp.up_proj.weight"]
        self.assertEqual(gate[2], (slice(None), slice(None, 3), slice(None)))
        self.assertEqual(up[2], (slice(None), slice(3, None), slice(None)))
        # Router gate is passed through untouched.
        self.assertEqual(out["mtp.layers.0.mlp.gate.weight"], "router")

    def test_moe_unpack_renames_separate_gate_up_experts(self) -> None:
        # Some MoE checkpoints ship already-unpacked expert stacks (separate
        # gate/up rather than a packed gate_up). They must be renamed to the
        # canonical mlp.{gate,up,down}_proj.weight the loader reads.
        raw = {
            "mtp.layers.0.mlp.experts.gate_proj": "g",
            "mtp.layers.0.mlp.experts.up_proj": "u",
            "mtp.layers.0.mlp.experts.down_proj": "d",
            "mtp.layers.0.mlp.gate.weight": "router",
        }
        out = prepare._normalize_qwen_moe_packed(raw)
        self.assertEqual(out["mtp.layers.0.mlp.gate_proj.weight"], "g")
        self.assertEqual(out["mtp.layers.0.mlp.up_proj.weight"], "u")
        self.assertEqual(out["mtp.layers.0.mlp.down_proj.weight"], "d")
        self.assertEqual(out["mtp.layers.0.mlp.gate.weight"], "router")
        # No raw experts.* keys should survive.
        self.assertFalse(any(".experts." in k for k in out))

    def test_detect_arch_dispatch(self) -> None:
        self.assertEqual(
            prepare._detect_arch({"mtp.layers.0.mlp.experts.gate_up_proj": 1}, {})[0],
            "qwen-moe-packed",
        )
        dense_id, _, dense_moe = prepare._detect_arch(
            {"mtp.layers.0.mlp.gate_proj.weight": 1}, {}
        )
        self.assertEqual(dense_id, "qwen-dense")
        self.assertFalse(dense_moe)


class RuntimeContractTests(unittest.TestCase):
    def test_runtime_contract_required_fields(self) -> None:
        contract = prepare._runtime_contract(
            arch_id="qwen3-next-mtp",
            depth_max=3,
            base_repo="mlx-community/Qwen3.6-27B-4bit",
            tensor_count=15,
            quant_bits=None,
        )
        self.assertEqual(contract["arch_id"], "qwen3-next-mtp")
        self.assertEqual(contract["mtp_tensor_count"], 15)
        self.assertEqual(contract["mtp_depth_max"], 3)
        for key in ("mtplx_version", "exactness_baseline", "verified_on"):
            self.assertIn(key, contract)
        self.assertNotIn("mtp_sidecar", contract)  # bf16 sidecar has no quant hint

    def test_runtime_contract_quant_hint(self) -> None:
        contract = prepare._runtime_contract(
            arch_id="qwen-dense",
            depth_max=1,
            base_repo="local/base",
            tensor_count=15,
            quant_bits=4,
        )
        # The loader scans this free-text field for INT4/INT8.
        self.assertIn("INT4", contract["mtp_sidecar"])


class QuantizeGateTests(unittest.TestCase):
    def test_quantizable_2d_weight(self) -> None:
        self.assertTrue(prepare._should_quantize("mtp.fc.weight", (5120, 4096), 64))
        # Last dim exactly == group_size is one valid group.
        self.assertTrue(prepare._should_quantize("mtp.fc.weight", (4096, 64), 64))
        # Smaller output dim must not exclude a quantizable last dim.
        self.assertTrue(prepare._should_quantize("mtp.fc.weight", (2, 4096), 64))

    def test_last_dim_must_be_multiple_of_group_size(self) -> None:
        # Regression: last dim 192 is a multiple of 64 but NOT of 128 — mx.quantize
        # would raise, so it must be skipped rather than attempted.
        self.assertFalse(prepare._should_quantize("mtp.fc.weight", (4096, 192), 128))
        self.assertTrue(prepare._should_quantize("mtp.fc.weight", (4096, 192), 64))
        # Last dim smaller than the group cannot be quantized.
        self.assertFalse(prepare._should_quantize("mtp.fc.weight", (4096, 32), 64))

    def test_non_2d_and_special_tensors_skip(self) -> None:
        # Norms (1-D), 3-D expert stacks, and the shared-expert gate stay FP.
        self.assertFalse(prepare._should_quantize("mtp.norm.weight", (4096,), 64))
        self.assertFalse(
            prepare._should_quantize("mtp.layers.0.mlp.gate_proj.weight", (128, 4096, 2048), 64)
        )
        self.assertFalse(
            prepare._should_quantize("mtp.layers.0.mlp.shared_expert_gate.weight", (1, 4096), 64)
        )


class ResolveBaseDirTests(unittest.TestCase):
    def test_local_cache_style_path_resolves_from_given_path(self) -> None:
        # A cache-style dir (…/<model>/snapshots/<hash>) that is NOT under the
        # global HF cache must resolve via the given path, not HF_CACHE/<name>.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models--org--Some-4bit"
            snapshot = model_dir / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            resolved = prepare._resolve_base_dir(str(model_dir))
            self.assertEqual(resolved, snapshot.resolve())

    def test_direct_model_dir_with_config_resolves_as_is(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "flat-model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")
            self.assertEqual(
                prepare._resolve_base_dir(str(model_dir)), model_dir.resolve()
            )


class PatchConfigTests(unittest.TestCase):
    def test_num_nextn_predict_layers_is_physical_count_not_draft_depth(self) -> None:
        # A base config that already (wrongly) declares 0 physical MTP layers.
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "config.json").write_text(
                json.dumps({"model_type": "qwen3_next", "num_nextn_predict_layers": 0})
            )
            prepare._patch_config(out_dir, tensor_count=15)
            patched = json.loads((out_dir / "config.json").read_text())
            # Physical MTP layer count is always 1, regardless of draft depth.
            self.assertEqual(patched["num_nextn_predict_layers"], 1)
            self.assertEqual(patched["mtp_num_hidden_layers"], 1)
            self.assertEqual(
                patched["mlx_lm_extra_tensors"],
                {"mtp_file": "mtp.safetensors", "mtp_tensor_count": 15},
            )


class ProvenanceContractTests(unittest.TestCase):
    """The generated manifest must satisfy the existing provenance checker."""

    def _manifest(self, *, base_model_id: str, source_model_id: str) -> dict:
        runtime = prepare._runtime_contract(
            arch_id="qwen-dense",
            depth_max=1,
            base_repo=base_model_id,
            tensor_count=15,
            quant_bits=None,
        )
        return prepare._provenance_manifest(
            model_key="Qwen3.6-27B-4bit",
            arch_id="qwen-dense",
            is_moe=False,
            base_model_id=base_model_id,
            base_snapshot_dir="/tmp/base",
            base_config_rec=_fake_record("config.json"),
            base_index_rec=_fake_record("index.json"),
            source_model_id=source_model_id,
            shard_recs=[{"name": "model-00013.safetensors", **_fake_record("shard")}],
            output_dir="/tmp/out",
            mtp_rec=_fake_record("mtp.safetensors"),
            runtime_rec=_fake_record("mtplx_runtime.json"),
            config_rec=_fake_record("config.json"),
            quantize=None,
            group_size=64,
            runtime=runtime,
        )

    def test_manifest_passes_provenance_validator(self) -> None:
        manifest = self._manifest(
            base_model_id="mlx-community/Qwen3.6-27B-4bit",
            source_model_id="Qwen/Qwen3.6-27B",
        )
        summary = provenance.validate_manifest(manifest, strict_local=False)
        self.assertEqual(summary["schema_version"], provenance.SCHEMA_VERSION)
        self.assertEqual(summary["mtp_tensor_count"], 15)
        self.assertEqual(summary["norm_policy"], "shift_mtp_norm_weights_by_1")

    def test_manifest_passes_fair_base_only_for_standard_ids(self) -> None:
        manifest = self._manifest(
            base_model_id="mlx-community/Qwen3.6-27B-4bit",
            source_model_id="Qwen/Qwen3.6-27B",
        )
        # Should not raise under the stricter fair-benchmark gate.
        provenance.validate_manifest(manifest, strict_local=False, fair_base_only=True)


if __name__ == "__main__":
    unittest.main()
