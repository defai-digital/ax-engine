#!/usr/bin/env python3
"""Unit tests for report_quantization_recipe_inventory.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("report_quantization_recipe_inventory.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "report_quantization_recipe_inventory", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["report_quantization_recipe_inventory"] = mod
MODULE_SPEC.loader.exec_module(mod)


def _write_model(
    model_dir: Path,
    *,
    config: dict | None = None,
    manifest: dict | None = None,
    weight_bytes: int = 0,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (model_dir / "config.json").write_text(json.dumps(config))
    if manifest is not None:
        (model_dir / "model-manifest.json").write_text(json.dumps(manifest))
    if weight_bytes > 0:
        (model_dir / "model.safetensors").write_bytes(b"\x00" * weight_bytes)


class ReportQuantizationRecipeInventoryTests(unittest.TestCase):
    def test_dense_4bit_uniform(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            _write_model(
                root,
                config={
                    "model_type": "qwen3",
                    "quantization": {"bits": 4, "group_size": 64, "layout": "uniform"},
                },
                weight_bytes=5_000_000_000,
            )

            inventory = mod.build_inventory([root])

        self.assertEqual(
            inventory["schema_version"], "ax.quantization_recipe_inventory.v1"
        )
        self.assertEqual(inventory["model_count"], 1)
        entry = inventory["entries"][0]
        self.assertEqual(entry["model_type"], "qwen3")
        self.assertEqual(entry["quantization_recipe"]["bits"], 4)
        self.assertEqual(entry["quantization_recipe"]["group_size"], 64)
        self.assertEqual(entry["quantization_recipe"]["layout"], "uniform")
        self.assertIsNone(entry["active_expert_accounting"])
        self.assertEqual(entry["estimate_kind"], "dense_weight_total")

    def test_moe_with_active_experts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            _write_model(
                root,
                config={
                    "model_type": "gemma4",
                    "quantization": {"bits": 4, "group_size": 64},
                },
                manifest={
                    "schema_version": "ax.native_model_manifest.v1",
                    "model_family": "gemma4",
                    "moe": {"expert_count": 8, "experts_per_token": 2},
                    "tensors": [
                        {"role": "self_attn.q_proj", "length_bytes": 3_000_000},
                        {
                            "role": "block_sparse_moe._exps.weight",
                            "length_bytes": 6_000_000,
                        },
                        {
                            "role": "block_sparse_moe.shared_expert.weight",
                            "length_bytes": 1_000_000,
                        },
                    ],
                },
                weight_bytes=10_000_000,
            )

            inventory = mod.build_inventory([root])

        entry = inventory["entries"][0]
        self.assertIsNotNone(entry["active_expert_accounting"])
        self.assertTrue(entry["active_expert_accounting"]["valid"])
        self.assertEqual(entry["estimate_kind"], "moe_active_estimate")
        non_routed = 3_000_000 + 1_000_000
        active_routed = int(6_000_000 * (2 / 8))
        self.assertEqual(
            entry["bytes_used_for_bandwidth_estimate"], non_routed + active_routed
        )

    def test_per_layer_overrides_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            _write_model(
                root,
                config={
                    "model_type": "gemma4",
                    "quantization": {
                        "bits": 4,
                        "group_size": 64,
                        "model.layers.0.mlp.gate_proj": {"bits": 8, "group_size": 64},
                        "model.layers.0.mlp.up_proj": {"bits": 8, "group_size": 64},
                    },
                },
                weight_bytes=5_000_000_000,
            )

            inventory = mod.build_inventory([root])

        entry = inventory["entries"][0]
        overrides = entry["quantization_recipe"]["per_layer_overrides"]
        self.assertEqual(len(overrides), 2)
        self.assertEqual(overrides["model.layers.0.mlp.gate_proj"]["bits"], 8)

    def test_missing_directory(self) -> None:
        inventory = mod.build_inventory([Path("/nonexistent/path")])
        entry = inventory["entries"][0]
        self.assertEqual(entry["error"], "directory_not_found")

    def test_missing_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            _write_model(root, weight_bytes=1_000_000)

            inventory = mod.build_inventory([root])

        entry = inventory["entries"][0]
        self.assertFalse(entry["has_config"])
        self.assertIsNone(entry["quantization_recipe"]["bits"])

    def test_multiple_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m1 = root / "model_a"
            m2 = root / "model_b"
            _write_model(
                m1,
                config={"model_type": "qwen3", "quantization": {"bits": 4}},
                weight_bytes=1_000,
            )
            _write_model(
                m2,
                config={"model_type": "gemma4", "quantization": {"bits": 8}},
                weight_bytes=2_000,
            )

            inventory = mod.build_inventory([m1, m2])

        self.assertEqual(inventory["model_count"], 2)
        self.assertEqual(inventory["entries"][0]["model_type"], "qwen3")
        self.assertEqual(inventory["entries"][1]["model_type"], "gemma4")

    def test_output_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            _write_model(root, config={"model_type": "qwen3"}, weight_bytes=100)
            out_path = Path(tmp) / "out" / "inventory.json"

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
                doc["schema_version"], "ax.quantization_recipe_inventory.v1"
            )


if __name__ == "__main__":
    unittest.main()
