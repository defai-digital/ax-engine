#!/usr/bin/env python3
"""Unit tests for quantize_direct_mlx.py.

No companion test file previously existed; scope limited to the specific
bug fixed here rather than full coverage of the script.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

SCRIPT_PATH = Path(__file__).with_name("quantize_direct_mlx.py")
MODULE_SPEC = importlib.util.spec_from_file_location("quantize_direct_mlx", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


class QuantizeSafetensorsTests(unittest.TestCase):
    def test_top_level_bits_is_the_dominant_low_bits(self) -> None:
        # Regression test: the top-level quant_config["bits"] was set to
        # high_bits (the rare, sensitive-layer bit width) instead of
        # low_bits (the dominant default applied to most weights). Every
        # quantized tensor gets its own explicit per-tensor override, so
        # this didn't break MLX weight loading, but it inverted the
        # semantic meaning of the recorded default: a tool reading only
        # config.json's top-level quantization.bits (e.g.
        # report_quantization_recipe_inventory.py) would report the model
        # as predominantly high-bit when it's actually predominantly
        # low-bit.
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            output_dir = Path(tmp) / "output"
            source_dir.mkdir()

            # "q_proj" is a low_bits (dominant) layer; not in
            # HIGH_BITS_LAYER_SUFFIXES.
            tensors = {"model.layers.0.self_attn.q_proj.weight": mx.zeros((64, 64))}
            mx.save_safetensors(str(source_dir / "model.safetensors"), tensors)

            low_bits, high_bits = mod.RECIPE_PARAMS["mixed_3_6"]
            self.assertNotEqual(low_bits, high_bits)

            quant_config, _ = mod.quantize_safetensors(source_dir, output_dir, "mixed_3_6")

        self.assertEqual(quant_config["bits"], low_bits)
        self.assertEqual(
            quant_config["model.layers.0.self_attn.q_proj"]["bits"], low_bits
        )


if __name__ == "__main__":
    unittest.main()
