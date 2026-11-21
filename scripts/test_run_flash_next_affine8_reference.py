"""Small affine8 table controls for the independent Flash Next diagnostic."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HAS_MLX = importlib.util.find_spec("mlx") is not None
if HAS_MLX:
    import mlx.core as mx
    import numpy as np

    from scripts.run_flash_next_affine8_reference import Affine8Table


@unittest.skipUnless(HAS_MLX, "requires MLX")
class Affine8TableTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.expected = []
        self.weights = {}
        self.aux = {}
        self.index = {}
        self.config = {"quantization": {"bits": 8, "group_size": 32, "mode": "affine"},
                       "text_config": {"split_ngram_parts": 3}}
        for shard in range(3):
            source = ((mx.arange(480).reshape(3, 160) + 37 * shard) % 91 - 38).astype(mx.bfloat16)
            params = mx.quantize(source, group_size=32, bits=8)
            self.expected.append(mx.dequantize(*params, group_size=32, bits=8))
            prefix = f"layer.1.ple.ple_embedding.ngram_embedding.shards.{shard}"
            for label, value in zip(("weight", "scales", "biases"), params, strict=True):
                key = f"{prefix}.{label}"
                tensors, name = ((self.weights, "packed.safetensors") if label == "weight"
                                 else (self.aux, "planes.safetensors"))
                tensors[key] = value
                self.index[key] = name
        mx.save_safetensors(str(self.root / "packed.safetensors"), self.weights)
        mx.save_safetensors(str(self.root / "planes.safetensors"), self.aux)
        self.write_metadata()

    def write_metadata(self):
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": self.index}))
        (self.root / "config.json").write_text(json.dumps(self.config))

    def test_cross_shard_order_and_deduplicated_reads(self):
        table = Affine8Table(self.root)
        indices = mx.array([[8, 3, 2], [3, 0, 2]], dtype=mx.int64)
        actual = table(indices)
        expected = mx.concatenate(self.expected)[indices]
        self.assertEqual(actual.shape, (2, 3, 160))
        self.assertEqual(actual.dtype, mx.bfloat16)
        np.testing.assert_array_equal(actual.view(mx.uint16), expected.view(mx.uint16))
        self.assertEqual(table.row_reads, 4)
        self.assertEqual(table.bytes_read, 4 * 180)
        self.assertEqual(table(mx.zeros((2, 0), dtype=mx.int32)).shape, (2, 0, 160))
        self.assertEqual(table.row_reads, 4)

    def test_rejects_out_of_range_or_noninteger_ids(self):
        table = Affine8Table(self.root)
        for bad in (-1, 9, 1.25):
            with self.subTest(value=bad), self.assertRaises(IndexError):
                table(mx.array([bad]))
        self.assertEqual(table.row_reads, 0)

    def test_rejects_quantization_and_truncated_planes(self):
        self.config["quantization"]["bits"] = 4
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "unsupported PLE quantization"):
            Affine8Table(self.root)
        self.config["quantization"]["bits"] = 8
        self.write_metadata()
        path = self.root / "planes.safetensors"
        with path.open("r+b") as handle:
            handle.truncate(path.stat().st_size - 1)
        with self.assertRaisesRegex(ValueError, "invalid affine8 row layout"):
            Affine8Table(self.root)


if __name__ == "__main__":
    unittest.main()
