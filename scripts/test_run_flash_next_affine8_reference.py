"""Small affine8 table controls for the independent Flash Next diagnostic."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HAS_MLX = importlib.util.find_spec("mlx") is not None
if HAS_MLX:
    import mlx.core as mx
    import numpy as np

    from scripts import run_flash_next_affine8_reference as diagnostic
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


@unittest.skipUnless(HAS_MLX, "requires MLX")
class FullWorkloadTests(unittest.TestCase):
    SIGNED = (Path(__file__).resolve().parents[1] / "benchmarks/results/qualification"
              / "2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0")

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        original = json.loads((self.SIGNED / "reference/result.json").read_text())
        vocab = original["matrix"]["vocab_size"]
        (self.root / "config.json").write_text(json.dumps({"vocab_size": vocab}))
        self.matrix = self.SIGNED / "generate-matrix/result.json"

    def test_preflight_binds_both_native_prompt_cohorts(self):
        workload = diagnostic.prepare_full_workload(self.root, self.matrix)
        self.assertEqual(len(workload.items), 105)
        self.assertEqual(len(workload.prompt_hashes), 15)
        altered = json.loads(self.matrix.read_text())
        altered["runs"]["default"]["prompt_id_hashes"]["cell-512-0"] = "0" * 64
        bad = self.root / "bad-matrix.json"
        bad.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "reference prompt window differs"):
            diagnostic.prepare_full_workload(self.root, bad)

    def test_full_workload_uses_frozen_budgets_and_all_repeats(self):
        workload = diagnostic.prepare_full_workload(self.root, self.matrix)
        tokens_by_prompt = {row["rendered"]: row["input_ids"] for row in workload.prompts}
        tokenizer = SimpleNamespace(
            encode=lambda text, **_: tokens_by_prompt[text],
            decode=lambda ids, **_: str(ids[0]),
        )
        generation_calls = []

        def generate(model, mx, ids, budget, stop_on_eos=True):
            generation_calls.append((len(ids), budget, stop_on_eos))
            count = 1 if stop_on_eos else budget
            return {"generated_ids": [1] * count, "generated_tokens": count, "normal_stop": stop_on_eos,
                    "ttft_seconds": 1.0, "decode_tok_s": 1.0}

        record = {}
        saves = []
        with patch.object(diagnostic.reference, "generate", generate):
            diagnostic.run_full_workload(None, tokenizer, workload, record, saves.append)
        self.assertEqual(len(record["qa"]["rows"]), 105)
        self.assertEqual(len(record["qa"]["grades"]), 105)
        self.assertTrue(record["qa"]["normal_stops"])
        self.assertEqual(generation_calls[104], (29774, 64, True))
        self.assertTrue(all(budget == 256 and stop for _, budget, stop in generation_calls[:104]))
        self.assertEqual(generation_calls[105:], [(length, 128, False)
                                                  for length in (512, 2048, 8192) for _ in range(5)])
        self.assertEqual(len(record["matrix"]["cells"]), 15)
        self.assertTrue(all(summary["full_length_runs"] == 3 for summary in record["matrix"]["summary"].values()))
        self.assertEqual(saves.count("matrix"), 15)


if __name__ == "__main__":
    unittest.main()
