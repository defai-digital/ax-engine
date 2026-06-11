#!/usr/bin/env python3
"""Unit tests for report_direct_model_weight_bytes.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("report_direct_model_weight_bytes.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "report_direct_model_weight_bytes", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["report_direct_model_weight_bytes"] = mod
MODULE_SPEC.loader.exec_module(mod)


def _write_fake_safetensors(model_dir: Path, files: dict[str, int]) -> None:
    for name, size in files.items():
        path = model_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * size)


def _write_model_manifest(model_dir: Path, manifest: dict) -> None:
    (model_dir / "model-manifest.json").write_text(json.dumps(manifest))


class ReportDirectModelWeightBytesTests(unittest.TestCase):
    def test_dense_model_safetensor_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_safetensors(
                root,
                {
                    "model-00001-of-00003.safetensors": 5_000_000_000,
                    "model-00002-of-00003.safetensors": 5_000_000_000,
                    "model-00003-of-00003.safetensors": 5_000_000_000,
                },
            )
            report = mod.build_report(root)

        self.assertEqual(report["schema_version"], "ax.direct_weight_bytes.v1")
        self.assertEqual(report["safetensor_bytes"], 15_000_000_000)
        self.assertEqual(report["safetensor_files"], 3)
        self.assertTrue(report["dense_estimate_supported"])
        self.assertFalse(report["moe_active_bytes_supported"])
        self.assertIsNone(report["moe_active_bytes"])
        self.assertIsNone(report["moe_block"])

    def test_symlink_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real_dir = Path(tmp) / "real_model"
            real_dir.mkdir()
            _write_fake_safetensors(real_dir, {"weights.safetensors": 1_000_000})

            link_dir = Path(tmp) / "link_model"
            link_dir.symlink_to(real_dir)

            report = mod.build_report(link_dir)

        self.assertEqual(report["safetensor_bytes"], 1_000_000)
        self.assertTrue(report["symlinks_followed"])
        self.assertIn("real_model", report["resolved_model_dir"])

    def test_moe_active_expert_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_safetensors(
                root,
                {
                    "model.safetensors": 10_000_000,
                },
            )
            _write_model_manifest(
                root,
                {
                    "schema_version": "ax.native_model_manifest.v1",
                    "model_family": "gemma4",
                    "moe": {
                        "expert_count": 8,
                        "experts_per_token": 2,
                    },
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
            )
            report = mod.build_report(root)

        self.assertTrue(report["moe_active_bytes_supported"])
        self.assertIsNotNone(report["moe_active_bytes"])
        self.assertIsNotNone(report["moe_block"])
        self.assertEqual(report["moe_block"]["expert_count"], 8)
        self.assertEqual(report["moe_block"]["experts_per_token"], 2)
        non_routed = 3_000_000 + 1_000_000
        active_routed = int(6_000_000 * (2 / 8))
        self.assertEqual(report["moe_active_bytes"], non_routed + active_routed)

    def test_no_safetensors_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}")
            with self.assertRaises(SystemExit):
                mod.build_report(root)

    def test_nonexistent_dir_raises(self) -> None:
        with self.assertRaises(SystemExit):
            mod.build_report(Path("/nonexistent/path/that/does/not/exist"))

    def test_file_list_is_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_safetensors(
                root,
                {
                    "shard/model-00001.safetensors": 100,
                    "model-00002.safetensors": 200,
                },
            )
            report = mod.build_report(root)

        self.assertEqual(report["safetensor_files"], 2)
        for f in report["safetensor_file_list"]:
            self.assertFalse(Path(f).is_absolute())

    def test_output_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            _write_fake_safetensors(root, {"w.safetensors": 42})
            out_path = Path(tmp) / "out" / "report.json"

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
            self.assertEqual(doc["schema_version"], "ax.direct_weight_bytes.v1")
            self.assertEqual(doc["safetensor_bytes"], 42)


if __name__ == "__main__":
    unittest.main()
