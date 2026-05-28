#!/usr/bin/env python3
"""Tests for check_mtp_sidecar_provenance.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_mtp_sidecar_provenance.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mtp_sidecar_provenance",
    SCRIPT_PATH,
)
assert MODULE_SPEC and MODULE_SPEC.loader
check = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["check_mtp_sidecar_provenance"] = check
MODULE_SPEC.loader.exec_module(check)


class MtpSidecarProvenanceTests(unittest.TestCase):
    def test_valid_manifest_passes_fair_base_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_manifest(root)
            summary = check.validate_manifest(manifest, fair_base_only=True)

        self.assertEqual(summary["base_model_id"], "mlx-community/Qwen3.6-27B-4bit")
        self.assertEqual(summary["source_model_id"], "Qwen/Qwen3.6-27B")
        self.assertEqual(summary["source_shard_count"], 2)

    def test_strict_local_detects_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_manifest(root)
            (root / "mtp.safetensors").write_text("bad")

            with self.assertRaisesRegex(check.ProvenanceError, "sha256 mismatch"):
                check.validate_manifest(manifest)

    def test_fair_base_rejects_youssofal_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_manifest(root)
            manifest["source"]["model_id"] = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"

            with self.assertRaisesRegex(check.ProvenanceError, "standard Qwen source"):
                check.validate_manifest(manifest, fair_base_only=True)


def record(path: Path, content: str) -> dict:
    path.write_text(content)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": check.sha256_file(path),
    }


def build_manifest(root: Path) -> dict:
    return {
        "schema_version": "ax.mtp_sidecar_provenance.v1",
        "generated_by": "scripts/prepare_qwen36_mtp_sidecar.py",
        "model_key": "27b",
        "base": {
            "model_id": "mlx-community/Qwen3.6-27B-4bit",
            "snapshot_dir": str(root),
            "snapshot": "test",
            "config": record(root / "base-config.json", "{}"),
            "index": record(root / "base-index.json", "{}"),
        },
        "source": {
            "model_id": "Qwen/Qwen3.6-27B",
            "mtp_shards": [
                {
                    "name": "model-00013-of-00015.safetensors",
                    **record(root / "shard-a", "a"),
                },
                {
                    "name": "model-00015-of-00015.safetensors",
                    **record(root / "shard-b", "b"),
                },
            ],
        },
        "output": {
            "dir": str(root),
            "mtp": record(root / "mtp.safetensors", "mtp"),
            "runtime": record(root / "mtplx_runtime.json", "{}"),
            "config": record(root / "config.json", "{}"),
        },
        "transform": {
            "norm_policy": "scale_selected_mtp_norm_weights_by_2",
            "moe_expert_unpack": False,
        },
        "runtime": {
            "arch_id": "qwen3-next-mtp",
            "mtp_depth_max": 3,
            "mtp_tensor_count": 27,
        },
    }


if __name__ == "__main__":
    unittest.main()
