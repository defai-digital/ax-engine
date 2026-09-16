#!/usr/bin/env python3
"""Unit tests for qualify_qwen38_flash_next.py (no weights)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = Path(__file__).with_name("qualify_qwen38_flash_next.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "qualify_qwen38_flash_next", MODULE_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


def _product_manifest() -> dict[str, object]:
    return {
        "model_family": "qwen4_exp",
        "layer_count": 1,
        "hidden_size": 8,
        "runtime_status": {"ready": True, "blockers": []},
        "tensors": [
            {
                "name": "layers.0.mlp.experts.gate_proj",
                "role": "ffn_gate_exps",
                "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
            }
        ],
    }


class QualifyFlashNextTest(unittest.TestCase):
    def test_contract_is_candidate_on_studio_sku(self) -> None:
        payload = mod.contract()
        self.assertEqual(payload["family"], "qwen4_exp")
        self.assertEqual(payload["alias"], "qwen3.8-flash-next:axq")
        self.assertEqual(
            payload["repo_id"],
            "AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-4bit-MTP",
        )
        self.assertEqual(payload["host_class"], "Mac Studio M5 Ultra, 256 GB")
        self.assertFalse(payload["fail_closed"])
        self.assertTrue(payload["ready"])
        self.assertIsNone(payload["load_blocker"])
        self.assertEqual(
            payload["experimental_opt_in"], "AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1"
        )
        self.assertEqual(payload["experimental_2bit_opt_in"], "AX_ENGINE_2BIT_EXPERIMENTAL=1")
        self.assertEqual(
            payload["product_expert_layouts"],
            [
                {"bits": 4, "group_size": 64},
                {"bits": 6, "group_size": 64},
            ],
        )
        self.assertEqual(
            payload["experimental_expert_layouts"],
            [{"bits": 2, "group_size": 32}],
        )
        self.assertIn("qwen3.8-27b:axq", payload["not"])
        self.assertIn("Candidate", payload["status"])
        self.assertIn("M2 evidence", payload["status"])

    def test_dry_run_cli_json(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--dry-run", "--json"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        payload = json.loads(proc.stdout)
        self.assertFalse(payload["fail_closed"])
        self.assertTrue(payload["ready"])

    def test_live_preflight_rejects_incomplete_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                mod._live_preflight(model_dir)
            self.assertIn("model-manifest.json", str(raised.exception))

    def test_live_preflight_accepts_ready_product_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model-manifest.json").write_text(
                json.dumps(_product_manifest()),
                encoding="utf-8",
            )
            mod._live_preflight(model_dir)

    def test_live_preflight_rejects_unknown_layout_and_mxfp4(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            blocked = _product_manifest()
            blocked["runtime_status"] = {
                "ready": False,
                "blockers": ["qwen4_exp_weight_layout_unknown"],
            }
            (model_dir / "model-manifest.json").write_text(
                json.dumps(blocked), encoding="utf-8"
            )
            with self.assertRaises(SystemExit) as raised:
                mod._live_preflight(model_dir)
            self.assertIn("qwen4_exp_weight_layout_unknown", str(raised.exception))

            mxfp4 = _product_manifest()
            mxfp4["tensors"][0]["quantization"] = {  # type: ignore[index]
                "mode": "mxfp4",
                "bits": 4,
                "group_size": 32,
            }
            (model_dir / "model-manifest.json").write_text(
                json.dumps(mxfp4), encoding="utf-8"
            )
            with self.assertRaises(SystemExit) as raised:
                mod._live_preflight(model_dir)
            self.assertIn("MXFP4", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
