#!/usr/bin/env python3
"""Unit tests for qualify_qwen38_flash_next.py (no weights)."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = Path(__file__).with_name("qualify_qwen38_flash_next.py")
MODULE_SPEC = importlib.util.spec_from_file_location("qualify_qwen38_flash_next", MODULE_PATH)
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
                "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
            }
        ],
    }


def _write_sidecar(root: Path) -> None:
    (root / "mtplx_runtime.json").write_text(json.dumps({"mtp_norm_layout": "raw_hf_delta"}))
    tensors = {f"mtp.tensor.{i}": {"dtype": "BF16", "shape": [1],
                "data_offsets": [2*i, 2*i+2]} for i in range(31)}
    header = json.dumps(tensors).encode()
    (root / "mtp.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(62))


class QualifyFlashNextTest(unittest.TestCase):
    def test_preflight_rejects_boolean_geometry_and_falsy_invalid_blockers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}")
            _write_sidecar(root)
            for field in ("layer_count", "hidden_size", "blockers"):
                manifest = _product_manifest()
                if field == "blockers":
                    manifest["runtime_status"][field] = False
                else:
                    manifest[field] = True
                (root / "model-manifest.json").write_text(json.dumps(manifest))
                with self.subTest(field=field), self.assertRaises(SystemExit):
                    mod._live_preflight(root)

    def test_contract_distinguishes_mxfp4_target_from_affine_compatibility(self) -> None:
        payload = mod.contract()
        self.assertEqual(payload["family"], "qwen4_exp")
        self.assertEqual(payload["alias"], "qwen3.8-flash-next:mxfp4")
        self.assertFalse(payload["sixbit_in_target_scope"])
        self.assertEqual(payload["format_scope"], ["mxfp4", "mxfp8"])
        self.assertTrue(payload["mtp_sidecar_required"])
        self.assertIn("pending", payload["mxfp8_status"])
        self.assertEqual(payload["experimental_expert_layouts"], [{"mode": "mxfp4", "bits": 4, "group_size": 32}])
        self.assertIn("qwen3.8-27b:axq", payload["not"])
        self.assertIn("Candidate", payload["status"])
        self.assertIn("MXFP4 MTP target", payload["status"])

    def test_dry_run_cli_json(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--dry-run", "--json"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["fail_closed"])
        self.assertFalse(payload["ready"])

    def test_live_preflight_rejects_incomplete_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                mod._live_preflight(model_dir)
            self.assertIn("model-manifest.json", str(raised.exception))

    def test_live_preflight_accepts_mxfp4_metadata_without_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model-manifest.json").write_text(
                json.dumps(_product_manifest()),
                encoding="utf-8",
            )
            _write_sidecar(model_dir)
            mod._live_preflight(model_dir)

    def test_live_preflight_mxfp4_needs_no_opt_in_and_rejects_unknown_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            blocked = _product_manifest()
            blocked["runtime_status"] = {
                "ready": False,
                "blockers": ["qwen4_exp_weight_layout_unknown"],
            }
            (model_dir / "model-manifest.json").write_text(json.dumps(blocked), encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                mod._live_preflight(model_dir)
            self.assertIn("qwen4_exp_weight_layout_unknown", str(raised.exception))

            mxfp4 = _product_manifest()
            mxfp4["tensors"][0]["quantization"] = {  # type: ignore[index]
                "mode": "mxfp4",
                "bits": 4,
                "group_size": 32,
            }
            (model_dir / "model-manifest.json").write_text(json.dumps(mxfp4), encoding="utf-8")
            # MXFP4/group32 preflights without the experimental opt-in; the
            # certification record (Candidate, gates open) tracks its status.
            _write_sidecar(model_dir)
            mod._live_preflight(model_dir)
            with patch.dict("os.environ", {"AX_ENGINE_FLASH_NEXT_EXPERIMENTAL": "1"}):
                mod._live_preflight(model_dir)
            # The same bits/group tuple cannot relabel affine as MXFP4.
            mxfp4["tensors"][0]["quantization"]["mode"] = "affine"
            (model_dir / "model-manifest.json").write_text(json.dumps(mxfp4))
            with self.assertRaisesRegex(SystemExit, "unsupported expert layout"):
                mod._live_preflight(model_dir)

    def test_sidecar_metadata_fails_closed_before_any_payload_load(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(SystemExit, "requires mtp.safetensors"):
                mod._mtp_metadata(root)
            _write_sidecar(root)
            self.assertEqual(mod._mtp_metadata(root)["tensor_count"], 31)
            (root / "mtplx_runtime.json").write_text('{}')
            with self.assertRaisesRegex(SystemExit, "mtp_norm_layout"):
                mod._mtp_metadata(root)
            _write_sidecar(root)
            raw = (root / "mtp.safetensors").read_bytes()
            for label, corrupt in [("truncated length", raw[:4]),
                                   ("oversized header", struct.pack("<Q", 64*1024*1024+1)),
                                   ("truncated payload", raw[:-1])]:
                with self.subTest(label=label):
                    (root / "mtp.safetensors").write_bytes(corrupt)
                    with self.assertRaises(SystemExit):
                        mod._mtp_metadata(root)
            _write_sidecar(root)
            raw = (root / "mtp.safetensors").read_bytes().replace(b'BF16', b'U32 ')
            (root / "mtp.safetensors").write_bytes(raw)
            with self.assertRaisesRegex(SystemExit, "unsupported sidecar tensor"):
                mod._mtp_metadata(root)

    def test_metadata_preflight_never_establishes_qualification(self) -> None:
        self.assertFalse(mod.contract()["release_ready"])
        self.assertFalse(mod.contract()["qualification"])
        self.assertIn("metadata only", mod.contract()["validation_scope"])

    def test_mtp_gates_remain_unassessed_after_ready_metadata_preflight(self) -> None:
        expected = {gate: "not_assessed" for gate in ("MTP-S", "MTP-P", "MTP-D")}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}")
            (root / "model-manifest.json").write_text(json.dumps(_product_manifest()))
            _write_sidecar(root)
            mod._live_preflight(root)
        payload = mod.contract()
        self.assertEqual(payload["mtp_certification"], expected)
        self.assertEqual(set(payload["mtp_gates"]), set(expected))
        self.assertFalse(payload["qualification"])
        self.assertFalse(payload["release_ready"])
        self.assertIn("same-state verifier", payload["mtp_gates"]["MTP-S"])
        self.assertIn("does not change defaults", payload["mtp_gates"]["MTP-P"])
        self.assertIn("release tag", payload["mtp_gates"]["MTP-D"])
        self.assertIn("neither fail nor establish MTP-S", payload["diagnostic_only"][0])

    def test_both_dry_run_formats_disclose_unassessed_mtp_gates(self) -> None:
        for as_json in (False, True):
            with self.subTest(as_json=as_json), contextlib.redirect_stdout(io.StringIO()) as out:
                mod._print_contract(as_json)
            text = out.getvalue()
            if as_json:
                self.assertEqual(
                    json.loads(text)["mtp_certification"],
                    {gate: "not_assessed" for gate in ("MTP-S", "MTP-P", "MTP-D")},
                )
            else:
                for gate in ("MTP-S", "MTP-P", "MTP-D"):
                    self.assertIn(f"{gate} [not_assessed]", text)

    def test_live_preflight_json_keeps_stdout_pure_json(self) -> None:
        # Human preflight lines must not precede the JSON document on stdout.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}")
            (root / "model-manifest.json").write_text(json.dumps(_product_manifest()))
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                _write_sidecar(root)
                code = mod.main(["--model-dir", str(root), "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["family"], mod.contract()["family"])
        self.assertIn("live preflight ok", err.getvalue())

    def test_ready_flag_cannot_hide_other_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}")
            manifest = _product_manifest()
            manifest["runtime_status"]["blockers"] = ["geometry_not_validated"]
            (root / "model-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(SystemExit, "geometry_not_validated"):
                mod._live_preflight(root)

    def test_missing_experts_and_ungated_two_bit_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict("os.environ", {}, clear=True):
            root = Path(td)
            (root / "config.json").write_text("{}")
            manifest = _product_manifest()
            manifest["tensors"][0]["quantization"] = {
                "mode": "affine",
                "bits": 2,
                "group_size": 32,
            }
            (root / "model-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(SystemExit, "unsupported expert"):
                mod._live_preflight(root)
            manifest["tensors"][0]["role"] = "other"
            (root / "model-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(SystemExit, "no quantized expert"):
                mod._live_preflight(root)


if __name__ == "__main__":
    unittest.main()
