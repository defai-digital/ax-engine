#!/usr/bin/env python3
"""Unit tests for qualify_qwen38_27b.py (no weights)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = Path(__file__).with_name("qualify_qwen38_27b.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "qualify_qwen38_27b", MODULE_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


sys.path.insert(0, str(ROOT / "scripts"))
from qwen38_live_gate import run_live, validate_cells, validate_host, validate_paired_greedy


class LiveGateTest(unittest.TestCase):
    def test_only_designated_hardware_is_admitted(self):
        host = dict(system="Darwin", arch="arm64", chip="Apple M4 Pro",
                    memory_bytes=64 * 1024**3, model="Mac16,11")
        validate_host(host)
        for changes in ({"chip": "Apple M5 Max"}, {"memory_bytes": 32 * 1024**3},
                        {"model": "MacBookPro"}, {"system": "Linux"}):
            with self.assertRaises(ValueError):
                validate_host({**host, **changes})

    def test_missing_skipped_partial_or_fallback_never_passes(self):
        direct = dict(mode="direct", status="ok", surface_passed=True, qa_items=32, qa_hard_passed=32, mtp_draft_tokens=0, mtp_verify_tokens=0)
        mtp = {**direct, "mode": "mtp", "mtp_draft_tokens": 3, "mtp_verify_tokens": 4}
        validate_cells([direct, mtp])
        for cells in ([], [direct], [direct, direct], [direct, mtp, mtp],
                      [direct, {**mtp, "status": "skip"}],
                      [direct, {**mtp, "status": "model_quality"}],
                      [direct, {**mtp, "surface_passed": False}],
                      [direct, {**mtp, "mtp_draft_tokens": 0}]):
            with self.assertRaises(ValueError):
                validate_cells(cells)

    def test_paired_greedy_rejects_divergence_after_individual_qa_passes(self):
        direct = dict(status="finished", prompt_tokens=list(range(1, 17)),
                      output_tokens=list(range(64)))
        self.assertTrue(validate_paired_greedy(direct, direct)["matched"])
        changed = list(direct["output_tokens"])
        changed[25] = 999
        with self.assertRaisesRegex(ValueError, "differs at output index 25"):
            validate_paired_greedy(direct, {**direct, "output_tokens": changed})

    def test_paired_greedy_rejects_empty_failed_partial_and_wrong_inputs(self):
        good = dict(status="finished", prompt_tokens=list(range(1, 17)),
                    output_tokens=list(range(64)))
        bad_cases = [{}, {**good, "status": "failed"},
                     {**good, "prompt_tokens": list(range(2, 18))},
                     {**good, "output_tokens": []},
                     {**good, "output_tokens": list(range(63))},
                     {**good, "output_tokens": [False] * 64},
                     {**good, "output_tokens": [-1] * 64}]
        for bad in bad_cases:
            for pair in ((bad, good), (good, bad), (bad, bad)):
                with self.subTest(pair=pair):
                    with self.assertRaisesRegex(ValueError, "missing, failed, or incomplete"):
                        validate_paired_greedy(*pair)

    def test_wrong_host_persists_failed_result(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "result"
            with patch("qwen38_live_gate.platform.system", return_value="Darwin"), \
                 patch("qwen38_live_gate.platform.machine", return_value="arm64"), \
                 patch("qwen38_live_gate.subprocess.check_output", side_effect=[
                     "Apple M5 Max", "MacBookPro", str(128 * 1024**3), "26.6.2"]):
                code = run_live(SimpleNamespace(output=out), mod.contract(), ROOT)
            self.assertEqual(code, 1)
            result = json.loads((out / "qualification.json").read_text())
            self.assertEqual(result["schema"], 2)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["cells"], [])
            self.assertIn("requires Mac mini", result["error"])

    def test_run_requires_all_provenance_arguments(self):
        with self.assertRaises(SystemExit):
            mod.parse_args(["--run"])


class QualifyQwen38Test(unittest.TestCase):
    def test_contract_pins_primary_pack(self) -> None:
        payload = mod.contract()
        self.assertEqual(payload["alias"], "qwen3.8-27b:axq")
        self.assertEqual(
            payload["repo_id"],
            "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
        )
        self.assertEqual(
            payload["revision"],
            "3e290738e96972307c6aeb9934ab170ca0eae1c1",
        )
        self.assertIn("27B weights", payload["ci"])
        self.assertEqual(payload["host_class"], "Mac mini M4 Pro, 64 GB")

    def test_dry_run_cli_json(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--dry-run", "--json"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["alias"], "qwen3.8-27b:axq")

    def test_live_preflight_rejects_wrong_revision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "wrongrev"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                mod._live_preflight(model_dir)

    def test_live_preflight_accepts_pinned_snapshot_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / mod.PRIMARY_REVISION
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            mod._live_preflight(model_dir)


if __name__ == "__main__":
    unittest.main()
