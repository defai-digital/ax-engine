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

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = Path(__file__).with_name("qualify_qwen38_27b.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "qualify_qwen38_27b", MODULE_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


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
        self.assertEqual(payload["host_class"], "Mac mini M5, 64 GB")

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
