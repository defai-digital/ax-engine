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


class QualifyFlashNextTest(unittest.TestCase):
    def test_contract_is_incubating_on_studio_sku(self) -> None:
        payload = mod.contract()
        self.assertEqual(payload["family"], "qwen4_exp")
        self.assertEqual(payload["host_class"], "Mac Studio M5 Ultra, 256 GB")
        self.assertTrue(payload["fail_closed"])
        self.assertEqual(payload["experimental_opt_in"], "AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1")
        self.assertEqual(payload["experimental_2bit_opt_in"], "AX_ENGINE_2BIT_EXPERIMENTAL=1")
        self.assertEqual(payload["experimental_expert_layouts"], [
            {"bits": 2, "group_size": 32},
            {"bits": 4, "group_size": 64},
            {"bits": 6, "group_size": 64},
        ])
        self.assertIn("qwen3.8-27b:axq", payload["not"])

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

    def test_live_preflight_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                mod._live_preflight(model_dir)
            self.assertIn("incubating", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
