#!/usr/bin/env python3
"""Tests for the Rapid-MLX prompt-suite benchmark adapter."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("bench_rapid_mlx_prompt_suites.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_rapid_mlx_prompt_suites", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
rapid = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bench_rapid_mlx_prompt_suites"] = rapid
MODULE_SPEC.loader.exec_module(rapid)


class RapidMlxPromptSuiteTests(unittest.TestCase):
    def test_prepare_rapid_mtp_compat_site_writes_sitecustomize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lightning = root / "lightning-mlx"
            patch = lightning / "vllm_mlx" / "patches" / "qwen3_next_mtp.py"
            patch.parent.mkdir(parents=True)
            patch.write_text("# patch\n")

            compat = rapid.prepare_rapid_mtp_compat_site(
                output_dir=root / "out",
                lightning_source=lightning,
                mode="lightning",
            )

            sitecustomize = Path(compat["sitecustomize"])
            self.assertEqual(compat["mode"], "lightning")
            self.assertEqual(Path(compat["patch_path"]), patch)
            self.assertTrue(sitecustomize.is_file())
            text = sitecustomize.read_text()
            self.assertIn("AX_RAPID_MLX_QWEN3_NEXT_MTP_PATCH", text)
            self.assertIn("vllm_mlx.patches.qwen3_next_mtp", text)

    def test_prepare_rapid_mtp_compat_site_none_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compat = rapid.prepare_rapid_mtp_compat_site(
                output_dir=Path(tmp) / "out",
                lightning_source=Path(tmp) / "missing",
                mode="none",
            )

        self.assertEqual(compat, {"mode": "none"})


if __name__ == "__main__":
    unittest.main()
