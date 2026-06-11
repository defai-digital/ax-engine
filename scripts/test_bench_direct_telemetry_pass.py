#!/usr/bin/env python3
"""Unit tests for bench_direct_telemetry_pass.py."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT_PATH = Path(__file__).with_name("bench_direct_telemetry_pass.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "bench_direct_telemetry_pass", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bench_direct_telemetry_pass"] = mod
MODULE_SPEC.loader.exec_module(mod)


def _make_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "model_dir": Path("/tmp/model"),
        "bench_script": Path("/tmp/bench.py"),
        "prompt_tokens": 128,
        "generation_tokens": 128,
        "repetitions": 5,
        "output": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class BenchDirectTelemetryPassTests(unittest.TestCase):
    def test_extract_telemetry_summary_basic(self) -> None:
        ax_results = [
            {
                "engine": "ax_engine_mlx",
                "decode_s": 1.5,
                "prefill_s": 0.5,
                "ax_mlx_telemetry": {"ax_mlx_decode_steps": 128},
            },
            {
                "engine": "ax_engine_mlx",
                "decode_s": 1.3,
                "prefill_s": 0.4,
                "ax_mlx_telemetry": {
                    "ax_mlx_decode_steps": 128,
                    "ax_mlx_prefix_cache_hits": 5,
                },
            },
        ]
        summary = mod.extract_telemetry_summary(ax_results)

        self.assertEqual(summary["ax_row_count"], 2)
        self.assertAlmostEqual(summary["avg_decode_s"], 1.4, places=2)
        self.assertAlmostEqual(summary["avg_prefill_s"], 0.45, places=2)
        self.assertIn("ax_mlx_decode_steps", summary["telemetry_keys"])
        self.assertIn("ax_mlx_prefix_cache_hits", summary["telemetry_keys"])

    def test_extract_telemetry_summary_empty(self) -> None:
        summary = mod.extract_telemetry_summary([])
        self.assertEqual(summary["ax_row_count"], 0)
        self.assertEqual(summary["avg_decode_s"], 0)

    def test_extract_telemetry_summary_zero_values(self) -> None:
        ax_results = [
            {
                "engine": "ax_engine_mlx",
                "decode_s": 0,
                "prefill_s": 0,
                "ax_mlx_telemetry": {},
            },
        ]
        summary = mod.extract_telemetry_summary(ax_results)
        self.assertEqual(summary["avg_decode_s"], 0)

    @patch("subprocess.run")
    def test_run_telemetry_pass_timeout(self, mock_run: MagicMock) -> None:
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=600)
        result = mod.run_telemetry_pass(_make_args())
        self.assertEqual(result["status"], "timeout")

    @patch("subprocess.run")
    def test_run_telemetry_pass_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stderr="error message")
        result = mod.run_telemetry_pass(_make_args())
        self.assertEqual(result["status"], "error")
        self.assertIn("exit code 1", result["error"])

    @patch("subprocess.run")
    def test_run_telemetry_pass_success(self, mock_run: MagicMock) -> None:
        import json

        mock_output = json.dumps(
            {
                "results": [
                    {
                        "engine": "ax_engine_mlx",
                        "decode_s": 1.5,
                        "prefill_s": 0.5,
                        "ax_mlx_telemetry": {"ax_mlx_decode_steps": 128},
                    },
                ],
                "bandwidth_accounting": {"safetensor_bytes": 1000},
            }
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_output, stderr="")
        result = mod.run_telemetry_pass(_make_args())
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["ax_row_count"], 1)
        self.assertIsNotNone(result["bandwidth_accounting"])

    @patch("subprocess.run")
    def test_run_telemetry_pass_no_ax_results(self, mock_run: MagicMock) -> None:
        import json

        mock_output = json.dumps({"results": [{"engine": "mlx_lm"}]})
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_output, stderr="")
        result = mod.run_telemetry_pass(_make_args())
        self.assertEqual(result["status"], "error")
        self.assertIn("no AX engine results", result["error"])


if __name__ == "__main__":
    unittest.main()
