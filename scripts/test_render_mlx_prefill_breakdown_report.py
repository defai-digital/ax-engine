#!/usr/bin/env python3
"""Unit tests for rendering AX MLX prefill breakdown reports."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "render_mlx_prefill_breakdown_report.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_mlx_prefill_breakdown_report",
    SCRIPT_PATH,
)
assert MODULE_SPEC and MODULE_SPEC.loader
renderer = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = renderer
MODULE_SPEC.loader.exec_module(renderer)


def metric(median: float) -> dict[str, float]:
    return {"mean": median, "median": median, "min": median, "max": median}


def ax_row(prompt_tokens: int, prefill_tok_s: float) -> dict[str, object]:
    return {
        "engine": "ax_engine_mlx",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": 128,
        "prefill_tok_s": metric(prefill_tok_s),
        "ax_mlx_telemetry": {
            "ax_mlx_prefill_wall_us": 100_000,
            "ax_mlx_prefill_forward_wall_us": 70_000,
            "ax_mlx_prefill_prefix_cache_wall_us": 20_000,
            "ax_mlx_prefill_generation_state_wall_us": 5_000,
            "ax_mlx_prefill_eval_barriers": 1,
            "ax_mlx_prefill_drain_async_evals": 2,
        },
    }


def mlx_row(prompt_tokens: int) -> dict[str, object]:
    return {
        "engine": "mlx_lm",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": 128,
        "prefill_tok_s": metric(1_000.0),
    }


def llama_row(prompt_tokens: int, prefill_tok_s: float) -> dict[str, object]:
    return {
        "engine": "llama_cpp_metal",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": 128,
        "prefill_tok_s": metric(prefill_tok_s),
    }


class PrefillBreakdownReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_artifacts(self) -> tuple[Path, Path]:
        ax_path = self.root / "qwen.json"
        ax_path.write_text(
            json.dumps(
                {
                    "schema_version": "ax.mlx_inference_stack.v2",
                    "model": "mlx-community/Qwen",
                    "results": [
                        mlx_row(128),
                        ax_row(128, 1_500.0),
                        mlx_row(512),
                        ax_row(512, 2_400.0),
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        llama_dir = self.root / "llama"
        llama_dir.mkdir()
        (llama_dir / "qwen.json").write_text(
            json.dumps(
                {
                    "schema_version": "ax.mlx_inference_stack.v2",
                    "results": [
                        llama_row(128, 2_000.0),
                        llama_row(512, 2_000.0),
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        return ax_path, llama_dir

    def test_renders_breakdown_table_with_ratios(self) -> None:
        ax_path, llama_dir = self.write_artifacts()

        rows = renderer.build_rows(ax_path, llama_dir=llama_dir)
        report = renderer.render_report(rows, title="Prefill Slice")

        self.assertIn("# Prefill Slice", report)
        self.assertIn("mlx-community/Qwen | 128 | 1,500.0 | 1.500x | 0.750x", report)
        self.assertIn("70.0 | 20.0 | 5.0 | 5.0 | 70.0%", report)
        self.assertIn("Worst AX/llama.cpp row", report)
        self.assertIn("prompt=128", report)

    def test_cli_writes_report_from_directory(self) -> None:
        ax_path, llama_dir = self.write_artifacts()
        output = self.root / "report.md"

        exit_code = renderer.main_with_args_for_test(
            [str(ax_path.parent), "--llama-dir", str(llama_dir), "--output", str(output)]
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("AX MLX Prefill Breakdown Report", output.read_text())

    def test_missing_prefill_telemetry_fails_closed(self) -> None:
        path = self.root / "bad.json"
        payload = {
            "results": [
                {
                    "engine": "ax_engine_mlx",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "prefill_tok_s": metric(1_000.0),
                }
            ]
        }
        path.write_text(json.dumps(payload) + "\n")

        with self.assertRaisesRegex(renderer.PrefillBreakdownReportError, "lacks ax_mlx_telemetry"):
            renderer.build_rows(path)


if __name__ == "__main__":
    unittest.main()
