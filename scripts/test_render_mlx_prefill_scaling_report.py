#!/usr/bin/env python3
"""Unit tests for rendering prefill scaling reports."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "render_mlx_prefill_scaling_report.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_mlx_prefill_scaling_report", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
renderer = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = renderer
MODULE_SPEC.loader.exec_module(renderer)


HASH_1K = "e" * 64
HASH_8K = "f" * 64


def metric(median: float, *, max_value: float | None = None) -> dict[str, float]:
    return {
        "mean": median,
        "median": median,
        "min": median,
        "max": max_value if max_value is not None else median,
    }


def row(
    *,
    engine: str,
    context_tokens: int,
    prompt_hash: str,
    prefill_tok_s: float,
    ttft_ms: float,
    peak_memory_gb: float,
    ratios: dict[str, float] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "engine": engine,
        "context_tokens": context_tokens,
        "generation_tokens": 1,
        "prompt_token_ids_sha256": prompt_hash,
        "repetitions": 3,
        "prefill_tok_s": metric(prefill_tok_s),
        "ttft_ms": metric(ttft_ms),
        "peak_memory_gb": metric(peak_memory_gb, max_value=peak_memory_gb),
    }
    if engine == "mlx_lm":
        payload["baseline"] = {"role": "primary_reference", "method": "mlx_lm.benchmark"}
    else:
        payload["ax_decode_policy"] = "direct_no_ngram_acceleration"
        payload["route"] = {"selected_backend": "mlx"}
        payload["ratios_to_mlx_lm"] = ratios or {
            "prefill_tok_s": 1.0,
            "ttft_ms": 1.0,
        }
    return payload


def artifact() -> dict[str, object]:
    return {
        "schema_version": "ax.mlx_prefill_scaling.v1",
        "model": {"id": "mlx-community/Qwen3.5-9B-4bit"},
        "host": {"chip": "Apple M5 Max", "memory_gb": 128},
        "benchmark": {
            "batch_size": 1,
            "temperature": 0.0,
            "prefill_step_size": 2048,
            "repetitions": 3,
        },
        "rows": [
            row(
                engine="mlx_lm",
                context_tokens=1024,
                prompt_hash=HASH_1K,
                prefill_tok_s=3000.0,
                ttft_ms=350.0,
                peak_memory_gb=20.0,
            ),
            row(
                engine="ax_engine_mlx",
                context_tokens=1024,
                prompt_hash=HASH_1K,
                prefill_tok_s=3200.0,
                ttft_ms=320.0,
                peak_memory_gb=21.0,
                ratios={"prefill_tok_s": 3200.0 / 3000.0, "ttft_ms": 320.0 / 350.0},
            ),
            row(
                engine="mlx_lm",
                context_tokens=8192,
                prompt_hash=HASH_8K,
                prefill_tok_s=2500.0,
                ttft_ms=3300.0,
                peak_memory_gb=32.0,
            ),
            row(
                engine="ax_engine_mlx",
                context_tokens=8192,
                prompt_hash=HASH_8K,
                prefill_tok_s=1200.0,
                ttft_ms=6800.0,
                peak_memory_gb=33.0,
                ratios={"prefill_tok_s": 1200.0 / 2500.0, "ttft_ms": 6800.0 / 3300.0},
            ),
        ],
    }


class PrefillScalingReportTests(unittest.TestCase):
    def write_artifact(self) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "prefill-scaling.json"
        path.write_text(json.dumps(artifact(), indent=2) + "\n")
        return path

    def test_renders_markdown_table_and_bend_marker(self) -> None:
        path = self.write_artifact()

        report = renderer.render_report(path, bend_drop_ratio=0.75)

        self.assertIn("# MLX Prefill Scaling Report", report)
        self.assertIn("| 8,192 | 1 | 2,500.0 | 1,200.0 | 0.480x |", report)
        self.assertIn("drop <= 0.75x previous", report)
        self.assertIn("direct AX prefill/TTFT scaling, not n-gram", report)

    def test_cli_writes_report(self) -> None:
        path = self.write_artifact()
        output = path.with_suffix(".md")

        exit_code = renderer.main_with_args_for_test([str(path), "--output", str(output)])

        self.assertEqual(exit_code, 0)
        self.assertIn("MLX Prefill Scaling Report", output.read_text())


if __name__ == "__main__":
    unittest.main()
