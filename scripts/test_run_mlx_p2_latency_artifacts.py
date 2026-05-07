#!/usr/bin/env python3
"""Unit tests for the P2 latency artifact runner helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCRIPT_PATH = Path(__file__).with_name("run_mlx_p2_latency_artifacts.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "run_mlx_p2_latency_artifacts", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
runner = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runner
MODULE_SPEC.loader.exec_module(runner)


def prompt(index: int = 0) -> object:
    return runner.PromptDoc(
        prompt_tokens=8192,
        generation_tokens=1,
        vocab_size=1000,
        token_ids=[1 + index, 2 + index, 3 + index],
        token_ids_sha256=f"{index + 1:064x}"[-64:],
        token_ids_path=f"/tmp/prompt-{index}.json",
    )


class P2LatencyRunnerTests(unittest.TestCase):
    def test_parse_concurrency_levels_adds_single_request_baseline(self) -> None:
        self.assertEqual(runner.parse_concurrency_levels("2,4"), [1, 2, 4])

    def test_concurrent_row_computes_ratios_and_overlap_classification(self) -> None:
        single_row = runner.concurrent_row(
            prompts=[prompt(0)],
            trials=[
                {
                    "request_ttft_ms": 100.0,
                    "total_wall_ms": 120.0,
                    "queue_delay_ms": 20.0,
                    "failure_count": 0.0,
                    "peak_memory_gb": 20.0,
                    "observations": [],
                },
                {
                    "request_ttft_ms": 110.0,
                    "total_wall_ms": 130.0,
                    "queue_delay_ms": 20.0,
                    "failure_count": 0.0,
                    "peak_memory_gb": 21.0,
                    "observations": [],
                },
                {
                    "request_ttft_ms": 120.0,
                    "total_wall_ms": 140.0,
                    "queue_delay_ms": 20.0,
                    "failure_count": 0.0,
                    "peak_memory_gb": 22.0,
                    "observations": [],
                },
            ],
            single_row=None,
        )
        multi_row = runner.concurrent_row(
            prompts=[prompt(0), prompt(1), prompt(2), prompt(3)],
            trials=[
                {
                    "request_ttft_ms": 300.0,
                    "total_wall_ms": 260.0,
                    "queue_delay_ms": 80.0,
                    "failure_count": 0.0,
                    "peak_memory_gb": 30.0,
                    "observations": [],
                },
                {
                    "request_ttft_ms": 330.0,
                    "total_wall_ms": 280.0,
                    "queue_delay_ms": 90.0,
                    "failure_count": 0.0,
                    "peak_memory_gb": 31.0,
                    "observations": [],
                },
                {
                    "request_ttft_ms": 360.0,
                    "total_wall_ms": 300.0,
                    "queue_delay_ms": 100.0,
                    "failure_count": 0.0,
                    "peak_memory_gb": 32.0,
                    "observations": [],
                },
            ],
            single_row=single_row,
        )

        self.assertEqual(multi_row["concurrent_requests"], 4)
        self.assertAlmostEqual(
            multi_row["ratios_to_single_request"]["request_ttft_ms"],
            3.0,
        )
        self.assertEqual(
            multi_row["prefill_overlap"]["classification"],
            "partial_overlap",
        )

    def test_startup_rows_keep_warm_load_metrics_out_of_benchmark_warm(self) -> None:
        row = runner.startup_phase_row(
            phase="benchmark_warm",
            prompt=prompt(0),
            observations=[
                {
                    "ttft_ms": 100.0,
                    "decode_tok_s": 50.0,
                    "wall_ms": 150.0,
                    "peak_memory_gb": 20.0,
                }
            ],
        )

        self.assertNotIn("model_load_ms", row)
        self.assertNotIn("server_ready_ms", row)

    def test_dry_run_does_not_require_real_model_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen"}))
            output_dir = root / "out"

            code = runner.main_with_args_for_test(
                [
                    "--model-dir",
                    str(model_dir),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ]
            )

            self.assertEqual(code, 0)

    def test_dry_run_reports_markdown_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen"}))
            output_dir = root / "out"

            code = runner.main_with_args_for_test(
                [
                    "--model-dir",
                    str(model_dir),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
