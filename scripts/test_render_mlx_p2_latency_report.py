#!/usr/bin/env python3
"""Unit tests for rendering P2 latency reports."""

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

SCRIPT_PATH = SCRIPT_DIR / "render_mlx_p2_latency_report.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_mlx_p2_latency_report", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
renderer = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = renderer
MODULE_SPEC.loader.exec_module(renderer)


PROMPT_HASH = "a" * 64


def metric(median: float, *, max_value: float | None = None) -> dict[str, float]:
    return {
        "mean": median,
        "median": median,
        "p75": median * 1.1,
        "min": median * 0.9,
        "max": max_value if max_value is not None else median * 1.2,
    }


def startup_row(
    phase: str,
    *,
    ttft_ms: float,
    decode_tok_s: float,
    ratios: dict[str, float] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "phase": phase,
        "engine": "ax_engine_mlx",
        "ax_decode_policy": "direct_no_ngram_acceleration",
        "route": {"selected_backend": "mlx"},
        "context_tokens": 2048,
        "generation_tokens": 128,
        "prompt_token_ids_sha256": PROMPT_HASH,
        "repetitions": 3,
        "ttft_ms": metric(ttft_ms),
        "decode_tok_s": metric(decode_tok_s),
        "peak_memory_gb": metric(24.0, max_value=25.0),
    }
    if phase == "process_cold":
        row["process_start_ms"] = metric(100.0)
        row["server_ready_ms"] = metric(1500.0)
        row["model_load_ms"] = metric(1400.0)
        row["first_request_ttft_ms"] = metric(ttft_ms)
    elif phase == "model_warm":
        row["model_load_ms"] = metric(800.0)
    if ratios is not None:
        row["ratios_to_benchmark_warm"] = ratios
    return row


def startup_artifact() -> dict[str, object]:
    return {
        "schema_version": "ax.mlx_startup_latency.v1",
        "model": {"id": "mlx-community/Qwen3.5-9B-4bit"},
        "host": {"chip": "Apple M5 Max"},
        "benchmark": {
            "batch_size": 1,
            "temperature": 0.0,
            "context_tokens": 2048,
            "generation_tokens": 128,
            "prompt_token_ids_sha256": PROMPT_HASH,
            "repetitions": 3,
        },
        "rows": [
            startup_row(
                "process_cold",
                ttft_ms=1800.0,
                decode_tok_s=80.0,
                ratios={"ttft_ms": 3.0, "decode_tok_s": 0.8},
            ),
            startup_row(
                "model_warm",
                ttft_ms=900.0,
                decode_tok_s=95.0,
                ratios={"ttft_ms": 1.5, "decode_tok_s": 0.95},
            ),
            startup_row("benchmark_warm", ttft_ms=600.0, decode_tok_s=100.0),
        ],
    }


def concurrent_row(
    requests: int,
    *,
    request_ttft_ms: float,
    total_wall_ms: float,
    peak_memory_gb: float,
    ratios: dict[str, float] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "engine": "ax_engine_mlx",
        "ax_decode_policy": "direct_no_ngram_acceleration",
        "route": {"selected_backend": "mlx"},
        "concurrent_requests": requests,
        "context_tokens": 8192,
        "generation_tokens": 1,
        "prompt_token_ids_sha256": [f"{index + 1:064x}"[-64:] for index in range(requests)],
        "repetitions": 3,
        "request_ttft_ms": metric(request_ttft_ms),
        "total_wall_ms": metric(total_wall_ms),
        "queue_delay_ms": metric(25.0, max_value=50.0),
        "failure_count": metric(0.0, max_value=0.0),
        "peak_memory_gb": metric(peak_memory_gb, max_value=peak_memory_gb),
        "prefill_overlap": {
            "classification": "partial_overlap" if requests > 1 else "serialized",
            "overlap_efficiency": metric(0.5 if requests > 1 else 0.0),
        },
        "scheduler_evidence": {
            "scheduled_prefill_tokens": 8192 * requests,
            "scheduled_decode_tokens": max(requests - 1, 0),
            "skipped_prefill_tokens": 2048 if requests > 1 else 0,
            "skipped_decode_tokens": 0,
            "mixed_prefill_decode_batches": 1 if requests > 1 else 0,
        },
    }
    if ratios is not None:
        row["ratios_to_single_request"] = ratios
    return row


def concurrent_artifact() -> dict[str, object]:
    return {
        "schema_version": "ax.mlx_concurrent_prefill.v1",
        "model": {"id": "mlx-community/Qwen3.5-9B-4bit"},
        "host": {"chip": "Apple M5 Max"},
        "benchmark": {
            "batch_size": 1,
            "temperature": 0.0,
            "context_tokens": 8192,
            "generation_tokens": 1,
            "repetitions": 3,
        },
        "rows": [
            concurrent_row(
                1,
                request_ttft_ms=900.0,
                total_wall_ms=950.0,
                peak_memory_gb=30.0,
            ),
            concurrent_row(
                4,
                request_ttft_ms=1800.0,
                total_wall_ms=2400.0,
                peak_memory_gb=42.0,
                ratios={
                    "request_ttft_ms": 2.0,
                    "total_wall_ms": 2400.0 / 950.0,
                    "peak_memory_gb": 1.4,
                },
            ),
        ],
    }


class P2LatencyReportTests(unittest.TestCase):
    def write_artifacts(self) -> tuple[Path, Path]:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        startup = root / "startup-latency.json"
        concurrent = root / "concurrent-prefill.json"
        startup.write_text(json.dumps(startup_artifact(), indent=2) + "\n")
        concurrent.write_text(json.dumps(concurrent_artifact(), indent=2) + "\n")
        return startup, concurrent

    def test_renders_both_sections(self) -> None:
        startup, concurrent = self.write_artifacts()

        report = renderer.render_report(
            startup_artifact=startup,
            concurrent_artifact=concurrent,
        )

        self.assertIn("# MLX P2 Latency Report", report)
        self.assertIn("## Startup Latency", report)
        self.assertIn("process_cold", report)
        self.assertIn("## Concurrent Prefill", report)
        self.assertIn("| 4 | 1,800.0 | 2.000x |", report)
        self.assertIn("not proof of continuous batching", report)

    def test_cli_writes_report(self) -> None:
        startup, concurrent = self.write_artifacts()
        output = startup.with_name("p2-latency.md")

        exit_code = renderer.main_with_args_for_test(
            [
                "--startup-artifact",
                str(startup),
                "--concurrent-artifact",
                str(concurrent),
                "--output",
                str(output),
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("MLX P2 Latency Report", output.read_text())

    def test_requires_at_least_one_artifact(self) -> None:
        self.assertEqual(renderer.main_with_args_for_test([]), 1)


if __name__ == "__main__":
    unittest.main()
