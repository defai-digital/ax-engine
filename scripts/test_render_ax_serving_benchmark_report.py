#!/usr/bin/env python3
"""Unit tests for AX serving benchmark Markdown reports."""

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

SCRIPT_PATH = SCRIPT_DIR / "render_ax_serving_benchmark_report.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_ax_serving_benchmark_report", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
renderer = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = renderer
MODULE_SPEC.loader.exec_module(renderer)


def dist(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[len(ordered) // 2],
        "p75": ordered[-1],
        "p90": ordered[-1],
        "p95": ordered[-1],
        "p99": ordered[-1],
        "max": ordered[-1],
    }


def category_summary() -> dict[str, object]:
    return {
        "requests": 2,
        "ok_requests": 2,
        "error_requests": 0,
        "request_throughput_rps": 1.0,
        "output_token_throughput_tok_s": 10.0,
        "ttft_ms": dist([1000.0, 1200.0]),
        "client_tpot_ms": dist([40.0, 50.0]),
        "stream_step_interval_ms": dist([40.0, 50.0]),
        "e2e_latency_ms": dist([1300.0, 1500.0]),
        "queue_delay_ms": dist([0.0, 0.0]),
        "input_tokens": dist([8192.0, 9000.0]),
        "output_tokens": dist([5.0, 5.0]),
        "goodput": {
            "requests": 2,
            "ratio": 1.0,
            "ttft_slo_ms": 2000.0,
            "client_tpot_slo_ms": 100.0,
            "e2e_slo_ms": 15000.0,
        },
    }


def artifact() -> dict[str, object]:
    observations = [
        {
            "prompt_id": "p1",
            "category": "long_rag",
            "phase": "measured",
            "status": 200,
            "ok": True,
            "error": None,
            "scheduled_at_s": 0.0,
            "started_at_s": 0.0,
            "queue_delay_ms": 0.0,
            "e2e_latency_ms": 1300.0,
            "ttft_ms": 1000.0,
            "client_tpot_ms": 40.0,
            "stream_step_interval_ms": [40.0],
            "input_tokens": 8192,
            "max_output_tokens": 5,
            "output_tokens": 5,
            "output_chunks": 5,
            "events": 7,
            "metadata": {},
        },
        {
            "prompt_id": "p2",
            "category": "long_rag",
            "phase": "measured",
            "status": 200,
            "ok": True,
            "error": None,
            "scheduled_at_s": 0.0,
            "started_at_s": 0.0,
            "queue_delay_ms": 0.0,
            "e2e_latency_ms": 1500.0,
            "ttft_ms": 1200.0,
            "client_tpot_ms": 50.0,
            "stream_step_interval_ms": [50.0],
            "input_tokens": 9000,
            "max_output_tokens": 5,
            "output_tokens": 5,
            "output_chunks": 5,
            "events": 7,
            "metadata": {},
        },
    ]
    return {
        "schema_version": "ax.serving_benchmark.v1",
        "created_at": "2026-05-14T00:00:00+00:00",
        "methodology": {
            "scope": "online_serving_streaming_latency",
            "endpoint": "/v1/generate/stream",
            "timing_scope": "client_observed_sse",
            "notes": ["unit fixture"],
        },
        "target": {
            "base_url": "http://127.0.0.1:8080",
            "model_id": "qwen3_dense",
            "input_kind": "tokens",
        },
        "load": {
            "concurrency": 2,
            "request_rate_rps": None,
            "warmup_requests": 0,
            "measured_requests": 2,
        },
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0},
        "corpus": {
            "path": "benchmarks/corpora/serving/long.jsonl",
            "sha256": "a" * 64,
            "prompt_count": 2,
            "categories": ["long_rag"],
        },
        "summary": category_summary(),
        "by_category": {"long_rag": category_summary()},
        "observations": observations,
    }


class AxServingBenchmarkReportTests(unittest.TestCase):
    def write_artifact(self) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "serving.json"
        path.write_text(json.dumps(artifact(), indent=2) + "\n")
        return path

    def test_renders_summary_and_category_tables(self) -> None:
        path = self.write_artifact()

        report = renderer.render_report(
            path,
            min_requests=2,
            min_concurrency=2,
            require_slo=True,
            min_input_tokens_p95=8192,
        )

        self.assertIn("# AX Serving Benchmark Report", report)
        self.assertIn("Goodput: 1.000", report)
        self.assertIn("| TTFT ms | 2 | 1,200.0 |", report)
        self.assertIn("| long_rag | 2 | 0 | 1,200.0 |", report)
        self.assertIn("client-observed SSE timings", report)

    def test_cli_writes_report(self) -> None:
        path = self.write_artifact()
        output = path.with_suffix(".md")

        exit_code = renderer.main_with_args_for_test(
            [
                str(path),
                "--output",
                str(output),
                "--min-input-tokens-p95",
                "8192",
                "--require-slo",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("AX Serving Benchmark Report", output.read_text())


if __name__ == "__main__":
    unittest.main()
