#!/usr/bin/env python3
"""Unit tests for the AX serving benchmark artifact checker."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_ax_serving_benchmark_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_ax_serving_benchmark_artifact", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


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


def observation(prompt_id: str, input_tokens: int, ttft_ms: float) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "category": "long_rag",
        "phase": "measured",
        "status": 200,
        "ok": True,
        "error": None,
        "scheduled_at_s": 0.0,
        "started_at_s": 0.0,
        "queue_delay_ms": 0.0,
        "e2e_latency_ms": ttft_ms + 200.0,
        "ttft_ms": ttft_ms,
        "client_tpot_ms": 50.0,
        "stream_step_interval_ms": [50.0, 55.0],
        "input_tokens": input_tokens,
        "max_output_tokens": 5,
        "output_tokens": 5,
        "output_chunks": 5,
        "events": 7,
        "metadata": {},
    }


def artifact() -> dict[str, object]:
    observations = [
        observation("p1", 8192, 1000.0),
        observation("p2", 9000, 1200.0),
    ]
    intervals = [50.0, 55.0, 50.0, 55.0]
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
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": 0,
        },
        "corpus": {
            "path": "benchmarks/corpora/serving/long.jsonl",
            "sha256": "a" * 64,
            "prompt_count": 2,
            "categories": ["long_rag"],
        },
        "summary": {
            "requests": 2,
            "ok_requests": 2,
            "error_requests": 0,
            "request_throughput_rps": 1.0,
            "output_token_throughput_tok_s": 10.0,
            "ttft_ms": dist([1000.0, 1200.0]),
            "client_tpot_ms": dist([50.0, 50.0]),
            "stream_step_interval_ms": dist(intervals),
            "e2e_latency_ms": dist([1200.0, 1400.0]),
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
        },
        "by_category": {
            "long_rag": {
                "requests": 2,
                "ok_requests": 2,
                "error_requests": 0,
            }
        },
        "observations": observations,
    }


class AxServingBenchmarkArtifactTests(unittest.TestCase):
    def write_artifact(self, payload: dict[str, object]) -> Path:
        root = Path(self.tmp.name)
        path = root / "serving.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_artifact_passes_with_long_prompt_gate(self) -> None:
        path = self.write_artifact(artifact())

        result = checker.validate_serving_benchmark_artifact(
            path,
            min_requests=2,
            min_concurrency=2,
            require_zero_errors=True,
            require_slo=True,
            min_goodput_ratio=0.99,
            min_input_tokens_p95=8192,
        )

        self.assertEqual(result["schema_version"], "ax.serving_benchmark.v1")

    def test_error_requests_fail_by_default(self) -> None:
        payload = artifact()
        summary = payload["summary"]
        assert isinstance(summary, dict)
        summary["ok_requests"] = 1
        summary["error_requests"] = 1
        observations = payload["observations"]
        assert isinstance(observations, list)
        bad = observations[1]
        assert isinstance(bad, dict)
        bad["ok"] = False
        bad["status"] = 500
        bad["error"] = "server failed"
        path = self.write_artifact(payload)

        with self.assertRaisesRegex(checker.ArtifactCheckError, "failed measured requests"):
            checker.validate_serving_benchmark_artifact(
                path,
                min_requests=1,
                min_concurrency=1,
                require_zero_errors=True,
                require_slo=False,
                min_goodput_ratio=None,
                min_input_tokens_p95=None,
            )

    def test_long_prompt_gate_rejects_short_corpus(self) -> None:
        payload = artifact()
        summary = payload["summary"]
        assert isinstance(summary, dict)
        summary["input_tokens"] = dist([512.0, 1024.0])
        path = self.write_artifact(payload)

        with self.assertRaisesRegex(checker.ArtifactCheckError, "below required 8192"):
            checker.validate_serving_benchmark_artifact(
                path,
                min_requests=1,
                min_concurrency=1,
                require_zero_errors=True,
                require_slo=False,
                min_goodput_ratio=None,
                min_input_tokens_p95=8192,
            )

    def test_cli_validates_artifact_file(self) -> None:
        path = self.write_artifact(artifact())

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                str(path),
                "--min-requests",
                "2",
                "--min-concurrency",
                "2",
                "--require-slo",
                "--min-goodput-ratio",
                "1.0",
                "--min-input-tokens-p95",
                "8192",
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("AX serving benchmark artifact check passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()
