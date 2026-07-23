#!/usr/bin/env python3
"""Tests for multi-model serving artifact checker."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import check_ax_multimodel_serving_artifact as checker


def _dist(value: float = 10.0, count: int = 2) -> dict[str, float | int]:
    return {
        "count": count,
        "min": value,
        "mean": value,
        "p50": value,
        "p75": value,
        "p90": value,
        "p95": value,
        "p99": value,
        "max": value,
    }


def _summary(*, requests: int = 2, errors: int = 0) -> dict[str, object]:
    ok = requests - errors
    return {
        "requests": requests,
        "ok_requests": ok,
        "error_requests": errors,
        "request_throughput_rps": 1.0,
        "output_token_throughput_tok_s": 20.0,
        "ttft_ms": _dist(100.0),
        "client_tpot_ms": _dist(15.0),
        "stream_step_interval_ms": _dist(16.0),
        "e2e_latency_ms": _dist(200.0),
        "queue_delay_ms": _dist(1.0),
        "input_tokens": _dist(32.0),
        "output_tokens": _dist(64.0),
    }


def _artifact() -> dict[str, object]:
    summary = _summary()
    return {
        "schema_version": checker.SCHEMA_VERSION,
        "methodology": {
            "scope": "timed_multi_model_serving_and_lifecycle",
            "request_endpoint": "/v1/generate/stream",
            "timing_scope": "client_observed",
        },
        "target": {
            "base_url": "http://127.0.0.1:31418",
            "models": ["qwen3.5-9b", "gemma-4-12b-it"],
        },
        "scenario": {
            "path": "scenario.jsonl",
            "sha256": "a" * 64,
            "events": 2,
        },
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0},
        "summary": summary,
        "by_model": {
            "qwen3.5-9b": _summary(requests=1),
            "gemma-4-12b-it": _summary(requests=1),
        },
        "lifecycle": {
            "events": 0,
            "ok_events": 0,
            "error_events": 0,
            "latency_ms": None,
        },
        "focus": {"families": ["gemma4", "qwen3"], "policy": "qwen3_gemma4_primary"},
        "interactive_stream_gap_ms": _dist(18.0, count=10),
        "availability": {"request_http_503": 0, "request_http_5xx": 0, "request_error_rate": 0.0},
        "observations": [
            {
                "event_id": "qwen",
                "kind": "request",
                "model_id": "qwen3.5-9b",
                "category": "interactive_decode",
                "ok": True,
                "status": 200,
                "scheduled_at_s": 0.0,
                "started_at_s": 0.0,
                "e2e_latency_ms": 200.0,
                "stream_step_interval_ms": [18.0, 18.0],
            },
            {
                "event_id": "gemma",
                "kind": "request",
                "model_id": "gemma-4-12b-it",
                "category": "interactive_decode",
                "ok": True,
                "status": 200,
                "scheduled_at_s": 0.05,
                "started_at_s": 0.05,
                "e2e_latency_ms": 220.0,
                "stream_step_interval_ms": [19.0, 19.0],
            },
        ],
    }


class MultiModelArtifactCheckerTests(unittest.TestCase):
    def test_accepts_focus_pair_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "artifact.json"
            path.write_text(json.dumps(_artifact()))
            checker.validate_multimodel_serving_artifact(
                path,
                min_requests=2,
                require_zero_errors=True,
                require_focus_families=["qwen3", "gemma4"],
                max_interactive_stream_gap_p95_ms=50.0,
                max_request_http_503=0,
            )

    def test_rejects_missing_focus_family(self) -> None:
        payload = _artifact()
        payload["target"]["models"] = ["qwen3.5-9b"]  # type: ignore[index]
        payload["by_model"] = {"qwen3.5-9b": _summary(requests=2)}  # type: ignore[index]
        payload["observations"] = [  # type: ignore[index]
            {
                "event_id": "qwen",
                "kind": "request",
                "model_id": "qwen3.5-9b",
                "category": "interactive_decode",
                "ok": True,
                "status": 200,
                "scheduled_at_s": 0.0,
                "started_at_s": 0.0,
                "e2e_latency_ms": 200.0,
                "stream_step_interval_ms": [18.0],
            },
            {
                "event_id": "qwen-2",
                "kind": "request",
                "model_id": "qwen3.5-9b",
                "category": "interactive_decode",
                "ok": True,
                "status": 200,
                "scheduled_at_s": 0.1,
                "started_at_s": 0.1,
                "e2e_latency_ms": 210.0,
                "stream_step_interval_ms": [18.0],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "artifact.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(checker.ArtifactCheckError, "gemma4"):
                checker.validate_multimodel_serving_artifact(
                    path,
                    min_requests=1,
                    require_zero_errors=True,
                    require_focus_families=["qwen3", "gemma4"],
                    max_interactive_stream_gap_p95_ms=None,
                    max_request_http_503=None,
                )

    def test_rejects_stream_gap_over_cap(self) -> None:
        payload = _artifact()
        payload["interactive_stream_gap_ms"] = _dist(80.0, count=10)  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "artifact.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(checker.ArtifactCheckError, "stream-gap"):
                checker.validate_multimodel_serving_artifact(
                    path,
                    min_requests=1,
                    require_zero_errors=True,
                    require_focus_families=[],
                    max_interactive_stream_gap_p95_ms=50.0,
                    max_request_http_503=None,
                )

    def test_cli_passes_on_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "artifact.json"
            path.write_text(json.dumps(_artifact()))
            code = checker.main(
                [
                    str(path),
                    "--require-focus-family",
                    "qwen3",
                    "--require-focus-family",
                    "gemma4",
                ]
            )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
