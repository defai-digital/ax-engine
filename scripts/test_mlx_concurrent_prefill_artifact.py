#!/usr/bin/env python3
"""Unit tests for concurrent prefill artifact checks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_mlx_concurrent_prefill_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mlx_concurrent_prefill_artifact", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def prompt_hash(index: int) -> str:
    return f"{index:064x}"[-64:]


def metric(median: float, *, max_value: float | None = None) -> dict[str, float]:
    return {
        "median": median,
        "p75": median * 1.1,
        "min": median * 0.9,
        "max": max_value if max_value is not None else median * 1.2,
    }


def row(
    *,
    concurrent_requests: int,
    request_ttft_ms: float,
    total_wall_ms: float,
    peak_memory_gb: float,
    ratios: dict[str, float] | None = None,
    failure_count: float = 0.0,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "engine": "ax_engine_mlx",
        "ax_decode_policy": "direct_no_ngram_acceleration",
        "route": {"selected_backend": "mlx"},
        "concurrent_requests": concurrent_requests,
        "context_tokens": 8192,
        "generation_tokens": 1,
        "prompt_token_ids_sha256": [
            prompt_hash(index + 1) for index in range(concurrent_requests)
        ],
        "repetitions": 3,
        "request_ttft_ms": metric(request_ttft_ms),
        "total_wall_ms": metric(total_wall_ms),
        "queue_delay_ms": metric(20.0, max_value=50.0),
        "failure_count": metric(failure_count, max_value=failure_count),
        "peak_memory_gb": metric(peak_memory_gb, max_value=peak_memory_gb),
        "prefill_overlap": {
            "classification": "partial_overlap" if concurrent_requests > 1 else "serialized",
            "overlap_efficiency": metric(0.45 if concurrent_requests > 1 else 0.0),
        },
        "scheduler_evidence": {
            "scheduled_prefill_tokens": 8192 * concurrent_requests,
            "scheduled_decode_tokens": max(concurrent_requests - 1, 0),
            "skipped_prefill_tokens": 2048 if concurrent_requests > 1 else 0,
            "skipped_decode_tokens": 0,
            "mixed_prefill_decode_batches": 1 if concurrent_requests > 1 else 0,
        },
    }
    if ratios is not None:
        payload["ratios_to_single_request"] = ratios
    return payload


def valid_artifact() -> dict[str, object]:
    return {
        "schema_version": checker.SCHEMA_VERSION,
        "model": {
            "id": "mlx-community/Qwen3.5-9B-4bit",
            "quantization": "4-bit",
        },
        "host": {
            "chip": "Apple M5 Max",
            "memory_gb": 128,
            "os": "macOS 26.4.1",
        },
        "benchmark": {
            "batch_size": 1,
            "temperature": 0.0,
            "context_tokens": 8192,
            "generation_tokens": 1,
            "repetitions": 3,
        },
        "claim_scope": "concurrent_long_context_prefill",
        "rows": [
            row(
                concurrent_requests=1,
                request_ttft_ms=900.0,
                total_wall_ms=950.0,
                peak_memory_gb=30.0,
            ),
            row(
                concurrent_requests=4,
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


class ConcurrentPrefillArtifactTests(unittest.TestCase):
    def write_fixture(self, artifact: dict[str, object]) -> Path:
        root = Path(tempfile.mkdtemp())
        path = root / "concurrent-prefill.json"
        path.write_text(json.dumps(artifact, indent=2) + "\n")
        self.addCleanup(lambda: root.rmdir())
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_valid_artifact_passes(self) -> None:
        path = self.write_fixture(valid_artifact())

        checked = checker.validate_mlx_concurrent_prefill_artifact(path)

        self.assertEqual(checked, ["concurrency=1", "concurrency=4"])

    def test_missing_single_request_baseline_fails(self) -> None:
        artifact = valid_artifact()
        artifact["rows"] = [
            payload for payload in artifact["rows"] if payload["concurrent_requests"] != 1
        ]
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.ConcurrentPrefillArtifactError, "concurrency=1"):
            checker.validate_mlx_concurrent_prefill_artifact(path)

    def test_prompt_hash_count_must_match_concurrency(self) -> None:
        artifact = valid_artifact()
        artifact["rows"][1]["prompt_token_ids_sha256"] = [prompt_hash(1)]
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.ConcurrentPrefillArtifactError, "one hash per request"):
            checker.validate_mlx_concurrent_prefill_artifact(path)

    def test_failure_count_must_be_zero(self) -> None:
        artifact = valid_artifact()
        artifact["rows"][1] = row(
            concurrent_requests=4,
            request_ttft_ms=1800.0,
            total_wall_ms=2400.0,
            peak_memory_gb=42.0,
            ratios={
                "request_ttft_ms": 2.0,
                "total_wall_ms": 2400.0 / 950.0,
                "peak_memory_gb": 1.4,
            },
            failure_count=1.0,
        )
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.ConcurrentPrefillArtifactError, "failure_count"):
            checker.validate_mlx_concurrent_prefill_artifact(path)

    def test_missing_scheduler_evidence_fails(self) -> None:
        artifact = valid_artifact()
        del artifact["rows"][1]["scheduler_evidence"]
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.ConcurrentPrefillArtifactError, "scheduler_evidence"):
            checker.validate_mlx_concurrent_prefill_artifact(path)

    def test_missing_scheduler_evidence_can_validate_legacy_boundary(self) -> None:
        artifact = valid_artifact()
        del artifact["rows"][1]["scheduler_evidence"]
        path = self.write_fixture(artifact)

        checked = checker.validate_mlx_concurrent_prefill_artifact(
            path,
            require_scheduler_evidence=False,
        )

        self.assertEqual(checked, ["concurrency=1", "concurrency=4"])

    def test_stale_ratio_fails(self) -> None:
        artifact = valid_artifact()
        artifact["rows"][1]["ratios_to_single_request"] = {
            "request_ttft_ms": 1.0,
            "total_wall_ms": 2400.0 / 950.0,
            "peak_memory_gb": 1.4,
        }
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.ConcurrentPrefillArtifactError, "stale request_ttft_ms"):
            checker.validate_mlx_concurrent_prefill_artifact(path)

    def test_cli_reports_concurrency_levels(self) -> None:
        path = self.write_fixture(valid_artifact())

        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("ax.mlx_concurrent_prefill.v1", completed.stdout)
        self.assertIn("concurrency=4", completed.stdout)


if __name__ == "__main__":
    unittest.main()
