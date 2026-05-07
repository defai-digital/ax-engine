#!/usr/bin/env python3
"""Unit tests for cold-vs-warm startup latency artifact checks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_mlx_startup_latency_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mlx_startup_latency_artifact", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


PROMPT_HASH = "c" * 64


def metric(median: float, *, max_value: float | None = None) -> dict[str, float]:
    return {
        "median": median,
        "p75": median * 1.1,
        "min": median * 0.9,
        "max": max_value if max_value is not None else median * 1.2,
    }


def row(
    *,
    phase: str,
    ttft_ms: float,
    decode_tok_s: float,
    ratios: dict[str, float] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
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
        "peak_memory_gb": metric(32.0, max_value=34.0),
    }
    if phase == "process_cold":
        payload["process_start_ms"] = metric(500.0)
        payload["server_ready_ms"] = metric(2100.0)
        payload["model_load_ms"] = metric(1600.0)
        payload["first_request_ttft_ms"] = metric(ttft_ms)
    elif phase == "model_warm":
        payload["model_load_ms"] = metric(900.0)

    if ratios is not None:
        payload["ratios_to_benchmark_warm"] = ratios
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
            "context_tokens": 2048,
            "generation_tokens": 128,
            "prompt_token_ids_sha256": PROMPT_HASH,
            "repetitions": 3,
        },
        "claim_scope": "cold_warm_startup_latency",
        "rows": [
            row(
                phase="process_cold",
                ttft_ms=1800.0,
                decode_tok_s=80.0,
                ratios={"ttft_ms": 3.0, "decode_tok_s": 0.8},
            ),
            row(
                phase="model_warm",
                ttft_ms=900.0,
                decode_tok_s=95.0,
                ratios={"ttft_ms": 1.5, "decode_tok_s": 0.95},
            ),
            row(phase="benchmark_warm", ttft_ms=600.0, decode_tok_s=100.0),
        ],
    }


class StartupLatencyArtifactTests(unittest.TestCase):
    def write_fixture(self, artifact: dict[str, object]) -> Path:
        root = Path(tempfile.mkdtemp())
        path = root / "startup-latency.json"
        path.write_text(json.dumps(artifact, indent=2) + "\n")
        self.addCleanup(lambda: root.rmdir())
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_valid_artifact_passes(self) -> None:
        path = self.write_fixture(valid_artifact())

        checked = checker.validate_mlx_startup_latency_artifact(path)

        self.assertEqual(
            checked,
            ["phase=process_cold", "phase=model_warm", "phase=benchmark_warm"],
        )

    def test_missing_phase_fails(self) -> None:
        artifact = valid_artifact()
        artifact["rows"] = [
            payload for payload in artifact["rows"] if payload["phase"] != "model_warm"
        ]
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.StartupLatencyArtifactError, "lacks required"):
            checker.validate_mlx_startup_latency_artifact(path)

    def test_stale_ratio_fails(self) -> None:
        artifact = valid_artifact()
        artifact["rows"][0]["ratios_to_benchmark_warm"] = {
            "ttft_ms": 1.0,
            "decode_tok_s": 0.8,
        }
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.StartupLatencyArtifactError, "stale ttft_ms ratio"):
            checker.validate_mlx_startup_latency_artifact(path)

    def test_mismatched_prompt_hash_fails(self) -> None:
        artifact = valid_artifact()
        artifact["rows"][1]["prompt_token_ids_sha256"] = "d" * 64
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.StartupLatencyArtifactError, "prompt hash"):
            checker.validate_mlx_startup_latency_artifact(path)

    def test_benchmark_warm_cannot_carry_startup_metrics(self) -> None:
        artifact = valid_artifact()
        artifact["rows"][2]["model_load_ms"] = metric(100.0)
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.StartupLatencyArtifactError, "benchmark_warm"):
            checker.validate_mlx_startup_latency_artifact(path)

    def test_cli_reports_validated_phases(self) -> None:
        path = self.write_fixture(valid_artifact())

        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("ax.mlx_startup_latency.v1", completed.stdout)
        self.assertIn("phase=process_cold", completed.stdout)


if __name__ == "__main__":
    unittest.main()
