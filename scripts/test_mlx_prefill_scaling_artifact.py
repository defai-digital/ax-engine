#!/usr/bin/env python3
"""Unit tests for long-context prefill scaling artifact checks."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_mlx_prefill_scaling_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mlx_prefill_scaling_artifact", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


HASH_1K = "a" * 64
HASH_8K = "b" * 64


def metric(median: float, *, max_value: float | None = None) -> dict[str, float]:
    return {
        "median": median,
        "p75": median * 1.1,
        "min": median * 0.9,
        "max": max_value if max_value is not None else median * 1.2,
    }


def row(
    *,
    engine: str,
    context_tokens: int,
    prompt_hash: str,
    prefill_tok_s: float,
    ttft_ms: float,
    memory_gb: float,
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
        "peak_memory_gb": metric(memory_gb, max_value=memory_gb),
    }
    if engine == "mlx_lm":
        payload["baseline"] = {"role": "primary_reference", "method": "mlx_lm.benchmark"}
    elif engine == "ax_engine_mlx":
        payload["ax_decode_policy"] = "direct_no_ngram_acceleration"
        payload["route"] = {"selected_backend": "mlx"}
        payload["ratios_to_mlx_lm"] = ratios or {
            "prefill_tok_s": prefill_tok_s / (prefill_tok_s / 1.1),
            "ttft_ms": ttft_ms / (ttft_ms / 0.9),
        }
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
            "prefill_step_size": 2048,
            "repetitions": 3,
        },
        "claim_scope": "long_context_prefill_ttft_scaling",
        "rows": [
            row(
                engine="mlx_lm",
                context_tokens=1024,
                prompt_hash=HASH_1K,
                prefill_tok_s=3000.0,
                ttft_ms=350.0,
                memory_gb=20.0,
            ),
            row(
                engine="ax_engine_mlx",
                context_tokens=1024,
                prompt_hash=HASH_1K,
                prefill_tok_s=3300.0,
                ttft_ms=315.0,
                memory_gb=19.5,
                ratios={"prefill_tok_s": 1.1, "ttft_ms": 0.9},
            ),
            row(
                engine="mlx_lm",
                context_tokens=8192,
                prompt_hash=HASH_8K,
                prefill_tok_s=2100.0,
                ttft_ms=2400.0,
                memory_gb=36.0,
            ),
            row(
                engine="ax_engine_mlx",
                context_tokens=8192,
                prompt_hash=HASH_8K,
                prefill_tok_s=2520.0,
                ttft_ms=2160.0,
                memory_gb=34.0,
                ratios={"prefill_tok_s": 1.2, "ttft_ms": 0.9},
            ),
        ],
    }


class PrefillScalingArtifactTests(unittest.TestCase):
    def write_fixture(self, artifact: dict[str, object]) -> Path:
        root = Path(tempfile.mkdtemp())
        path = root / "prefill-scaling.json"
        path.write_text(json.dumps(artifact, indent=2) + "\n")
        self.addCleanup(lambda: root.rmdir())
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_valid_artifact_passes(self) -> None:
        path = self.write_fixture(valid_artifact())

        checked = checker.validate_prefill_scaling_artifact(path)

        self.assertEqual(
            checked,
            ["context=1024:generation=1", "context=8192:generation=1"],
        )

    def test_missing_mlx_lm_baseline_fails(self) -> None:
        artifact = valid_artifact()
        artifact["rows"] = [
            row for row in artifact["rows"] if not (
                row["engine"] == "mlx_lm" and row["context_tokens"] == 8192
            )
        ]
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.PrefillScalingArtifactError, "lacks required engines"):
            checker.validate_prefill_scaling_artifact(path)

    def test_stale_ratio_fails(self) -> None:
        artifact = valid_artifact()
        for payload in artifact["rows"]:
            if payload["engine"] == "ax_engine_mlx" and payload["context_tokens"] == 8192:
                payload["ratios_to_mlx_lm"] = {"prefill_tok_s": 1.0, "ttft_ms": 0.9}
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.PrefillScalingArtifactError, "stale prefill_tok_s ratio"):
            checker.validate_prefill_scaling_artifact(path)

    def test_insufficient_context_coverage_fails(self) -> None:
        artifact = valid_artifact()
        artifact["rows"] = [
            row for row in artifact["rows"] if int(row["context_tokens"]) < 8192
        ]
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.PrefillScalingArtifactError, "context points"):
            checker.validate_prefill_scaling_artifact(path)

    def test_invalid_prompt_hash_fails(self) -> None:
        artifact = valid_artifact()
        artifact["rows"][0]["prompt_token_ids_sha256"] = "not-a-hash"
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.PrefillScalingArtifactError, "invalid prompt"):
            checker.validate_prefill_scaling_artifact(path)


if __name__ == "__main__":
    unittest.main()
