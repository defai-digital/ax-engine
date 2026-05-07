#!/usr/bin/env python3
"""Unit tests for building prefill scaling artifacts from MLX inference runs."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "build_mlx_prefill_scaling_artifact.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "build_mlx_prefill_scaling_artifact", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
builder = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = builder
MODULE_SPEC.loader.exec_module(builder)


HASH_1K = "c" * 64
HASH_8K = "d" * 64


def metric(median: float, *, max_value: float | None = None) -> dict[str, float]:
    return {
        "mean": median,
        "median": median,
        "min": median * 0.9,
        "max": max_value if max_value is not None else median * 1.1,
    }


def source_row(
    *,
    engine: str,
    prompt_tokens: int,
    prompt_hash: str,
    prefill_tok_s: float,
    prefill_s: float,
    memory_gb: float,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "engine": engine,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": 1,
        "prompt_token_ids_sha256": prompt_hash,
        "prompt_token_ids_path": f"prompts/prompt-{prompt_tokens}.json",
        "prefill_tok_s": metric(prefill_tok_s),
        "decode_tok_s": metric(1.0),
        "peak_memory_gb": metric(memory_gb, max_value=memory_gb),
        "trials": [{}, {}, {}],
    }
    if engine == "mlx_lm":
        payload["method"] = "mlx_lm.benchmark"
        payload["baseline"] = {"role": "primary_reference"}
    elif engine == "ax_engine_mlx":
        payload["method"] = "server_sse_runner_time_us"
        payload["ax_decode_policy"] = "direct_no_ngram_acceleration"
        payload["ax_decode_claim_status"] = "direct_same_policy_baseline"
        payload["prefill_s"] = metric(prefill_s)
        payload["ttft_source"] = "ax_engine_runner_prefill_time"
        payload["memory_source"] = "server_process_rss_after_stream"
    return payload


def source_artifact() -> dict[str, object]:
    return {
        "schema_version": builder.SOURCE_SCHEMA_VERSION,
        "host": {"chip": "Apple M5 Max", "memory_gb": 128},
        "model": "mlx-community/Qwen3.5-9B-4bit",
        "model_dir": ".internal/models/Qwen3.5-9B-4bit",
        "model_config": {"model_type": "qwen3_5", "quantization": {"bits": 4}},
        "prefill_step_size": 2048,
        "repetitions": 3,
        "results": [
            source_row(
                engine="mlx_lm",
                prompt_tokens=1024,
                prompt_hash=HASH_1K,
                prefill_tok_s=2048.0,
                prefill_s=0.5,
                memory_gb=18.0,
            ),
            source_row(
                engine="ax_engine_mlx",
                prompt_tokens=1024,
                prompt_hash=HASH_1K,
                prefill_tok_s=2560.0,
                prefill_s=0.4,
                memory_gb=19.0,
            ),
            source_row(
                engine="mlx_lm",
                prompt_tokens=8192,
                prompt_hash=HASH_8K,
                prefill_tok_s=2048.0,
                prefill_s=4.0,
                memory_gb=26.0,
            ),
            source_row(
                engine="ax_engine_mlx",
                prompt_tokens=8192,
                prompt_hash=HASH_8K,
                prefill_tok_s=4096.0,
                prefill_s=2.0,
                memory_gb=27.0,
            ),
        ],
    }


class BuildPrefillScalingArtifactTests(unittest.TestCase):
    def write_source(self, payload: dict[str, object]) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "source.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def test_builds_and_validates_scaling_artifact(self) -> None:
        source = self.write_source(source_artifact())

        artifact = builder.build_prefill_scaling_artifact(source)
        rows = artifact["rows"]

        self.assertEqual(artifact["schema_version"], "ax.mlx_prefill_scaling.v1")
        self.assertEqual(len(rows), 4)
        ax_8k = [
            row
            for row in rows
            if row["engine"] == "ax_engine_mlx" and row["context_tokens"] == 8192
        ][0]
        self.assertEqual(ax_8k["ttft_ms"]["median"], 2000.0)
        self.assertEqual(ax_8k["ratios_to_mlx_lm"]["prefill_tok_s"], 2.0)
        self.assertEqual(ax_8k["ratios_to_mlx_lm"]["ttft_ms"], 0.5)

    def test_missing_ax_memory_fails(self) -> None:
        payload = source_artifact()
        for row in payload["results"]:
            if row["engine"] == "ax_engine_mlx" and row["prompt_tokens"] == 8192:
                row.pop("peak_memory_gb")
        source = self.write_source(payload)

        with self.assertRaisesRegex(builder.PrefillScalingBuildError, "peak_memory_gb"):
            builder.build_prefill_scaling_artifact(source)

    def test_min_context_filter_requires_remaining_baseline(self) -> None:
        payload = source_artifact()
        payload["results"] = [
            row
            for row in payload["results"]
            if not (row["engine"] == "mlx_lm" and row["prompt_tokens"] == 8192)
        ]
        source = self.write_source(payload)

        with self.assertRaisesRegex(builder.PrefillScalingBuildError, "missing mlx_lm baseline"):
            builder.build_prefill_scaling_artifact(source)


if __name__ == "__main__":
    unittest.main()
