#!/usr/bin/env python3
"""Unit tests for long-context comparison artifact tooling."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = load_module(
    "check_long_context_comparison_artifact",
    "check_long_context_comparison_artifact.py",
)
builder = load_module(
    "build_long_context_comparison_artifact",
    "build_long_context_comparison_artifact.py",
)
renderer = load_module(
    "render_long_context_comparison_report",
    "render_long_context_comparison_report.py",
)


HASH_1K = "a" * 64
HASH_8K = "b" * 64


def metric(median: float, *, max_value: float | None = None) -> dict[str, float]:
    return {
        "mean": median,
        "median": median,
        "p75": median * 1.1,
        "min": median * 0.9,
        "max": max_value if max_value is not None else median * 1.2,
    }


def comparison_row(
    *,
    engine: str,
    context_tokens: int,
    prompt_hash: str,
    prefill_tok_s: float,
    ttft_ms: float,
    ratio_prefill: float | None = None,
    ratio_ttft: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "engine": engine,
        "context_tokens": context_tokens,
        "generation_tokens": 1,
        "repetitions": 3,
        "prompt_token_ids_sha256": prompt_hash,
        "prefill_tok_s": metric(prefill_tok_s),
        "ttft_ms": metric(ttft_ms),
    }
    if engine == "mlx_lm":
        row["baseline"] = {"role": "primary_reference", "method": "mlx_lm.benchmark"}
        row["prompt_contract"] = "mlx_lm_random_tokens_seed_0"
    elif engine == "ax_engine_mlx":
        row["ax_decode_policy"] = "direct_no_ngram_acceleration"
        row["route"] = {"selected_backend": "mlx"}
        row["ratios_to_mlx_lm"] = {
            "prefill_tok_s": ratio_prefill if ratio_prefill is not None else 1.0,
            "ttft_ms": ratio_ttft if ratio_ttft is not None else 1.0,
        }
        row["prompt_contract"] = "mlx_lm_random_tokens_seed_0"
    elif engine == "llama_cpp_metal":
        row["prompt_contract"] = "shape_compatible_llama_bench_internal_tokens"
        row["claim_boundary"] = "Shape-compatible external GGUF baseline, not prompt-hash parity evidence."
        row["ratios_to_mlx_lm"] = {
            "prefill_tok_s": ratio_prefill if ratio_prefill is not None else 1.0,
            "ttft_ms": ratio_ttft if ratio_ttft is not None else 1.0,
        }
    return row


def valid_artifact() -> dict[str, object]:
    return {
        "schema_version": checker.SCHEMA_VERSION,
        "claim_scope": "long_context_cold_prefill_comparison",
        "model": {"id": "qwen3.5"},
        "host": {"chip": "Apple M5 Max", "memory_gb": 128},
        "benchmark": {"batch_size": 1, "temperature": 0.0, "prefill_step_size": 2048, "repetitions": 3},
        "rows": [
            comparison_row(
                engine="mlx_lm",
                context_tokens=1024,
                prompt_hash=HASH_1K,
                prefill_tok_s=3000.0,
                ttft_ms=340.0,
            ),
            comparison_row(
                engine="ax_engine_mlx",
                context_tokens=1024,
                prompt_hash=HASH_1K,
                prefill_tok_s=3300.0,
                ttft_ms=306.0,
                ratio_prefill=1.1,
                ratio_ttft=0.9,
            ),
            comparison_row(
                engine="llama_cpp_metal",
                context_tokens=1024,
                prompt_hash=HASH_1K,
                prefill_tok_s=3600.0,
                ttft_ms=255.0,
                ratio_prefill=1.2,
                ratio_ttft=0.75,
            ),
            comparison_row(
                engine="mlx_lm",
                context_tokens=8192,
                prompt_hash=HASH_8K,
                prefill_tok_s=2000.0,
                ttft_ms=4096.0,
            ),
            comparison_row(
                engine="ax_engine_mlx",
                context_tokens=8192,
                prompt_hash=HASH_8K,
                prefill_tok_s=1600.0,
                ttft_ms=5120.0,
                ratio_prefill=0.8,
                ratio_ttft=1.25,
            ),
            comparison_row(
                engine="llama_cpp_metal",
                context_tokens=8192,
                prompt_hash=HASH_8K,
                prefill_tok_s=2400.0,
                ttft_ms=3413.3333333333335,
                ratio_prefill=1.2,
                ratio_ttft=3413.3333333333335 / 4096.0,
            ),
        ],
    }


def source_row(
    *,
    engine: str,
    prompt_tokens: int,
    prompt_hash: str,
    prefill_tok_s: float,
    ttft_ms: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "engine": engine,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": 1,
        "prompt_token_ids_sha256": prompt_hash,
        "repetitions": 3,
        "prefill_tok_s": metric(prefill_tok_s),
        "ttft_ms": metric(ttft_ms),
        "timing_scope": "test",
    }
    if engine == "ax_engine_mlx":
        row["ax_decode_policy"] = "direct_no_ngram_acceleration"
        row["route"] = {"selected_backend": "mlx"}
        row["peak_memory_gb"] = metric(10.0)
    if engine == "llama_cpp_metal":
        row["prompt_contract"] = "shape_compatible_llama_bench_internal_tokens"
        row["claim_boundary"] = "Shape-compatible external GGUF baseline, not prompt-hash parity evidence."
    return row


def source_artifact() -> dict[str, object]:
    return {
        "schema_version": "ax.mlx_inference_stack.v2",
        "model": "qwen3.5",
        "model_dir": ".internal/models/qwen",
        "model_config": {"model_family": "qwen3_5"},
        "host": {"chip": "Apple M5 Max", "memory_gb": 128},
        "prefill_step_size": 2048,
        "repetitions": 3,
        "results": [
            source_row(engine="mlx_lm", prompt_tokens=1024, prompt_hash=HASH_1K, prefill_tok_s=3000.0, ttft_ms=340.0),
            source_row(engine="ax_engine_mlx", prompt_tokens=1024, prompt_hash=HASH_1K, prefill_tok_s=3300.0, ttft_ms=306.0),
            source_row(engine="llama_cpp_metal", prompt_tokens=1024, prompt_hash=HASH_1K, prefill_tok_s=3600.0, ttft_ms=255.0),
            source_row(engine="mlx_lm", prompt_tokens=8192, prompt_hash=HASH_8K, prefill_tok_s=2000.0, ttft_ms=4096.0),
            source_row(engine="ax_engine_mlx", prompt_tokens=8192, prompt_hash=HASH_8K, prefill_tok_s=1600.0, ttft_ms=5120.0),
            source_row(engine="llama_cpp_metal", prompt_tokens=8192, prompt_hash=HASH_8K, prefill_tok_s=2400.0, ttft_ms=3413.3333333333335),
        ],
    }


class LongContextComparisonTests(unittest.TestCase):
    def write_json(self, payload: dict[str, object], name: str = "artifact.json") -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / name
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def test_valid_artifact_passes_with_llama_cpp_required(self) -> None:
        path = self.write_json(valid_artifact())

        checked = checker.validate_long_context_comparison_artifact(
            path,
            require_llama_cpp=True,
        )

        self.assertEqual(
            checked,
            ["context=1024:generation=1", "context=8192:generation=1"],
        )

    def test_prompt_hash_mismatch_between_ax_and_mlx_fails(self) -> None:
        artifact = valid_artifact()
        for row in artifact["rows"]:
            if row["engine"] == "ax_engine_mlx" and row["context_tokens"] == 8192:
                row["prompt_token_ids_sha256"] = "c" * 64
        path = self.write_json(artifact)

        with self.assertRaisesRegex(
            checker.LongContextComparisonArtifactError,
            "one prompt hash",
        ):
            checker.validate_long_context_comparison_artifact(path)

    def test_llama_cpp_boundary_is_required(self) -> None:
        artifact = valid_artifact()
        for row in artifact["rows"]:
            if row["engine"] == "llama_cpp_metal":
                row["claim_boundary"] = "external"
        path = self.write_json(artifact)

        with self.assertRaisesRegex(
            checker.LongContextComparisonArtifactError,
            "claim boundary",
        ):
            checker.validate_long_context_comparison_artifact(path)

    def test_builder_preserves_external_llama_cpp_ratios(self) -> None:
        source_path = self.write_json(source_artifact(), name="source.json")

        artifact = builder.build_long_context_comparison_artifact(source_path)

        path = self.write_json(artifact, name="built.json")
        checker.validate_long_context_comparison_artifact(path, require_llama_cpp=True)
        llama_8k = [
            row
            for row in artifact["rows"]
            if row["engine"] == "llama_cpp_metal" and row["context_tokens"] == 8192
        ][0]
        self.assertAlmostEqual(llama_8k["ratios_to_mlx_lm"]["prefill_tok_s"], 1.2)
        self.assertEqual(
            artifact["comparison_contract"]["shape_compatible_external_engines"],
            ["llama_cpp_metal"],
        )

    def test_renderer_outputs_external_column(self) -> None:
        path = self.write_json(valid_artifact())

        report = renderer.render_report(path, require_llama_cpp=True)

        self.assertIn("# Long-Context Comparison Report", report)
        self.assertIn("llama.cpp prefill tok/s", report)
        self.assertIn("| 8,192 | 1 | 2,000.0 | 1,600.0 | 0.800x | 2,400.0 | 1.200x |", report)
        self.assertIn("not prompt-hash parity", report)

    def test_renderer_cli_writes_report(self) -> None:
        path = self.write_json(valid_artifact())
        output = path.with_suffix(".md")

        exit_code = renderer.main_with_args_for_test(
            [str(path), "--output", str(output), "--require-llama-cpp"]
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("Long-Context Comparison Report", output.read_text())


if __name__ == "__main__":
    unittest.main()
