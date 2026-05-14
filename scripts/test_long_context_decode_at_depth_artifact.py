#!/usr/bin/env python3
"""Unit tests for long-context decode-at-depth artifact tooling."""

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
    "check_long_context_decode_at_depth_artifact",
    "check_long_context_decode_at_depth_artifact.py",
)
builder = load_module(
    "build_long_context_decode_at_depth_artifact",
    "build_long_context_decode_at_depth_artifact.py",
)
renderer = load_module(
    "render_long_context_decode_at_depth_report",
    "render_long_context_decode_at_depth_report.py",
)


HASH_1K = "a" * 64
HASH_8K = "b" * 64


def metric(median: float) -> dict[str, float]:
    return {
        "mean": median,
        "median": median,
        "p75": median * 1.1,
        "min": median * 0.9,
        "max": median * 1.2,
    }


def decode_row(
    *,
    engine: str,
    context_depth_tokens: int,
    prompt_hash: str,
    decode_tok_s: float,
    ratio_decode: float | None = None,
    llama_depth_contract: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "engine": engine,
        "context_depth_tokens": context_depth_tokens,
        "generation_tokens": 128,
        "repetitions": 3,
        "prompt_token_ids_sha256": prompt_hash,
        "decode_tok_s": metric(decode_tok_s),
    }
    if engine == "mlx_lm":
        row["baseline"] = {"role": "primary_reference", "method": "mlx_lm.benchmark"}
        row["depth_contract"] = "generation_after_prompt_hash_parity_prefill"
    elif engine == "ax_engine_mlx":
        row["ax_decode_policy"] = "direct_no_ngram_acceleration"
        row["route"] = {"selected_backend": "mlx"}
        row["ratios_to_mlx_lm"] = {
            "decode_tok_s": ratio_decode if ratio_decode is not None else 1.0,
        }
        row["depth_contract"] = "generation_after_prompt_hash_parity_prefill"
    elif engine == "llama_cpp_metal":
        if llama_depth_contract:
            row["depth_contract"] = checker.LLAMA_CPP_DEPTH_CONTRACT
        row["claim_boundary"] = (
            "Shape-compatible external GGUF decode-depth baseline, "
            "not prompt-hash parity evidence."
        )
        row["ratios_to_mlx_lm"] = {
            "decode_tok_s": ratio_decode if ratio_decode is not None else 1.0,
        }
    return row


def valid_artifact() -> dict[str, object]:
    return {
        "schema_version": checker.SCHEMA_VERSION,
        "claim_scope": "long_context_decode_at_existing_depth",
        "model": {"id": "qwen3.5"},
        "host": {"chip": "Apple M5 Max", "memory_gb": 128},
        "benchmark": {"batch_size": 1, "temperature": 0.0, "repetitions": 3},
        "rows": [
            decode_row(
                engine="mlx_lm",
                context_depth_tokens=1024,
                prompt_hash=HASH_1K,
                decode_tok_s=120.0,
            ),
            decode_row(
                engine="ax_engine_mlx",
                context_depth_tokens=1024,
                prompt_hash=HASH_1K,
                decode_tok_s=132.0,
                ratio_decode=1.1,
            ),
            decode_row(
                engine="llama_cpp_metal",
                context_depth_tokens=1024,
                prompt_hash=HASH_1K,
                decode_tok_s=144.0,
                ratio_decode=1.2,
            ),
            decode_row(
                engine="mlx_lm",
                context_depth_tokens=8192,
                prompt_hash=HASH_8K,
                decode_tok_s=80.0,
            ),
            decode_row(
                engine="ax_engine_mlx",
                context_depth_tokens=8192,
                prompt_hash=HASH_8K,
                decode_tok_s=72.0,
                ratio_decode=0.9,
            ),
            decode_row(
                engine="llama_cpp_metal",
                context_depth_tokens=8192,
                prompt_hash=HASH_8K,
                decode_tok_s=96.0,
                ratio_decode=1.2,
            ),
        ],
    }


def source_row(
    *,
    engine: str,
    prompt_tokens: int,
    prompt_hash: str,
    decode_tok_s: float,
    llama_depth_contract: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "engine": engine,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": 128,
        "prompt_token_ids_sha256": prompt_hash,
        "repetitions": 3,
        "decode_tok_s": metric(decode_tok_s),
        "timing_scope": "test",
    }
    if engine == "ax_engine_mlx":
        row["ax_decode_policy"] = "direct_no_ngram_acceleration"
        row["route"] = {"selected_backend": "mlx"}
    if engine == "llama_cpp_metal" and llama_depth_contract:
        row["decode_at_depth_contract"] = checker.LLAMA_CPP_DEPTH_CONTRACT
        row["decode_at_depth_tok_s"] = metric(decode_tok_s)
        row["llama_cpp_depth"] = {"n_depth": prompt_tokens, "backends": "Metal"}
        row["claim_boundary"] = (
            "Shape-compatible external GGUF decode-depth baseline, "
            "not prompt-hash parity evidence."
        )
    return row


def source_artifact(*, llama_depth_contract: bool) -> dict[str, object]:
    return {
        "schema_version": "ax.mlx_inference_stack.v2",
        "model": "qwen3.5",
        "model_dir": ".internal/models/qwen",
        "model_config": {"model_family": "qwen3_5"},
        "host": {"chip": "Apple M5 Max", "memory_gb": 128},
        "repetitions": 3,
        "results": [
            source_row(engine="mlx_lm", prompt_tokens=1024, prompt_hash=HASH_1K, decode_tok_s=120.0),
            source_row(engine="ax_engine_mlx", prompt_tokens=1024, prompt_hash=HASH_1K, decode_tok_s=132.0),
            source_row(
                engine="llama_cpp_metal",
                prompt_tokens=1024,
                prompt_hash=HASH_1K,
                decode_tok_s=144.0,
                llama_depth_contract=llama_depth_contract,
            ),
            source_row(engine="mlx_lm", prompt_tokens=8192, prompt_hash=HASH_8K, decode_tok_s=80.0),
            source_row(engine="ax_engine_mlx", prompt_tokens=8192, prompt_hash=HASH_8K, decode_tok_s=72.0),
            source_row(
                engine="llama_cpp_metal",
                prompt_tokens=8192,
                prompt_hash=HASH_8K,
                decode_tok_s=96.0,
                llama_depth_contract=llama_depth_contract,
            ),
        ],
    }


class LongContextDecodeAtDepthTests(unittest.TestCase):
    def write_json(self, payload: dict[str, object], name: str = "artifact.json") -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / name
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def test_valid_artifact_passes_with_llama_cpp_required(self) -> None:
        path = self.write_json(valid_artifact())

        checked = checker.validate_long_context_decode_at_depth_artifact(
            path,
            require_llama_cpp=True,
        )

        self.assertEqual(
            checked,
            ["depth=1024:generation=128", "depth=8192:generation=128"],
        )

    def test_prompt_hash_mismatch_between_ax_and_mlx_fails(self) -> None:
        artifact = valid_artifact()
        for row in artifact["rows"]:
            if row["engine"] == "ax_engine_mlx" and row["context_depth_tokens"] == 8192:
                row["prompt_token_ids_sha256"] = "c" * 64
        path = self.write_json(artifact)

        with self.assertRaisesRegex(
            checker.LongContextDecodeAtDepthArtifactError,
            "one prompt hash",
        ):
            checker.validate_long_context_decode_at_depth_artifact(path)

    def test_llama_cpp_depth_contract_is_required(self) -> None:
        artifact = valid_artifact()
        for row in artifact["rows"]:
            if row["engine"] == "llama_cpp_metal":
                row.pop("depth_contract")
        path = self.write_json(artifact)

        with self.assertRaisesRegex(
            checker.LongContextDecodeAtDepthArtifactError,
            "llama_bench_n_depth",
        ):
            checker.validate_long_context_decode_at_depth_artifact(path)

    def test_builder_skips_shape_only_llama_cpp_rows(self) -> None:
        source_path = self.write_json(
            source_artifact(llama_depth_contract=False),
            name="source.json",
        )

        artifact = builder.build_long_context_decode_at_depth_artifact(source_path)

        path = self.write_json(artifact, name="built.json")
        checker.validate_long_context_decode_at_depth_artifact(path)
        self.assertNotIn("llama_cpp_metal", {row["engine"] for row in artifact["rows"]})

    def test_builder_preserves_llama_cpp_depth_ratios_when_contract_is_present(self) -> None:
        source_path = self.write_json(
            source_artifact(llama_depth_contract=True),
            name="source.json",
        )

        artifact = builder.build_long_context_decode_at_depth_artifact(source_path)

        path = self.write_json(artifact, name="built.json")
        checker.validate_long_context_decode_at_depth_artifact(path, require_llama_cpp=True)
        llama_8k = [
            row
            for row in artifact["rows"]
            if row["engine"] == "llama_cpp_metal" and row["context_depth_tokens"] == 8192
        ][0]
        self.assertAlmostEqual(llama_8k["ratios_to_mlx_lm"]["decode_tok_s"], 1.2)
        self.assertEqual(llama_8k["llama_cpp_depth"]["n_depth"], 8192)

    def test_renderer_outputs_external_column(self) -> None:
        path = self.write_json(valid_artifact())

        report = renderer.render_report(path, require_llama_cpp=True)

        self.assertIn("# Long-Context Decode-at-Depth Report", report)
        self.assertIn("llama.cpp decode tok/s", report)
        self.assertIn("| 8,192 | 128 | 80.0 | 72.0 | 0.900x | 96.0 | 1.200x |", report)
        self.assertIn("llama-bench n_depth", report)

    def test_renderer_cli_writes_report(self) -> None:
        path = self.write_json(valid_artifact())
        output = path.with_suffix(".md")

        exit_code = renderer.main_with_args_for_test(
            [str(path), "--output", str(output), "--require-llama-cpp"]
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("Long-Context Decode-at-Depth Report", output.read_text())


if __name__ == "__main__":
    unittest.main()
