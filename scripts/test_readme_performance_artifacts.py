#!/usr/bin/env python3
"""Unit tests for README performance artifact provenance checks."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_readme_performance_artifacts.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_readme_performance_artifacts", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def metric(value: float) -> dict[str, float]:
    return {"median": value, "mean": value, "min": value, "max": value}


class ReadmePerformanceArtifactTests(unittest.TestCase):
    def write_fixture(self, root: Path, *, stale_readme_value: bool = False) -> None:
        artifact_dir = root / "benchmarks/results/mlx-inference/local"
        prompt_dir = artifact_dir / "gemma-4-e2b-it-4bit-prompts"
        prompt_dir.mkdir(parents=True)
        tokens = [1, 2, 3, 4]
        prompt_hash = checker.token_sha256(tokens)
        prompt_path = prompt_dir / f"prompt-4-gen-2-{prompt_hash[:12]}.json"
        prompt_path.write_text(
            json.dumps(
                {
                    "schema_version": "ax.mlx_reference_prompt.v1",
                    "source": "mlx_lm.benchmark",
                    "random_seed": 0,
                    "prompt_distribution": "mx.random.randint(0, vocab_size, (1, prompt_tokens))",
                    "vocab_size": 10,
                    "prompt_tokens": 4,
                    "generation_tokens": 2,
                    "sha256": prompt_hash,
                    "token_ids": tokens,
                },
                indent=2,
            )
            + "\n"
        )

        def row(
            engine: str,
            prefill: float,
            decode: float,
            *,
            method: str = "server_sse_runner_time_us",
        ) -> dict[str, object]:
            payload: dict[str, object] = {
                "engine": engine,
                "method": method,
                "batch_size": 1,
                "prefill_step_size": 2048,
                "prompt_tokens": 4,
                "generation_tokens": 2,
                "prompt_token_ids_sha256": prompt_hash,
                "prefill_tok_s": metric(prefill),
                "decode_tok_s": metric(decode),
                "trials": [{}, {}, {}],
            }
            if engine == "mlx_lm":
                payload["method"] = "mlx_lm.benchmark"
                payload["baseline"] = {
                    "engine": "mlx_lm",
                    "method": "mlx_lm.benchmark",
                    "role": "primary_reference",
                }
            elif engine == "mlx_swift_lm":
                payload["method"] = "mlx_swift_lm_benchmark_adapter"
                payload["secondary_reference_role"] = (
                    "mlx-swift-lm BenchmarkHelpers/MLXLMCommon generation adapter"
                )
            elif engine == "ax_engine_mlx":
                payload["ax_decode_policy"] = "direct_no_ngram_acceleration"
                payload["ax_decode_claim_status"] = "direct_same_policy_baseline"
            elif engine == "ax_engine_mlx_ngram_accel":
                payload["ax_decode_policy"] = "ngram_acceleration_kv_trim"
                payload["ax_decode_claim_status"] = "ngram_acceleration_effective_throughput"
                payload["ngram_acceleration_telemetry"] = {
                    "ax_ngram_draft_attempts": 1,
                    "ax_ngram_draft_tokens": 2,
                    "ax_ngram_accepted_tokens": 2,
                }
            return payload

        artifact = {
            "schema_version": "ax.mlx_inference_stack.v2",
            "reference_contract": {
                "prompt_contract": {
                    "artifacts": [
                        {
                            "prompt_tokens": 4,
                            "generation_tokens": 2,
                            "token_ids_path": str(prompt_path.relative_to(root)),
                            "token_ids_sha256": prompt_hash,
                        }
                    ]
                }
            },
            "prefill_step_size": 2048,
            "repetitions": 3,
            "results": [
                row("mlx_lm", 100.0, 10.0),
                row("mlx_swift_lm", 90.0, 9.0),
                row("ax_engine_mlx", 80.0, 8.0),
                row("ax_engine_mlx_ngram_accel", 82.0, 12.0),
            ],
        }
        (artifact_dir / "gemma-4-e2b-it-4bit.json").write_text(
            json.dumps(artifact, indent=2) + "\n"
        )

        direct_decode = "8.7" if stale_readme_value else "8.0"
        (root / "README.md").write_text(
            "\n".join(
                [
                    "# Test",
                    "`benchmarks/results/mlx-inference/local/`",
                    "### Decode throughput (tok/s) - generation=2 tokens, temp=0",
                    "| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax direct baseline | ax default n-gram |",
                    "|---|---|---:|---:|---:|---:|---:|",
                    f"| Gemma 4 E2B | 4-bit · group=64 · affine | 4 | 10.0 | 9.0 (-10.0%) | {direct_decode} (-20.0%) | **12.0 (+20.0%)** |",
                    "### Prefill throughput (tok/s) - percentages vs mlx_lm",
                    "| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |",
                    "|---|---|---:|---:|---:|---:|",
                    "| Gemma 4 E2B | 4-bit · group=64 · affine | 4 | 100.0 | 90.0 (-10.0%) | 80.0 (-20.0%) |",
                    "",
                ]
            )
        )

    def test_readme_metrics_match_artifact_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=7,
            )

        self.assertEqual(len(checked), 7)

    def test_stale_readme_metric_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root, stale_readme_value=True)

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "README value mismatch",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=7,
                )


if __name__ == "__main__":
    unittest.main()
