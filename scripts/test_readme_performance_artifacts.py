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


def ngram_telemetry(*, attempts: int, accepted: int, fallback_steps: int = 0) -> dict[str, int]:
    return {
        "ax_ngram_draft_attempts": attempts,
        "ax_ngram_draft_tokens": accepted,
        "ax_ngram_accepted_tokens": accepted,
        "ax_ngram_rejected_tokens": 0,
        "ax_ngram_full_accepts": attempts if accepted else 0,
        "ax_ngram_partial_rejects": 0,
        "ax_ngram_complete_misses": 0,
        "ax_ngram_no_draft_steps": fallback_steps,
        "ax_ngram_cooldown_steps": 0,
        "ax_ngram_cooldown_events": 0,
        "ax_ngram_cooldown_steps_scheduled": 0,
        "ax_ngram_request_disable_events": 0,
        "ax_ngram_request_disabled_steps": 0,
        "ax_ngram_fallback_no_candidate_steps": fallback_steps,
        "ax_ngram_fallback_confidence_filtered_steps": 0,
        "ax_ngram_fallback_short_output_steps": 0,
        "ax_ngram_fallback_linear_no_draft_steps": 0,
        "ax_ngram_policy_variant": 1,
        "ax_ngram_adaptive_draft_len_steps": attempts,
        "ax_ngram_adaptive_draft_len_total": accepted,
    }


def ax_mlx_telemetry() -> dict[str, int]:
    return {
        "ax_mlx_prefill_steps": 1,
        "ax_mlx_decode_steps": 2,
    }


def write_hot_prefix_artifact(
    root: Path,
    *,
    warmup_tokens: int = 0,
    miss_count: int = 0,
    blocked_count: int = 0,
    tokens_match: bool = True,
) -> Path:
    artifact_path = (
        root
        / "benchmarks/results/mlx-inference/local-hot-prefix/qwen3-5-9b.json"
    )
    artifact_path.parent.mkdir(parents=True)
    per_prompt = []
    for index, reused_tokens in enumerate([16, 32, 48, 32, 48], start=1):
        per_prompt.append(
            {
                "id": f"p{index}",
                "tokens_match": tokens_match,
                "warm_telemetry": {
                    "ax_mlx_prefix_cache_hits": 1,
                    "ax_mlx_prefix_cache_reused_tokens": reused_tokens,
                    "ax_mlx_prefix_cache_warmup_tokens": warmup_tokens,
                    "ax_mlx_prefix_cache_misses": miss_count,
                    "ax_mlx_prefix_cache_blocked": blocked_count,
                },
            }
        )
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": checker.PREFIX_REUSE_EQUIVALENCE_SCHEMA_VERSION,
                "config": {"mode": "warm_repeat"},
                "aggregate": {
                    "prompts_matching_exactly": 5 if tokens_match else 4,
                    "prompts_total": 5,
                    "verdict": "PASS",
                },
                "per_prompt": per_prompt,
            },
            indent=2,
        )
        + "\n"
    )
    return artifact_path


def add_hot_prefix_readme_claim(root: Path, artifact_path: Path, text: str | None = None) -> None:
    readme_path = root / "README.md"
    relative_path = artifact_path.relative_to(root)
    claim_text = text or (
        "Hot-prefix physical reuse restored physical prefix snapshots on "
        "5/5 prompts, reused 176 tokens, and used 0 warmup-substitution tokens."
    )
    readme_path.write_text(
        readme_path.read_text()
        + f"<!-- readme-hot-prefix-artifact: {relative_path} -->\n"
        + claim_text
        + "\n"
    )


def write_prefill_scaling_artifact(root: Path, *, ratio: float = 0.84) -> Path:
    artifact_path = root / "benchmarks/results/mlx-inference/local-p1/prefill-scaling.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": checker.PREFILL_SCALING_SCHEMA_VERSION,
                "rows": [
                    {
                        "engine": "ax_engine_mlx",
                        "context_tokens": 8192,
                        "ratios_to_mlx_lm": {"prefill_tok_s": ratio},
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return artifact_path


def write_concurrent_prefill_artifact(
    root: Path, *, classification: str = "serialized"
) -> Path:
    artifact_path = (
        root / "benchmarks/results/mlx-inference/local-p2/concurrent-prefill.json"
    )
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": checker.CONCURRENT_PREFILL_SCHEMA_VERSION,
                "rows": [
                    {
                        "engine": "ax_engine_mlx",
                        "concurrent_requests": 4,
                        "prefill_overlap": {"classification": classification},
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return artifact_path


def add_boundary_readme_claims(
    root: Path,
    *,
    prefill_artifact_path: Path,
    concurrent_artifact_path: Path,
    prefill_ratio_text: str = "0.840x",
    concurrent_classification: str = "serialized",
) -> None:
    readme_path = root / "README.md"
    prefill_relative_path = prefill_artifact_path.relative_to(root)
    concurrent_relative_path = concurrent_artifact_path.relative_to(root)
    readme_path.write_text(
        readme_path.read_text()
        + f"<!-- readme-long-context-boundary-artifact: {prefill_relative_path} -->\n"
        + f"<!-- readme-concurrent-prefill-boundary-artifact: {concurrent_relative_path} -->\n"
        + f"The 8k P1 AX/MLX prefill ratio was {prefill_ratio_text}, and "
        + "the 4-request P2 concurrent prefill row was classified as "
        + f"{concurrent_classification}.\n"
    )


class ReadmePerformanceArtifactTests(unittest.TestCase):
    def write_fixture(
        self,
        root: Path,
        *,
        stale_readme_value: bool = False,
        stale_readme_percent: bool = False,
    ) -> None:
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
                "ttft_ms": metric(30.0),
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
                payload["timing_scope"] = "ax_engine_runner_time_us"
                payload["runtime_identity"] = {
                    "selected_backend": "mlx",
                    "route_identity": "repo_owned_mlx",
                    "resolution_policy": "mlx_only",
                }
                payload["ttft_source"] = "ax_engine_runner_prefill_time"
                payload["prefill_s"] = metric(0.1)
                payload["decode_s"] = metric(0.2)
                payload["ax_mlx_telemetry"] = ax_mlx_telemetry()
                payload["ax_decode_policy"] = "direct_no_ngram_acceleration"
                payload["ax_decode_claim_status"] = "direct_same_policy_baseline"
                payload["ngram_acceleration_telemetry"] = ngram_telemetry(
                    attempts=0,
                    accepted=0,
                )
            elif engine == "ax_engine_mlx_ngram_accel":
                payload["timing_scope"] = "ax_engine_runner_time_us"
                payload["runtime_identity"] = {
                    "selected_backend": "mlx",
                    "route_identity": "repo_owned_mlx",
                    "resolution_policy": "mlx_only",
                }
                payload["ttft_source"] = "ax_engine_runner_prefill_time"
                payload["prefill_s"] = metric(0.1)
                payload["decode_s"] = metric(0.2)
                payload["ax_mlx_telemetry"] = ax_mlx_telemetry()
                payload["ax_decode_policy"] = "ngram_acceleration_kv_trim"
                payload["ax_decode_claim_status"] = "ngram_acceleration_effective_throughput"
                payload["ngram_acceleration_telemetry"] = ngram_telemetry(
                    attempts=1,
                    accepted=2,
                )
            return payload

        artifact = {
            "schema_version": "ax.mlx_inference_stack.v2",
            "claim_gate": {
                "schema_version": checker.PHASE0_CLAIM_GATE_SCHEMA_VERSION,
                "scope": "mlx_inference_stack_public_readme",
            },
            "concurrent_prefill_overlap_classification": {
                "classification": "single_request_no_overlap",
                "continuous_batching_claim": False,
                "concurrency": 1,
            },
            "prefix_reuse_evidence": {
                "hit_count": 0,
                "miss_count": 0,
                "blocked_count": 0,
                "blocked_policy_disabled_count": 0,
                "blocked_unsupported_layout_count": 0,
                "blocked_trim_failure_count": 0,
                "stored_prefix_count": 0,
                "eviction_count": 0,
                "reused_token_count": 0,
                "warmup_token_count": 0,
                "cache_entry_count": 0,
                "cache_bytes_kib": 0,
                "physical_snapshot_hit_observed": False,
                "physical_snapshot_miss_warmup_observed": False,
                "physical_snapshot_blocked_observed": False,
                "physical_snapshot_coverage": "none_observed",
                "blocked_reason_count": 0,
                "blocked_reason_accounting_gap_count": 0,
            },
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
        direct_delta = "-19.0%" if stale_readme_percent else "-20.0%"
        (root / "README.md").write_text(
            "\n".join(
                [
                    "# Test",
                    "`benchmarks/results/mlx-inference/local/`",
                    "### Decode throughput (tok/s) - generation=2 tokens, temp=0",
                    "| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax direct baseline | ax default n-gram |",
                    "|---|---|---:|---:|---:|---:|---:|",
                    f"| Gemma 4 E2B | 4-bit | 4 | 10.0 | 9.0 (-10.0%) | {direct_decode} ({direct_delta}) | **12.0 (+20.0%)** |",
                    "### Prefill throughput (tok/s) - percentages vs mlx_lm",
                    "| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |",
                    "|---|---|---:|---:|---:|---:|",
                    "| Gemma 4 E2B | 4-bit | 4 | 100.0 | 90.0 (-10.0%) | 80.0 (-20.0%) |",
                    "### Time to first token (ms) - generation=2 tokens, temp=0",
                    "| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |",
                    "|---|---|---:|---:|---:|---:|",
                    "| Gemma 4 E2B | 4-bit | 4 | 40.0 | 44.4 (+11.1%) | **30.0 (-25.0%)** |",
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
                expected_metric_count=10,
            )

        self.assertEqual(len(checked), 10)

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
                    expected_metric_count=10,
                )

    def test_stale_readme_percentage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root, stale_readme_percent=True)

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "README percentage mismatch",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_decode_table_generation_heading_must_match_artifact_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            readme_path = root / "README.md"
            readme_path.write_text(
                readme_path.read_text().replace(
                    "generation=2 tokens",
                    "generation=64 tokens",
                )
            )

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "generation_tokens=64",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=readme_path,
                    expected_metric_count=10,
                )

    def test_phase0_ax_ttft_requires_positive_prefill_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["prefill_s"] = metric(0.0)
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "prefill_s\\.median must be positive",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_phase0_ax_ttft_requires_positive_ttft_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["ttft_ms"] = metric(0.0)
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
            readme_path = root / "README.md"
            readme_path.write_text(
                readme_path.read_text().replace(
                    "| Gemma 4 E2B | 4-bit | 4 | 40.0 | 44.4 (+11.1%) | **30.0 (-25.0%)** |",
                    "| Gemma 4 E2B | 4-bit | 4 | 40.0 | 44.4 (+11.1%) | **0.0 (-100.0%)** |",
                )
            )

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "ttft_ms\\.median must be positive",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=readme_path,
                    expected_metric_count=10,
                )

    def test_phase0_ax_ttft_requires_positive_prefill_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["ax_mlx_telemetry"]["ax_mlx_prefill_steps"] = 0
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "positive ax_mlx_prefill_steps",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_reused_reference_rows_may_have_source_repetition_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["repetitions"] = 5
            artifact["ax_only_refresh"] = {
                "method": "reuse_existing_reference_rows_and_rerun_ax_engine_rows",
                "reference_results_source": "benchmarks/results/mlx-inference/reference/gemma-4-e2b-it-4bit.json",
                "reference_rows_reused": 2,
                "ax_rows_refreshed": 2,
            }
            for row in artifact["results"]:
                if row["engine"] in {"ax_engine_mlx", "ax_engine_mlx_ngram_accel"}:
                    row["trials"] = [{}, {}, {}, {}, {}]
                else:
                    row["trials"] = [{}, {}, {}]
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=10,
            )

        self.assertEqual(len(checked), 10)

    def test_readme_can_validate_reference_and_ax_overlay_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            base_dir = root / "benchmarks/results/mlx-inference/local"
            overlay_dir = root / "benchmarks/results/mlx-inference/ax-overlay"
            overlay_dir.mkdir(parents=True)
            overlay_path = overlay_dir / "gemma-4-e2b-it-4bit.json"
            artifact = json.loads((base_dir / "gemma-4-e2b-it-4bit.json").read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["prefill_tok_s"] = metric(85.0)
                    row["decode_tok_s"] = metric(8.5)
                elif row["engine"] == "ax_engine_mlx_ngram_accel":
                    row["decode_tok_s"] = metric(12.5)
            overlay_path.write_text(json.dumps(artifact, indent=2) + "\n")

            readme_path = root / "README.md"
            readme_path.write_text(
                readme_path.read_text()
                .replace(
                    "`benchmarks/results/mlx-inference/local/`",
                    "<!-- readme-performance-artifacts: "
                    "reference=benchmarks/results/mlx-inference/local/; "
                    "ax-overlay=benchmarks/results/mlx-inference/ax-overlay/ -->\n"
                    "`benchmarks/results/mlx-inference/local/`",
                )
                .replace("8.0 (-20.0%)", "8.5 (-15.0%)")
                .replace("**12.0 (+20.0%)**", "**12.5 (+25.0%)**")
                .replace("80.0 (-20.0%)", "85.0 (-15.0%)")
            )

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=readme_path,
                expected_metric_count=10,
            )

        self.assertEqual(len(checked), 10)

    def test_non_reference_rows_must_match_artifact_repetition_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["repetitions"] = 5
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["trials"] = [{}, {}, {}]
                else:
                    row["trials"] = [{}, {}, {}, {}, {}]
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "ax_engine_mlx prompt=4 lacks repetition trials",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_phase0_artifact_requires_ax_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row.pop("runtime_identity")
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "runtime_identity",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_ngram_effective_claim_requires_draft_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx_ngram_accel":
                    row["ngram_acceleration_telemetry"] = ngram_telemetry(
                        attempts=0,
                        accepted=0,
                        fallback_steps=2,
                    )
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "without draft acceptance",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_ngram_no_accept_fallback_allows_attempts_without_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx_ngram_accel":
                    row["ax_decode_claim_status"] = "ngram_no_accept_fallback"
                    row["ngram_acceleration_telemetry"] = ngram_telemetry(
                        attempts=2,
                        accepted=0,
                        fallback_steps=2,
                    )
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=10,
            )

            self.assertEqual(len(checked), 10)

    def test_public_prefix_reuse_claim_requires_physical_snapshot_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["public_claims"] = ["prefix_reuse"]
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "physical snapshot hit evidence",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_readme_hot_prefix_artifact_accepts_physical_hit_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = write_hot_prefix_artifact(root)
            add_hot_prefix_readme_claim(root, artifact_path)

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=10,
            )

        self.assertEqual(len(checked), 10)

    def test_readme_hot_prefix_artifact_rejects_warmup_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = write_hot_prefix_artifact(root, warmup_tokens=1)
            add_hot_prefix_readme_claim(root, artifact_path)

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "not hit-only physical reuse",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_readme_hot_prefix_claim_rejects_stale_reused_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = write_hot_prefix_artifact(root)
            add_hot_prefix_readme_claim(
                root,
                artifact_path,
                text=(
                    "Hot-prefix physical reuse restored physical prefix snapshots "
                    "on 5/5 prompts, reused 175 tokens, and used 0 warmup tokens."
                ),
            )

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "stale reused token count",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_readme_boundary_artifacts_accept_current_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            prefill_artifact_path = write_prefill_scaling_artifact(root)
            concurrent_artifact_path = write_concurrent_prefill_artifact(root)
            add_boundary_readme_claims(
                root,
                prefill_artifact_path=prefill_artifact_path,
                concurrent_artifact_path=concurrent_artifact_path,
            )

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=10,
            )

        self.assertEqual(len(checked), 10)

    def test_summary_reports_narrative_claim_checks_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            hot_prefix_artifact_path = write_hot_prefix_artifact(root)
            prefill_artifact_path = write_prefill_scaling_artifact(root)
            concurrent_artifact_path = write_concurrent_prefill_artifact(root)
            add_hot_prefix_readme_claim(root, hot_prefix_artifact_path)
            add_boundary_readme_claims(
                root,
                prefill_artifact_path=prefill_artifact_path,
                concurrent_artifact_path=concurrent_artifact_path,
            )

            checked = checker.check_readme_performance_summary(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=10,
            )

        self.assertEqual(len(checked.metric_checks), 10)
        self.assertEqual(len(checked.narrative_claim_checks), 3)
        self.assertIn(
            "hot-prefix:qwen3-5-9b.json:5/5:176",
            checked.narrative_claim_checks,
        )
        self.assertIn(
            "long-context-boundary:prefill-scaling.json:8192:0.840",
            checked.narrative_claim_checks,
        )
        self.assertIn(
            "concurrent-prefill-boundary:concurrent-prefill.json:4:serialized",
            checked.narrative_claim_checks,
        )

    def test_readme_boundary_artifacts_reject_stale_prefill_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            prefill_artifact_path = write_prefill_scaling_artifact(root, ratio=0.913)
            concurrent_artifact_path = write_concurrent_prefill_artifact(root)
            add_boundary_readme_claims(
                root,
                prefill_artifact_path=prefill_artifact_path,
                concurrent_artifact_path=concurrent_artifact_path,
            )

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "long-context boundary claim is stale",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_readme_boundary_artifacts_reject_stale_concurrency_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            prefill_artifact_path = write_prefill_scaling_artifact(root)
            concurrent_artifact_path = write_concurrent_prefill_artifact(
                root,
                classification="partial_overlap",
            )
            add_boundary_readme_claims(
                root,
                prefill_artifact_path=prefill_artifact_path,
                concurrent_artifact_path=concurrent_artifact_path,
            )

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "concurrent-prefill boundary claim is stale",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_phase0_claim_gate_requires_prefix_coverage_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["prefix_reuse_evidence"].pop("physical_snapshot_coverage")
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "invalid physical_snapshot_coverage",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_phase0_claim_gate_rejects_inconsistent_prefix_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["prefix_reuse_evidence"]["physical_snapshot_coverage"] = "hit_only"
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "physical_snapshot_coverage is inconsistent",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_phase0_claim_gate_rejects_inconsistent_blocked_reason_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            evidence = artifact["prefix_reuse_evidence"]
            evidence["blocked_count"] = 3
            evidence["blocked_policy_disabled_count"] = 1
            evidence["blocked_unsupported_layout_count"] = 1
            evidence["physical_snapshot_blocked_observed"] = True
            evidence["physical_snapshot_coverage"] = "blocked_only"
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "blocked_reason_count is inconsistent",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_phase0_claim_gate_rejects_negative_prefix_counter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["prefix_reuse_evidence"]["hit_count"] = -1
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "hit_count must be non-negative",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_public_claim_requires_matching_artifact_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["public_claims"] = ["prefix_reuse"]
            artifact.pop("prefix_reuse_evidence")
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "claims prefix_reuse",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_unknown_public_claim_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["public_claims"] = [{"name": "speculative_superpower"}]
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "unknown public claim",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_public_continuous_batching_claim_requires_positive_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["public_claims"] = ["continuous_batching"]
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "positive overlap evidence",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_public_continuous_batching_claim_accepts_positive_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["public_claims"] = ["continuous_batching"]
            artifact["concurrent_prefill_overlap_classification"] = {
                "classification": "partial_overlap",
                "continuous_batching_claim": True,
                "concurrency": 2,
            }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=10,
            )

        self.assertEqual(len(checked), 10)

    def test_phase0_claim_gate_rejects_invalid_overlap_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["concurrent_prefill_overlap_classification"]["classification"] = "unknown"
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "overlap classification is invalid",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )

    def test_phase0_claim_gate_rejects_inconsistent_overlap_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["concurrent_prefill_overlap_classification"] = {
                "classification": "serialized",
                "continuous_batching_claim": True,
                "concurrency": 4,
            }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "overlap claim is inconsistent",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=10,
                )


if __name__ == "__main__":
    unittest.main()
