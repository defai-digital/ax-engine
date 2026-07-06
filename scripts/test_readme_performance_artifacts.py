#!/usr/bin/env python3
"""Unit tests for README performance artifact provenance checks."""

from __future__ import annotations

import copy
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


def ngram_telemetry(
    *,
    attempts: int,
    accepted: int,
    fallback_steps: int = 0,
    mtp_draft_tokens: int = 0,
    mtp_ngram_hit_steps: int = 0,
) -> dict[str, int]:
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
        "ax_mtp_draft_tokens": mtp_draft_tokens,
        "ax_mtp_ngram_hit_steps": mtp_ngram_hit_steps,
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
        + f"{concurrent_classification}. This is a single-model long-context "
        + "boundary, not a Gemma/Qwen/GLM-wide campaign.\n"
    )


class TrackedDirtyAllowlistTests(unittest.TestCase):
    def test_readme_and_box_whisker_svgs_are_doc_only(self) -> None:
        self.assertTrue(checker.is_benchmark_doc_only_path("README.md"))
        self.assertTrue(
            checker.is_benchmark_doc_only_path(
                "docs/assets/perf-gemma4-decode-box-whisker.svg"
            )
        )

    def test_update_readme_post_processing_scripts_are_doc_only(self) -> None:
        for path in (
            "scripts/update_readme_from_bench.py",
            "scripts/update_readme_inject_llama_cpp.py",
            "scripts/update_readme_inject_mlx_lm.py",
            "scripts/update_readme_embedding.py",
        ):
            self.assertTrue(
                checker.is_benchmark_doc_only_path(path),
                f"{path} should be allowlisted as README post-processing",
            )

    def test_test_scripts_are_doc_only(self) -> None:
        self.assertTrue(
            checker.is_benchmark_doc_only_path("scripts/test_update_readme_from_bench.py")
        )
        self.assertTrue(
            checker.is_benchmark_doc_only_path(
                "scripts/test_bench_llama_cpp_metal_sweep.py"
            )
        )

    def test_llama_cpp_sweep_orchestrator_is_doc_only(self) -> None:
        # bench_llama_cpp_metal_sweep.py orchestrates llama.cpp Metal full-stack
        # runs only; AX-only and mlx_lm-only artifacts (the ones this checker
        # validates against README markers) do not invoke it.
        self.assertTrue(
            checker.is_benchmark_doc_only_path("scripts/bench_llama_cpp_metal_sweep.py")
        )

    def test_bench_mlx_inference_stack_is_not_doc_only(self) -> None:
        # bench_mlx_inference_stack.py is the single source of bench-output JSON
        # for every artifact this checker validates. A dirty diff on it MUST
        # fail the tracked-dirty gate.
        self.assertFalse(
            checker.is_benchmark_doc_only_path("scripts/bench_mlx_inference_stack.py")
        )

    def test_engine_code_is_not_doc_only(self) -> None:
        self.assertFalse(checker.is_benchmark_doc_only_path("crates/ax-engine-mlx/src/runner.rs"))
        self.assertFalse(checker.is_benchmark_doc_only_path("scripts/bench_ax_only_sweep.py"))

    def test_pyproject_toml_is_doc_only(self) -> None:
        self.assertTrue(checker.is_benchmark_doc_only_path("pyproject.toml"))

    def test_benchmark_results_json_are_doc_only(self) -> None:
        self.assertTrue(
            checker.is_benchmark_doc_only_path(
                "benchmarks/results/mtp-fair/2026-06-01/27b-4bit/flappy/ax_engine.json"
            )
        )
        self.assertTrue(
            checker.is_benchmark_doc_only_path(
                "benchmarks/results/mlx-inference/2026-06-01-ax-direct/gemma-4-e2b-it-4bit.json"
            )
        )
        # non-JSON files under benchmarks/results/ are not auto-exempted
        self.assertFalse(
            checker.is_benchmark_doc_only_path(
                "benchmarks/results/mtp-fair/run.sh"
            )
        )

    def test_tracked_dirty_delete_status_accepted_for_doc_only_path(self) -> None:
        # Deletion of benchmark result artifacts is irrelevant to a separate AX bench run.
        self.assertTrue(
            checker.tracked_dirty_is_benchmark_doc_only([
                " D benchmarks/results/mtp-fair/2026-06-01/27b-4bit/flappy/ax_engine.json",
                " M pyproject.toml",
            ])
        )
        # Deletion of non-doc-only files must still fail.
        self.assertFalse(
            checker.tracked_dirty_is_benchmark_doc_only([
                " D crates/ax-engine-core/src/kv.rs",
            ])
        )

    def test_tracked_dirty_aggregate_accepts_post_processing_only(self) -> None:
        status = [
            " M README.md",
            " M docs/assets/perf-qwen-decode-box-whisker.svg",
            " M scripts/update_readme_from_bench.py",
            " M scripts/test_update_readme_from_bench.py",
            " M scripts/bench_llama_cpp_metal_sweep.py",
        ]
        self.assertTrue(checker.tracked_dirty_is_benchmark_doc_only(status))

    def test_tracked_dirty_aggregate_rejects_bench_producer(self) -> None:
        status = [
            " M README.md",
            " M scripts/bench_mlx_inference_stack.py",
        ]
        self.assertFalse(checker.tracked_dirty_is_benchmark_doc_only(status))


class ReadmePerformanceArtifactTests(unittest.TestCase):
    def test_unavailable_cells_allow_annotated_no_decode_text(self) -> None:
        self.assertTrue(checker.is_unavailable_cell("— (no decode)†"))
        self.assertTrue(checker.is_unavailable_cell("—"))
        self.assertFalse(checker.is_unavailable_cell("12.7 (-55.0%)"))

    def test_metric_median_raises_clean_error_on_null_median(self) -> None:
        # README delta verification paths feed bench medians through
        # metric_median(). Cache-warm AX rows emit {"median": None}; that must
        # surface as ArtifactCheckError rather than float(None) TypeError.
        for table, key in (("prefill", "prefill_tok_s"), ("decode", "decode_tok_s"), ("ttft", "ttft_ms")):
            with self.assertRaisesRegex(checker.ArtifactCheckError, rf"lacks {key}\.median"):
                checker.metric_median(
                    {"engine": "ax_engine_mlx", "prompt_tokens": 128, key: {"median": None}},
                    table,
                )

    def test_metric_summary_median_raises_clean_error_on_null_median(self) -> None:
        # When the bench runs with --ax-enable-prefix-cache and every trial
        # turns out to be a cache-warm hit, summarize_runs emits
        # {"median": None, ...}. The validator must surface a clean
        # ArtifactCheckError rather than a TypeError from float(None).
        with self.assertRaisesRegex(checker.ArtifactCheckError, r"lacks prefill_s\.median"):
            checker.metric_summary_median(
                {"engine": "ax_engine_mlx", "prefill_s": {"median": None}},
                "prefill_s",
            )
        with self.assertRaisesRegex(checker.ArtifactCheckError, r"lacks prefill_s\.median"):
            checker.validate_positive_metric_summary(
                artifact_path=Path("artifact.json"),
                row={"engine": "ax_engine_mlx", "prefill_s": {"median": None}},
                key="prefill_s",
            )

    def test_repo_internal_absolute_prompt_path_resolves_to_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_path = (
                root
                / "benchmarks/results/mlx-inference/local/model-prompts/"
                "prompt-4-gen-2.json"
            )
            prompt_path.parent.mkdir(parents=True)
            prompt_hash = checker.token_sha256([1, 2, 3, 4])
            prompt_path.write_text(
                json.dumps(
                    {
                        "schema_version": "ax.mlx_reference_prompt.v1",
                        "prompt_tokens": 4,
                        "generation_tokens": 2,
                        "sha256": prompt_hash,
                        "token_ids": [1, 2, 3, 4],
                    },
                    indent=2,
                )
                + "\n"
            )
            old_absolute_path = (
                Path("/Users/akiralam/code/ax-engine")
                / prompt_path.relative_to(root)
            )

            resolved_hash = checker.validate_prompt_artifact(
                repo_root=root,
                artifact_path=root / "benchmarks/results/mlx-inference/local/model.json",
                prompt_doc={
                    "token_ids_path": str(old_absolute_path),
                    "token_ids_sha256": prompt_hash,
                    "prompt_tokens": 4,
                    "generation_tokens": 2,
                },
                prompt_tokens=4,
                generation_tokens=2,
            )

        self.assertEqual(resolved_hash, prompt_hash)

    def test_legacy_mlx_inference_prompt_path_resolves_to_categorized_checkout(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_path = (
                root
                / "benchmarks/results/inference/mlx-inference/local/model-prompts/"
                "prompt-4-gen-2.json"
            )
            prompt_path.parent.mkdir(parents=True)
            prompt_hash = checker.token_sha256([1, 2, 3, 4])
            prompt_path.write_text(
                json.dumps(
                    {
                        "schema_version": "ax.mlx_reference_prompt.v1",
                        "prompt_tokens": 4,
                        "generation_tokens": 2,
                        "sha256": prompt_hash,
                        "token_ids": [1, 2, 3, 4],
                    },
                    indent=2,
                )
                + "\n"
            )

            resolved_hash = checker.validate_prompt_artifact(
                repo_root=root,
                artifact_path=(
                    root
                    / "benchmarks/results/inference/mlx-inference/local/model.json"
                ),
                prompt_doc={
                    "token_ids_path": (
                        "benchmarks/results/mlx-inference/local/model-prompts/"
                        "prompt-4-gen-2.json"
                    ),
                    "token_ids_sha256": prompt_hash,
                    "prompt_tokens": 4,
                    "generation_tokens": 2,
                },
                prompt_tokens=4,
                generation_tokens=2,
            )

        self.assertEqual(resolved_hash, prompt_hash)

    def test_llama_cpp_rows_accept_current_metadata_key(self) -> None:
        checker.validate_delegated_metrics_if_present(
            artifact_path=Path("artifact.json"),
            row={"engine": "llama_cpp_metal", "llama_cpp": {"build_commit": "abc"}},
            require_phase0=True,
        )

    def test_ax_split_allows_explicit_no_decode_steps(self) -> None:
        row = {
            "engine": "ax_engine_mlx",
            "generation_tokens": 128,
            "prefill_s": metric(1.0),
            "decode_s": metric(0.0),
            "ax_mlx_telemetry": {
                "ax_mlx_prefill_steps": 1,
                "ax_mlx_decode_steps": 0,
            },
            "ax_mlx_decode_route": {"classification": "no_decode_steps"},
        }
        checker.validate_ax_prefill_decode_split(
            artifact_path=Path("artifact.json"),
            row=row,
            require_phase0=False,
        )

    def test_ngram_telemetry_allows_no_observed_draft_path_for_no_decode(self) -> None:
        row = {
            "ax_decode_claim_status": "ngram_no_observed_draft_path",
            "ax_mlx_decode_route": {"classification": "no_decode_steps"},
            "ngram_acceleration_telemetry": ngram_telemetry(
                attempts=0,
                accepted=0,
            ),
        }
        checker.validate_ngram_claim_telemetry(
            artifact_path=Path("artifact.json"),
            row=row,
            require_phase0=True,
        )

    def write_fixture(
        self,
        root: Path,
        *,
        stale_readme_value: bool = False,
        stale_readme_percent: bool = False,
        claim_gate: bool = True,
        claim_status: bool = True,
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
                payload["ttft_ms"] = metric(40.0)
                payload["baseline"] = {
                    "engine": "mlx_lm",
                    "method": "mlx_lm.benchmark",
                    "role": "primary_reference",
                }
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
                payload["prefill_work_contract"] = (
                    checker.HISTORICAL_PREFILL_WORK_CONTRACT
                )
                if claim_status:
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
                payload["prefill_work_contract"] = (
                    checker.HISTORICAL_PREFILL_WORK_CONTRACT
                )
                if claim_status:
                    payload["ax_decode_claim_status"] = (
                        "ngram_acceleration_effective_throughput"
                    )
                payload["ngram_acceleration_telemetry"] = ngram_telemetry(
                    attempts=1,
                    accepted=2,
                )
            return payload

        artifact = {
            "schema_version": checker.MLX_INFERENCE_STACK_SCHEMA_VERSION,
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
                row("ax_engine_mlx", 80.0, 8.0),
                row("ax_engine_mlx_ngram_accel", 82.0, 12.0),
            ],
        }
        if claim_gate:
            artifact["claim_gate"] = {
                "schema_version": checker.PHASE0_CLAIM_GATE_SCHEMA_VERSION,
                "scope": "mlx_inference_stack_public_readme",
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
                    "#### Decode throughput (tok/s) - generation=2 tokens, temp=0",
                    "| Model | MLX quantization | Prompt tok | mlx_lm | ax direct baseline |",
                    "|---|---|---:|---:|---:|",
                    f"| Gemma 4 E2B | 4-bit | 4 | 10.0 | {direct_decode} ({direct_delta}) |",
                    "#### Prefill throughput (tok/s) - percentages vs mlx_lm",
                    "| Model | MLX quantization | Prompt tok | mlx_lm | ax engine |",
                    "|---|---|---:|---:|---:|",
                    "| Gemma 4 E2B | 4-bit | 4 | 100.0 | 80.0 (-20.0%) |",
                    "#### Time to first token (ms) - generation=2 tokens, temp=0",
                    "| Model | MLX quantization | Prompt tok | mlx_lm | ax engine |",
                    "|---|---|---:|---:|---:|",
                    "| Gemma 4 E2B | 4-bit | 4 | 40.0 | **30.0 (-25.0%)** |",
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
                expected_metric_count=6,
            )

        self.assertEqual(len(checked), 6)

    def test_ax_overlay_allows_ax_only_artifact_without_mlx_lm_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            reference_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            overlay_path = (
                root
                / "benchmarks/results/mlx-inference/local-ax/gemma-4-e2b-it-4bit.json"
            )
            overlay_path.parent.mkdir(parents=True)
            overlay = json.loads(reference_path.read_text())
            overlay["results"] = [
                row for row in overlay["results"] if row["engine"] == "ax_engine_mlx"
            ]
            overlay_path.write_text(json.dumps(overlay, indent=2) + "\n")
            readme_path = root / "README.md"
            readme_path.write_text(
                readme_path.read_text().replace(
                    "`benchmarks/results/mlx-inference/local/`",
                    "<!-- readme-performance-artifacts: "
                    "reference=benchmarks/results/mlx-inference/local/; "
                    "ax-overlay=benchmarks/results/mlx-inference/local-ax/ -->",
                )
            )

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=readme_path,
                expected_metric_count=6,
            )

        self.assertEqual(len(checked), 6)

    def test_phase0_ax_row_rejects_stale_long_prefill_work_contract(self) -> None:
        row = {
            "engine": "ax_engine_mlx",
            "prompt_tokens": 2048,
            "sampler_settings": None,
            "prefill_work_contract": checker.HISTORICAL_PREFILL_WORK_CONTRACT,
        }

        with self.assertRaisesRegex(
            checker.ArtifactCheckError,
            "expected 'mlx_lm_style_cache_only_prefix_plus_final_prompt_token'",
        ):
            checker.validate_ax_prefill_work_contract(
                artifact_path=Path("artifact.json"),
                row=row,
            )

    def test_phase0_ax_row_accepts_long_greedy_prefill_work_contract(self) -> None:
        row = {
            "engine": "ax_engine_mlx_ngram_accel",
            "prompt_tokens": 2048,
            "sampler_settings": "greedy",
            "prefill_work_contract": checker.MLX_LM_STYLE_PREFILL_WORK_CONTRACT,
        }

        checker.validate_ax_prefill_work_contract(
            artifact_path=Path("artifact.json"),
            row=row,
        )

    def test_phase0_pure_mtp_row_accepts_mtp_only_contract(self) -> None:
        prompt_hash = checker.token_sha256([1, 2, 3, 4])
        row = {
            "engine": "ax_engine_mlx_pure_mtp",
            "method": "server_sse_runner_time_us",
            "timing_scope": "ax_engine_runner_time_us",
            "runtime_identity": {
                "selected_backend": "mlx",
                "route_identity": "repo_owned_mlx",
            },
            "batch_size": 1,
            "prefill_step_size": 2048,
            "prompt_tokens": 4,
            "generation_tokens": 2,
            "prompt_token_ids_sha256": prompt_hash,
            "prefill_tok_s": metric(80.0),
            "decode_tok_s": metric(11.0),
            "ttft_ms": metric(30.0),
            "prefill_s": metric(0.1),
            "decode_s": metric(0.2),
            "trials": [{}, {}, {}],
            "ttft_source": "ax_engine_runner_prefill_time",
            "ax_mlx_telemetry": ax_mlx_telemetry(),
            "ax_decode_policy": "mtp_head_only_no_ngram_stacking",
            "ax_decode_claim_status": "mtp_head_only_effective",
            "ax_decode_effective_route": "mtp_head_only_verify_loop",
            "run_stability": {
                "schema_version": checker.RUN_STABILITY_SCHEMA_VERSION,
                "metric": "decode_tok_s",
                "classification": "stable_enough",
                "trial_count": 3,
            },
            "ngram_acceleration_telemetry": ngram_telemetry(
                attempts=0,
                accepted=0,
                mtp_draft_tokens=2,
            ),
        }

        checker.validate_artifact_row(
            artifact_path=Path("artifact.json"),
            artifact={
                "prefill_step_size": 2048,
                "repetitions": 3,
                "claim_gate": {
                    "schema_version": checker.PHASE0_CLAIM_GATE_SCHEMA_VERSION,
                },
            },
            row=row,
            prompt_hashes={(4, 2): prompt_hash},
        )

    def test_phase0_mtp_row_rejects_ngram_stacking_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            direct = next(
                row for row in artifact["results"] if row["engine"] == "ax_engine_mlx"
            )
            mtp_row = dict(direct)
            mtp_row.update(
                {
                    "engine": "ax_engine_mlx_pure_mtp",
                    "ax_decode_policy": "mtp_head_only_no_ngram_stacking",
                    "ax_decode_claim_status": "mtp_head_only_effective",
                    "ax_decode_effective_route": "mtp_head_only_verify_loop",
                    "ngram_acceleration_telemetry": ngram_telemetry(
                        attempts=0,
                        accepted=0,
                        mtp_draft_tokens=2,
                        mtp_ngram_hit_steps=1,
                    ),
                }
            )
            artifact["results"].append(mtp_row)
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "MTP AX row has n-gram stacking hits",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_phase0_mtp_row_rejects_missing_effective_claim(self) -> None:
        row = {
            "engine": "ax_engine_gemma4_assistant_mtp",
            "ax_decode_policy": "gemma4_assistant_mtp_no_ngram_stacking",
            "ax_decode_claim_status": "mtp_head_only_no_observed_draft_path",
            "ax_decode_effective_route": "mtp_head_only_not_observed",
            "ngram_acceleration_telemetry": ngram_telemetry(
                attempts=0,
                accepted=0,
            ),
        }

        with self.assertRaisesRegex(
            checker.ArtifactCheckError,
            "MTP AX row lacks effective MTP claim",
        ):
            checker.validate_mtp_row_contract(
                artifact_path=Path("artifact.json"),
                row=row,
                require_phase0=True,
            )

    def test_build_provenance_rejects_tracked_dirty_artifact(self) -> None:
        with self.assertRaisesRegex(
            checker.ArtifactCheckError,
            r"tracked-dirty source tree.*scripts/bench_mlx_inference_stack.py",
        ):
            checker.validate_build_provenance(
                artifact_path=Path("artifact.json"),
                artifact={
                    "build": {
                        "git_tracked_dirty": True,
                        "git_tracked_status": [
                            " M scripts/bench_mlx_inference_stack.py"
                        ],
                    }
                },
            )

    def test_build_provenance_allows_clean_or_legacy_artifact(self) -> None:
        checker.validate_build_provenance(
            artifact_path=Path("artifact.json"),
            artifact={"build": {"git_tracked_dirty": False}},
        )
        checker.validate_build_provenance(
            artifact_path=Path("legacy.json"),
            artifact={"build": {"commit": "abc123"}},
        )

    def test_host_performance_conditions_validate_optional_shape(self) -> None:
        checker.validate_host_performance_conditions(
            artifact_path=Path("artifact.json"),
            artifact={
                "host": {
                    "performance_conditions": {
                        "power_source": "AC Power",
                        "battery_status": "80%; AC attached",
                        "thermal_status_lines": [
                            "Note: No thermal warning level has been recorded"
                        ],
                        "load_average": {
                            "one_minute": 1.0,
                            "five_minutes": 1.5,
                            "fifteen_minutes": 2.0,
                        },
                        "thermal_warning_recorded": False,
                        "performance_warning_recorded": True,
                        "cpu_power_status_recorded": False,
                    }
                }
            },
        )
        checker.validate_host_performance_conditions(
            artifact_path=Path("legacy.json"),
            artifact={"host": {"chip": "Apple M5 Max"}},
        )

    def test_host_performance_conditions_rejects_malformed_shape(self) -> None:
        with self.assertRaisesRegex(
            checker.ArtifactCheckError,
            "performance_warning_recorded must be boolean",
        ):
            checker.validate_host_performance_conditions(
                artifact_path=Path("artifact.json"),
                artifact={
                    "host": {
                        "performance_conditions": {
                            "performance_warning_recorded": "no"
                        }
                    }
                },
            )

        with self.assertRaisesRegex(
            checker.ArtifactCheckError,
            r"load_average.one_minute must be numeric",
        ):
            checker.validate_host_performance_conditions(
                artifact_path=Path("artifact.json"),
                artifact={
                    "host": {
                        "performance_conditions": {
                            "load_average": {
                                "one_minute": "1",
                                "five_minutes": 1.5,
                                "fifteen_minutes": 2.0,
                            }
                        }
                    }
                },
            )

    def test_benchmark_window_validates_optional_shape(self) -> None:
        checker.validate_benchmark_window(
            artifact_path=Path("artifact.json"),
            artifact={
                "benchmark_window": {
                    "started_at": "2026-07-06T10:00:00-0400",
                    "finished_at": "2026-07-06T10:01:00-0400",
                    "elapsed_seconds": 60.0,
                    "performance_conditions_start": {
                        "load_average": {
                            "one_minute": 1.0,
                            "five_minutes": 1.5,
                            "fifteen_minutes": 2.0,
                        }
                    },
                    "performance_conditions_end": {
                        "thermal_warning_recorded": False
                    },
                }
            },
        )
        checker.validate_benchmark_window(
            artifact_path=Path("legacy.json"),
            artifact={},
        )

    def test_benchmark_window_rejects_malformed_shape(self) -> None:
        with self.assertRaisesRegex(
            checker.ArtifactCheckError,
            "benchmark_window.elapsed_seconds must be numeric",
        ):
            checker.validate_benchmark_window(
                artifact_path=Path("artifact.json"),
                artifact={
                    "benchmark_window": {
                        "started_at": "2026-07-06T10:00:00-0400",
                        "finished_at": "2026-07-06T10:01:00-0400",
                        "elapsed_seconds": "60",
                    }
                },
            )

        with self.assertRaisesRegex(
            checker.ArtifactCheckError,
            r"load_average.one_minute must be numeric",
        ):
            checker.validate_benchmark_window(
                artifact_path=Path("artifact.json"),
                artifact={
                    "benchmark_window": {
                        "started_at": "2026-07-06T10:00:00-0400",
                        "finished_at": "2026-07-06T10:01:00-0400",
                        "elapsed_seconds": 60.0,
                        "performance_conditions_start": {
                            "load_average": {
                                "one_minute": "1",
                                "five_minutes": 1.5,
                                "fifteen_minutes": 2.0,
                            }
                        },
                    }
                },
            )

    def test_build_provenance_allows_benchmark_doc_only_dirty_artifact(self) -> None:
        checker.validate_build_provenance(
            artifact_path=Path("artifact.json"),
            artifact={
                "build": {
                    "git_tracked_dirty": True,
                    "git_tracked_status": [
                        " M README.md",
                        " M docs/assets/perf-gemma4-decode-box-whisker.svg",
                        " M docs/assets/perf-qwen-prefill-box-whisker.svg",
                    ],
                }
            },
        )

    def test_build_provenance_allows_dirty_artifact_with_accepted_flag(self) -> None:
        # git_tracked_dirty_accepted=True is an explicit author override for cases
        # where the dirty changes are non-doc-only but were committed shortly after
        # the bench run and are now part of the canonical codebase.
        checker.validate_build_provenance(
            artifact_path=Path("artifact.json"),
            artifact={
                "build": {
                    "git_tracked_dirty": True,
                    "git_tracked_dirty_accepted": True,
                    "git_tracked_status": [
                        " M crates/ax-engine-core/src/kv.rs",
                        " M metal/kernels/phase1_dense_path.metal",
                    ],
                }
            },
        )

    def test_build_provenance_rejects_dirty_without_accepted_flag(self) -> None:
        with self.assertRaisesRegex(
            checker.ArtifactCheckError,
            r"tracked-dirty source tree",
        ):
            checker.validate_build_provenance(
                artifact_path=Path("artifact.json"),
                artifact={
                    "build": {
                        "git_tracked_dirty": True,
                        "git_tracked_dirty_accepted": False,
                        "git_tracked_status": [
                            " M crates/ax-engine-core/src/kv.rs",
                        ],
                    }
                },
            )

    def test_legacy_non_gated_rows_allow_missing_claim_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root, claim_gate=False, claim_status=False)

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=6,
            )

        self.assertEqual(len(checked), 6)

    def test_degenerate_direct_claim_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["ax_decode_claim_status"] = "legacy_removed_status"
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "direct AX row lacks claim status",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_direct_ax_row_rejects_unstable_run_stability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["run_stability"] = {
                        "schema_version": checker.RUN_STABILITY_SCHEMA_VERSION,
                        "metric": "decode_tok_s",
                        "classification": "tail_regression",
                        "trial_count": 3,
                        "last_vs_first_pct": -12.5,
                    }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "unstable benchmark row: tail_regression",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_run_stability_summary_accepts_current_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if str(row["engine"]).startswith("ax_engine"):
                    row["run_stability"] = {
                        "schema_version": checker.RUN_STABILITY_SCHEMA_VERSION,
                        "metric": "decode_tok_s",
                        "classification": "stable_enough",
                        "trial_count": 3,
                    }
            artifact["run_stability_summary"] = {
                "schema_version": checker.RUN_STABILITY_SUMMARY_SCHEMA_VERSION,
                "scope": "ax_engine_rows",
                "row_count": 2,
                "stable_enough_count": 2,
                "unstable_count": 0,
                "missing_count": 0,
                "classification_counts": {"stable_enough": 2},
                "unstable_rows": [],
                "publication_candidate": True,
            }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=6,
            )

            self.assertEqual(len(checked), 6)

    def test_run_stability_summary_rejects_stale_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if str(row["engine"]).startswith("ax_engine"):
                    row["run_stability"] = {
                        "schema_version": checker.RUN_STABILITY_SCHEMA_VERSION,
                        "metric": "decode_tok_s",
                        "classification": "stable_enough",
                        "trial_count": 3,
                    }
            artifact["run_stability_summary"] = {
                "schema_version": checker.RUN_STABILITY_SUMMARY_SCHEMA_VERSION,
                "scope": "ax_engine_rows",
                "row_count": 2,
                "stable_enough_count": 1,
                "unstable_count": 0,
                "missing_count": 0,
                "classification_counts": {"stable_enough": 2},
                "unstable_rows": [],
                "publication_candidate": True,
            }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "run_stability_summary stable_enough_count is inconsistent",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_run_stability_summary_rejects_stale_row_schema_without_phase0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root, claim_gate=False)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if str(row["engine"]).startswith("ax_engine"):
                    row["run_stability"] = {
                        "schema_version": "ax.benchmark_run_stability.v0",
                        "metric": "decode_tok_s",
                        "classification": "stable_enough",
                        "trial_count": 3,
                    }
            artifact["run_stability_summary"] = {
                "schema_version": checker.RUN_STABILITY_SUMMARY_SCHEMA_VERSION,
                "scope": "ax_engine_rows",
                "row_count": 2,
                "stable_enough_count": 2,
                "unstable_count": 0,
                "missing_count": 0,
                "classification_counts": {"stable_enough": 2},
                "unstable_rows": [],
                "publication_candidate": True,
            }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "has stale run_stability schema",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_run_stability_summary_rejects_publication_candidate_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["run_stability"] = {
                        "schema_version": checker.RUN_STABILITY_SCHEMA_VERSION,
                        "metric": "decode_tok_s",
                        "classification": "tail_regression",
                        "trial_count": 3,
                        "last_vs_first_pct": -12.5,
                    }
                elif row["engine"] == "ax_engine_mlx_ngram_accel":
                    row["run_stability"] = {
                        "schema_version": checker.RUN_STABILITY_SCHEMA_VERSION,
                        "metric": "decode_tok_s",
                        "classification": "stable_enough",
                        "trial_count": 3,
                    }
            artifact["run_stability_summary"] = {
                "schema_version": checker.RUN_STABILITY_SUMMARY_SCHEMA_VERSION,
                "scope": "ax_engine_rows",
                "row_count": 2,
                "stable_enough_count": 1,
                "unstable_count": 1,
                "missing_count": 0,
                "classification_counts": {
                    "stable_enough": 1,
                    "tail_regression": 1,
                },
                "unstable_rows": [
                    {
                        "engine": "ax_engine_mlx",
                        "prompt_tokens": 4,
                        "generation_tokens": 2,
                        "classification": "tail_regression",
                        "last_vs_first_pct": -12.5,
                    }
                ],
                "publication_candidate": True,
            }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "run_stability_summary publication_candidate is inconsistent",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_run_stability_summary_rejects_non_publication_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            artifact["run_stability_summary"] = {
                "schema_version": checker.RUN_STABILITY_SUMMARY_SCHEMA_VERSION,
                "scope": "ax_engine_rows",
                "row_count": 2,
                "stable_enough_count": 0,
                "unstable_count": 0,
                "missing_count": 2,
                "classification_counts": {},
                "unstable_rows": [],
                "publication_candidate": False,
            }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "run_stability_summary is not a publication candidate",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_run_stability_summary_rejects_empty_ax_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            artifact["results"] = [
                row for row in artifact["results"] if row["engine"] == "mlx_lm"
            ]
            artifact["run_stability_summary"] = {
                "schema_version": checker.RUN_STABILITY_SUMMARY_SCHEMA_VERSION,
                "scope": "ax_engine_rows",
                "row_count": 0,
                "stable_enough_count": 0,
                "unstable_count": 0,
                "missing_count": 0,
                "classification_counts": {},
                "unstable_rows": [],
                "publication_candidate": False,
            }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "run_stability_summary is not a publication candidate",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_ax_only_refresh_regression_summary_accepts_reference_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            reference_path = (
                root
                / "benchmarks/results/mlx-inference/reference/gemma-4-e2b-it-4bit.json"
            )
            reference_path.parent.mkdir(parents=True)
            reference_path.write_text(json.dumps(artifact, indent=2) + "\n")
            artifact["ax_only_refresh"] = {
                "method": "reuse_existing_reference_rows_and_rerun_ax_engine_rows",
                "reference_results_source": str(reference_path.relative_to(root)),
            }
            artifact["ax_only_refresh"]["ax_reference_regression_summary"] = (
                checker.expected_ax_only_refresh_regression_summary(
                    artifact=artifact,
                    reference_artifact=json.loads(reference_path.read_text()),
                )
            )
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=6,
            )

            self.assertEqual(len(checked), 6)

    def test_ax_only_refresh_schema_requires_regression_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            reference_path = (
                root
                / "benchmarks/results/mlx-inference/reference/gemma-4-e2b-it-4bit.json"
            )
            reference_path.parent.mkdir(parents=True)
            reference_path.write_text(json.dumps(artifact, indent=2) + "\n")
            artifact["ax_only_refresh"] = {
                "schema_version": checker.AX_ONLY_REFRESH_SCHEMA_VERSION,
                "method": "reuse_existing_reference_rows_and_rerun_ax_engine_rows",
                "reference_results_source": str(reference_path.relative_to(root)),
            }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "ax_only_refresh lacks ax_reference_regression_summary",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_ax_only_refresh_rejects_wrong_reference_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            reference_path = (
                root
                / "benchmarks/results/mlx-inference/reference/gemma-4-e2b-it-4bit.json"
            )
            reference_path.parent.mkdir(parents=True)
            reference_path.write_text(
                json.dumps({"schema_version": "wrong", "results": []}, indent=2)
                + "\n"
            )
            artifact["ax_only_refresh"] = {
                "schema_version": checker.AX_ONLY_REFRESH_SCHEMA_VERSION,
                "method": "reuse_existing_reference_rows_and_rerun_ax_engine_rows",
                "reference_results_source": str(reference_path.relative_to(root)),
                "ax_reference_regression_summary": {},
            }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "ax_only_refresh reference has unexpected schema_version",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_ax_only_refresh_rejects_reference_without_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            reference_path = (
                root
                / "benchmarks/results/mlx-inference/reference/gemma-4-e2b-it-4bit.json"
            )
            reference_path.parent.mkdir(parents=True)
            reference_path.write_text(
                json.dumps(
                    {"schema_version": checker.MLX_INFERENCE_STACK_SCHEMA_VERSION},
                    indent=2,
                )
                + "\n"
            )
            artifact["ax_only_refresh"] = {
                "schema_version": checker.AX_ONLY_REFRESH_SCHEMA_VERSION,
                "method": "reuse_existing_reference_rows_and_rerun_ax_engine_rows",
                "reference_results_source": str(reference_path.relative_to(root)),
                "ax_reference_regression_summary": {},
            }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "ax_only_refresh reference lacks results list",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_ax_only_refresh_regression_summary_rejects_decode_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            reference = copy.deepcopy(artifact)
            for row in reference["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["decode_tok_s"] = metric(10.0)
            reference_path = (
                root
                / "benchmarks/results/mlx-inference/reference/gemma-4-e2b-it-4bit.json"
            )
            reference_path.parent.mkdir(parents=True)
            reference_path.write_text(json.dumps(reference, indent=2) + "\n")
            artifact["ax_only_refresh"] = {
                "method": "reuse_existing_reference_rows_and_rerun_ax_engine_rows",
                "reference_results_source": str(reference_path.relative_to(root)),
            }
            regression_summary = checker.expected_ax_only_refresh_regression_summary(
                artifact=artifact,
                reference_artifact=reference,
            )
            artifact["ax_only_refresh"]["ax_reference_regression_summary"] = (
                regression_summary
            )
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "ax_reference_regression_summary is not a publication candidate",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_ax_only_refresh_regression_summary_rejects_missing_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            reference = copy.deepcopy(artifact)
            reference["results"] = [
                row for row in reference["results"] if row["engine"] == "mlx_lm"
            ]
            reference_path = (
                root
                / "benchmarks/results/mlx-inference/reference/gemma-4-e2b-it-4bit.json"
            )
            reference_path.parent.mkdir(parents=True)
            reference_path.write_text(json.dumps(reference, indent=2) + "\n")
            artifact["ax_only_refresh"] = {
                "method": "reuse_existing_reference_rows_and_rerun_ax_engine_rows",
                "reference_results_source": str(reference_path.relative_to(root)),
            }
            regression_summary = checker.expected_ax_only_refresh_regression_summary(
                artifact=artifact,
                reference_artifact=reference,
            )
            self.assertEqual(
                regression_summary["missing_reference_rows"],
                [
                    {
                        "engine": "ax_engine_mlx",
                        "prompt_tokens": 4,
                        "generation_tokens": 2,
                        "classification": "missing_reference",
                    },
                    {
                        "engine": "ax_engine_mlx_ngram_accel",
                        "prompt_tokens": 4,
                        "generation_tokens": 2,
                        "classification": "missing_reference",
                    },
                ],
            )
            artifact["ax_only_refresh"]["ax_reference_regression_summary"] = (
                regression_summary
            )
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "ax_reference_regression_summary is not a publication candidate",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_ax_only_refresh_regression_summary_rejects_duplicate_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            reference = copy.deepcopy(artifact)
            duplicate = next(
                row for row in reference["results"] if row["engine"] == "ax_engine_mlx"
            )
            reference["results"].append(copy.deepcopy(duplicate))
            reference_path = (
                root
                / "benchmarks/results/mlx-inference/reference/gemma-4-e2b-it-4bit.json"
            )
            reference_path.parent.mkdir(parents=True)
            reference_path.write_text(json.dumps(reference, indent=2) + "\n")
            artifact["ax_only_refresh"] = {
                "schema_version": checker.AX_ONLY_REFRESH_SCHEMA_VERSION,
                "method": "reuse_existing_reference_rows_and_rerun_ax_engine_rows",
                "reference_results_source": str(reference_path.relative_to(root)),
            }
            regression_summary = checker.expected_ax_only_refresh_regression_summary(
                artifact=artifact,
                reference_artifact=reference,
            )
            self.assertEqual(
                regression_summary["duplicate_reference_rows"],
                [
                    {
                        "engine": "ax_engine_mlx",
                        "prompt_tokens": 4,
                        "generation_tokens": 2,
                        "classification": "duplicate_reference",
                    }
                ],
            )
            artifact["ax_only_refresh"]["ax_reference_regression_summary"] = (
                regression_summary
            )
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "ax_reference_regression_summary is not a publication candidate",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_ax_only_refresh_regression_summary_rejects_duplicate_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            reference = copy.deepcopy(artifact)
            duplicate = next(
                row for row in artifact["results"] if row["engine"] == "ax_engine_mlx"
            )
            artifact["results"].append(copy.deepcopy(duplicate))
            reference_path = (
                root
                / "benchmarks/results/mlx-inference/reference/gemma-4-e2b-it-4bit.json"
            )
            reference_path.parent.mkdir(parents=True)
            reference_path.write_text(json.dumps(reference, indent=2) + "\n")
            artifact["ax_only_refresh"] = {
                "schema_version": checker.AX_ONLY_REFRESH_SCHEMA_VERSION,
                "method": "reuse_existing_reference_rows_and_rerun_ax_engine_rows",
                "reference_results_source": str(reference_path.relative_to(root)),
            }
            regression_summary = checker.expected_ax_only_refresh_regression_summary(
                artifact=artifact,
                reference_artifact=reference,
            )
            self.assertEqual(
                regression_summary["duplicate_current_rows"],
                [
                    {
                        "engine": "ax_engine_mlx",
                        "prompt_tokens": 4,
                        "generation_tokens": 2,
                        "classification": "duplicate_current",
                    }
                ],
            )
            artifact["ax_only_refresh"]["ax_reference_regression_summary"] = (
                regression_summary
            )
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "ax_reference_regression_summary is not a publication candidate",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_phase0_artifact_rejects_unvalidated_ax_engine_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            direct = next(
                row for row in artifact["results"] if row["engine"] == "ax_engine_mlx"
            )
            unknown_row = dict(direct)
            unknown_row["engine"] = "ax_engine_future_fastpath"
            artifact["results"].append(unknown_row)
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "unvalidated AX row engine: ax_engine_future_fastpath",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_phase0_artifact_rejects_non_object_result_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            artifact["results"].append("bad-row")
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "has non-object result row",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_direct_ax_row_rejects_hidden_hotpath_fallback_counters(self) -> None:
        fallback_keys = [
            "ax_mlx_single_decode_steps",
            "ax_mlx_ngram_decode_steps",
            "ax_mlx_dense_ffn_split_gate_up_layers",
            "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks",
            "ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked",
            "ax_mlx_direct_cpp_linear_attention_post_input_fallbacks",
            "ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked",
        ]
        for key in fallback_keys:
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    self.write_fixture(root)
                    artifact_path = (
                        root
                        / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
                    )
                    artifact = json.loads(artifact_path.read_text())
                    for row in artifact["results"]:
                        if row["engine"] == "ax_engine_mlx":
                            row["ax_mlx_telemetry"][key] = 1
                    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

                    with self.assertRaisesRegex(
                        checker.ArtifactCheckError,
                        "hidden hotpath fallback counters",
                    ):
                        checker.check_readme_performance(
                            repo_root=root,
                            readme_path=root / "README.md",
                        expected_metric_count=6,
                    )

    def test_direct_ax_row_allows_split_ffn_for_5bit_model(self) -> None:
        # 5-bit gate/up packing is intentionally disabled in the engine; the
        # dense_ffn_split_gate_up_layers counter is expected to be non-zero for
        # these models and must not fire the hotpath-fallback gate.
        row = {
            "ax_mlx_telemetry": {
                "ax_mlx_dense_ffn_split_gate_up_layers": 175,
                "ax_mlx_dense_ffn_gate_up_packed_layers": 0,
            }
        }
        checker.validate_direct_hotpath_no_hidden_fallbacks(
            artifact_path=Path("artifact.json"),
            row=row,
            require_phase0=True,
            model_repo_id="mlx-community/gemma-4-e2b-it-5bit",
        )
        # Other counters must still be rejected even for 5-bit models.
        for key in (
            "ax_mlx_single_decode_steps",
            "ax_mlx_ngram_decode_steps",
            "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks",
        ):
            bad_row = {
                "ax_mlx_telemetry": {
                    "ax_mlx_dense_ffn_split_gate_up_layers": 175,
                    key: 1,
                }
            }
            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "hidden hotpath fallback counters",
            ):
                checker.validate_direct_hotpath_no_hidden_fallbacks(
                    artifact_path=Path("artifact.json"),
                    row=bad_row,
                    require_phase0=True,
                    model_repo_id="mlx-community/gemma-4-e2b-it-5bit",
                )

    def test_direct_ax_row_allows_split_ffn_for_qwen3_dense_model(self) -> None:
        # Qwen3 dense FFN gate/up packing is intentionally guarded in weights.rs
        # until the packed path is token-exact against mlx_lm.
        row = {
            "ax_mlx_telemetry": {
                "ax_mlx_dense_ffn_split_gate_up_layers": 64,
                "ax_mlx_dense_ffn_gate_up_packed_layers": 0,
            }
        }
        checker.validate_direct_hotpath_no_hidden_fallbacks(
            artifact_path=Path("artifact.json"),
            row=row,
            require_phase0=True,
            model_repo_id="mlx-community/Qwen3.6-27B-4bit",
        )

        bad_row = {
            "ax_mlx_telemetry": {
                "ax_mlx_dense_ffn_split_gate_up_layers": 64,
                "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks": 1,
            }
        }
        with self.assertRaisesRegex(
            checker.ArtifactCheckError,
            "hidden hotpath fallback counters",
        ):
            checker.validate_direct_hotpath_no_hidden_fallbacks(
                artifact_path=Path("artifact.json"),
                row=bad_row,
                require_phase0=True,
                model_repo_id="mlx-community/Qwen3.6-27B-4bit",
            )

    def test_direct_ax_variant_rows_reject_hidden_hotpath_fallback_counters(
        self,
    ) -> None:
        variant_engines = sorted(checker.AX_DIRECT_ENGINE_KEYS - {"ax_engine_mlx"})
        for engine in variant_engines:
            with self.subTest(engine=engine):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    self.write_fixture(root)
                    artifact_path = (
                        root
                        / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
                    )
                    artifact = json.loads(artifact_path.read_text())
                    direct_row = next(
                        row
                        for row in artifact["results"]
                        if row["engine"] == "ax_engine_mlx"
                    )
                    variant_row = dict(direct_row)
                    variant_row["engine"] = engine
                    variant_row["ax_mlx_telemetry"] = dict(
                        direct_row["ax_mlx_telemetry"]
                    )
                    variant_row["ax_mlx_telemetry"][
                        "ax_mlx_single_decode_steps"
                    ] = 1
                    artifact["results"].append(variant_row)
                    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

                    with self.assertRaisesRegex(
                        checker.ArtifactCheckError,
                        "hidden hotpath fallback counters",
                    ):
                        checker.check_readme_performance(
                            repo_root=root,
                            readme_path=root / "README.md",
                            expected_metric_count=6,
                        )

    def test_direct_ax_row_accepts_direct_cpp_linear_attention_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["ax_mlx_telemetry"].update(
                        {
                            "ax_mlx_direct_cpp_linear_attention_inputs_attempts": 4,
                            "ax_mlx_direct_cpp_linear_attention_inputs_hits": 4,
                            "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks": 0,
                            "ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked": 0,
                            "ax_mlx_direct_cpp_linear_attention_post_input_attempts": 4,
                            "ax_mlx_direct_cpp_linear_attention_post_input_hits": 4,
                            "ax_mlx_direct_cpp_linear_attention_post_input_fallbacks": 0,
                            "ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked": 0,
                        }
                    )
                    row["ax_mlx_direct_cpp_linear_attention_inputs"] = {
                        "schema_version": "ax.mlx_direct_cpp_linear_attention_inputs.v1",
                        "classification": "all_hits",
                        "attempts": 4,
                        "hits": 4,
                        "fallbacks": 0,
                        "profile_blocked": 0,
                        "hit_rate_micros": 1_000_000,
                    }
                    row["ax_mlx_direct_cpp_linear_attention_post_input"] = {
                        "schema_version": "ax.mlx_direct_cpp_linear_attention_post_input.v1",
                        "classification": "all_hits",
                        "attempts": 4,
                        "hits": 4,
                        "fallbacks": 0,
                        "profile_blocked": 0,
                        "hit_rate_micros": 1_000_000,
                    }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            result = checker.check_readme_performance(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=6,
            )

        self.assertEqual(len(result), 6)

    def test_direct_ax_row_rejects_direct_cpp_linear_attention_without_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["ax_mlx_telemetry"].update(
                        {
                            "ax_mlx_direct_cpp_linear_attention_inputs_attempts": 4,
                            "ax_mlx_direct_cpp_linear_attention_inputs_hits": 4,
                            "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks": 0,
                            "ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked": 0,
                        }
                    )
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "lacks direct C\\+\\+ linear-attention summary",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_direct_ax_row_rejects_direct_cpp_linear_attention_post_input_without_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["ax_mlx_telemetry"].update(
                        {
                            "ax_mlx_direct_cpp_linear_attention_post_input_attempts": 4,
                            "ax_mlx_direct_cpp_linear_attention_post_input_hits": 4,
                            "ax_mlx_direct_cpp_linear_attention_post_input_fallbacks": 0,
                            "ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked": 0,
                        }
                    )
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "lacks direct C\\+\\+ linear-attention post-input summary",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_direct_ax_row_rejects_incomplete_direct_cpp_linear_attention_hits(
        self,
    ) -> None:
        cases = [
            (
                "ax_mlx_direct_cpp_linear_attention_inputs",
                {
                    "ax_mlx_direct_cpp_linear_attention_inputs_attempts": 4,
                    "ax_mlx_direct_cpp_linear_attention_inputs_hits": 3,
                    "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks": 0,
                    "ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked": 0,
                },
                "ax.mlx_direct_cpp_linear_attention_inputs.v1",
                "direct C\\+\\+ linear-attention summary is not all_hits",
            ),
            (
                "ax_mlx_direct_cpp_linear_attention_post_input",
                {
                    "ax_mlx_direct_cpp_linear_attention_post_input_attempts": 4,
                    "ax_mlx_direct_cpp_linear_attention_post_input_hits": 3,
                    "ax_mlx_direct_cpp_linear_attention_post_input_fallbacks": 0,
                    "ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked": 0,
                },
                "ax.mlx_direct_cpp_linear_attention_post_input.v1",
                "direct C\\+\\+ linear-attention post-input summary is not all_hits",
            ),
        ]
        for summary_key, telemetry, schema_version, expected_error in cases:
            with self.subTest(summary_key=summary_key):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    self.write_fixture(root)
                    artifact_path = (
                        root
                        / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
                    )
                    artifact = json.loads(artifact_path.read_text())
                    for row in artifact["results"]:
                        if row["engine"] == "ax_engine_mlx":
                            row["ax_mlx_telemetry"].update(telemetry)
                            row[summary_key] = {
                                "schema_version": schema_version,
                                "classification": "incomplete_accounting",
                                "attempts": 4,
                                "hits": 3,
                                "fallbacks": 0,
                                "profile_blocked": 0,
                                "hit_rate_micros": 750_000,
                            }
                    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

                    with self.assertRaisesRegex(
                        checker.ArtifactCheckError,
                        expected_error,
                    ):
                        checker.check_readme_performance(
                            repo_root=root,
                            readme_path=root / "README.md",
                            expected_metric_count=6,
                        )

    def test_direct_ax_row_rejects_fused_kv_decode_fallback_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["kv_compression_claim_status"] = "integrated_fused_compressed_decode"
                    row["kv_compression_telemetry"] = {
                        "ax_mlx_kv_compression_fused_decode_fallbacks": 1,
                    }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "claims fused KV decode with fallback telemetry",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

    def test_degenerate_ngram_claim_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = (
                root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            )
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx_ngram_accel":
                    row["ax_decode_claim_status"] = "legacy_removed_status"
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "n-gram row lacks claim status",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                expected_metric_count=6,
            )

        self.assertEqual(len(checked), 6)

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
                expected_metric_count=6,
            )

        self.assertEqual(len(checked), 6)

    def test_readme_can_scope_ax_decode_overlay_to_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            base_dir = root / "benchmarks/results/mlx-inference/local"
            overlay_dir = root / "benchmarks/results/mlx-inference/decode-overlay"
            overlay_dir.mkdir(parents=True)
            overlay_path = overlay_dir / "gemma-4-e2b-it-4bit.json"
            artifact = json.loads((base_dir / "gemma-4-e2b-it-4bit.json").read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx":
                    row["prefill_tok_s"] = metric(999.0)
                    row["decode_tok_s"] = metric(8.5)
                    row["ttft_ms"] = metric(999.0)
            overlay_path.write_text(json.dumps(artifact, indent=2) + "\n")

            readme_path = root / "README.md"
            readme_path.write_text(
                readme_path.read_text()
                .replace(
                    "`benchmarks/results/mlx-inference/local/`",
                    "<!-- readme-performance-artifacts: "
                    "reference=benchmarks/results/mlx-inference/local/; "
                    "ax-base=benchmarks/results/mlx-inference/local/; "
                    "ax-decode-overlay@p4="
                    "benchmarks/results/mlx-inference/decode-overlay/ -->\n"
                    "`benchmarks/results/mlx-inference/local/`",
                )
                .replace("8.0 (-20.0%)", "8.5 (-15.0%)")
            )

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=readme_path,
                expected_metric_count=6,
            )

        self.assertEqual(len(checked), 6)

    def test_readme_rejects_unscoped_base_in_composite_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            readme_path = root / "README.md"
            readme_path.write_text(
                readme_path.read_text().replace(
                    "`benchmarks/results/mlx-inference/local/`",
                    "<!-- readme-performance-artifacts: "
                    "base=benchmarks/results/mlx-inference/local/; "
                    "ax-overlay=benchmarks/results/mlx-inference/local/ -->\n"
                    "`benchmarks/results/mlx-inference/local/`",
                )
            )

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "base= is only allowed as a single legacy source",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=readme_path,
                    expected_metric_count=6,
                )

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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                expected_metric_count=6,
            )

            self.assertEqual(len(checked), 6)

    def test_ngram_no_draft_fallback_allows_matching_effective_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx_ngram_accel":
                    row["ax_decode_claim_status"] = "ngram_no_draft_direct_fallback"
                    row["ax_decode_effective_route"] = (
                        "linear_no_draft_direct_pipeline_fallback"
                    )
                    row["ngram_acceleration_telemetry"] = ngram_telemetry(
                        attempts=0,
                        accepted=0,
                        fallback_steps=2,
                    )
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=6,
            )

            self.assertEqual(len(checked), 6)

    def test_ngram_no_draft_fallback_allows_request_disabled_direct_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx_ngram_accel":
                    row["ax_decode_claim_status"] = "ngram_no_draft_direct_fallback"
                    row["ax_decode_effective_route"] = "no_draft_fallback"
                    row["ngram_acceleration_telemetry"] = ngram_telemetry(
                        attempts=0,
                        accepted=0,
                        fallback_steps=2,
                    )
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            checked = checker.check_readme_performance(
                repo_root=root,
                readme_path=root / "README.md",
                expected_metric_count=6,
            )

            self.assertEqual(len(checked), 6)

    def test_ngram_no_draft_fallback_rejects_inconsistent_effective_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_fixture(root)
            artifact_path = root / "benchmarks/results/mlx-inference/local/gemma-4-e2b-it-4bit.json"
            artifact = json.loads(artifact_path.read_text())
            for row in artifact["results"]:
                if row["engine"] == "ax_engine_mlx_ngram_accel":
                    row["ax_decode_claim_status"] = "ngram_no_draft_direct_fallback"
                    row["ax_decode_effective_route"] = "ngram_verified_bonus_tokens"
                    row["ngram_acceleration_telemetry"] = ngram_telemetry(
                        attempts=0,
                        accepted=0,
                        fallback_steps=2,
                    )
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "inconsistent effective route",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=root / "README.md",
                    expected_metric_count=6,
                )

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
                    expected_metric_count=6,
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
                expected_metric_count=6,
            )

        self.assertEqual(len(checked), 6)

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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                expected_metric_count=6,
            )

        self.assertEqual(len(checked), 6)

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
                expected_metric_count=6,
            )

        self.assertEqual(len(checked.metric_checks), 6)
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
                    expected_metric_count=6,
                )

    def test_readme_boundary_artifacts_reject_missing_campaign_scope(self) -> None:
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
            readme_path = root / "README.md"
            readme_path.write_text(
                readme_path.read_text().replace(
                    "This is a single-model long-context boundary, not a "
                    "Gemma/Qwen/GLM-wide campaign.",
                    "",
                )
            )

            with self.assertRaisesRegex(
                checker.ArtifactCheckError,
                "must not imply a family-wide campaign",
            ):
                checker.check_readme_performance(
                    repo_root=root,
                    readme_path=readme_path,
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
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
                expected_metric_count=6,
            )

        self.assertEqual(len(checked), 6)

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
                    expected_metric_count=6,
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
                    expected_metric_count=6,
                )


if __name__ == "__main__":
    unittest.main()
