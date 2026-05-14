#!/usr/bin/env python3
"""Unit tests for TurboQuant quality gate artifact validation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_turboquant_quality_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location("check_turboquant_quality_artifact", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(checker)

BUILDER_PATH = Path(__file__).with_name("build_turboquant_quality_artifact.py")
BUILDER_SPEC = importlib.util.spec_from_file_location("build_turboquant_quality_artifact", BUILDER_PATH)
assert BUILDER_SPEC and BUILDER_SPEC.loader
builder = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(builder)

METRICS_PATH = Path(__file__).with_name("build_turboquant_quality_metrics.py")
METRICS_SPEC = importlib.util.spec_from_file_location("build_turboquant_quality_metrics", METRICS_PATH)
assert METRICS_SPEC and METRICS_SPEC.loader
metrics_builder = importlib.util.module_from_spec(METRICS_SPEC)
METRICS_SPEC.loader.exec_module(metrics_builder)

DECODE_OUTPUTS_PATH = Path(__file__).with_name("build_turboquant_decode_outputs.py")
DECODE_OUTPUTS_SPEC = importlib.util.spec_from_file_location(
    "build_turboquant_decode_outputs", DECODE_OUTPUTS_PATH
)
assert DECODE_OUTPUTS_SPEC and DECODE_OUTPUTS_SPEC.loader
decode_outputs_builder = importlib.util.module_from_spec(DECODE_OUTPUTS_SPEC)
DECODE_OUTPUTS_SPEC.loader.exec_module(decode_outputs_builder)

READINESS_PATH = Path(__file__).with_name("check_turboquant_promotion_readiness.py")
READINESS_SPEC = importlib.util.spec_from_file_location(
    "check_turboquant_promotion_readiness", READINESS_PATH
)
assert READINESS_SPEC and READINESS_SPEC.loader
readiness = importlib.util.module_from_spec(READINESS_SPEC)
READINESS_SPEC.loader.exec_module(readiness)

SHA = "a" * 64


def valid_artifact(root: Path) -> dict:
    manifest = root / "benchmarks/manifests/scenario/long_context_qwen_8k.json"
    baseline = root / "benchmarks/results/turboquant/baseline.json"
    candidate = root / "benchmarks/results/turboquant/candidate.json"
    manifest.parent.mkdir(parents=True)
    baseline.parent.mkdir(parents=True)
    manifest.write_text("{}")
    baseline.write_text("{}")
    candidate.write_text("{}")

    return {
        "schema_version": checker.SCHEMA_VERSION,
        "model": {
            "id": "qwen3_5_9b_q4",
            "family": "qwen3_dense",
            "revision": "test-revision",
            "head_dim": 128,
        },
        "workload": {
            "manifest": "benchmarks/manifests/scenario/long_context_qwen_8k.json",
            "context_tokens": 8192,
            "generation_tokens": 256,
            "prompt_sha256": SHA,
        },
        "baseline": {
            "backend": "mlx",
            "kv_compression_mode": "disabled",
        },
        "candidate": {
            "backend": "mlx",
            "kv_compression_mode": checker.REQUIRED_CANDIDATE_COMPRESSION_MODE,
            "preset": "k8v4",
            "quality_profile": "reference_k8v4",
            "decode_path": "fused_compressed_decode",
        },
        "metrics": {
            "max_abs_diff": 0.03,
            "mean_abs_diff": 0.01,
            "min_cosine_similarity": 0.999,
            "decode_tok_s_ratio_to_baseline": 0.9,
            "kv_saved_kib": 1024,
        },
        "route_metadata": {
            "crossover_decisions": {
                "ax_mlx_kv_compression_route_metadata_schema": 2,
                "ax_mlx_kv_compression_production_ready": 0,
                "ax_mlx_kv_compression_production_blockers": 1,
                "ax_mlx_kv_compression_preset": 1,
                "ax_mlx_kv_compression_key_bits": 8,
                "ax_mlx_kv_compression_value_bits": 4,
                "ax_mlx_kv_compression_eligible_layers": 20,
                "ax_mlx_kv_compression_candidate_token_layers": 120000,
                "ax_mlx_kv_compression_estimated_saved_kib": 4096,
                "ax_mlx_kv_compression_runtime_storage_written_slots": 5000,
                "ax_mlx_kv_compression_decode_path": 2,
                "ax_mlx_kv_compression_fused_decode_candidates": 1,
                "ax_mlx_kv_compression_fused_decode_attempts": 1,
                "ax_mlx_kv_compression_fused_decode_successes": 1,
                "ax_mlx_kv_compression_fused_decode_fallbacks": 0,
                "ax_mlx_kv_compression_fused_decode_fallback_reason": 0,
            }
        },
        "artifacts": [
            {
                "role": "baseline",
                "path": "benchmarks/results/turboquant/baseline.json",
                "sha256": SHA,
            },
            {
                "role": "candidate",
                "path": "benchmarks/results/turboquant/candidate.json",
                "sha256": SHA,
            },
        ],
        "decision": {
            "passed": True,
            "quality_gate_passed": True,
            "performance_promotion_ready": True,
            "performance_blockers": [],
            "public_support_docs_approved": False,
        },
    }


def benchmark_doc(
    *,
    compressed: bool,
    decode_path: str | None = None,
    compression_mode: str = checker.REQUIRED_CANDIDATE_COMPRESSION_MODE,
    output_token_ids: list[int] | None = None,
    decode_tok_s: float | None = None,
) -> dict:
    resolved_decode_path = decode_path or "fused_compressed_decode"
    decode_path_code = {
        "full_precision_shadow": 1,
        "fused_compressed_decode": 2,
        "cpu_oracle_compressed_decode": 3,
    }.get(resolved_decode_path, 1)
    row = {
        "engine": "ax_engine_mlx",
        "prompt_tokens": 8192,
        "generation_tokens": 256,
        "prompt_token_ids_sha256": SHA,
        "decode_tok_s": {
            "median": (
                decode_tok_s
                if decode_tok_s is not None
                else 90.0
                if compressed
                else 100.0
            )
        },
    }
    if output_token_ids is not None:
        row["trials"] = [
            {
                "prefill_tok_s": 1.0,
                "decode_tok_s": 1.0,
                "output_tokens": float(len(output_token_ids)),
                "output_token_ids": output_token_ids,
            }
        ]
    if compressed:
        row.update(
            {
                "experimental_mlx_kv_compression": compression_mode,
                "kv_compression_decode_path": resolved_decode_path,
                "kv_compression_telemetry": {
                    "ax_mlx_kv_compression_route_metadata_schema": 2,
                    "ax_mlx_kv_compression_production_ready": 0,
                    "ax_mlx_kv_compression_production_blockers": 1,
                    "ax_mlx_kv_compression_preset": 1,
                    "ax_mlx_kv_compression_key_bits": 8,
                    "ax_mlx_kv_compression_value_bits": 4,
                    "ax_mlx_kv_compression_eligible_layers": 20,
                    "ax_mlx_kv_compression_candidate_token_layers": 120000,
                    "ax_mlx_kv_compression_estimated_saved_kib": 4096,
                    "ax_mlx_kv_compression_runtime_storage_written_slots": 5000,
                    "ax_mlx_kv_compression_decode_path": decode_path_code,
                    "ax_mlx_kv_compression_fused_decode_candidates": 1,
                    "ax_mlx_kv_compression_fused_decode_attempts": 1
                    if decode_path_code == 2
                    else 0,
                    "ax_mlx_kv_compression_fused_decode_successes": 1
                    if decode_path_code == 2
                    else 0,
                    "ax_mlx_kv_compression_fused_decode_fallbacks": 0,
                    "ax_mlx_kv_compression_fused_decode_fallback_reason": 0
                    if decode_path_code == 2
                    else 1,
                },
            }
        )
    return {
        "schema_version": "ax.mlx_inference_stack.v2",
        "results": [row],
    }


class TurboQuantQualityArtifactTests(unittest.TestCase):
    def test_metrics_builder_compares_decode_vectors(self) -> None:
        report = metrics_builder.compare_vectors(
            [[1.0, 2.0, 3.0], [0.0, 1.0]],
            [[1.0, 2.1, 2.9], [0.0, 1.0]],
        )

        self.assertEqual(report["schema_version"], metrics_builder.SCHEMA_VERSION)
        self.assertAlmostEqual(report["metrics"]["max_abs_diff"], 0.1)
        self.assertAlmostEqual(report["metrics"]["mean_abs_diff"], 0.04)
        self.assertGreater(report["metrics"]["min_cosine_similarity"], 0.999)

    def test_metrics_builder_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(
            metrics_builder.QualityMetricsBuildError,
            "dimension mismatch",
        ):
            metrics_builder.compare_vectors([[1.0, 2.0]], [[1.0]])

    def test_metrics_builder_cli_writes_quality_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline-outputs.json"
            candidate = root / "candidate-outputs.json"
            output = root / "quality-metrics.json"
            baseline.write_text(json.dumps({"decode_outputs": [[1.0, 2.0]]}))
            candidate.write_text(json.dumps({"decode_outputs": [[1.0, 2.01]]}))

            self.assertEqual(
                metrics_builder.main(
                    [
                        "--baseline-outputs",
                        str(baseline),
                        "--candidate-outputs",
                        str(candidate),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            payload = json.loads(output.read_text())
            self.assertIn("metrics", payload)

    def test_metrics_builder_cli_returns_one_on_invalid_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline-outputs.json"
            candidate = root / "candidate-outputs.json"
            output = root / "quality-metrics.json"
            baseline.write_text(json.dumps({"decode_outputs": [[1.0, 2.0]]}))
            candidate.write_text(json.dumps({"decode_outputs": [[1.0]]}))

            self.assertEqual(
                metrics_builder.main(
                    [
                        "--baseline-outputs",
                        str(baseline),
                        "--candidate-outputs",
                        str(candidate),
                        "--output",
                        str(output),
                    ]
                ),
                1,
            )
            self.assertFalse(output.exists())

    def test_decode_outputs_builder_extracts_captured_ax_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark.json"
            tokens = list(range(256))
            benchmark.write_text(
                json.dumps(benchmark_doc(compressed=False, output_token_ids=tokens))
            )

            doc = decode_outputs_builder.build_decode_outputs(
                benchmark,
                engine="ax_engine_mlx",
                context_tokens=8192,
                generation_tokens=256,
                compression_mode="disabled",
            )

            self.assertEqual(doc["schema_version"], decode_outputs_builder.SCHEMA_VERSION)
            self.assertEqual(doc["prompt_token_ids_sha256"], SHA)
            self.assertEqual(doc["decode_outputs"], [[float(token) for token in tokens]])

    def test_decode_outputs_builder_fails_closed_without_captured_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark.json"
            doc = benchmark_doc(compressed=False)
            doc["results"][0]["trials"] = [{"output_tokens": 256.0}]
            benchmark.write_text(json.dumps(doc))

            with self.assertRaisesRegex(
                decode_outputs_builder.DecodeOutputBuildError,
                "--capture-output-token-ids",
            ):
                decode_outputs_builder.build_decode_outputs(
                    benchmark,
                    engine="ax_engine_mlx",
                    context_tokens=8192,
                    generation_tokens=256,
                    compression_mode="disabled",
                )

    def test_valid_artifact_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checker.validate_artifact(valid_artifact(root), root=root)

    def test_valid_head_dim_256_artifact_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            artifact["model"]["head_dim"] = 256
            checker.validate_artifact(artifact, root=root)

    def test_valid_head_dim_512_artifact_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            artifact["model"]["head_dim"] = 512
            checker.validate_artifact(artifact, root=root)

    def test_short_context_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            artifact["workload"]["context_tokens"] = 4096
            with self.assertRaisesRegex(checker.ArtifactValidationError, "context_tokens"):
                checker.validate_artifact(artifact, root=root)

    def test_quality_metric_regression_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            artifact["metrics"]["min_cosine_similarity"] = 0.99
            with self.assertRaisesRegex(checker.ArtifactValidationError, "min_cosine_similarity"):
                checker.validate_artifact(artifact, root=root)

    def test_decode_speed_regression_does_not_fail_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            artifact["metrics"]["decode_tok_s_ratio_to_baseline"] = 0.1
            artifact["decision"]["performance_promotion_ready"] = False
            artifact["decision"]["performance_blockers"] = [
                "metrics.decode_tok_s_ratio_to_baseline must be >= 0.85"
            ]

            checker.validate_artifact(artifact, root=root)
            self.assertEqual(
                checker.performance_gate_blockers(artifact["metrics"]),
                ["metrics.decode_tok_s_ratio_to_baseline must be >= 0.85"],
            )

    def test_missing_runtime_storage_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            del artifact["route_metadata"]["crossover_decisions"][
                "ax_mlx_kv_compression_runtime_storage_written_slots"
            ]
            with self.assertRaisesRegex(checker.ArtifactValidationError, "missing required keys"):
                checker.validate_artifact(artifact, root=root)

    def test_stale_route_schema_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            artifact["route_metadata"]["crossover_decisions"][
                "ax_mlx_kv_compression_route_metadata_schema"
            ] = 1
            with self.assertRaisesRegex(checker.ArtifactValidationError, "schema must be >= 2"):
                checker.validate_artifact(artifact, root=root)

    def test_cpu_oracle_route_fails_closed_as_promotion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            artifact["route_metadata"]["crossover_decisions"][
                "ax_mlx_kv_compression_decode_path"
            ] = 3
            with self.assertRaisesRegex(checker.ArtifactValidationError, "path code 2"):
                checker.validate_artifact(artifact, root=root)

    def test_shadow_mode_fails_closed_as_promotion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            artifact["candidate"]["kv_compression_mode"] = "turboquant-shadow"
            with self.assertRaisesRegex(checker.ArtifactValidationError, "kv_compression_mode"):
                checker.validate_artifact(artifact, root=root)

    def test_missing_fused_decode_success_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            artifact["route_metadata"]["crossover_decisions"][
                "ax_mlx_kv_compression_fused_decode_successes"
            ] = 0
            with self.assertRaisesRegex(checker.ArtifactValidationError, "successes"):
                checker.validate_artifact(artifact, root=root)

    def test_fused_decode_fallback_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            artifact["route_metadata"]["crossover_decisions"][
                "ax_mlx_kv_compression_fused_decode_fallbacks"
            ] = 1
            artifact["route_metadata"]["crossover_decisions"][
                "ax_mlx_kv_compression_fused_decode_fallback_reason"
            ] = 5
            with self.assertRaisesRegex(checker.ArtifactValidationError, "zero fused decode fallbacks"):
                checker.validate_artifact(artifact, root=root)

    def test_stale_runtime_production_blockers_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            artifact["route_metadata"]["crossover_decisions"][
                "ax_mlx_kv_compression_production_blockers"
            ] = checker.MAX_RUNTIME_PRODUCTION_BLOCKERS + 1
            with self.assertRaisesRegex(checker.ArtifactValidationError, "unexpected production blockers"):
                checker.validate_artifact(artifact, root=root)

    def test_public_docs_approval_is_not_part_of_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            artifact["decision"]["public_support_docs_approved"] = True
            with self.assertRaisesRegex(checker.ArtifactValidationError, "public_support_docs_approved"):
                checker.validate_artifact(artifact, root=root)

    def test_cli_validates_artifact_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "quality.json"
            artifact_path.write_text(json.dumps(valid_artifact(root)))
            self.assertEqual(checker.main(["--root", str(root), str(artifact_path)]), 0)

    def test_builder_compiles_and_validates_quality_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "benchmarks/manifests/scenario/long_context_qwen_8k.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}")
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            metrics = root / "metrics.json"
            baseline.write_text(json.dumps(benchmark_doc(compressed=False)))
            candidate.write_text(json.dumps(benchmark_doc(compressed=True)))
            metrics.write_text(
                json.dumps(
                    {
                        "max_abs_diff": 0.03,
                        "mean_abs_diff": 0.01,
                        "min_cosine_similarity": 0.999,
                    }
                )
            )

            artifact = builder.build_quality_artifact(
                baseline_benchmark=baseline,
                candidate_benchmark=candidate,
                quality_metrics=metrics,
                manifest=manifest,
                model_id="qwen3_5_9b_q4",
                model_family="qwen3_dense",
                model_revision="test",
                head_dim=128,
                context_tokens=8192,
                generation_tokens=256,
                baseline_engine="ax_engine_mlx",
                candidate_engine="ax_engine_mlx",
                root=root,
            )

            self.assertEqual(artifact["candidate"]["decode_path"], "fused_compressed_decode")
            self.assertEqual(artifact["metrics"]["decode_tok_s_ratio_to_baseline"], 0.9)
            self.assertTrue(artifact["decision"]["performance_promotion_ready"])
            checker.validate_artifact(artifact, root=root)

    def test_builder_compiles_quality_artifact_when_performance_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "benchmarks/manifests/scenario/long_context_qwen_8k.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}")
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            metrics = root / "metrics.json"
            baseline.write_text(json.dumps(benchmark_doc(compressed=False)))
            candidate.write_text(
                json.dumps(benchmark_doc(compressed=True, decode_tok_s=10.0))
            )
            metrics.write_text(
                json.dumps(
                    {
                        "max_abs_diff": 0.03,
                        "mean_abs_diff": 0.01,
                        "min_cosine_similarity": 0.999,
                    }
                )
            )

            artifact = builder.build_quality_artifact(
                baseline_benchmark=baseline,
                candidate_benchmark=candidate,
                quality_metrics=metrics,
                manifest=manifest,
                model_id="qwen3_5_9b_q4",
                model_family="qwen3_dense",
                model_revision="test",
                head_dim=128,
                context_tokens=8192,
                generation_tokens=256,
                baseline_engine="ax_engine_mlx",
                candidate_engine="ax_engine_mlx",
                root=root,
            )

            self.assertEqual(artifact["metrics"]["decode_tok_s_ratio_to_baseline"], 0.1)
            self.assertFalse(artifact["decision"]["performance_promotion_ready"])
            self.assertTrue(artifact["decision"]["performance_blockers"])
            checker.validate_artifact(artifact, root=root)

    def test_readiness_keeps_public_claim_blocked_on_performance_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            model_dir = models_root / "gemma"
            model_dir.mkdir(parents=True)
            (model_dir / "model-manifest.json").write_text(
                json.dumps(
                    {
                        "model_family": "gemma4",
                        "attention_head_dim": 256,
                        "global_head_dim": 512,
                        "attention_head_count": 8,
                        "kv_head_count": 1,
                        "layer_types": ["full_attention"],
                    }
                )
            )
            artifact_path = root / "quality-gate.json"
            artifact = valid_artifact(root)
            artifact["metrics"]["decode_tok_s_ratio_to_baseline"] = 0.1
            artifact["decision"]["performance_promotion_ready"] = False
            artifact["decision"]["performance_blockers"] = [
                "metrics.decode_tok_s_ratio_to_baseline must be >= 0.85"
            ]
            artifact_path.write_text(json.dumps(artifact))

            report = readiness.build_report(
                models_root=models_root,
                results_root=root / "empty-results",
                artifacts=[artifact_path],
                require_artifact_files=True,
                root=root,
            )

            self.assertFalse(report["decision"]["can_make_public_support_claim"])
            self.assertEqual(
                report["decision"]["blockers"],
                [
                    "no passing long-context fused-path performance promotion artifact was found"
                ],
            )
            self.assertTrue(report["quality_artifacts"][0]["passes_quality_gate"])
            self.assertFalse(report["quality_artifacts"][0]["passes_performance_gate"])
            self.assertEqual(
                report["quality_artifacts"][0]["promotion_gap"],
                {
                    "observed_decode_tok_s_ratio_to_baseline": 0.1,
                    "required_min_decode_tok_s_ratio_to_baseline": 0.85,
                    "performance_promotion_ready": False,
                    "next_action": (
                        "rerun or improve fused compressed decode until "
                        "performance blockers clear"
                    ),
                },
            )

    def test_readiness_reports_passing_promotion_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            model_dir = models_root / "gemma"
            model_dir.mkdir(parents=True)
            (model_dir / "model-manifest.json").write_text(
                json.dumps(
                    {
                        "model_family": "gemma4",
                        "attention_head_dim": 256,
                        "global_head_dim": 512,
                        "attention_head_count": 8,
                        "kv_head_count": 1,
                        "layer_types": ["full_attention"],
                    }
                )
            )
            artifact_path = root / "quality-gate.json"
            artifact_path.write_text(json.dumps(valid_artifact(root)))

            report = readiness.build_report(
                models_root=models_root,
                results_root=root / "empty-results",
                artifacts=[artifact_path],
                require_artifact_files=True,
                root=root,
            )

            self.assertTrue(report["decision"]["can_make_public_support_claim"])
            self.assertEqual(
                report["quality_artifacts"][0]["promotion_gap"]["next_action"],
                "ready_for_companion_prd_review",
            )
            self.assertTrue(
                report["quality_artifacts"][0]["promotion_gap"][
                    "performance_promotion_ready"
                ]
            )

    def test_readiness_derives_dense_full_attention_layers_from_tensor_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            model_dir = models_root / "qwen"
            model_dir.mkdir(parents=True)
            (model_dir / "model-manifest.json").write_text(
                json.dumps(
                    {
                        "model_family": "qwen3_dense",
                        "attention_head_dim": 128,
                        "attention_head_count": 16,
                        "kv_head_count": 8,
                        "layer_count": 2,
                        "tensors": [
                            {"role": "attention_q", "layer_index": 0},
                            {"role": "attention_o", "layer_index": 0},
                        ],
                    }
                )
            )

            report = readiness.build_report(
                models_root=models_root,
                results_root=root / "empty-results",
                artifacts=[],
                require_artifact_files=True,
                root=root,
            )

            self.assertTrue(report["models"][0]["eligible_for_current_fused_gate"])
            self.assertEqual(report["models"][0]["full_attention_layers"], 2)

    def test_readiness_blocks_manifest_without_layer_coverage_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            model_dir = models_root / "unknown"
            model_dir.mkdir(parents=True)
            (model_dir / "model-manifest.json").write_text(
                json.dumps(
                    {
                        "model_family": "qwen3_dense",
                        "attention_head_dim": 128,
                        "attention_head_count": 16,
                        "kv_head_count": 8,
                        "layer_count": 2,
                        "tensors": [],
                    }
                )
            )

            report = readiness.build_report(
                models_root=models_root,
                results_root=root / "empty-results",
                artifacts=[],
                require_artifact_files=True,
                root=root,
            )

            self.assertFalse(report["models"][0]["eligible_for_current_fused_gate"])
            self.assertEqual(report["models"][0]["full_attention_layers"], None)
            self.assertIn(
                "full_attention layer coverage could not be determined",
                report["models"][0]["blockers"],
            )

    def test_builder_rejects_shadow_decode_path_as_promotion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "benchmarks/manifests/scenario/long_context_qwen_8k.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}")
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            metrics = root / "metrics.json"
            baseline.write_text(json.dumps(benchmark_doc(compressed=False)))
            candidate.write_text(
                json.dumps(
                    benchmark_doc(
                        compressed=True,
                        decode_path="full_precision_shadow",
                    )
                )
            )
            metrics.write_text(
                json.dumps(
                    {
                        "max_abs_diff": 0.03,
                        "mean_abs_diff": 0.01,
                        "min_cosine_similarity": 0.999,
                    }
                )
            )

            with self.assertRaisesRegex(
                builder.checker.ArtifactValidationError,
                "decode_path",
            ):
                builder.build_quality_artifact(
                    baseline_benchmark=baseline,
                    candidate_benchmark=candidate,
                    quality_metrics=metrics,
                    manifest=manifest,
                    model_id="qwen3_5_9b_q4",
                    model_family="qwen3_dense",
                    model_revision="test",
                    head_dim=128,
                    context_tokens=8192,
                    generation_tokens=256,
                    baseline_engine="ax_engine_mlx",
                    candidate_engine="ax_engine_mlx",
                    root=root,
                )

    def test_builder_rejects_shadow_mode_as_promotion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "benchmarks/manifests/scenario/long_context_qwen_8k.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}")
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            metrics = root / "metrics.json"
            baseline.write_text(json.dumps(benchmark_doc(compressed=False)))
            candidate.write_text(
                json.dumps(
                    benchmark_doc(
                        compressed=True,
                        compression_mode="turboquant-shadow",
                    )
                )
            )
            metrics.write_text(
                json.dumps(
                    {
                        "max_abs_diff": 0.03,
                        "mean_abs_diff": 0.01,
                        "min_cosine_similarity": 0.999,
                    }
                )
            )

            with self.assertRaisesRegex(
                builder.checker.ArtifactValidationError,
                "kv_compression_mode",
            ):
                builder.build_quality_artifact(
                    baseline_benchmark=baseline,
                    candidate_benchmark=candidate,
                    quality_metrics=metrics,
                    manifest=manifest,
                    model_id="qwen3_5_9b_q4",
                    model_family="qwen3_dense",
                    model_revision="test",
                    head_dim=128,
                    context_tokens=8192,
                    generation_tokens=256,
                    baseline_engine="ax_engine_mlx",
                    candidate_engine="ax_engine_mlx",
                    root=root,
                )

    def test_builder_cli_returns_one_on_shadow_decode_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "benchmarks/manifests/scenario/long_context_qwen_8k.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}")
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            metrics = root / "metrics.json"
            output = root / "quality.json"
            baseline.write_text(json.dumps(benchmark_doc(compressed=False)))
            candidate.write_text(
                json.dumps(
                    benchmark_doc(
                        compressed=True,
                        decode_path="full_precision_shadow",
                    )
                )
            )
            metrics.write_text(
                json.dumps(
                    {
                        "max_abs_diff": 0.03,
                        "mean_abs_diff": 0.01,
                        "min_cosine_similarity": 0.999,
                    }
                )
            )

            self.assertEqual(
                builder.main(
                    [
                        "--baseline-benchmark",
                        str(baseline),
                        "--candidate-benchmark",
                        str(candidate),
                        "--quality-metrics",
                        str(metrics),
                        "--output",
                        str(output),
                        "--manifest",
                        str(manifest),
                        "--model-id",
                        "qwen3_5_9b_q4",
                        "--model-family",
                        "qwen3_dense",
                        "--model-revision",
                        "test",
                        "--root",
                        str(root),
                    ]
                ),
                1,
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
