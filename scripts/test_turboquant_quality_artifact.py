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
                "ax_mlx_kv_compression_route_metadata_schema": 1,
                "ax_mlx_kv_compression_production_ready": 0,
                "ax_mlx_kv_compression_production_blockers": 2,
                "ax_mlx_kv_compression_preset": 1,
                "ax_mlx_kv_compression_key_bits": 8,
                "ax_mlx_kv_compression_value_bits": 4,
                "ax_mlx_kv_compression_eligible_layers": 20,
                "ax_mlx_kv_compression_candidate_token_layers": 120000,
                "ax_mlx_kv_compression_estimated_saved_kib": 4096,
                "ax_mlx_kv_compression_runtime_storage_written_slots": 5000,
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
            "public_support_docs_approved": False,
        },
    }


def benchmark_doc(*, compressed: bool, decode_path: str | None = None) -> dict:
    row = {
        "engine": "ax_engine_mlx",
        "prompt_tokens": 8192,
        "generation_tokens": 256,
        "prompt_token_ids_sha256": SHA,
        "decode_tok_s": {"median": 100.0 if not compressed else 90.0},
    }
    if compressed:
        row.update(
            {
                "experimental_mlx_kv_compression": "turboquant-shadow",
                "kv_compression_decode_path": decode_path or "fused_compressed_decode",
                "kv_compression_telemetry": {
                    "ax_mlx_kv_compression_route_metadata_schema": 1,
                    "ax_mlx_kv_compression_production_ready": 0,
                    "ax_mlx_kv_compression_production_blockers": 2,
                    "ax_mlx_kv_compression_preset": 1,
                    "ax_mlx_kv_compression_key_bits": 8,
                    "ax_mlx_kv_compression_value_bits": 4,
                    "ax_mlx_kv_compression_eligible_layers": 20,
                    "ax_mlx_kv_compression_candidate_token_layers": 120000,
                    "ax_mlx_kv_compression_estimated_saved_kib": 4096,
                    "ax_mlx_kv_compression_runtime_storage_written_slots": 5000,
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

    def test_valid_artifact_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checker.validate_artifact(valid_artifact(root), root=root)

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

    def test_missing_runtime_storage_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = valid_artifact(root)
            del artifact["route_metadata"]["crossover_decisions"][
                "ax_mlx_kv_compression_runtime_storage_written_slots"
            ]
            with self.assertRaisesRegex(checker.ArtifactValidationError, "missing required keys"):
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
            checker.validate_artifact(artifact, root=root)

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
