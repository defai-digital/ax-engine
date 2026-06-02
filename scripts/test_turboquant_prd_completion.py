#!/usr/bin/env python3
"""Unit tests for TurboQuant PRD completion reporting."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_turboquant_prd_completion.py")
MODULE_SPEC = importlib.util.spec_from_file_location("check_turboquant_prd_completion", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(checker)

QUALITY_TEST_PATH = Path(__file__).with_name("test_turboquant_quality_artifact.py")
QUALITY_TEST_SPEC = importlib.util.spec_from_file_location(
    "test_turboquant_quality_artifact",
    QUALITY_TEST_PATH,
)
assert QUALITY_TEST_SPEC and QUALITY_TEST_SPEC.loader
quality_fixtures = importlib.util.module_from_spec(QUALITY_TEST_SPEC)
QUALITY_TEST_SPEC.loader.exec_module(quality_fixtures)

MICROBENCH_TEST_PATH = Path(__file__).with_name("test_turboquant_microbench_artifact.py")
MICROBENCH_TEST_SPEC = importlib.util.spec_from_file_location(
    "test_turboquant_microbench_artifact",
    MICROBENCH_TEST_PATH,
)
assert MICROBENCH_TEST_SPEC and MICROBENCH_TEST_SPEC.loader
microbench_fixtures = importlib.util.module_from_spec(MICROBENCH_TEST_SPEC)
MICROBENCH_TEST_SPEC.loader.exec_module(microbench_fixtures)


def write_eligible_manifest(models_root: Path) -> None:
    model_dir = models_root / "qwen-test"
    model_dir.mkdir(parents=True)
    (model_dir / "model-manifest.json").write_text(
        json.dumps(
            {
                "model_family": "qwen3",
                "attention_head_dim": 128,
                "attention_head_count": 8,
                "kv_head_count": 2,
                "layer_types": ["full_attention", "full_attention"],
            }
        )
    )


def write_short_decode_artifact(path: Path, *, speedup: float = 2.1) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "ax.turboquant_short_decode_speedup.v1",
                "new_tokens": 1,
                "cold_tokens": 1024,
                "short_decode_speedup": speedup,
            }
        )
    )


class TurboQuantPrdCompletionTests(unittest.TestCase):
    def test_completion_report_passes_when_all_evidence_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            results_root = root / "results"
            results_root.mkdir()
            write_eligible_manifest(models_root)

            quality_path = results_root / "quality-gate.json"
            quality_path.write_text(json.dumps(quality_fixtures.valid_artifact(root)))
            microbench_path = results_root / "microbench.json"
            microbench_path.write_text(json.dumps(microbench_fixtures.microbench_artifact()))
            short_decode_path = results_root / "short-decode.json"
            write_short_decode_artifact(short_decode_path)

            report = checker.build_report(
                models_root=models_root,
                results_root=results_root,
                quality_artifacts=[quality_path],
                microbench_artifacts=[microbench_path],
                short_decode_artifacts=[short_decode_path],
                required_model_families=["qwen3"],
                require_artifact_files=False,
                min_microbench_cold_tokens=8192,
                d3_head_dim=128,
                min_d3_speedup_vs_dim=1.5,
                min_d4_speedup=2.0,
            )

        self.assertTrue(report["decision"]["prd_complete"])
        self.assertEqual(report["decision"]["blockers"], [])

    def test_completion_report_fails_closed_without_short_decode_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            results_root = root / "results"
            results_root.mkdir()
            write_eligible_manifest(models_root)

            quality_path = results_root / "quality-gate.json"
            quality_path.write_text(json.dumps(quality_fixtures.valid_artifact(root)))
            microbench_path = results_root / "microbench.json"
            microbench_path.write_text(json.dumps(microbench_fixtures.microbench_artifact()))

            report = checker.build_report(
                models_root=models_root,
                results_root=results_root,
                quality_artifacts=[quality_path],
                microbench_artifacts=[microbench_path],
                short_decode_artifacts=[],
                required_model_families=["qwen3"],
                require_artifact_files=False,
                min_microbench_cold_tokens=8192,
                d3_head_dim=128,
                min_d3_speedup_vs_dim=1.5,
                min_d4_speedup=2.0,
            )

        self.assertFalse(report["decision"]["prd_complete"])
        self.assertIn("missing D4 short decode speedup evidence", report["decision"]["blockers"][0])

    def test_completion_report_requires_dim_parallel_for_d3_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            results_root = root / "results"
            results_root.mkdir()
            write_eligible_manifest(models_root)

            quality_path = results_root / "quality-gate.json"
            quality_path.write_text(json.dumps(quality_fixtures.valid_artifact(root)))
            microbench = deepcopy(microbench_fixtures.microbench_artifact())
            microbench["rows"][1]["kernel_variants"] = [
                variant
                for variant in microbench["rows"][1]["kernel_variants"]
                if variant["name"] != "dim_parallel"
            ]
            microbench_path = results_root / "microbench.json"
            microbench_path.write_text(json.dumps(microbench))
            short_decode_path = results_root / "short-decode.json"
            write_short_decode_artifact(short_decode_path)

            report = checker.build_report(
                models_root=models_root,
                results_root=results_root,
                quality_artifacts=[quality_path],
                microbench_artifacts=[microbench_path],
                short_decode_artifacts=[short_decode_path],
                required_model_families=["qwen3"],
                require_artifact_files=False,
                min_microbench_cold_tokens=8192,
                d3_head_dim=128,
                min_d3_speedup_vs_dim=1.5,
                min_d4_speedup=2.0,
            )

        self.assertFalse(report["decision"]["prd_complete"])
        self.assertTrue(
            any(
                "missing D3 fused decode microbench evidence" in blocker
                for blocker in report["decision"]["blockers"]
            )
        )
        self.assertEqual(
            report["evidence"]["microbench_artifacts"][0]["blocker"],
            "D3 evidence must include dim_parallel comparison variant",
        )

    def test_completion_report_validates_selected_d3_row_not_largest_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            results_root = root / "results"
            results_root.mkdir()
            write_eligible_manifest(models_root)

            quality_path = results_root / "quality-gate.json"
            quality_path.write_text(json.dumps(quality_fixtures.valid_artifact(root)))
            microbench = deepcopy(microbench_fixtures.microbench_artifact())
            microbench["rows"][1]["kernel_variants"] = [
                microbench_fixtures.variant("dim_parallel", median_us=1200),
                microbench_fixtures.variant("two_stage_scores", median_us=1000),
            ]
            larger_non_d3_row = deepcopy(microbench["rows"][1])
            larger_non_d3_row["cold_tokens"] = 16384
            larger_non_d3_row["head_dim"] = 512
            larger_non_d3_row["kernel_variants"] = [
                microbench_fixtures.variant("dim_parallel", median_us=3000),
                microbench_fixtures.variant("two_stage_scores", median_us=1000),
            ]
            microbench["rows"].append(larger_non_d3_row)
            microbench_path = results_root / "microbench.json"
            microbench_path.write_text(json.dumps(microbench))
            short_decode_path = results_root / "short-decode.json"
            write_short_decode_artifact(short_decode_path)

            report = checker.build_report(
                models_root=models_root,
                results_root=results_root,
                quality_artifacts=[quality_path],
                microbench_artifacts=[microbench_path],
                short_decode_artifacts=[short_decode_path],
                required_model_families=["qwen3"],
                require_artifact_files=False,
                min_microbench_cold_tokens=8192,
                d3_head_dim=128,
                min_d3_speedup_vs_dim=1.5,
                min_d4_speedup=2.0,
            )

        self.assertFalse(report["decision"]["prd_complete"])
        self.assertEqual(
            report["evidence"]["microbench_artifacts"][0]["blocker"],
            "two_stage_scores speedup versus dim_parallel is below 1.5",
        )

    def test_completion_report_accepts_any_passing_d3_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            results_root = root / "results"
            results_root.mkdir()
            write_eligible_manifest(models_root)

            quality_path = results_root / "quality-gate.json"
            quality_path.write_text(json.dumps(quality_fixtures.valid_artifact(root)))
            microbench = deepcopy(microbench_fixtures.microbench_artifact())
            larger_failing_d3_row = deepcopy(microbench["rows"][1])
            larger_failing_d3_row["cold_tokens"] = 16384
            larger_failing_d3_row["kernel_variants"] = [
                microbench_fixtures.variant("dim_parallel", median_us=1200),
                microbench_fixtures.variant("two_stage_scores", median_us=1000),
            ]
            microbench["rows"].append(larger_failing_d3_row)
            microbench_path = results_root / "microbench.json"
            microbench_path.write_text(json.dumps(microbench))
            short_decode_path = results_root / "short-decode.json"
            write_short_decode_artifact(short_decode_path)

            report = checker.build_report(
                models_root=models_root,
                results_root=results_root,
                quality_artifacts=[quality_path],
                microbench_artifacts=[microbench_path],
                short_decode_artifacts=[short_decode_path],
                required_model_families=["qwen3"],
                require_artifact_files=False,
                min_microbench_cold_tokens=8192,
                d3_head_dim=128,
                min_d3_speedup_vs_dim=1.5,
                min_d4_speedup=2.0,
            )

        self.assertTrue(report["decision"]["prd_complete"])
        self.assertTrue(report["evidence"]["microbench_artifacts"][0]["passes_gate"])


if __name__ == "__main__":
    unittest.main()
