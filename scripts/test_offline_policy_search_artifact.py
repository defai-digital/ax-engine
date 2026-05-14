#!/usr/bin/env python3
"""Unit tests for offline policy-search artifact validation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_offline_policy_search_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_offline_policy_search_artifact", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def candidate(
    *,
    policy_id: str = "k8v4-hot-256",
    selected_backend: str = "mlx",
    fallback_count: int | None = 0,
    fallback_tokens: int | None = 0,
    quality_gate_passed: bool | None = True,
    deterministic_replay_passed: bool | None = True,
    prompt_tokens: int = 8192,
    generation_tokens: int = 256,
) -> dict[str, object]:
    row: dict[str, object] = {
        "policy_id": policy_id,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "seed": 42,
        "policy": {
            "kv_preset": "TurboQuantK8V4",
            "hot_window_tokens": 256,
            "eligible_layer_mask": "full_attention_only",
        },
        "route": {
            "selected_backend": selected_backend,
            "support_tier": "repo_owned_runtime",
        },
        "metrics": {
            "decode_tok_s": 110.0,
            "ttft_ms": 120.0,
            "kv_saved_bytes": 1048576,
        },
        "route_metadata": {
            "ax_mlx_kv_compression_decode_path": "fused_compressed_decode",
        },
    }
    if quality_gate_passed is not None:
        row["quality_gate_passed"] = quality_gate_passed
    if deterministic_replay_passed is not None:
        row["deterministic_replay_passed"] = deterministic_replay_passed
    if fallback_count is not None:
        row["fallback_count"] = fallback_count
    if fallback_tokens is not None:
        row["fallback_tokens"] = fallback_tokens
    return row


def artifact(
    *,
    selected_backend: str = "mlx",
    baseline: dict[str, object] | None = None,
    candidates: list[dict[str, object]] | None = None,
    decision_classification: str = "diagnostic_only",
    promotion_evidence: dict[str, object] | None = None,
    confirmation_evidence: dict[str, object] | None = None,
    dirty: bool = False,
    changed_files: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": checker.SCHEMA_VERSION,
        "target": "turboquant_kv_policy",
        "status": "diagnostic_only",
        "created_at": "2026-05-14T00:00:00Z",
        "repo": {
            "commit": "abc1234",
            "dirty": dirty,
        },
        "model": {
            "id": "gemma-4-e2b-it-4bit",
            "family": "gemma4",
            "artifacts_dir": ".internal/models/gemma-4-e2b-it-4bit",
            "manifest_digest": "sha256:test",
        },
        "route": {
            "selected_backend": selected_backend,
            "support_tier": "repo_owned_runtime",
        },
        "search": {
            "algorithm": "grid",
            "seed": 42,
            "budget": {
                "max_candidates": 8,
                "max_wall_time_seconds": 3600,
            },
            "space": {
                "kv_preset": ["disabled", "TurboQuantK8V4"],
                "hot_window_tokens": [256, 512],
            },
        },
        "objective": {
            "maximize": ["decode_tok_s", "kv_saved_bytes"],
            "minimize": ["ttft_ms", "fallback_tokens"],
            "hard_constraints": [
                "quality_gate_pass",
                "deterministic_replay_pass",
                "selected_backend_mlx",
            ],
        },
        "baseline": baseline
        if baseline is not None
        else {
            "policy_id": "disabled",
            "route": {
                "selected_backend": "mlx",
                "support_tier": "repo_owned_runtime",
            },
            "prompt_tokens": 8192,
            "generation_tokens": 256,
            "metrics": {
                "decode_tok_s": 100.0,
                "ttft_ms": 120.0,
            },
        },
        "candidates": candidates if candidates is not None else [candidate()],
        "best_candidate": {
            "policy_id": "k8v4-hot-256",
        },
        "decision": {
            "classification": decision_classification,
            "reason": "diagnostic evidence only",
        },
    }
    if dirty:
        payload["repo"] = {
            "commit": "abc1234",
            "dirty": True,
            **({"changed_files": changed_files} if changed_files is not None else {}),
        }
    if promotion_evidence is not None:
        payload["promotion_evidence"] = promotion_evidence
    if confirmation_evidence is not None:
        payload["confirmation_evidence"] = confirmation_evidence
    return payload


def confirmation_evidence(
    *,
    classification_hint: str = "candidate_win_needs_repeat",
    candidate_policy_id: str = "k8v4-hot-256",
    runs: int = 3,
    cooldown_seconds: int = 20,
) -> dict[str, object]:
    return {
        "baseline_policy_id": "disabled",
        "candidate_policy_id": candidate_policy_id,
        "repeated_measurements": {
            "runs": runs,
            "cooldown_seconds": cooldown_seconds,
        },
        "decision_metric": "decode_tok_s",
        "baseline_median": 100.0,
        "candidate_median": 112.0,
        "relative_delta": 0.12,
        "noise_band": 0.03,
        "classification_hint": classification_hint,
    }


class OfflinePolicySearchArtifactTests(unittest.TestCase):
    def write_fixture(self, payload: dict[str, object]) -> Path:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = Path(root.name) / "offline-policy-search.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def test_valid_diagnostic_artifact_passes(self) -> None:
        path = self.write_fixture(artifact())

        classification = checker.validate_offline_policy_search_artifact(path)

        self.assertEqual(classification, "diagnostic_only")

    def test_summary_counts_diagnostic_negative_and_promotion_review(self) -> None:
        diagnostic = self.write_fixture(artifact())
        negative = self.write_fixture(
            artifact(
                decision_classification="negative_result",
                confirmation_evidence=confirmation_evidence(
                    classification_hint="negative_result",
                ),
            )
        )
        promotion = self.write_fixture(
            artifact(
                decision_classification="promotion_ready_for_companion_prd_review",
                promotion_evidence={
                    "deterministic_replay_passed": True,
                    "quality_gate_passed": True,
                    "repeated_measurements": {
                        "runs": 3,
                        "cooldown_seconds": 20,
                    },
                    "companion_prd_gates": [
                        "TURBOQUANT-PROMOTION-PRD Gate B",
                        "TURBOQUANT-PROMOTION-PRD Gate C",
                    ],
                },
            )
        )

        result = checker.check_offline_policy_search_artifacts(
            [diagnostic, negative, promotion]
        )

        self.assertEqual(result.artifact_count, 3)
        self.assertEqual(result.diagnostic_count, 1)
        self.assertEqual(result.negative_count, 1)
        self.assertEqual(result.promotion_review_count, 1)

    def test_wrong_top_level_route_fails_closed(self) -> None:
        path = self.write_fixture(artifact(selected_backend="llama_cpp"))

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "route.selected_backend must be mlx",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_candidate_delegated_route_fails_closed(self) -> None:
        path = self.write_fixture(
            artifact(candidates=[candidate(selected_backend="mlx_lm_delegated")])
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            r"candidates\[0\]\.route\.selected_backend must be mlx",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_missing_baseline_fails_closed(self) -> None:
        payload = artifact()
        del payload["baseline"]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "baseline must be an object",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_missing_fallback_accounting_fails_closed(self) -> None:
        path = self.write_fixture(
            artifact(candidates=[candidate(fallback_count=None)])
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            r"candidates\[0\]\.fallback_count must be an integer",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_candidate_shape_must_match_baseline(self) -> None:
        path = self.write_fixture(
            artifact(candidates=[candidate(prompt_tokens=4096)])
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "shape must match baseline",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_search_space_must_define_dimensions(self) -> None:
        payload = artifact()
        payload["search"]["space"] = {}  # type: ignore[index]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "search.space must define at least one search dimension",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_search_space_dimension_must_be_non_empty_list(self) -> None:
        payload = artifact()
        payload["search"]["space"] = {"kv_preset": []}  # type: ignore[index]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "search.space.kv_preset must not be empty",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_search_space_dimension_values_must_be_unique_scalars(self) -> None:
        payload = artifact()
        payload["search"]["space"] = {"kv_preset": ["TurboQuantK8V4", "TurboQuantK8V4"]}  # type: ignore[index]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "duplicates an earlier value",
        ):
            checker.validate_offline_policy_search_artifact(path)

        payload = artifact()
        payload["search"]["space"] = {"kv_preset": [True]}  # type: ignore[index]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "must be a non-empty string or integer",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_duplicate_candidate_policy_ids_fail_closed(self) -> None:
        path = self.write_fixture(
            artifact(candidates=[candidate(policy_id="dup"), candidate(policy_id="dup")])
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "duplicates an earlier candidate",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_budget_must_cover_candidate_rows(self) -> None:
        payload = artifact(
            candidates=[candidate(policy_id="first"), candidate(policy_id="second")]
        )
        payload["search"]["budget"]["max_candidates"] = 1  # type: ignore[index]
        payload["best_candidate"]["policy_id"] = "first"  # type: ignore[index]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "max_candidates must be >= number of candidate rows",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_candidate_win_requires_confirmation_evidence(self) -> None:
        path = self.write_fixture(
            artifact(decision_classification="candidate_win_needs_repeat")
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "confirmation_evidence must be an object",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_candidate_win_with_confirmation_evidence_passes(self) -> None:
        path = self.write_fixture(
            artifact(
                decision_classification="candidate_win_needs_repeat",
                confirmation_evidence=confirmation_evidence(),
            )
        )

        self.assertEqual(
            checker.validate_offline_policy_search_artifact(path),
            "candidate_win_needs_repeat",
        )

    def test_non_diagnostic_decision_requires_candidate_quality_evidence(self) -> None:
        path = self.write_fixture(
            artifact(
                candidates=[candidate(quality_gate_passed=None)],
                decision_classification="candidate_win_needs_repeat",
                confirmation_evidence=confirmation_evidence(),
            )
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            r"candidates\[0\]\.quality_gate_passed must be a boolean",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_non_diagnostic_decision_requires_candidate_replay_evidence(self) -> None:
        path = self.write_fixture(
            artifact(
                candidates=[candidate(deterministic_replay_passed=None)],
                decision_classification="negative_result",
                confirmation_evidence=confirmation_evidence(classification_hint="negative_result"),
            )
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            r"candidates\[0\]\.deterministic_replay_passed must be a boolean",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_rejected_quality_can_record_failed_quality_gate(self) -> None:
        path = self.write_fixture(
            artifact(
                candidates=[candidate(quality_gate_passed=False)],
                decision_classification="rejected_quality",
            )
        )

        self.assertEqual(
            checker.validate_offline_policy_search_artifact(path),
            "rejected_quality",
        )

    def test_confirmation_evidence_requires_repeated_measurements(self) -> None:
        path = self.write_fixture(
            artifact(
                decision_classification="candidate_win_needs_repeat",
                confirmation_evidence=confirmation_evidence(runs=1),
            )
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "runs must be >= 2",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_confirmation_evidence_must_reference_candidate(self) -> None:
        path = self.write_fixture(
            artifact(
                decision_classification="candidate_win_needs_repeat",
                confirmation_evidence=confirmation_evidence(candidate_policy_id="missing"),
            )
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "candidate_policy_id must refer to a candidate row",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_confirmation_evidence_hint_must_match_decision(self) -> None:
        path = self.write_fixture(
            artifact(
                decision_classification="negative_result",
                confirmation_evidence=confirmation_evidence(
                    classification_hint="candidate_win_needs_repeat"
                ),
            )
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "classification_hint must match decision.classification",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_promotion_review_requires_evidence(self) -> None:
        path = self.write_fixture(
            artifact(decision_classification="promotion_ready_for_companion_prd_review")
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "promotion_evidence must be an object",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_promotion_review_requires_true_evidence_flags(self) -> None:
        path = self.write_fixture(
            artifact(
                decision_classification="promotion_ready_for_companion_prd_review",
                promotion_evidence={
                    "deterministic_replay_passed": False,
                    "quality_gate_passed": True,
                    "repeated_measurements": {
                        "runs": 3,
                        "cooldown_seconds": 20,
                    },
                    "companion_prd_gates": [
                        "TURBOQUANT-PROMOTION-PRD Gate B",
                    ],
                },
            )
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "deterministic_replay_passed must be true",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_promotion_review_requires_best_candidate_quality_flag(self) -> None:
        path = self.write_fixture(
            artifact(
                candidates=[candidate(quality_gate_passed=False)],
                decision_classification="promotion_ready_for_companion_prd_review",
                promotion_evidence={
                    "deterministic_replay_passed": True,
                    "quality_gate_passed": True,
                    "repeated_measurements": {
                        "runs": 3,
                        "cooldown_seconds": 20,
                    },
                    "companion_prd_gates": [
                        "TURBOQUANT-PROMOTION-PRD Gate B",
                    ],
                },
            )
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "quality gate",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_promotion_review_requires_zero_best_candidate_fallbacks(self) -> None:
        path = self.write_fixture(
            artifact(
                candidates=[candidate(fallback_count=1)],
                decision_classification="promotion_ready_for_companion_prd_review",
                promotion_evidence={
                    "deterministic_replay_passed": True,
                    "quality_gate_passed": True,
                    "repeated_measurements": {
                        "runs": 3,
                        "cooldown_seconds": 20,
                    },
                    "companion_prd_gates": [
                        "TURBOQUANT-PROMOTION-PRD Gate B",
                    ],
                },
            )
        )

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "zero fallbacks",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_dirty_repo_requires_changed_files(self) -> None:
        path = self.write_fixture(artifact(dirty=True))

        with self.assertRaisesRegex(
            checker.OfflinePolicySearchArtifactError,
            "repo.changed_files",
        ):
            checker.validate_offline_policy_search_artifact(path)

    def test_dirty_repo_with_changed_files_passes(self) -> None:
        path = self.write_fixture(
            artifact(dirty=True, changed_files=["scripts/example.py"])
        )

        self.assertEqual(
            checker.validate_offline_policy_search_artifact(path),
            "diagnostic_only",
        )

    def test_cli_reports_success(self) -> None:
        path = self.write_fixture(artifact())

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(path)],
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("artifact check passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
