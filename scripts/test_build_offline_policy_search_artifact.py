#!/usr/bin/env python3
"""Unit tests for building offline policy-search artifacts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("build_offline_policy_search_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "build_offline_policy_search_artifact", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
builder = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = builder
MODULE_SPEC.loader.exec_module(builder)

CHECKER_PATH = Path(__file__).with_name("check_offline_policy_search_artifact.py")
CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_offline_policy_search_artifact", CHECKER_PATH
)
assert CHECKER_SPEC and CHECKER_SPEC.loader
checker = importlib.util.module_from_spec(CHECKER_SPEC)
sys.modules[CHECKER_SPEC.name] = checker
CHECKER_SPEC.loader.exec_module(checker)


def metadata() -> dict[str, object]:
    return {
        "target": "turboquant_kv_policy",
        "status": "diagnostic_only",
        "created_at": "2026-05-14T00:00:00Z",
        "repo": {
            "commit": "abc1234",
            "dirty": False,
        },
        "model": {
            "id": "gemma-4-e2b-it-4bit",
            "family": "gemma4",
            "artifacts_dir": ".internal/models/gemma-4-e2b-it-4bit",
            "manifest_digest": "sha256:test",
        },
        "route": {
            "selected_backend": "mlx",
            "support_tier": "repo_owned_runtime",
        },
        "space": {
            "kv_preset": ["disabled", "TurboQuantK8V4"],
            "hot_window_tokens": [256],
        },
    }


def baseline() -> dict[str, object]:
    return {
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
    }


def candidate(policy_id: str = "k8v4-hot-256") -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "prompt_tokens": 8192,
        "generation_tokens": 256,
        "seed": 42,
        "policy": {
            "kv_preset": "TurboQuantK8V4",
            "hot_window_tokens": 256,
            "eligible_layer_mask": "full_attention_only",
        },
        "route": {
            "selected_backend": "mlx",
            "support_tier": "repo_owned_runtime",
        },
        "metrics": {
            "decode_tok_s": 110.0,
            "ttft_ms": 118.0,
            "kv_saved_bytes": 1048576,
        },
        "route_metadata": {
            "ax_mlx_kv_compression_decode_path": "fused_compressed_decode",
        },
        "fallback_count": 0,
        "fallback_tokens": 0,
        "quality_gate_passed": True,
        "deterministic_replay_passed": True,
    }


def confirmation_evidence(
    classification_hint: str = "candidate_win_needs_repeat",
) -> dict[str, object]:
    return {
        "baseline_policy_id": "disabled",
        "candidate_policy_id": "k8v4-hot-256",
        "repeated_measurements": {
            "runs": 3,
            "cooldown_seconds": 20,
        },
        "decision_metric": "decode_tok_s",
        "baseline_median": 100.0,
        "candidate_median": 110.0,
        "relative_delta": 0.10,
        "noise_band": 0.03,
        "classification_hint": classification_hint,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


class BuildOfflinePolicySearchArtifactTests(unittest.TestCase):
    def test_builds_valid_minimal_artifact(self) -> None:
        artifact = builder.build_offline_policy_search_artifact(
            metadata=metadata(),
            baseline=baseline(),
            candidates=[candidate()],
            decision_classification="diagnostic_only",
            decision_reason="diagnostic evidence only",
        )

        self.assertEqual(artifact["schema"], checker.SCHEMA_VERSION)
        self.assertEqual(artifact["best_candidate"]["policy_id"], "k8v4-hot-256")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            write_json(path, artifact)
            self.assertEqual(
                checker.validate_offline_policy_search_artifact(path),
                "diagnostic_only",
            )

    def test_builds_valid_candidate_win_with_confirmation_evidence(self) -> None:
        search_metadata = metadata()
        search_metadata["confirmation_evidence"] = confirmation_evidence()

        artifact = builder.build_offline_policy_search_artifact(
            metadata=search_metadata,
            baseline=baseline(),
            candidates=[candidate()],
            decision_classification="candidate_win_needs_repeat",
            decision_reason="candidate win requires repeated measurement review",
        )

        self.assertIn("confirmation_evidence", artifact)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            write_json(path, artifact)
            self.assertEqual(
                checker.validate_offline_policy_search_artifact(path),
                "candidate_win_needs_repeat",
            )

    def test_best_policy_id_required_for_multiple_candidates(self) -> None:
        with self.assertRaisesRegex(builder.OfflinePolicySearchBuildError, "best-policy-id"):
            builder.build_offline_policy_search_artifact(
                metadata=metadata(),
                baseline=baseline(),
                candidates=[candidate("first"), candidate("second")],
                decision_classification="diagnostic_only",
                decision_reason="diagnostic evidence only",
            )

    def test_best_policy_id_must_match_candidate(self) -> None:
        with self.assertRaisesRegex(builder.OfflinePolicySearchBuildError, "does not match"):
            builder.build_offline_policy_search_artifact(
                metadata=metadata(),
                baseline=baseline(),
                candidates=[candidate("first")],
                best_policy_id="missing",
                decision_classification="diagnostic_only",
                decision_reason="diagnostic evidence only",
            )

    def test_cli_writes_valid_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "metadata.json"
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            output_path = root / "artifact.json"
            write_json(metadata_path, metadata())
            write_json(baseline_path, baseline())
            write_json(candidate_path, candidate())

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--metadata",
                    str(metadata_path),
                    "--baseline",
                    str(baseline_path),
                    "--candidate",
                    str(candidate_path),
                    "--output",
                    str(output_path),
                    "--skip-git-repo-metadata",
                ],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output_path.exists())
            self.assertEqual(
                checker.validate_offline_policy_search_artifact(output_path),
                "diagnostic_only",
            )

    def test_cli_writes_candidate_win_with_confirmation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_payload = metadata()
            metadata_payload["confirmation_evidence"] = confirmation_evidence()
            metadata_path = root / "metadata.json"
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            output_path = root / "artifact.json"
            write_json(metadata_path, metadata_payload)
            write_json(baseline_path, baseline())
            write_json(candidate_path, candidate())

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--metadata",
                    str(metadata_path),
                    "--baseline",
                    str(baseline_path),
                    "--candidate",
                    str(candidate_path),
                    "--output",
                    str(output_path),
                    "--decision-classification",
                    "candidate_win_needs_repeat",
                    "--decision-reason",
                    "candidate win requires repeated measurement review",
                    "--skip-git-repo-metadata",
                ],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output_path.exists())
            self.assertEqual(
                checker.validate_offline_policy_search_artifact(output_path),
                "candidate_win_needs_repeat",
            )

    def test_cli_rejects_dirty_repo_without_allow_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            (root / "tracked.txt").write_text("v1\n")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-m",
                    "init",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            (root / "tracked.txt").write_text("v2\n")

            with self.assertRaisesRegex(builder.OfflinePolicySearchBuildError, "dirty"):
                builder.default_repo_metadata(allow_dirty=False, root=root)

            repo = builder.default_repo_metadata(allow_dirty=True, root=root)
            self.assertTrue(repo["dirty"])
            self.assertIn("tracked.txt", repo["changed_files"])


if __name__ == "__main__":
    unittest.main()
