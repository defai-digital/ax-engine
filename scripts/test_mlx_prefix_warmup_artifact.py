#!/usr/bin/env python3
"""Unit tests for MLX prefix miss/warmup correctness artifact checks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_mlx_prefix_warmup_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mlx_prefix_warmup_artifact", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


PROMPT_HASH = "a" * 64
OUTPUT_HASH = "b" * 64


def valid_observation() -> dict[str, object]:
    return {
        "request_id": "request-b",
        "prompt_token_ids_sha256": PROMPT_HASH,
        "prompt_digest_kind": "token_ids",
        "route": {
            "selected_backend": "mlx",
            "route_identity": "repo_owned_mlx",
        },
        "logical_prefix_reuse": {
            "matched_token_count": 256,
            "reused_block_count": 4,
        },
        "physical_prefix_snapshot": {
            "hit_count": 0,
            "miss_count": 1,
            "warmup_token_count": 256,
            "reused_token_count": 0,
            "blocked_count": 0,
            "physical_snapshot_coverage": "miss_warmup_only",
        },
        "correctness": {
            "status": "passed",
            "deterministic_replay": True,
            "output_token_ids_sha256": OUTPUT_HASH,
        },
    }


def valid_artifact() -> dict[str, object]:
    return {
        "schema_version": checker.SCHEMA_VERSION,
        "claim_scope": "physical_prefix_miss_warmup_correctness",
        "model": {
            "id": "mlx-community/Qwen3.5-9B-4bit",
            "quantization": "4-bit",
        },
        "host": {
            "chip": "Apple M5 Max",
            "memory_gb": 128,
            "os": "macOS 26.4.1",
        },
        "benchmark": {
            "shared_prefix_tokens": 256,
            "generation_tokens": 16,
            "repetitions": 1,
        },
        "observations": [valid_observation()],
    }


class PrefixWarmupArtifactTests(unittest.TestCase):
    def write_fixture(self, artifact: dict[str, object]) -> Path:
        root = Path(tempfile.mkdtemp())
        path = root / "prefix-warmup.json"
        path.write_text(json.dumps(artifact, indent=2) + "\n")
        self.addCleanup(lambda: root.rmdir())
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_valid_artifact_passes(self) -> None:
        path = self.write_fixture(valid_artifact())

        checked = checker.validate_prefix_warmup_artifact(path)

        self.assertEqual(checked, ["request-b:matched=256:warmup=256"])

    def test_requires_physical_miss(self) -> None:
        artifact = valid_artifact()
        artifact["observations"][0]["physical_prefix_snapshot"]["miss_count"] = 0
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.PrefixWarmupArtifactError, "miss_count"):
            checker.validate_prefix_warmup_artifact(path)

    def test_accepts_prompt_ref_digest(self) -> None:
        artifact = valid_artifact()
        observation = artifact["observations"][0]
        del observation["prompt_token_ids_sha256"]
        observation["prompt_ref_sha256"] = PROMPT_HASH
        observation["prompt_digest_kind"] = "prompt_ref_bytes"
        path = self.write_fixture(artifact)

        checked = checker.validate_prefix_warmup_artifact(path)

        self.assertEqual(checked, ["request-b:matched=256:warmup=256"])

    def test_requires_one_prompt_digest(self) -> None:
        artifact = valid_artifact()
        artifact["observations"][0]["prompt_ref_sha256"] = PROMPT_HASH
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(
            checker.PrefixWarmupArtifactError,
            "exactly one prompt digest",
        ):
            checker.validate_prefix_warmup_artifact(path)

    def test_rejects_physical_hit_evidence(self) -> None:
        artifact = valid_artifact()
        artifact["observations"][0]["physical_prefix_snapshot"]["hit_count"] = 1
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.PrefixWarmupArtifactError, "hit_count"):
            checker.validate_prefix_warmup_artifact(path)

    def test_requires_correctness_pass(self) -> None:
        artifact = valid_artifact()
        artifact["observations"][0]["correctness"]["status"] = "failed"
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(checker.PrefixWarmupArtifactError, "status"):
            checker.validate_prefix_warmup_artifact(path)

    def test_requires_deterministic_replay(self) -> None:
        artifact = valid_artifact()
        artifact["observations"][0]["correctness"]["deterministic_replay"] = False
        path = self.write_fixture(artifact)

        with self.assertRaisesRegex(
            checker.PrefixWarmupArtifactError,
            "deterministic_replay",
        ):
            checker.validate_prefix_warmup_artifact(path)

    def test_cli_reports_checked_observations(self) -> None:
        path = self.write_fixture(valid_artifact())

        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("ax.mlx_prefix_warmup.v1", completed.stdout)
        self.assertIn("request-b:matched=256:warmup=256", completed.stdout)


if __name__ == "__main__":
    unittest.main()
