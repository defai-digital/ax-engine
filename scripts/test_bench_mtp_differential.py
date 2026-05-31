#!/usr/bin/env python3
"""Tests for the MTP differential benchmark artifact builder."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("bench_mtp_differential.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_mtp_differential", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
diff = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bench_mtp_differential"] = diff
MODULE_SPEC.loader.exec_module(diff)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


class MtpDifferentialTests(unittest.TestCase):
    def test_first_diff_index_handles_equal_prefix_and_length_mismatch(self) -> None:
        self.assertIsNone(diff.first_diff_index([1, 2, 3], [1, 2, 3]))
        self.assertEqual(diff.first_diff_index([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(diff.first_diff_index([1, 2], [1, 2, 3]), 2)
        self.assertIsNone(diff.first_diff_index(None, [1, 2, 3]))

    def test_harness_contract_id_is_stable_for_same_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = write_jsonl(root / "flappy.jsonl")
            config = diff.RunConfig(
                mode="sampled",
                depth=3,
                max_tokens=1000,
                repetitions=5,
                warmup_repetitions=1,
                cooldown_s=15.0,
                sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                enable_thinking=False,
            )
            first = diff.build_contract(config, ["flappy"], {"flappy": suite})
            second = diff.build_contract(config, ["flappy"], {"flappy": suite})

        self.assertEqual(diff.sha256_json(first), diff.sha256_json(second))

    def test_build_differential_artifact_from_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = write_jsonl(root / "flappy.jsonl")
            ax_path = write_json(root / "ax.json", fake_ax_artifact())
            mtplx_path = write_json(root / "mtplx.json", fake_mtplx_artifact())
            config = diff.RunConfig(
                mode="greedy",
                depth=3,
                max_tokens=4,
                repetitions=2,
                warmup_repetitions=1,
                cooldown_s=1.0,
                sampling={"temperature": 0.0, "top_p": 1.0, "top_k": 0},
                enable_thinking=False,
            )

            artifact = diff.build_differential_artifact(
                config=config,
                suites=["flappy"],
                suite_files={"flappy": suite},
                ax_artifacts={"flappy": ax_path},
                mtplx_artifacts={"flappy": mtplx_path},
            )

        self.assertEqual(artifact["schema"], "ax.mtp_differential.v1")
        self.assertEqual(artifact["summary"]["ax_wins"], 1)
        suite_result = artifact["suites"][0]
        self.assertEqual(suite_result["summary"]["matched_case_count"], 1)
        self.assertAlmostEqual(suite_result["summary"]["speedup_ratio"], 1.25)
        self.assertEqual(suite_result["cases"][0]["token_diff_first_index"], 2)
        self.assertEqual(suite_result["cases"][0]["ax"]["acceptance_by_depth"], [1.0, 1.0, 0.5])
        self.assertEqual(suite_result["cases"][0]["mtplx"]["acceptance_by_depth"], [1.0, 1.0, 0.5])

    def test_dirty_and_repetition_mismatch_emit_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = write_jsonl(root / "flappy.jsonl")
            ax = fake_ax_artifact()
            ax["build"]["git_tracked_dirty"] = True
            ax["repetitions"] = 3
            ax_path = write_json(root / "ax.json", ax)
            mtplx_path = write_json(root / "mtplx.json", fake_mtplx_artifact())
            config = diff.RunConfig(
                mode="sampled",
                depth=3,
                max_tokens=4,
                repetitions=2,
                warmup_repetitions=1,
                cooldown_s=1.0,
                sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                enable_thinking=False,
            )

            artifact = diff.build_differential_artifact(
                config=config,
                suites=["flappy"],
                suite_files={"flappy": suite},
                ax_artifacts={"flappy": ax_path},
                mtplx_artifacts={"flappy": mtplx_path},
            )

        joined = "\n".join(artifact["warnings"])
        self.assertIn("dirty tracked worktree", joined)
        self.assertIn("AX repetitions=3", joined)

    def test_model_provenance_summarizes_sidecar_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "ax_mtp_sidecar_manifest.json"
            manifest_path.write_text(json.dumps(fake_sidecar_manifest(), sort_keys=True) + "\n")

            summary = diff.model_provenance_for(root)

        self.assertEqual(summary["kind"], "ax_mtp_sidecar_manifest")
        self.assertEqual(summary["base_model_id"], "mlx-community/Qwen3.6-27B-4bit")
        self.assertEqual(summary["source_model_id"], "Qwen/Qwen3.6-27B")
        self.assertEqual(summary["source_shard_count"], 2)
        self.assertEqual(summary["output_mtp_sha256"], "m" * 64)


def write_jsonl(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "id": "case_1",
                "category": "coding",
                "prompt": "Write a small loop.",
                "max_tokens": 4,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return path


def fake_ax_artifact() -> dict:
    return {
        "schema_version": "ax.mlx_inference_stack.v2",
        "build": {"git_tracked_dirty": False, "commit": "abc123"},
        "repetitions": 2,
        "cooldown": 1.0,
        "generation_tokens": 4,
        "ax_mtp_max_depth": 3,
        "results": [
            {
                "engine": "ax_engine_mlx_ngram_accel",
                "prompt_case_id": "case_1",
                "prompt_category": "coding",
                "prompt_text_sha256": "prompt-hash",
                "prompt_token_ids_sha256": "token-hash",
                "decode_tok_s": {"median": 10.0},
                "client_wall_total_ms": {"median": 100.0},
                "ngram_acceleration_telemetry": {
                    "ax_mtp_draft_tokens": 6,
                    "ax_mtp_accepted_tokens": 5,
                    "ax_mtp_drafted_depth0": 2,
                    "ax_mtp_drafted_depth1": 2,
                    "ax_mtp_drafted_depth2": 2,
                    "ax_mtp_accepted_depth0": 2,
                    "ax_mtp_accepted_depth1": 2,
                    "ax_mtp_accepted_depth2": 1,
                    "ax_mtp_ngram_hit_steps": 0,
                },
                "trials": [{"output_token_ids": [1, 2, 3]}],
            }
        ],
    }


def fake_mtplx_artifact() -> dict:
    return {
        "schema": "ax.mtplx.prompt_suite_mtp.v1",
        "engine": "mtplx",
        "build": {"git_tracked_dirty": False},
        "repetitions": 2,
        "warmup_repetitions": 1,
        "cooldown_s": 1.0,
        "depth": 3,
        "max_tokens": 4,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0},
        "results": [
            {
                "prompt_id": "case_1",
                "category": "coding",
                "prompt_sha256": "prompt-hash",
                "prompt_tokens": 12,
                "summary": {
                    "decode_tok_s": {"median": 8.0},
                    "accepted_drafts": 5,
                    "drafted_tokens": 6,
                    "accept_rate": 5 / 6,
                },
                "runs": [
                    {
                        "tokens": [1, 2, 4],
                        "accepted_by_depth": [2, 2, 1],
                        "drafted_by_depth": [2, 2, 2],
                    }
                ],
            }
        ],
    }


def fake_sidecar_manifest() -> dict:
    return {
        "schema_version": "ax.mtp_sidecar_provenance.v1",
        "model_key": "27b",
        "base": {
            "model_id": "mlx-community/Qwen3.6-27B-4bit",
            "snapshot": "abc",
        },
        "source": {
            "model_id": "Qwen/Qwen3.6-27B",
            "mtp_shards": [
                {"name": "a.safetensors", "sha256": "a" * 64},
                {"name": "b.safetensors", "sha256": "b" * 64},
            ],
        },
        "output": {
            "mtp": {
                "sha256": "m" * 64,
                "size_bytes": 123,
            }
        },
        "transform": {
            "norm_policy": "shift_mtp_norm_weights_by_1",
        },
        "runtime": {
            "arch_id": "qwen3-next-mtp",
            "mtplx_version": "0.3.7",
            "mtp_depth_max": 3,
            "mtp_tensor_count": 27,
            "recommended_draft_sampler": {"temperature": 0.7, "top_p": 0.95, "top_k": 20},
            "sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "exactness_baseline": {"context": 2048, "max_abs_diff": 0.0},
            "verified_on": {"system": "Darwin", "machine": "arm64"},
        },
    }


if __name__ == "__main__":
    unittest.main()
