#!/usr/bin/env python3
"""Tests for the fair Qwen3.6 MTP benchmark summary builder."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("bench_qwen36_mtp_fair.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "bench_qwen36_mtp_fair", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
fair = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bench_qwen36_mtp_fair"] = fair
MODULE_SPEC.loader.exec_module(fair)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


class Qwen36MtpFairTests(unittest.TestCase):
    def test_depth_policy_native_uses_profile_depth(self) -> None:
        profile = fair.QWEN36_PROFILES["27b-4bit"]
        self.assertEqual(
            fair.effective_depth(profile, ["mtplx", "ax_engine"], "native", None), 3
        )
        self.assertEqual(
            fair.effective_depth(profile, ["mtplx", "ax_engine"], "fair-shared", None),
            3,
        )
        self.assertEqual(
            fair.effective_depth(profile, ["mtplx", "ax_engine"], "native", 2), 2
        )

    def test_validate_sidecar_for_profile_accepts_current_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hf_cache = Path(tmp) / "hf"
            make_sidecar_manifest(hf_cache)
            summary = fair.validate_sidecar_for_profile(
                fair.QWEN36_PROFILES["27b-4bit"], hf_cache
            )

        self.assertEqual(summary["model_key"], "27b")
        self.assertEqual(summary["norm_policy"], "shift_mtp_norm_weights_by_1")

    def test_validate_sidecar_for_profile_rejects_old_norm_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hf_cache = Path(tmp) / "hf"
            make_sidecar_manifest(
                hf_cache,
                norm_policy="scale_selected_mtp_norm_weights_by_2",
            )

            with self.assertRaisesRegex(ValueError, "norm_policy"):
                fair.validate_sidecar_for_profile(
                    fair.QWEN36_PROFILES["27b-4bit"], hf_cache
                )

    def test_build_summary_includes_two_engines_and_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_cache = root / "hf"
            make_sidecar_manifest(hf_cache)
            artifacts = {
                ("27b-4bit", "flappy", "ax_engine"): write_json(
                    root / "ax.json", fake_ax_artifact()
                ),
                ("27b-4bit", "flappy", "mtplx"): write_json(
                    root / "mtplx.json", fake_mtplx_artifact()
                ),
            }
            args = Namespace(
                models=["27b-4bit"],
                engines=["mtplx", "ax_engine"],
                suites=["flappy"],
                depth_policy="native",
                depth=None,
                mode="sampled",
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=128,
                repetitions=1,
                warmup_repetitions=1,
                cooldown=0.0,
                hf_cache=hf_cache,
                pure_mtp=False,
            )

            summary = fair.build_summary(args, artifacts)

        self.assertEqual(summary["schema"], "ax.qwen36_mtp_fair.v1")
        self.assertTrue(summary["contract"]["ax_pure_mtp"])
        self.assertEqual(
            summary["contract"]["ax_engine_modes"], {"ax_engine": "pure_mtp"}
        )
        row = summary["rows"][0]
        self.assertEqual(row["depth"], 3)
        self.assertAlmostEqual(row["engines"]["ax_engine"]["decode_tok_s"], 10.0)
        self.assertAlmostEqual(row["engines"]["mtplx"]["decode_tok_s"], 8.0)
        self.assertAlmostEqual(row["ratios"]["ax_engine_vs_mtplx"], 1.25)
        self.assertEqual(row["provenance"]["kind"], "ax_mtp_sidecar_manifest")

    def test_build_summary_records_pure_mtp_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_cache = root / "hf"
            make_sidecar_manifest(hf_cache)
            artifacts = {
                ("27b-4bit", "flappy", "ax_engine"): write_json(
                    root / "ax.json", fake_ax_artifact()
                ),
            }
            args = Namespace(
                models=["27b-4bit"],
                engines=["ax_engine"],
                suites=["flappy"],
                depth_policy="native",
                depth=None,
                mode="sampled",
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=128,
                repetitions=1,
                warmup_repetitions=1,
                cooldown=0.0,
                hf_cache=hf_cache,
                pure_mtp=True,
            )

            summary = fair.build_summary(args, artifacts)

        self.assertTrue(summary["contract"]["ax_pure_mtp"])
        self.assertEqual(
            summary["contract"]["ax_engine_modes"], {"ax_engine": "pure_mtp"}
        )

    def test_build_summary_rejects_pure_mtp_with_ngram_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_cache = root / "hf"
            make_sidecar_manifest(hf_cache)
            artifact = fake_ax_artifact()
            artifact["results"][0]["ngram_acceleration_telemetry"][
                "ax_mtp_ngram_hit_steps"
            ] = 3
            artifacts = {
                ("27b-4bit", "flappy", "ax_engine"): write_json(
                    root / "ax.json", artifact
                ),
            }
            args = Namespace(
                models=["27b-4bit"],
                engines=["ax_engine"],
                suites=["flappy"],
                depth_policy="native",
                depth=None,
                mode="sampled",
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=128,
                repetitions=1,
                warmup_repetitions=1,
                cooldown=0.0,
                hf_cache=hf_cache,
                pure_mtp=True,
            )

            with self.assertRaisesRegex(RuntimeError, "pure-MTP"):
                fair.build_summary(args, artifacts)

    def test_build_summary_records_stacked_ax_engine_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_cache = root / "hf"
            make_sidecar_manifest(hf_cache)
            artifacts = {
                ("27b-4bit", "flappy", "ax_engine_ngram"): write_json(
                    root / "ax-ngram.json", fake_ax_artifact()
                ),
            }
            args = Namespace(
                models=["27b-4bit"],
                engines=["ax_engine_ngram"],
                suites=["flappy"],
                depth_policy="native",
                depth=None,
                mode="sampled",
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=128,
                repetitions=1,
                warmup_repetitions=1,
                cooldown=0.0,
                hf_cache=hf_cache,
                pure_mtp=False,
            )

            summary = fair.build_summary(args, artifacts)

        self.assertFalse(summary["contract"]["ax_pure_mtp"])
        self.assertEqual(
            summary["contract"]["ax_engine_modes"],
            {"ax_engine_ngram": "mtp_ngram_stacked"},
        )

    def test_run_rapid_mlx_suite_forwards_mtp_tuning_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "lightning.json"
            config = fair.diff.RunConfig(
                mode="sampled",
                depth=3,
                max_tokens=64,
                repetitions=1,
                warmup_repetitions=1,
                cooldown_s=0.0,
                sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                enable_thinking=False,
            )
            with patch.object(fair, "run_subprocess") as run_subprocess:
                fair.run_rapid_mlx_suite(
                    python=Path("python"),
                    lightning_source=Path("lightning"),
                    suite="flappy",
                    suite_file=root / "suite.jsonl",
                    output_path=output,
                    model_dir=root / "model",
                    config=config,
                    mtp_optimistic=True,
                    mtp_draft_temperature=0.5,
                )

        cmd = run_subprocess.call_args.args[0]
        self.assertIn("--mtp-optimistic", cmd)
        self.assertEqual(cmd[cmd.index("--mtp-draft-temperature") + 1], "0.5")

    def test_error_artifact_keeps_table_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            error_path = fair.write_error_artifact(
                root / "mtplx.json",
                engine="mtplx",
                error=RuntimeError("server failed"),
                command=["mtplx"],
            )
            summary = fair.summarize_engine_artifact("mtplx", error_path)
            artifact = json.loads(error_path.read_text())

        self.assertEqual(summary["status"], "error")
        self.assertIn("server failed", summary["error"])
        self.assertEqual(artifact["command"], ["mtplx"])

    def test_error_artifact_uses_called_process_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            error_path = fair.write_error_artifact(
                Path(tmp) / "ax.json",
                engine="ax_engine",
                error=fair.subprocess.CalledProcessError(7, ["ax", "bench"]),
                command=None,
            )
            artifact = json.loads(error_path.read_text())

        self.assertEqual(artifact["command"], ["ax", "bench"])
        self.assertEqual(artifact["returncode"], 7)

    def test_validation_warnings_change_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = fake_mtplx_artifact()
            artifact["validations_passed"] = 1
            artifact["validations_total"] = 2
            path = write_json(Path(tmp) / "mtplx.json", artifact)
            summary = fair.summarize_engine_artifact("mtplx", path)

        self.assertEqual(summary["status"], "ok_validation_warnings")
        self.assertEqual(summary["validations_passed"], 1)
        self.assertEqual(summary["validations_total"], 2)


def sidecar_record(path: Path, content: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": fair.provenance_check.sha256_file(path),
    }


def make_sidecar_manifest(
    hf_cache: Path,
    *,
    norm_policy: str = "shift_mtp_norm_weights_by_1",
) -> None:
    sidecar = hf_cache / "models--ax-local--Qwen3.6-27B-MTP" / "snapshots" / "v1"
    sidecar.mkdir(parents=True)
    manifest = {
        "schema_version": "ax.mtp_sidecar_provenance.v1",
        "generated_by": "scripts/prepare_qwen36_mtp_sidecar.py",
        "model_key": "27b",
        "base": {
            "model_id": "mlx-community/Qwen3.6-27B-4bit",
            "snapshot": "abc",
            "config": sidecar_record(sidecar / "base-config.json", "{}"),
        },
        "source": {
            "model_id": "Qwen/Qwen3.6-27B",
            "mtp_shards": [
                {
                    "name": "model-00013-of-00015.safetensors",
                    **sidecar_record(sidecar / "source-shard-a.safetensors", "a"),
                },
                {
                    "name": "model-00015-of-00015.safetensors",
                    **sidecar_record(sidecar / "source-shard-b.safetensors", "b"),
                },
            ],
        },
        "output": {
            "mtp": sidecar_record(sidecar / "model.safetensors", "mtp"),
            "runtime": sidecar_record(sidecar / "mtplx_runtime.json", "{}"),
            "config": sidecar_record(sidecar / "config.json", "{}"),
        },
        "transform": {"norm_policy": norm_policy},
        "runtime": {
            "arch_id": "qwen3-next-mtp",
            "mtplx_version": "0.3.7",
            "mtp_depth_max": 3,
            "mtp_tensor_count": 15,
            "recommended_draft_sampler": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 20,
            },
            "sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "exactness_baseline": {"context": 2048, "max_abs_diff": 0.0},
            "verified_on": {"system": "Darwin", "machine": "arm64"},
        },
    }
    (sidecar / "ax_mtp_sidecar_manifest.json").write_text(json.dumps(manifest) + "\n")


def fake_ax_artifact() -> dict:
    return {
        "schema_version": "ax.mlx_inference_stack.v2",
        "build": {"git_tracked_dirty": False},
        "repetitions": 1,
        "ax_mtp_max_depth": 3,
        "results": [
            {
                "engine": "ax_engine_mlx_ngram_accel",
                "prompt_case_id": "case_1",
                "decode_tok_s": {"median": 10.0},
                "ngram_acceleration_telemetry": {
                    "ax_mtp_draft_tokens": 4,
                    "ax_mtp_accepted_tokens": 2,
                    "ax_mtp_drafted_depth0": 4,
                    "ax_mtp_accepted_depth0": 2,
                },
                "trials": [{"output_token_ids": [1, 2]}],
            }
        ],
    }


def fake_mtplx_artifact() -> dict:
    return {
        "schema": "ax.mtplx.prompt_suite_mtp.v1",
        "engine": "mtplx",
        "depth": 3,
        "results": [
            {
                "prompt_id": "case_1",
                "summary": {
                    "decode_tok_s": {"median": 8.0},
                    "accepted_drafts": 2,
                    "drafted_tokens": 4,
                    "accept_rate": 0.5,
                },
                "runs": [
                    {
                        "tokens": [1, 2],
                        "accepted_by_depth": [2],
                        "drafted_by_depth": [4],
                    }
                ],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
