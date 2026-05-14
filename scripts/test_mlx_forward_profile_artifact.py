#!/usr/bin/env python3
"""Unit tests for diagnostic MLX forward-profile artifact checks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_mlx_forward_profile_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mlx_forward_profile_artifact", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def metric(median: float) -> dict[str, float]:
    return {"mean": median, "median": median, "min": median, "max": median}


def profile(*, projection_wall_us: int = 9000) -> dict[str, int]:
    return {
        "ax_mlx_linear_attention_profile_enabled": 1,
        "ax_mlx_linear_attention_profile_layers": 30,
        "ax_mlx_linear_attention_profile_tokens": 3840,
        "ax_mlx_linear_attention_profile_projection_wall_us": projection_wall_us,
        "ax_mlx_linear_attention_profile_conv_wall_us": 1000,
        "ax_mlx_linear_attention_profile_qk_norm_wall_us": 500,
        "ax_mlx_linear_attention_profile_recurrent_wall_us": 3000,
        "ax_mlx_linear_attention_profile_output_wall_us": 1000,
    }


def ax_row(
    *,
    engine: str,
    prefill_tok_s: float,
    projection_wall_us: int,
    pack: bool,
) -> dict[str, object]:
    return {
        "engine": engine,
        "prompt_tokens": 128,
        "generation_tokens": 128,
        "ax_decode_policy": "direct_no_ngram_acceleration",
        "ax_linear_attention_projection_pack": pack,
        "prefill_tok_s": metric(prefill_tok_s),
        "ax_mlx_linear_attention_profile": profile(
            projection_wall_us=projection_wall_us,
        ),
    }


def artifact() -> dict[str, object]:
    return {
        "schema_version": checker.SCHEMA_VERSION,
        "model": "qwen3_6_35b_a3b_8bit",
        "ax_linear_attention_profile": True,
        "ax_linear_attention_projection_pack_compare": True,
        "results": [
            ax_row(
                engine="ax_engine_mlx",
                prefill_tok_s=1800.0,
                projection_wall_us=9000,
                pack=False,
            ),
            ax_row(
                engine="ax_engine_mlx_linear_pack",
                prefill_tok_s=1900.0,
                projection_wall_us=6000,
                pack=True,
            ),
        ],
    }


class MlxForwardProfileArtifactTests(unittest.TestCase):
    def write_fixture(self, payload: dict[str, object]) -> Path:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = Path(root.name) / "forward-profile.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def test_pack_comparison_artifact_passes_as_diagnostic(self) -> None:
        path = self.write_fixture(artifact())

        checked = checker.validate_mlx_forward_profile_artifact(
            path,
            require_pack_comparison=True,
        )

        self.assertEqual(len(checked), 1)
        self.assertEqual(checked[0].verdict, "candidate win")

    def test_summary_reports_pack_comparison_verdicts(self) -> None:
        path = self.write_fixture(artifact())

        checked = checker.check_mlx_forward_profile_artifacts(
            [path],
            require_pack_comparison=True,
        )

        self.assertEqual(checked.artifact_count, 1)
        self.assertEqual(checked.diagnostic_count, 1)
        self.assertEqual(
            checker.summarize_pack_comparisons(checked.pack_comparisons),
            "qwen3_6_35b_a3b_8bit prompt=128: candidate win",
        )

    def test_public_packed_claim_fails_closed(self) -> None:
        payload = artifact()
        payload["public_claims"] = ["packed_projection_performance"]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.MlxForwardProfileArtifactError,
            "public claims",
        ):
            checker.validate_mlx_forward_profile_artifact(
                path,
                require_pack_comparison=True,
            )

    def test_pack_comparison_requires_matched_split_row(self) -> None:
        payload = artifact()
        payload["results"] = [payload["results"][1]]  # type: ignore[index]
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.MlxForwardProfileArtifactError,
            "matched split and packed",
        ):
            checker.validate_mlx_forward_profile_artifact(
                path,
                require_pack_comparison=True,
            )

    def test_pack_comparison_requires_direct_ax_policy(self) -> None:
        payload = artifact()
        packed = payload["results"][1]  # type: ignore[index]
        assert isinstance(packed, dict)
        packed["ax_decode_policy"] = "ngram_acceleration_kv_trim"
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.MlxForwardProfileArtifactError,
            "direct_no_ngram_acceleration",
        ):
            checker.validate_mlx_forward_profile_artifact(
                path,
                require_pack_comparison=True,
            )

    def test_cli_reports_diagnostics(self) -> None:
        path = self.write_fixture(artifact())

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                str(path),
                "--require-pack-comparison",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("diagnostics validated", completed.stdout)
        self.assertIn(
            "qwen3_6_35b_a3b_8bit prompt=128: candidate win",
            completed.stdout,
        )


if __name__ == "__main__":
    unittest.main()
