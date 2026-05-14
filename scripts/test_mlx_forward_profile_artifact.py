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
    prompt_tokens: int = 128,
    generation_tokens: int = 128,
) -> dict[str, object]:
    return {
        "engine": engine,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "ax_decode_policy": "direct_no_ngram_acceleration",
        "ax_linear_attention_projection_pack": pack,
        "prefill_tok_s": metric(prefill_tok_s),
        "ax_mlx_linear_attention_profile": profile(
            projection_wall_us=projection_wall_us,
        ),
    }


def artifact(
    *,
    prompt_tokens: int = 128,
    generation_tokens: int = 128,
) -> dict[str, object]:
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
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
            ),
            ax_row(
                engine="ax_engine_mlx_linear_pack",
                prefill_tok_s=1900.0,
                projection_wall_us=6000,
                pack=True,
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
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
        self.assertEqual(checked.pack_candidate_win_count, 1)
        self.assertEqual(checked.pack_candidate_win_prompt_count, 1)
        self.assertEqual(
            checker.summarize_pack_comparisons(checked.pack_comparisons),
            "qwen3_6_35b_a3b_8bit prompt=128 gen=128: candidate win",
        )

    def test_summary_disambiguates_generation_tokens(self) -> None:
        first = self.write_fixture(artifact(prompt_tokens=128, generation_tokens=64))
        second = self.write_fixture(artifact(prompt_tokens=128, generation_tokens=128))

        checked = checker.check_mlx_forward_profile_artifacts(
            [first, second],
            require_pack_comparison=True,
        )

        summary = checker.summarize_pack_comparisons(checked.pack_comparisons)
        self.assertIn("prompt=128 gen=64: candidate win", summary)
        self.assertIn("prompt=128 gen=128: candidate win", summary)

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

    def test_pack_candidate_win_gate_rejects_regression(self) -> None:
        payload = artifact()
        packed = payload["results"][1]  # type: ignore[index]
        assert isinstance(packed, dict)
        packed["prefill_tok_s"] = metric(1700.0)
        packed["ax_mlx_linear_attention_profile"] = profile(projection_wall_us=11000)
        path = self.write_fixture(payload)

        with self.assertRaisesRegex(
            checker.MlxForwardProfileArtifactError,
            "not a candidate win",
        ):
            checker.validate_mlx_forward_profile_artifact(
                path,
                require_pack_candidate_win=True,
            )

    def test_pack_candidate_win_gate_implies_pack_comparison(self) -> None:
        path = self.write_fixture(artifact())

        checked = checker.validate_mlx_forward_profile_artifact(
            path,
            require_pack_candidate_win=True,
        )

        self.assertEqual(len(checked), 1)
        self.assertEqual(checked[0].verdict, "candidate win")

    def test_min_pack_candidate_wins_gate_rejects_thin_evidence(self) -> None:
        path = self.write_fixture(artifact())

        with self.assertRaisesRegex(
            checker.MlxForwardProfileArtifactError,
            "expected at least 2",
        ):
            checker.check_mlx_forward_profile_artifacts(
                [path],
                require_pack_candidate_win=True,
                min_pack_candidate_wins=2,
            )

    def test_min_pack_candidate_wins_must_be_non_negative(self) -> None:
        path = self.write_fixture(artifact())

        with self.assertRaisesRegex(
            checker.MlxForwardProfileArtifactError,
            "non-negative",
        ):
            checker.check_mlx_forward_profile_artifacts(
                [path],
                min_pack_candidate_wins=-1,
            )

    def test_min_pack_candidate_win_prompts_rejects_thin_prompt_coverage(self) -> None:
        path = self.write_fixture(artifact())

        with self.assertRaisesRegex(
            checker.MlxForwardProfileArtifactError,
            "prompt length",
        ):
            checker.check_mlx_forward_profile_artifacts(
                [path],
                require_pack_candidate_win=True,
                min_pack_candidate_win_prompts=2,
            )

    def test_min_pack_candidate_win_prompts_accepts_multiple_prompts(self) -> None:
        first = self.write_fixture(artifact(prompt_tokens=128))
        second = self.write_fixture(artifact(prompt_tokens=512))

        checked = checker.check_mlx_forward_profile_artifacts(
            [first, second],
            require_pack_candidate_win=True,
            min_pack_candidate_win_prompts=2,
        )

        self.assertEqual(checked.pack_candidate_win_count, 2)
        self.assertEqual(checked.pack_candidate_win_prompt_count, 2)

    def test_min_pack_candidate_win_prompts_must_be_non_negative(self) -> None:
        path = self.write_fixture(artifact())

        with self.assertRaisesRegex(
            checker.MlxForwardProfileArtifactError,
            "non-negative",
        ):
            checker.check_mlx_forward_profile_artifacts(
                [path],
                min_pack_candidate_win_prompts=-1,
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
        self.assertIn("1 candidate win", completed.stdout)
        self.assertIn("1 candidate-win prompt length", completed.stdout)
        self.assertIn(
            "qwen3_6_35b_a3b_8bit prompt=128 gen=128: candidate win",
            completed.stdout,
        )

    def test_cli_reports_missing_file_without_traceback(self) -> None:
        missing = Path(tempfile.mkdtemp()) / "missing.json"
        self.addCleanup(lambda: missing.parent.rmdir())

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                str(missing),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("failed to read", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_cli_accepts_min_pack_candidate_wins(self) -> None:
        path = self.write_fixture(artifact())

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                str(path),
                "--require-pack-candidate-win",
                "--min-pack-candidate-wins",
                "1",
                "--min-pack-candidate-win-prompts",
                "1",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("1 candidate win", completed.stdout)
        self.assertIn("1 candidate-win prompt length", completed.stdout)


if __name__ == "__main__":
    unittest.main()
