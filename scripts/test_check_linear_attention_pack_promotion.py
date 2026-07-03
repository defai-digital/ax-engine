#!/usr/bin/env python3
"""Unit tests for linear-attention projection-pack promotion checks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_linear_attention_pack_promotion.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_linear_attention_pack_promotion", SCRIPT_PATH
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
    prompt_tokens: int,
    prefill_tok_s: float,
    projection_wall_us: int,
    pack: bool,
) -> dict[str, object]:
    return {
        "engine": engine,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": 128,
        "ax_decode_policy": "direct_no_ngram_acceleration",
        "ax_linear_attention_projection_pack": pack,
        "prefill_tok_s": metric(prefill_tok_s),
        "ax_mlx_linear_attention_profile": profile(
            projection_wall_us=projection_wall_us,
        ),
    }


def pack_pair(
    *,
    prompt_tokens: int,
    packed_prefill_tok_s: float,
    packed_projection_wall_us: int,
) -> list[dict[str, object]]:
    return [
        ax_row(
            engine="ax_engine_mlx",
            prompt_tokens=prompt_tokens,
            prefill_tok_s=1800.0,
            projection_wall_us=9000,
            pack=False,
        ),
        ax_row(
            engine="ax_engine_mlx_linear_pack",
            prompt_tokens=prompt_tokens,
            prefill_tok_s=packed_prefill_tok_s,
            projection_wall_us=packed_projection_wall_us,
            pack=True,
        ),
    ]


def artifact(*, second_prompt_is_win: bool) -> dict[str, object]:
    second_prefill = 1900.0 if second_prompt_is_win else 1810.0
    second_projection = 6000 if second_prompt_is_win else 8900
    return {
        "schema_version": checker.forward_checker.SCHEMA_VERSION,
        "model": "qwen3_6_35b_a3b_8bit",
        "ax_linear_attention_profile": True,
        "ax_linear_attention_projection_pack_compare": True,
        "results": [
            *pack_pair(
                prompt_tokens=128,
                packed_prefill_tok_s=1900.0,
                packed_projection_wall_us=6000,
            ),
            *pack_pair(
                prompt_tokens=512,
                packed_prefill_tok_s=second_prefill,
                packed_projection_wall_us=second_projection,
            ),
        ],
    }


class LinearAttentionPackPromotionTests(unittest.TestCase):
    def write_fixture(self, payload: dict[str, object]) -> Path:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = Path(root.name) / "linear-pack-ab.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def test_curated_artifact_default_uses_inference_results_tree(self) -> None:
        self.assertEqual(len(checker.DEFAULT_ARTIFACTS), 1)
        self.assertIn(
            "benchmarks/results/inference/mlx-inference/",
            checker.DEFAULT_ARTIFACTS[0].as_posix(),
        )

    def test_curated_artifact_is_not_promoted(self) -> None:
        decision = checker.check_linear_attention_pack_promotion(
            list(checker.DEFAULT_ARTIFACTS),
            expect_decision=checker.NOT_PROMOTED,
        )

        self.assertEqual(decision.decision, checker.NOT_PROMOTED)
        self.assertEqual(decision.comparison_count, 2)
        self.assertEqual(decision.candidate_win_count, 1)
        self.assertEqual(decision.non_win_count, 1)

    def test_not_promoted_requires_non_win_or_thin_coverage(self) -> None:
        path = self.write_fixture(artifact(second_prompt_is_win=True))

        with self.assertRaisesRegex(
            checker.LinearAttentionPackPromotionError,
            "promotion candidate",
        ):
            checker.check_linear_attention_pack_promotion(
                [path],
                expect_decision=checker.NOT_PROMOTED,
            )

    def test_promotion_candidate_rejects_mixed_evidence(self) -> None:
        path = self.write_fixture(artifact(second_prompt_is_win=False))

        with self.assertRaisesRegex(
            checker.LinearAttentionPackPromotionError,
            "not a promotion candidate",
        ):
            checker.check_linear_attention_pack_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_promotion_candidate_requires_prompt_coverage(self) -> None:
        payload = artifact(second_prompt_is_win=True)
        payload["results"] = payload["results"][:2]  # type: ignore[index]
        path = self.write_fixture(payload)

        decision = checker.check_linear_attention_pack_promotion(
            [path],
            expect_decision=checker.NOT_PROMOTED,
        )
        self.assertEqual(decision.candidate_win_prompt_count, 1)

        with self.assertRaisesRegex(
            checker.LinearAttentionPackPromotionError,
            "not a promotion candidate",
        ):
            checker.check_linear_attention_pack_promotion(
                [path],
                expect_decision=checker.PROMOTION_CANDIDATE,
            )

    def test_cli_reports_not_promoted(self) -> None:
        path = self.write_fixture(artifact(second_prompt_is_win=False))

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--artifact",
                str(path),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("decision=not_promoted", completed.stdout)
        self.assertIn("1/2 candidate wins", completed.stdout)

    def test_cli_rejects_unexpected_promotion_candidate(self) -> None:
        path = self.write_fixture(artifact(second_prompt_is_win=True))

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--artifact",
                str(path),
                "--expect-decision",
                checker.NOT_PROMOTED,
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("promotion candidate", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
