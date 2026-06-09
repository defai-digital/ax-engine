#!/usr/bin/env python3
"""Tests for bench_gemma4_assistant_mtp.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load(name: str):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load("bench_gemma4_assistant_mtp")


class Gemma4AssistantMtpBenchTests(unittest.TestCase):
    def test_parse_name_path_map(self) -> None:
        parsed = bench.parse_name_path_map(["26b-a4b-4bit=/tmp/g26", "31b-4bit=/tmp/g31"])
        self.assertEqual(parsed["26b-a4b-4bit"], Path("/tmp/g26"))
        self.assertEqual(parsed["31b-4bit"], Path("/tmp/g31"))

    def test_select_bench_profiles_uses_safe_defaults_for_legacy_modes(self) -> None:
        profiles = bench.select_bench_profiles(modes=["mtp", "mtp-ngram"], profile_keys=[])

        self.assertEqual(
            [profile.key for profile in profiles],
            ["assistant_mtp_default", "assistant_mtp_ngram_default"],
        )
        self.assertEqual([profile.env for profile in profiles], [{}, {}])
        self.assertFalse(any(profile.experimental for profile in profiles))

    def test_select_bench_profiles_exposes_explicit_experimental_profile(self) -> None:
        profiles = bench.select_bench_profiles(
            modes=[],
            profile_keys=["target_softmax_topk256_experimental"],
        )

        self.assertEqual(len(profiles), 1)
        self.assertTrue(profiles[0].experimental)
        self.assertEqual(
            profiles[0].env,
            {"AX_MLX_MTP_TARGET_SOFTMAX_MODE": "topk-256"},
        )

    def test_summarize_artifact_extracts_assistant_and_ngram_telemetry(self) -> None:
        artifact = {
            "build": {"git_tracked_dirty": False},
            "results": [
                {
                    "engine": "ax_engine_gemma4_assistant_mtp_ngram",
                    "decode_tok_s": {"median": 40.0},
                    "prefill_tok_s": {"median": 100.0},
                    "ttft_ms": {"median": 25.0},
                    "ax_decode_claim_status": "ngram_acceleration_effective_throughput",
                    "ax_decode_effective_route": "ngram_verified_bonus_tokens",
                    "ax_mlx_gemma4_assistant_mtp": {
                        "ax_mlx_gemma4_assistant_mtp_draft_tokens": 10,
                        "ax_mlx_gemma4_assistant_mtp_accepted_tokens": 8,
                    },
                    "ngram_acceleration_telemetry": {
                        "ax_mtp_draft_tokens": 12,
                        "ax_mtp_accepted_tokens": 9,
                        "ax_mtp_source_mtp_submitted_tokens": 6,
                        "ax_mtp_source_mtp_accepted_tokens": 5,
                        "ax_mtp_source_mtp_rejected_tokens": 1,
                        "ax_mtp_source_mtp_cascade_rejected_tokens": 0,
                        "ax_mtp_source_assistant_submitted_tokens": 4,
                        "ax_mtp_source_assistant_accepted_tokens": 3,
                        "ax_mtp_source_assistant_rejected_tokens": 1,
                        "ax_mtp_source_assistant_cascade_rejected_tokens": 0,
                        "ax_ngram_draft_tokens": 99,
                        "ax_ngram_accepted_tokens": 88,
                        "ax_mtp_ngram_proposed_tokens": 5,
                        "ax_mtp_ngram_submitted_tokens": 4,
                        "ax_mtp_ngram_accepted_tokens": 3,
                        "ax_mtp_ngram_submitted_accepted_tokens": 3,
                        "ax_mtp_ngram_rejected_tokens": 1,
                        "ax_mtp_ngram_cascade_rejected_tokens": 0,
                        "ax_mtp_ngram_hit_steps": 2,
                        "ax_mtp_ngram_utility_gated_steps": 1,
                        "ax_mtp_ngram_utility_insufficient_sample_steps": 2,
                        "ax_mtp_ngram_safety_tightened_steps": 3,
                        "ax_mtp_ngram_lookup_wall_us": 40,
                        "ax_mtp_target_softmax_wall_us": 50,
                        "ax_mtp_verify_tokens": 16,
                        "ax_mtp_emitted_tokens": 8,
                        "ax_mtp_ngram_utility_baseline_cost_per_emitted_token_us": 100,
                        "ax_mtp_ngram_utility_stacked_cost_per_emitted_token_us": 120,
                        "ax_mtp_ngram_gate_policy": 1,
                        "ax_mtp_ngram_safety_reason": 3,
                    },
                },
                {
                    "engine": "ax_engine_gemma4_assistant_mtp_ngram",
                    "decode_tok_s": {"median": 50.0},
                    "prefill_tok_s": {"median": 120.0},
                    "ttft_ms": {"median": 20.0},
                    "ax_mlx_gemma4_assistant_mtp": {
                        "ax_mlx_gemma4_assistant_mtp_draft_tokens": 10,
                        "ax_mlx_gemma4_assistant_mtp_accepted_tokens": 7,
                    },
                    "ngram_acceleration_telemetry": {
                        "ax_mtp_draft_tokens": 8,
                        "ax_mtp_accepted_tokens": 6,
                        "ax_mtp_source_mtp_submitted_tokens": 5,
                        "ax_mtp_source_mtp_accepted_tokens": 4,
                        "ax_mtp_source_mtp_rejected_tokens": 0,
                        "ax_mtp_source_mtp_cascade_rejected_tokens": 1,
                        "ax_mtp_source_assistant_submitted_tokens": 3,
                        "ax_mtp_source_assistant_accepted_tokens": 2,
                        "ax_mtp_source_assistant_rejected_tokens": 0,
                        "ax_mtp_source_assistant_cascade_rejected_tokens": 1,
                        "ax_ngram_draft_tokens": 77,
                        "ax_ngram_accepted_tokens": 66,
                        "ax_mtp_ngram_proposed_tokens": 7,
                        "ax_mtp_ngram_submitted_tokens": 6,
                        "ax_mtp_ngram_accepted_tokens": 3,
                        "ax_mtp_ngram_submitted_accepted_tokens": 3,
                        "ax_mtp_ngram_rejected_tokens": 2,
                        "ax_mtp_ngram_cascade_rejected_tokens": 1,
                        "ax_mtp_ngram_hit_steps": 1,
                        "ax_mtp_ngram_utility_gated_steps": 0,
                        "ax_mtp_ngram_utility_insufficient_sample_steps": 1,
                        "ax_mtp_ngram_safety_disabled_steps": 1,
                        "ax_mtp_ngram_lookup_wall_us": 60,
                        "ax_mtp_target_softmax_wall_us": 70,
                        "ax_mtp_verify_tokens": 12,
                        "ax_mtp_emitted_tokens": 7,
                        "ax_mtp_ngram_utility_baseline_cost_per_emitted_token_us": 80,
                        "ax_mtp_ngram_utility_stacked_cost_per_emitted_token_us": 90,
                        "ax_mtp_ngram_gate_policy": 1,
                        "ax_mtp_ngram_safety_reason": 4,
                    },
                },
                {"engine": "unrelated", "decode_tok_s": {"median": 1.0}},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(json.dumps(artifact))
            summary = bench.summarize_artifact(
                path,
                expected_engine="ax_engine_gemma4_assistant_mtp_ngram",
            )

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["decode_tok_s_median"], 45.0)
        self.assertEqual(summary["prefill_tok_s_median"], 110.0)
        self.assertEqual(summary["ttft_ms_median"], 22.5)
        self.assertEqual(summary["assistant_accept_rate"], 15 / 20)
        self.assertEqual(summary["mtp_accept_rate"], 15 / 20)
        self.assertEqual(summary["ngram_accept_rate"], 6 / 10)
        self.assertEqual(summary["ngram_hit_steps"], 3)
        self.assertEqual(summary["source_mtp_submitted_tokens"], 11)
        self.assertEqual(summary["source_mtp_accepted_tokens"], 9)
        self.assertEqual(summary["source_mtp_rejected_tokens"], 1)
        self.assertEqual(summary["source_mtp_cascade_rejected_tokens"], 1)
        self.assertEqual(summary["source_assistant_submitted_tokens"], 7)
        self.assertEqual(summary["source_assistant_accepted_tokens"], 5)
        self.assertEqual(summary["source_assistant_rejected_tokens"], 1)
        self.assertEqual(summary["source_assistant_cascade_rejected_tokens"], 1)
        self.assertEqual(summary["ngram_proposed_tokens"], 12)
        self.assertEqual(summary["ngram_rejected_tokens"], 3)
        self.assertEqual(summary["ngram_cascade_rejected_tokens"], 1)
        self.assertEqual(summary["ngram_utility_gated_steps"], 1)
        self.assertEqual(summary["ngram_utility_insufficient_sample_steps"], 3)
        self.assertEqual(summary["ngram_safety_disabled_steps"], 1)
        self.assertEqual(summary["ngram_safety_tightened_steps"], 3)
        self.assertEqual(summary["ngram_lookup_wall_us"], 100)
        self.assertEqual(summary["target_softmax_wall_us"], 120)
        self.assertEqual(summary["verify_tokens"], 28)
        self.assertEqual(summary["emitted_tokens"], 15)
        self.assertEqual(summary["utility_baseline_cost_per_emitted_token_us_median"], 90.0)
        self.assertEqual(summary["utility_stacked_cost_per_emitted_token_us_median"], 105.0)
        self.assertEqual(summary["gate_policies"], [1])
        self.assertEqual(summary["safety_reasons"], [3, 4])

    def test_summarize_artifact_fails_closed_on_missing_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(json.dumps({"results": [{"engine": "other"}]}))
            with self.assertRaisesRegex(ValueError, "no rows"):
                bench.summarize_artifact(path, expected_engine="missing")

    def test_summarize_artifact_max_merges_affine_bits(self) -> None:
        artifact = {
            "results": [
                {
                    "engine": "ax_engine_mlx",
                    "decode_tok_s": {"median": 68.0},
                    "ax_mlx_telemetry": {
                        "ax_mlx_affine_tensor_count": 332,
                        "ax_mlx_affine_min_bits": 4,
                        "ax_mlx_affine_max_bits": 4,
                        "ax_mlx_affine_4bit_count": 332,
                        "ax_mlx_affine_8bit_count": 0,
                    },
                },
                {
                    "engine": "ax_engine_mlx",
                    "decode_tok_s": {"median": 64.0},
                    "ax_mlx_telemetry": {
                        "ax_mlx_affine_tensor_count": 332,
                        "ax_mlx_affine_min_bits": 4,
                        "ax_mlx_affine_max_bits": 4,
                        "ax_mlx_affine_4bit_count": 332,
                        "ax_mlx_affine_8bit_count": 0,
                    },
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "direct.json"
            path.write_text(json.dumps(artifact))
            summary = bench.summarize_artifact(path, expected_engine="ax_engine_mlx")

        self.assertEqual(summary["affine_tensor_count"], 332)
        self.assertEqual(summary["affine_min_bits"], 4)
        self.assertEqual(summary["affine_max_bits"], 4)
        self.assertEqual(summary["affine_4bit_count"], 332)
        self.assertEqual(summary["affine_8bit_count"], 0)
        # A direct row never drafts assistant tokens.
        self.assertEqual(summary["assistant_draft_tokens"], 0)

    def test_summarize_artifact_affine_defaults_when_absent(self) -> None:
        artifact = {"results": [{"engine": "ax_engine_mlx", "decode_tok_s": {"median": 1.0}}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "noaffine.json"
            path.write_text(json.dumps(artifact))
            summary = bench.summarize_artifact(path, expected_engine="ax_engine_mlx")
        self.assertIsNone(summary["affine_min_bits"])
        self.assertIsNone(summary["affine_max_bits"])
        self.assertEqual(summary["affine_8bit_count"], 0)


class BuildAxCmdTests(unittest.TestCase):
    def _cmd(self, mode: str) -> list[str]:
        return bench.build_ax_cmd(
            python=Path("/usr/bin/python3"),
            suite_file=Path("/suites/flappy.jsonl"),
            output_path=Path("/out/row.json"),
            model_dir=Path("/models/g12"),
            mode=mode,
            max_tokens=1000,
            repetitions=5,
            cooldown=30.0,
            inter_case_cooldown=10.0,
            sampling={"temperature": 0.6},
            depth=2,
            no_build=False,
        )

    def test_direct_mode_omits_all_mtp_flags(self) -> None:
        cmd = self._cmd("direct")
        self.assertNotIn("--ax-gemma4-assistant-mtp", cmd)
        self.assertNotIn("--ax-mtp-max-depth", cmd)
        self.assertNotIn("--ax-mtp-disable-ngram-stacking", cmd)

    def test_mtp_mode_disables_ngram_stacking(self) -> None:
        cmd = self._cmd("mtp")
        self.assertIn("--ax-gemma4-assistant-mtp", cmd)
        self.assertIn("--ax-mtp-disable-ngram-stacking", cmd)
        self.assertIn("--ax-mtp-max-depth", cmd)
        self.assertEqual(cmd[cmd.index("--ax-mtp-max-depth") + 1], "2")

    def test_ngram_mode_keeps_stacking_enabled(self) -> None:
        cmd = self._cmd("mtp-ngram")
        self.assertIn("--ax-gemma4-assistant-mtp", cmd)
        self.assertNotIn("--ax-mtp-disable-ngram-stacking", cmd)

    def test_direct_profile_is_selectable(self) -> None:
        profiles = bench.select_bench_profiles(modes=["direct"], profile_keys=[])
        self.assertEqual([profile.key for profile in profiles], ["direct"])
        self.assertEqual(profiles[0].mode, "direct")
        self.assertEqual(bench.ENGINE_KEYS["direct"], "ax_engine_mlx")


class ComparisonTests(unittest.TestCase):
    def test_classify_vs_direct_keep_default(self) -> None:
        self.assertEqual(
            bench.classify_vs_direct(0.07, -0.01, parity=True, drafted=True),
            "keep-default",
        )

    def test_classify_vs_direct_keep_opt_in_when_small_gain(self) -> None:
        self.assertEqual(
            bench.classify_vs_direct(0.02, 0.0, parity=True, drafted=True),
            "keep-opt-in",
        )

    def test_classify_vs_direct_remove_claim_when_slower(self) -> None:
        self.assertEqual(
            bench.classify_vs_direct(-0.04, -0.06, parity=True, drafted=True),
            "remove-claim",
        )

    def test_classify_vs_direct_keep_default_blocked_by_regression(self) -> None:
        # Strong aggregate gain but one suite regresses past the 3% guard.
        self.assertEqual(
            bench.classify_vs_direct(0.06, -0.05, parity=True, drafted=True),
            "keep-opt-in",
        )

    def test_classify_vs_direct_retest_on_parity_mismatch(self) -> None:
        self.assertEqual(
            bench.classify_vs_direct(0.20, 0.0, parity=False, drafted=True),
            "retest",
        )

    def test_classify_vs_direct_reject_when_not_drafted(self) -> None:
        self.assertEqual(
            bench.classify_vs_direct(0.20, 0.0, parity=True, drafted=False),
            "reject",
        )

    def test_classify_ngram_vs_mtp(self) -> None:
        self.assertEqual(bench.classify_ngram_vs_mtp(0.04, 0.0), "keep-default")
        self.assertEqual(bench.classify_ngram_vs_mtp(0.01, 0.0), "keep-opt-in")
        self.assertEqual(bench.classify_ngram_vs_mtp(-0.02, -0.02), "remove-claim")
        self.assertEqual(bench.classify_ngram_vs_mtp(0.04, -0.05), "keep-opt-in")

    def test_compare_suite_decodes_uses_common_suites(self) -> None:
        result = bench.compare_suite_decodes(
            {"flappy": 106.0, "long_code": 110.0, "extra": 999.0},
            {"flappy": 100.0, "long_code": 100.0},
        )
        self.assertEqual(result["suites"], ["flappy", "long_code"])
        self.assertAlmostEqual(result["delta"], 0.08)
        self.assertAlmostEqual(result["worst_suite_delta"], 0.06)

    def test_compare_suite_decodes_no_common(self) -> None:
        result = bench.compare_suite_decodes({"a": 1.0}, {"b": 2.0})
        self.assertIsNone(result["delta"])
        self.assertEqual(result["per_suite_delta"], {})

    def _row(
        self,
        *,
        profile: str,
        mode: str,
        suite: str,
        decode: float,
        max_bits: int,
        eightbit: int,
        drafted: int,
    ) -> dict:
        return {
            "model": "12b-4bit",
            "suite": suite,
            "mode": mode,
            "profile": profile,
            "decode_tok_s_median": decode,
            "assistant_draft_tokens": drafted,
            "affine_max_bits": max_bits,
            "affine_8bit_count": eightbit,
            "affine_4bit_count": 332,
            "affine_tensor_count": 332,
            "model_dir": f"/models/{profile}",
        }

    def test_build_comparisons_parity_mismatch_marks_retest(self) -> None:
        rows = []
        for suite in ("flappy", "long_code"):
            rows.append(
                self._row(profile="direct", mode="direct", suite=suite, decode=68.0, max_bits=4, eightbit=0, drafted=0)
            )
            # MTP runs the mixed 4/8-bit target: faster-looking but unfair.
            rows.append(
                self._row(
                    profile="assistant_mtp_default", mode="mtp", suite=suite, decode=80.0, max_bits=8, eightbit=144, drafted=500
                )
            )
        result = bench.build_comparisons(rows)
        self.assertFalse(result["parity_ok"])
        direct_cmp = next(c for c in result["comparisons"] if c["baseline"] == "direct")
        self.assertEqual(direct_cmp["classification"], "retest")
        self.assertTrue(any("parity mismatch" in w for w in result["warnings"]))

    def test_build_comparisons_keep_default_on_parity(self) -> None:
        rows = []
        for suite in ("flappy", "long_code"):
            rows.append(
                self._row(profile="direct", mode="direct", suite=suite, decode=100.0, max_bits=4, eightbit=0, drafted=0)
            )
            rows.append(
                self._row(
                    profile="assistant_mtp_default", mode="mtp", suite=suite, decode=106.0, max_bits=4, eightbit=0, drafted=500
                )
            )
        result = bench.build_comparisons(rows)
        self.assertTrue(result["parity_ok"])
        direct_cmp = next(c for c in result["comparisons"] if c["baseline"] == "direct")
        self.assertEqual(direct_cmp["classification"], "keep-default")
        self.assertTrue(direct_cmp["drafted"])
        self.assertAlmostEqual(direct_cmp["delta_vs_baseline"], 0.06)

    def test_build_comparisons_ngram_vs_mtp_baseline(self) -> None:
        rows = []
        for suite in ("flappy", "long_code"):
            rows.append(
                self._row(profile="direct", mode="direct", suite=suite, decode=100.0, max_bits=4, eightbit=0, drafted=0)
            )
            rows.append(
                self._row(
                    profile="assistant_mtp_default", mode="mtp", suite=suite, decode=110.0, max_bits=4, eightbit=0, drafted=500
                )
            )
            rows.append(
                self._row(
                    profile="assistant_mtp_ngram_default", mode="mtp-ngram", suite=suite, decode=99.0, max_bits=4, eightbit=0, drafted=500
                )
            )
        result = bench.build_comparisons(rows)
        ngram_cmp = next(
            c for c in result["comparisons"]
            if c["profile"] == "assistant_mtp_ngram_default" and c["baseline"] == "assistant_mtp"
        )
        self.assertEqual(ngram_cmp["baseline_profile"], "assistant_mtp_default")
        # 99 vs 110 is a regression -> n-gram should not be a default.
        self.assertEqual(ngram_cmp["classification"], "remove-claim")

    def test_build_comparisons_warns_when_no_direct_row(self) -> None:
        rows = [
            self._row(
                profile="assistant_mtp_default", mode="mtp", suite="flappy", decode=60.0, max_bits=4, eightbit=0, drafted=500
            )
        ]
        result = bench.build_comparisons(rows)
        self.assertTrue(any("no direct-decode row" in w for w in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
