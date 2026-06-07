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


if __name__ == "__main__":
    unittest.main()
