from __future__ import annotations

import argparse
import copy
import tempfile
import unittest
from pathlib import Path

from scripts import bench_mtp_6bit_ax_refresh as bench


class BenchMtpRefreshTests(unittest.TestCase):
    @staticmethod
    def exact_summary() -> dict[str, object]:
        rows: list[dict[str, object]] = []
        for target in bench.SUPPORTED_TARGETS:
            for suite in bench.DEFAULT_SUITES:
                rows.append(
                    {
                        "model_id": target.key,
                        "model": target.label,
                        "suite_id": suite,
                        "ax_direct_decode_tok_s": 50.0,
                        "ax_mtp_decode_tok_s": 100.0,
                        "ax_mtp_speedup_x": 2.0,
                        "ax_mtp_prefill_tok_s": 500.0,
                        "ax_mtp_ttft_ms": 250.0,
                        "ax_mtp_accept_rate_pct": 99.0,
                        "ax_mtp_step_coverage_pct": 100.0,
                        "ax_mtp_fallback_prompt_count": 0,
                        "ax_mtp_direct_fallback_steps": 0,
                        "publication_candidate": True,
                        "publication_reasons": [],
                        "ax_mtp_ngram_telemetry": {
                            key: 0 for key in bench.NGRAM_ZERO_KEYS
                        },
                    }
                )
        return {
            "schema": "ax.mtp_6bit_ax_acceleration_summary.v3",
            "publication_candidate": True,
            "claim_type": "exact_mtp_acceleration",
            "run_dir": (
                "benchmarks/results/speculative/mtp-6bit/"
                "2026-07-13-exact"
            ),
            "methodology": {
                "generated_tokens": 1000,
                "repetitions": 5,
                "warmup_repetitions": 2,
                "sampling": {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                },
            },
            "rows": rows,
        }

    @staticmethod
    def mtp_row(
        *,
        case_id: str,
        match_x1000: int,
        mtp_steps: int,
        fallback_steps: int,
        emitted_tokens: int,
    ) -> dict[str, object]:
        return {
            "prompt_case_id": case_id,
            "ngram_acceleration_telemetry": {
                "ax_mtp_mtp_only_accept_rate_ewma_samples": mtp_steps,
                "ax_mtp_mtp_only_accept_rate_ewma_x1000": match_x1000,
                "ax_mtp_decode_steps": mtp_steps,
                "ax_mtp_direct_fallback_steps": fallback_steps,
                "ax_mtp_emitted_tokens": emitted_tokens,
            },
        }

    def test_bench_command_records_two_warmups_and_sampled_exact_sampler(self) -> None:
        target = bench.Target(
            key="test",
            label="Test",
            mode="MTP",
            model_dir=Path("/models/test"),
            mtp_depth=2,
        )
        args = argparse.Namespace(
            suites_dir=Path("/prompts"),
            generated_tokens=32,
            repetitions=5,
            warmup_repetitions=2,
            cooldown=0.0,
            inter_case_cooldown=0.0,
            approximate_speed_ceiling=False,
        )

        command = bench.bench_cmd(
            target=target,
            suite="sample",
            mode="mtp",
            output_path=Path("/tmp/result.json"),
            args=args,
        )

        warmup_index = command.index("--warmup-repetitions")
        sampling_index = command.index("--ax-sampling")
        self.assertEqual(command[warmup_index + 1], "2")
        self.assertEqual(
            command[sampling_index + 1],
            '{"temperature":0.6,"top_p":0.95,"top_k":20}',
        )

    def test_exact_artifact_validation_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an exact MTP publication candidate"):
            bench.validate_exact_mtp_artifact(Path("artifact.json"), {})

        bench.validate_exact_mtp_artifact(
            Path("artifact.json"),
            {"mtp_correctness_summary": {"publication_candidate": True}},
        )

    def test_exact_publication_methodology_requires_clean_two_by_five(self) -> None:
        valid = {
            "warmup_repetitions": 2,
            "repetitions": 5,
            "build": {"git_tracked_dirty": False},
        }
        self.assertEqual(
            bench.exact_publication_methodology_reasons(valid, valid), []
        )

        smoke = {
            "warmup_repetitions": 0,
            "repetitions": 2,
            "build": {"git_tracked_dirty": True},
        }
        reasons = bench.exact_publication_methodology_reasons(smoke, smoke)
        self.assertIn("direct_requires_two_warmups", reasons)
        self.assertIn("mtp_requires_five_measurements", reasons)
        self.assertIn("mtp_requires_clean_tracked_build", reasons)

    def test_approximate_flag_is_only_added_to_mtp_rows(self) -> None:
        target = bench.Target(
            key="test",
            label="Test",
            mode="MTP",
            model_dir=Path("/models/test"),
            mtp_depth=2,
        )
        args = argparse.Namespace(
            suites_dir=Path("/prompts"),
            generated_tokens=32,
            repetitions=5,
            warmup_repetitions=2,
            cooldown=0.0,
            inter_case_cooldown=0.0,
            approximate_speed_ceiling=True,
        )

        direct_command = bench.bench_cmd(
            target=target,
            suite="sample",
            mode="direct",
            output_path=Path("/tmp/direct.json"),
            args=args,
        )
        mtp_command = bench.bench_cmd(
            target=target,
            suite="sample",
            mode="mtp",
            output_path=Path("/tmp/mtp.json"),
            args=args,
        )

        self.assertIn("--ax-direct", direct_command)
        self.assertNotIn("--ax-mtp-approximate-optimistic", direct_command)
        self.assertIn("--ax-mtp-approximate-optimistic", mtp_command)

    def test_approximate_artifact_is_explicit_and_non_publishable(self) -> None:
        artifact = {
            "results": [
                {
                    "prompt_case_id": "case",
                    "publication_candidate": False,
                    "ax_mtp_correctness": {
                        "effective_mode": "approximate_optimistic"
                    },
                }
            ]
        }
        bench.validate_approximate_mtp_artifact(Path("artifact.json"), artifact)

        artifact["results"][0]["publication_candidate"] = True
        with self.assertRaisesRegex(ValueError, "incorrectly marks"):
            bench.validate_approximate_mtp_artifact(Path("artifact.json"), artifact)

    def test_exact_seed_reproducibility_allows_cross_mode_sequence_difference(self) -> None:
        direct = {
            "results": [
                {
                    "prompt_case_id": "case",
                    "trials": [
                        {"output_token_ids": [1, 2, 3]},
                        {"output_token_ids": [1, 2, 3]},
                    ],
                }
            ]
        }
        mtp = {
            "results": [
                {
                    "prompt_case_id": "case",
                    "trials": [
                        {"output_token_ids": [1, 2, 4]},
                        {"output_token_ids": [1, 2, 4]},
                    ],
                }
            ]
        }

        bench.validate_exact_seed_reproducibility(
            Path("direct.json"), direct, Path("mtp.json"), mtp
        )

        mtp["results"][0]["trials"][1]["output_token_ids"] = [1, 2, 5]
        with self.assertRaisesRegex(ValueError, "seed-reproducibility oracle failed"):
            bench.validate_exact_seed_reproducibility(
                Path("direct.json"), direct, Path("mtp.json"), mtp
            )

    def test_qwen_draft_quality_uses_prompt_median_target_match_ewma(self) -> None:
        artifact = {
            "results": [
                self.mtp_row(
                    case_id="low",
                    match_x1000=200,
                    mtp_steps=8,
                    fallback_steps=982,
                    emitted_tokens=17,
                ),
                self.mtp_row(
                    case_id="high",
                    match_x1000=1000,
                    mtp_steps=499,
                    fallback_steps=0,
                    emitted_tokens=999,
                ),
            ]
        }

        quality, kind = bench.draft_quality(artifact, assistant_mtp=False)

        self.assertEqual(quality, 60.0)
        self.assertEqual(kind, "target_argmax_match_ewma")

    def test_qwen_draft_quality_fails_closed_without_match_ewma(self) -> None:
        artifact = {
            "results": [
                {
                    "prompt_case_id": "missing",
                    "ngram_acceleration_telemetry": {
                        "ax_mtp_mtp_only_accept_rate_ewma_samples": 8,
                    },
                }
            ]
        }

        with self.assertRaisesRegex(
            ValueError, "target-match EWMA telemetry is missing"
        ):
            bench.draft_quality(artifact, assistant_mtp=False)

    def test_mtp_coverage_exposes_direct_fallback(self) -> None:
        artifact = {
            "results": [
                self.mtp_row(
                    case_id="fallback",
                    match_x1000=200,
                    mtp_steps=8,
                    fallback_steps=982,
                    emitted_tokens=17,
                ),
                self.mtp_row(
                    case_id="effective",
                    match_x1000=1000,
                    mtp_steps=499,
                    fallback_steps=0,
                    emitted_tokens=999,
                ),
            ]
        }

        coverage = bench.mtp_coverage(artifact)

        self.assertEqual(coverage["fallback_prompt_count"], 1)
        self.assertEqual(coverage["prompt_count"], 2)
        self.assertEqual(coverage["decode_route_steps"], 1489)
        self.assertAlmostEqual(
            float(coverage["step_coverage_pct"]), 507 / 1489 * 100.0
        )

    def test_mtp_coverage_requires_prompt_case_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "no prompt-case rows"):
            bench.mtp_coverage({"results": []})

    def test_approximate_table_labels_policy_and_fallback_metrics(self) -> None:
        row = {
            "model_id": "qwen3.6-35b-a3b",
            "suite_id": "long_code",
            "ax_direct_decode_tok_s": 121.0,
            "ax_mtp_decode_tok_s": 121.6,
            "ax_mtp_speedup_x": 1.005,
            "ax_mtp_draft_quality_pct": 21.1,
            "ax_mtp_draft_quality_kind": "target_argmax_match_ewma",
            "ax_mtp_step_coverage_pct": 15.1,
            "ax_mtp_fallback_prompt_count": 3,
            "prompt_count": 4,
        }

        table = "\n".join(
            bench.table_lines([row], approximate_diagnostic=True)
        )

        self.assertIn("Approx. MTP decode", table)
        self.assertIn("21.1% match", table)
        self.assertIn("15.1%", table)
        self.assertIn("3/4", table)

    def test_exact_table_labels_acceleration_and_context_metrics(self) -> None:
        row = self.exact_summary()["rows"][0]

        table = "\n".join(
            bench.table_lines([row], approximate_diagnostic=False)
        )

        self.assertIn("AX MTP decode", table)
        self.assertIn("AX speedup", table)
        self.assertIn("AX MTP prefill", table)
        self.assertIn("AX MTP TTFT", table)
        self.assertIn("2.00x", table)

    def test_update_readme_replaces_legacy_diagnostic_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            readme = Path(tmp) / "README.md"
            readme.write_text(
                "before\n\n"
                "#### AX Engine 6-bit approximate MTP diagnostic (2026-07-11)\n\n"
                "legacy diagnostic\n\n"
                "#### Qwen3.6 MTP peer decode comparison (2026-07-09)\n\n"
                "peer section\n"
            )

            bench.update_readme(readme, self.exact_summary())

            updated = readme.read_text()
        self.assertIn(
            "#### AX Engine 6-bit exact sampled-MTP acceleration (2026-07-13)",
            updated,
        )
        self.assertIn("All 15 target/suite rows accelerate decode", updated)
        self.assertIn("perf-mtp-6bit-ax-acceleration.svg", updated)
        self.assertIn("#### Qwen3.6 MTP peer decode comparison", updated)
        self.assertNotIn("legacy diagnostic", updated)
        self.assertNotIn("approximate-diagnostic.svg", updated)

    def test_update_readme_fails_closed_for_ineligible_exact_summary(self) -> None:
        mutations = {
            "summary publication": lambda summary: summary.update(
                publication_candidate=False
            ),
            "row speedup": lambda summary: summary["rows"][0].update(
                ax_mtp_decode_tok_s=50.0,
                ax_mtp_speedup_x=1.0,
            ),
            "fallback": lambda summary: summary["rows"][0].update(
                ax_mtp_fallback_prompt_count=1
            ),
            "ngram": lambda summary: summary["rows"][0][
                "ax_mtp_ngram_telemetry"
            ].update(ax_mtp_ngram_hit_steps=1),
            "partial matrix": lambda summary: summary["rows"].pop(),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                summary = copy.deepcopy(self.exact_summary())
                mutate(summary)
                with self.assertRaises(ValueError):
                    bench.render_readme_section(summary)


if __name__ == "__main__":
    unittest.main()
