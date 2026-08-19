from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import bench_mtp_6bit_ax_refresh as bench


class BenchMtpRefreshTests(unittest.TestCase):
    @staticmethod
    def publication_artifact() -> dict[str, object]:
        conditions = {
            "load_average": {"one_minute": 1.0},
            "power_source": "AC Power",
            "thermal_warning_recorded": False,
            "performance_warning_recorded": False,
            "cpu_power_status_recorded": False,
            "top_processes_cpu": [{"cpu_percent": 10.0}],
        }
        return {
            "schema_version": bench.MLX_INFERENCE_STACK_SCHEMA,
            "warmup_repetitions": 2,
            "repetitions": 5,
            "cooldown": 15.0,
            "generation_tokens": 1000,
            "ax_prefix_cache_mode": "disabled_for_cold_prefill_benchmark",
            "build": {
                "git_tracked_dirty": False,
                "build_profile": "release",
            },
            "run_stability_summary": {"publication_candidate": True},
            "benchmark_window": {
                "performance_conditions_start": conditions,
                "performance_conditions_end": conditions,
            },
        }

    @staticmethod
    def exact_artifact(
        *,
        engine: str,
        suite: str = "flappy",
        model_dir: str = "/models/test",
    ) -> dict[str, object]:
        row = {
            "prompt_case_id": "case",
            "engine": engine,
            "prompt_source": "real",
            "prompt_suite_id": suite,
            "prompt_text_sha256": "a" * 64,
            "prompt_token_ids_sha256": "b" * 64,
            "prompt_tokens": 128,
            "generation_tokens": 1000,
            "sampler_settings": bench.MTP_SAMPLER_SIGNATURE,
            "seed": 0,
            "random_seed": 0,
            "run_stability": {"classification": "stable_enough"},
            "decode_tok_s": {"median": 50.0},
            "prefill_tok_s": {"median": 500.0},
            "ttft_ms": {"median": 250.0},
            "trials": [
                {
                    "output_token_ids": list(range(1000)),
                    "output_tokens": 1000.0,
                }
                for _ in range(5)
            ],
        }
        return {
            "model_dir": model_dir,
            "results": [row],
        }

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
            "schema": bench.MTP_6BIT_EXACT_SCHEMA,
            "publication_candidate": True,
            "claim_type": bench.MTP_6BIT_EXACT_CLAIM_TYPE,
            "engine_version": "6.9.0",
            "build_commit": "a" * 40,
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

    def test_skip_existing_requires_publishable_run_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "artifact.json"
            artifact = self.publication_artifact()
            artifact["results"] = [{"engine": "ax_engine_mlx"}]
            artifact_path.write_text(json.dumps(artifact))

            self.assertTrue(bench.existing_artifact_ok(artifact_path))

            artifact["benchmark_window"]["performance_conditions_end"][
                "load_average"
            ]["one_minute"] = bench.MAX_PUBLICATION_LOAD_AVERAGE + 0.1
            artifact_path.write_text(json.dumps(artifact))

            self.assertFalse(bench.existing_artifact_ok(artifact_path))

    def test_skip_existing_rejects_missing_or_malformed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "artifact.json"
            self.assertFalse(bench.existing_artifact_ok(artifact_path))

            artifact_path.write_text("{")
            self.assertFalse(bench.existing_artifact_ok(artifact_path))

            artifact_path.write_text(json.dumps({"results": []}))
            self.assertFalse(bench.existing_artifact_ok(artifact_path))

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
        self.assertEqual(
            command[command.index("--max-load-average") + 1],
            str(bench.MAX_PUBLICATION_LOAD_AVERAGE),
        )
        self.assertEqual(
            command[command.index("--max-top-process-cpu-percent") + 1],
            str(bench.MAX_PUBLICATION_PROCESS_CPU_PERCENT),
        )
        self.assertEqual(
            command[command.index("--load-average-wait-timeout") + 1],
            str(bench.DEFAULT_LOAD_WAIT_TIMEOUT_S),
        )
        self.assertEqual(
            command[command.index("--load-average-poll-interval") + 1],
            str(bench.DEFAULT_LOAD_POLL_INTERVAL_S),
        )
        self.assertIn("--ax-qwen-linear-mtp-exact", command)

    def test_formal_mtp_disables_runtime_bypasses(self) -> None:
        self.assertEqual(
            bench.FORMAL_MTP_ENV,
            {
                "AX_MLX_MTP_BYPASS_THRESHOLD": "0",
                "AX_MLX_MTP_MIN_REMAINING_TOKENS": "0",
            },
        )

    def test_maybe_run_case_forwards_formal_env_only_for_mtp(self) -> None:
        target = bench.Target(
            key="test",
            label="Test",
            mode="MTP",
            model_dir=Path("/models/test"),
            mtp_depth=1,
        )
        args = argparse.Namespace(
            skip_existing=False,
            suites_dir=Path("/prompts"),
            generated_tokens=32,
            repetitions=5,
            warmup_repetitions=2,
            cooldown=0.0,
            inter_case_cooldown=0.0,
            approximate_speed_ceiling=False,
        )

        with mock.patch.object(bench, "run_logged") as run_logged:
            bench.maybe_run_case(
                target=target,
                suite="sample",
                mode="direct",
                output_path=Path("/tmp/direct.json"),
                args=args,
            )
            self.assertIsNone(run_logged.call_args.kwargs["env_overrides"])

            bench.maybe_run_case(
                target=target,
                suite="sample",
                mode="mtp",
                output_path=Path("/tmp/mtp.json"),
                args=args,
            )
            self.assertEqual(
                run_logged.call_args.kwargs["env_overrides"], bench.FORMAL_MTP_ENV
            )

    def test_exact_artifact_validation_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an exact MTP publication candidate"):
            bench.validate_exact_mtp_artifact(Path("artifact.json"), {})

        bench.validate_exact_mtp_artifact(
            Path("artifact.json"),
            {"mtp_correctness_summary": {"publication_candidate": True}},
        )
        with self.assertRaisesRegex(ValueError, "exact verifier profile"):
            bench.validate_exact_mtp_artifact(
                Path("artifact.json"),
                {"mtp_correctness_summary": {"publication_candidate": True}},
                require_qwen_linear_exact=True,
            )
        bench.validate_exact_mtp_artifact(
            Path("artifact.json"),
            {
                "mtp_correctness_summary": {"publication_candidate": True},
                "ax_qwen_linear_mtp_exact": True,
                "ax_qwen_linear_mtp_exact_explicit_enable": True,
            },
            require_qwen_linear_exact=True,
        )

    def test_exact_publication_methodology_requires_clean_two_by_five(self) -> None:
        valid = self.publication_artifact()
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
        self.assertIn("direct_requires_release_build", reasons)

    def test_exact_publication_methodology_rejects_bad_run_conditions(self) -> None:
        artifact = self.publication_artifact()
        artifact["benchmark_window"]["performance_conditions_end"][
            "load_average"
        ]["one_minute"] = 2.1

        reasons = bench.exact_publication_methodology_reasons(artifact, artifact)

        self.assertIn(
            "direct_performance_conditions_end_load_above_limit",
            reasons,
        )

    def test_build_identity_comes_from_measured_artifacts(self) -> None:
        build = {
            "build": {
                "engine_version": "6.13.1",
                "commit": "b" * 40,
            }
        }
        identity = bench.matching_build_identity(
            Path("direct.json"),
            build,
            Path("mtp.json"),
            copy.deepcopy(build),
        )

        self.assertEqual(identity.engine_version, "6.13.1")
        self.assertEqual(identity.commit, "b" * 40)

    def test_build_identity_rejects_mixed_or_abbreviated_commits(self) -> None:
        direct = {
            "build": {
                "engine_version": "6.13.1",
                "commit": "b" * 40,
            }
        }
        mtp = copy.deepcopy(direct)
        mtp["build"]["commit"] = "c" * 40
        with self.assertRaisesRegex(ValueError, "build identity differs"):
            bench.matching_build_identity(
                Path("direct.json"), direct, Path("mtp.json"), mtp
            )

        direct["build"]["commit"] = "bff75300"
        with self.assertRaisesRegex(ValueError, "full measured build commit"):
            bench.artifact_build_identity(Path("direct.json"), direct)

    def test_exact_artifact_rows_require_complete_stable_trials(self) -> None:
        artifact = self.exact_artifact(engine="ax_engine_mlx")
        bench.validate_exact_artifact_rows(
            Path("direct.json"),
            artifact,
            expected_engines={"ax_engine_mlx"},
            expected_suite="flappy",
        )

        artifact["results"][0]["trials"][0]["output_token_ids"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete generated-token trial"):
            bench.validate_exact_artifact_rows(
                Path("direct.json"),
                artifact,
                expected_engines={"ax_engine_mlx"},
                expected_suite="flappy",
            )

    def test_exact_prompt_parity_requires_same_package_and_prompt_hashes(self) -> None:
        direct = self.exact_artifact(engine="ax_engine_mlx")
        mtp = self.exact_artifact(engine="ax_engine_mlx_pure_mtp")
        bench.validate_exact_prompt_parity(
            Path("direct.json"), direct, Path("mtp.json"), mtp
        )

        mtp["results"][0]["prompt_token_ids_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "decode contract differs"):
            bench.validate_exact_prompt_parity(
                Path("direct.json"), direct, Path("mtp.json"), mtp
            )

        mtp["results"][0]["prompt_token_ids_sha256"] = "b" * 64
        mtp["model_dir"] = "/models/other"
        with self.assertRaisesRegex(ValueError, "model packages differ"):
            bench.validate_exact_prompt_parity(
                Path("direct.json"), direct, Path("mtp.json"), mtp
            )

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
        self.assertNotIn("--ax-qwen-linear-mtp-exact", direct_command)
        self.assertNotIn("--ax-mtp-approximate-optimistic", direct_command)
        self.assertIn("--ax-mtp-approximate-optimistic", mtp_command)
        self.assertIn("--ax-qwen-linear-mtp-exact", mtp_command)

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

    def test_exact_table_labels_comparison_and_context_metrics(self) -> None:
        row = self.exact_summary()["rows"][0]

        table = "\n".join(
            bench.table_lines([row], approximate_diagnostic=False)
        )

        self.assertIn("AX MTP decode", table)
        self.assertIn("AX MTP/direct", table)
        self.assertIn("AX MTP prefill", table)
        self.assertIn("AX MTP TTFT", table)
        self.assertIn("2.00x", table)

    def test_update_readme_reports_wins_and_losses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            readme = Path(tmp) / "README.md"
            readme.write_text(
                "before\n\n"
                "#### AX Engine 6-bit approximate MTP diagnostic (2026-07-11)\n\n"
                "legacy diagnostic\n\n"
                "#### Qwen3.6 MTP peer decode comparison (2026-07-09)\n\n"
                "peer section\n"
            )

            summary = self.exact_summary()
            summary["rows"][0].update(
                ax_mtp_decode_tok_s=40.0,
                ax_mtp_speedup_x=0.8,
            )
            bench.update_readme(readme, summary)

            updated = readme.read_text()
        self.assertIn(
            "#### AX Engine v6.9.0 6-bit exact sampled-MTP comparison (2026-07-13)",
            updated,
        )
        self.assertIn(
            "Across 18 target/suite rows: 17 MTP wins, 0 ties, and 1 loss; "
            "MTP/direct ratios span 0.80x-2.00x.",
            updated,
        )
        self.assertIn("perf-mtp-6bit-ax-acceleration.svg", updated)
        self.assertIn(
            "](../benchmarks/results/speculative/mtp-6bit/"
            "2026-07-13-exact/summary.json)",
            updated,
        )
        self.assertIn("#### Qwen3.6 MTP peer decode comparison", updated)
        self.assertNotIn("legacy diagnostic", updated)
        self.assertNotIn("approximate-diagnostic.svg", updated)

    def test_update_readme_fails_closed_for_ineligible_exact_summary(self) -> None:
        mutations = {
            "summary publication": lambda summary: summary.update(
                publication_candidate=False
            ),
            "missing measured commit": lambda summary: summary.pop("build_commit"),
            "inconsistent ratio": lambda summary: summary["rows"][0].update(
                ax_mtp_decode_tok_s=50.0,
                ax_mtp_speedup_x=2.0,
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
