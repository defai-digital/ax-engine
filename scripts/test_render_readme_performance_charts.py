#!/usr/bin/env python3
"""Unit tests for README performance chart helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CHART_SCRIPT_PATH = Path(__file__).with_name("render_readme_performance_charts.py")
CHART_MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_readme_performance_charts", CHART_SCRIPT_PATH
)
assert CHART_MODULE_SPEC and CHART_MODULE_SPEC.loader
charts = importlib.util.module_from_spec(CHART_MODULE_SPEC)
sys.modules[CHART_MODULE_SPEC.name] = charts
CHART_MODULE_SPEC.loader.exec_module(charts)

GEMMA12_CHART_SCRIPT_PATH = Path(__file__).with_name(
    "render_gemma4_12b_direct_charts.py"
)
GEMMA12_CHART_MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_gemma4_12b_direct_charts", GEMMA12_CHART_SCRIPT_PATH
)
assert GEMMA12_CHART_MODULE_SPEC and GEMMA12_CHART_MODULE_SPEC.loader
gemma12_charts = importlib.util.module_from_spec(GEMMA12_CHART_MODULE_SPEC)
sys.modules[GEMMA12_CHART_MODULE_SPEC.name] = gemma12_charts
GEMMA12_CHART_MODULE_SPEC.loader.exec_module(gemma12_charts)

MTP_SCRIPT_PATH = Path(__file__).with_name("bench_mtp_6bit_ax_refresh.py")
MTP_MODULE_SPEC = importlib.util.spec_from_file_location(
    "bench_mtp_6bit_ax_refresh", MTP_SCRIPT_PATH
)
assert MTP_MODULE_SPEC and MTP_MODULE_SPEC.loader
mtp_refresh = importlib.util.module_from_spec(MTP_MODULE_SPEC)
sys.modules[MTP_MODULE_SPEC.name] = mtp_refresh
MTP_MODULE_SPEC.loader.exec_module(mtp_refresh)


class ReadmePerformanceChartTests(unittest.TestCase):
    @staticmethod
    def exact_mtp_chart_summary() -> dict[str, object]:
        rows = []
        for target in mtp_refresh.SUPPORTED_TARGETS:
            for suite in mtp_refresh.DEFAULT_SUITES:
                rows.append(
                    {
                        "model_id": target.key,
                        "model": target.label,
                        "suite_id": suite,
                        "ax_direct_decode_tok_s": 50.0,
                        "ax_mtp_decode_tok_s": 100.0,
                        "ax_mtp_speedup_x": 2.0,
                        "ax_mtp_step_coverage_pct": 100.0,
                        "ax_mtp_fallback_prompt_count": 0,
                        "ax_mtp_direct_fallback_steps": 0,
                        "publication_candidate": True,
                        "publication_reasons": [],
                        "ax_mtp_ngram_telemetry": {
                            key: 0 for key in mtp_refresh.NGRAM_ZERO_KEYS
                        },
                    }
                )
        return {
            "schema": charts.MTP_6BIT_EXACT_SCHEMA,
            "publication_candidate": True,
            "claim_type": "exact_mtp_comparison",
            "engine_version": "6.9.0",
            "build_commit": "a" * 40,
            "rows": rows,
        }

    @staticmethod
    def mtp_peer_summary() -> dict[str, object]:
        labels = {
            "27b-4bit": "Qwen3.6 27B 4-bit",
            "27b-6bit": "Qwen3.6 27B 6-bit",
            "35b-a3b-4bit": "Qwen3.6 35B-A3B 4-bit",
            "35b-a3b-6bit": "Qwen3.6 35B-A3B 6-bit",
        }
        rows = []
        for (target, engine), status in charts.MTP_PEER_EXPECTED_STATUS.items():
            supported = status == "supported"
            rows.append(
                {
                    "target": target,
                    "model_label": labels[target],
                    "suite": "flappy",
                    "engine": engine,
                    "status": status,
                    "artifact": f"benchmarks/results/{target}/{engine}.json",
                    "publication_candidate": supported,
                    "publication_reasons": [],
                    "metrics": (
                        {
                            "status": "ok",
                            "case_count": 4,
                            "decode_tok_s": 50.0,
                            "prefill_tok_s": 500.0,
                            "ttft_ms": 100.0,
                            "accept_rate": 0.5,
                            "degeneracy_gate": {
                                "degenerate": False,
                                "evidence_complete": True,
                            },
                        }
                        if supported
                        else {"status": "unsupported"}
                    ),
                }
            )
        return {
            "schema": charts.MTP_PEER_SCHEMA,
            "publication_candidate": True,
            "publication_reasons": [],
            "engine_identities": {
                "ax_engine": {
                    "name": "AX Engine",
                    "version": "6.13.2",
                    "commit": "a" * 40,
                },
                "mtplx": {
                    "name": "MTPLX",
                    "version": "2.1.0",
                    "commit": "b" * 40,
                },
                "lightning_mlx": {
                    "name": "lightning-mlx",
                    "version": "0.8.0",
                    "commit": "c" * 40,
                },
            },
            "contract": {
                "models": ["27b", "35b-a3b"],
                "bits": [4, 6],
                "engines": [
                    "ax_engine",
                    "mtplx",
                    "lightning_mlx",
                    "rapid_mlx",
                    "omlx",
                ],
                "suites": ["flappy"],
                "benchmark_contract": "apples-to-apples",
                "mode": "mtp",
                "max_tokens": 1000,
                "repetitions": 5,
                "warmup_repetitions": 2,
                "cooldown_s": 15.0,
                "inter_case_cooldown_s": 10.0,
                "seed": 0,
                "ax_mtp_optimistic": False,
                "lightning_mtp_optimistic": False,
                "lightning_prefix_cache_policy": "disabled_for_cold_prefill",
                "publication_load_gate": {
                    "max_load_average": 2.0,
                    "max_top_process_cpu_percent": 50.0,
                    "load_wait_timeout_seconds": 900.0,
                    "load_poll_interval_seconds": 5.0,
                },
                "sampling": {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                },
            },
            "rows": rows,
        }

    def test_family_boxplots_cover_all_direct_metrics(self) -> None:
        self.assertEqual(len(charts.CHARTS), 6)
        self.assertEqual(
            {(chart.family, chart.metric) for chart in charts.CHARTS},
            {
                ("gemma4", "decode"),
                ("gemma4", "prefill"),
                ("gemma4", "ttft"),
                ("qwen", "decode"),
                ("qwen", "prefill"),
                ("qwen", "ttft"),
            },
        )

    def test_ax_direct_snapshot_charts_are_complete_and_ax_only(self) -> None:
        snapshot_path = (
            charts.REPO_ROOT
            / "benchmarks/results/inference/ax-direct/"
            "2026-07-27-v6.12.0-m5max-ax-direct-only/sweep_results.json"
        )
        legacy_snapshot = json.loads(snapshot_path.read_text())
        with self.assertRaisesRegex(
            charts.ChartError, "README publication eligible"
        ):
            charts.load_ax_direct_snapshot(snapshot_path)

        expected_slugs = [row["slug"] for row in legacy_snapshot["rows"]]
        legacy_snapshot["readme_ax_direct_publication_candidate"] = True
        legacy_snapshot["ax_direct_matrix"] = {
            "schema_version": charts.AX_DIRECT_MATRIX_SCHEMA,
            "expected_slugs": expected_slugs,
            "expected_model_count": len(expected_slugs),
            "publication_model_count": len(expected_slugs),
            "expected_cell_count": len(expected_slugs) * len(charts.PROMPT_TOKENS),
            "publication_cell_count": len(expected_slugs)
            * len(charts.PROMPT_TOKENS),
            "publication_candidate": True,
        }
        for row in legacy_snapshot["rows"]:
            row["result_doc"]["build"]["git_tracked_dirty"] = False
            row["result_doc"]["run_stability_summary"][
                "publication_candidate"
            ] = True
        with tempfile.TemporaryDirectory() as root_name:
            gated_snapshot_path = Path(root_name) / "sweep_results.json"
            gated_snapshot_path.write_text(json.dumps(legacy_snapshot))
            snapshot = charts.load_ax_direct_snapshot(gated_snapshot_path)
            legacy_snapshot["rows"][0]["readme_model"] = "Gemma 4 E4B"
            gated_snapshot_path.write_text(json.dumps(legacy_snapshot))
            with self.assertRaisesRegex(charts.ChartError, "display label"):
                charts.load_ax_direct_snapshot(gated_snapshot_path)

        self.assertEqual(snapshot["engine_version"], "6.12.0")
        self.assertEqual(len(snapshot["rows"]), 12)
        chart_rows = charts.ax_direct_snapshot_chart_rows(snapshot, "decode")
        self.assertEqual(len(chart_rows), 36)
        self.assertAlmostEqual(
            chart_rows[0]["decode_tok_s"]["median"], 232.1, places=1
        )
        self.assertTrue(
            all(row["engine"] == "ax_engine_mlx" for row in chart_rows)
        )

        readme = charts.REPO_ROOT / "docs/PERFORMANCE-RESULTS.md"
        retained_mlx_rows = charts.load_retained_mlx_lm_rows(
            readme, charts.readme_model_slugs(readme)
        )
        self.assertTrue(all(row["engine"] == "mlx_lm" for row in retained_mlx_rows))
        gemma_decode_spec = next(
            spec
            for spec in charts.CHARTS
            if spec.family == "gemma4" and spec.metric == "decode"
        )
        boxplot = charts.render_family_chart(
            gemma_decode_spec,
            charts.collect_family_values(
                retained_mlx_rows
                + chart_rows
                + charts.load_llama_rows_from_readme(readme),
                gemma_decode_spec,
                ax_engine_version=str(snapshot["engine_version"]),
            ),
            ax_engine_version=str(snapshot["engine_version"]),
        )
        self.assertIn("AX Engine v6.12.0", boxplot)
        self.assertIn("retained mlx-lm 0.31.3", boxplot)
        self.assertIn("cross-run distribution", boxplot)

        readme_text = readme.read_text()
        for row in snapshot["rows"]:
            values = [
                row["metrics"][metric][prompt_tokens]
                for metric in ("decode_tok_s", "prefill_tok_s", "ttft_ms")
                for prompt_tokens in (128, 512, 2048)
            ]
            model_label, quant_label = row["label"].rsplit(" ", 1)
            table_row = "| {} | {} |".format(
                ("Gemma 4 " if row["family"] == "gemma" else "Qwen 3.6 ")
                + model_label,
                quant_label,
            )
            table_row += "".join(f" {value:,.1f} |" for value in values)
            self.assertIn(table_row, readme_text)

    def test_mlx_lm_direct_snapshot_requires_clean_complete_reference_matrix(
        self,
    ) -> None:
        self.assertNotIn(
            "gemma-4-e2b-it-6bit",
            {
                slug
                for slugs in charts.FAMILY_SLUGS.values()
                for slug in slugs
            },
        )
        self.assertIn("gemma-4-e2b-it-6bit", charts.AX_DIRECT_EXPECTED_SLUGS)
        source_path = (
            charts.REPO_ROOT
            / "benchmarks/results/inference/ax-direct/"
            "2026-07-27-v6.12.0-m5max-ax-direct-only/sweep_results.json"
        )
        snapshot = json.loads(source_path.read_text())
        expected_slugs = [
            slug for slugs in charts.FAMILY_SLUGS.values() for slug in slugs
        ]
        snapshot["rows"] = [
            row for row in snapshot["rows"] if row["slug"] in expected_slugs
        ]
        snapshot.update(
            ax_direct_only=False,
            mlx_lm_reference_only=True,
            readme_reference_publication_candidate=True,
        )
        snapshot["reference_matrix"] = {
            "schema_version": charts.MLX_LM_REFERENCE_MATRIX_SCHEMA,
            "expected_slugs": expected_slugs,
            "expected_model_count": len(expected_slugs),
            "publication_model_count": len(expected_slugs),
            "expected_cell_count": len(expected_slugs) * len(charts.PROMPT_TOKENS),
            "publication_cell_count": len(expected_slugs)
            * len(charts.PROMPT_TOKENS),
            "publication_candidate": True,
        }
        for row in snapshot["rows"]:
            result_doc = row["result_doc"]
            result_doc["build"]["git_tracked_dirty"] = False
            result_doc["host"]["toolchain"]["python_mlx_lm"] = "0.31.3"
            for result in result_doc["results"]:
                result["engine"] = "mlx_lm"

        with tempfile.TemporaryDirectory() as root_name:
            snapshot_path = Path(root_name) / "sweep_results.json"
            snapshot_path.write_text(json.dumps(snapshot))
            loaded = charts.load_mlx_lm_direct_snapshot(snapshot_path)
            self.assertEqual(loaded["version"], "0.31.3")
            self.assertEqual(
                len(loaded["rows"]),
                len(expected_slugs) * len(charts.PROMPT_TOKENS),
            )

            snapshot["rows"][0]["result_doc"]["build"]["git_tracked_dirty"] = True
            snapshot_path.write_text(json.dumps(snapshot))
            with self.assertRaisesRegex(charts.ChartError, "dirty tracked tree"):
                charts.load_mlx_lm_direct_snapshot(snapshot_path)

            snapshot["rows"][0]["result_doc"]["build"]["git_tracked_dirty"] = False
            snapshot["rows"][0]["result_doc"]["build"]["commit"] = "abcdef12"
            snapshot_path.write_text(json.dumps(snapshot))
            with self.assertRaisesRegex(charts.ChartError, "full source commit"):
                charts.load_mlx_lm_direct_snapshot(snapshot_path)

    def test_ax_direct_snapshot_marker_is_separate_from_legacy_sources(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text("{}\n")
            readme = root / "README.md"
            readme.write_text(
                "<!-- readme-ax-direct-snapshot: snapshot.json -->\n"
                "<!-- readme-performance-artifacts: reference=legacy/ -->\n"
            )

            self.assertEqual(
                charts.find_ax_direct_snapshot(readme), snapshot_path.resolve()
            )

    def test_mlx_lm_direct_snapshot_marker_is_separate_from_legacy_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            snapshot_path = root / "reference.json"
            snapshot_path.write_text("{}\n")
            readme = root / "README.md"
            readme.write_text(
                "<!-- readme-mlx-lm-direct-snapshot: reference.json -->\n"
                "<!-- readme-performance-artifacts: reference=legacy/ -->\n"
            )

            self.assertEqual(
                charts.find_mlx_lm_direct_snapshot(readme),
                snapshot_path.resolve(),
            )

    def test_llama_cpp_chart_label_comes_from_result_marker(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            readme = Path(root_name) / "README.md"
            readme.write_text(
                "<!-- readme-llama-cpp-build: b10050 -->\n",
                encoding="utf-8",
            )
            spec = next(
                item
                for item in charts.CHARTS
                if item.family == "gemma4" and item.metric == "decode"
            )

            build = charts.find_llama_cpp_build(readme)
            labels = {
                engine: label
                for engine, label, _color, _dot in charts.series_for_chart(
                    spec,
                    mlx_lm_version="0.31.4",
                    llama_cpp_build=build,
                )
            }

            self.assertEqual(build, "b10050")
            self.assertEqual(labels["mlx_lm"], "mlx-lm 0.31.4")
            self.assertEqual(labels["llama_cpp_metal"], "llama.cpp b10050")
            self.assertIn(
                "retained llama.cpp b10050",
                charts.direct_versions_footnote(
                    "6.13.2", llama_cpp_build=build
                ),
            )
            self.assertEqual(
                charts.direct_versions_footnote(
                    "6.13.3",
                    snapshot_date="2026-08-07",
                    mlx_lm_version="0.31.3",
                    mlx_lm_snapshot_date="2026-08-07",
                    llama_cpp_build=build,
                ),
                (
                    "AX v6.13.3 (2026-08-07) · mlx-lm 0.31.3 "
                    "(2026-08-07) · llama.cpp b10050 · separate runs"
                ),
            )

            readme.write_text(
                "<!-- readme-llama-cpp-build: b10050 -->\n"
                "<!-- readme-llama-cpp-build: malformed -->\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(charts.ChartError, "exactly one"):
                charts.find_llama_cpp_build(readme)

    def test_gemma4_12b_decode_uses_llama_matched_depth(self) -> None:
        row = {
            "engine": "llama_cpp_metal",
            "decode_tok_s": {"median": 57.1},
            "decode_at_depth_tok_s": {"median": 56.9},
        }

        self.assertEqual(gemma12_charts.metric_median(row, "decode_tok_s"), 56.9)

    def test_gemma4_12b_chart_preserves_values_and_peer_version(self) -> None:
        data = {
            "llama_cpp_metal": {128: 1244.49, 512: 1739.45, 2048: 1543.56},
            "ax_engine_mlx": {128: 1184.35, 512: 1867.30, 2048: 2049.06},
        }

        svg = gemma12_charts.render_chart(
            title="Prefill", unit="tok/s", lower_is_better=False, data=data
        )

        self.assertIn("llama.cpp Metal b9820", svg)
        self.assertNotIn("b9700", svg)
        self.assertIn(">1,244</text>", svg)
        self.assertIn(">1,184</text>", svg)

    def test_chart_merge_keeps_ax_high_water_row(self) -> None:
        rows = {
            ("gemma-4-e2b-it-4bit", "ax_engine_mlx", 128, 128): {
                "engine": "ax_engine_mlx",
                "prefill_tok_s": {"median": 100.0},
                "decode_tok_s": {"median": 10.0},
                "ttft_ms": {"median": 40.0},
            }
        }

        charts.merge_chart_row(
            rows,
            ("gemma-4-e2b-it-4bit", "ax_engine_mlx", 128, 128),
            {
                "engine": "ax_engine_mlx",
                "prefill_tok_s": {"median": 90.0},
                "decode_tok_s": {"median": 9.0},
                "ttft_ms": {"median": 30.0},
            },
            "prefill",
        )
        self.assertEqual(
            rows[("gemma-4-e2b-it-4bit", "ax_engine_mlx", 128, 128)][
                "prefill_tok_s"
            ]["median"],
            100.0,
        )

        charts.merge_chart_row(
            rows,
            ("gemma-4-e2b-it-4bit", "ax_engine_mlx", 128, 128),
            {
                "engine": "ax_engine_mlx",
                "prefill_tok_s": {"median": 90.0},
                "decode_tok_s": {"median": 9.0},
                "ttft_ms": {"median": 30.0},
            },
            "ttft",
        )
        self.assertEqual(
            rows[("gemma-4-e2b-it-4bit", "ax_engine_mlx", 128, 128)]["ttft_ms"][
                "median"
            ],
            30.0,
        )

    def test_mtp_6bit_summary_accepts_speculative_results_tree(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            summary_path = (
                root
                / "benchmarks/results/speculative/mtp-6bit/local-run/summary.json"
            )
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text('{"rows": []}\n')
            readme = root / "README.md"
            readme.write_text(
                "[summary](benchmarks/results/speculative/mtp-6bit/local-run/summary.json)\n"
            )

            self.assertEqual(
                charts.find_mtp_6bit_summary(readme), summary_path.resolve()
            )

        performance_results = charts.REPO_ROOT / "docs/PERFORMANCE-RESULTS.md"
        self.assertEqual(
            charts.find_mtp_6bit_summary(performance_results),
            charts.REPO_ROOT
            / "benchmarks/results/speculative/mtp-6bit/2026-08-06-v6.13.1-m5max-supported-mtp-ax-only/summary.json",
        )

    def test_mtp_6bit_refresh_defaults_to_speculative_results_tree(self) -> None:
        self.assertTrue(
            mtp_refresh.DEFAULT_OUTPUT_BASE.as_posix().endswith(
                "/benchmarks/results/speculative/mtp-6bit"
            )
        )

    def test_mtp_peer_summary_resolves_from_performance_results(self) -> None:
        performance_results = charts.REPO_ROOT / "docs/PERFORMANCE-RESULTS.md"
        self.assertEqual(
            charts.find_mtp_peer_summary(performance_results),
            charts.REPO_ROOT
            / "benchmarks/results/mtp-qwen36-matrix/2026-08-07-peer-comparison-apples-to-apples-refresh/summary.json",
        )

    def test_mtp_peer_chart_uses_measured_engine_identities(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            summary_path = Path(root_name) / "2026-08-06-peer" / "summary.json"
            summary_path.parent.mkdir()
            summary_path.write_text(json.dumps(self.mtp_peer_summary()))

            rows = charts.load_mtp_peer_rows(summary_path)
            chart = charts.render_mtp_peer_comparison_chart(
                rows, summary_path, "decode"
            )

        self.assertEqual(len(rows), 10)
        self.assertIn("AX Engine v6.13.2", chart)
        self.assertIn("MTPLX v2.1.0", chart)
        self.assertIn("lightning-mlx v0.8.0", chart)
        self.assertIn("(aaaaaaaa)", chart)

    def test_mtp_peer_chart_rejects_stitched_or_ineligible_summary(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            summary_path = Path(root_name) / "summary.json"
            summary = self.mtp_peer_summary()
            summary["schema"] = "ax.qwen36_mtp_peer_comparison_stitched.v1"
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(charts.ChartError, "summary.v2"):
                charts.load_mtp_peer_rows(summary_path)

            summary = self.mtp_peer_summary()
            summary["rows"][0]["publication_candidate"] = False
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(charts.ChartError, "not publication eligible"):
                charts.load_mtp_peer_rows(summary_path)

    def test_mtp_peer_chart_requires_full_measured_identity(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            summary_path = Path(root_name) / "summary.json"
            summary = self.mtp_peer_summary()
            summary["engine_identities"]["mtplx"]["commit"] = "short"
            summary_path.write_text(json.dumps(summary))

            with self.assertRaisesRegex(charts.ChartError, "full commit"):
                charts.load_mtp_peer_rows(summary_path)

    def test_mtp_approximate_summary_must_be_non_publishable(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            summary_path = Path(root_name) / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "schema": charts.MTP_6BIT_APPROXIMATE_SCHEMA,
                        "publication_candidate": True,
                        "claim_type": "approximate_optimistic_diagnostic",
                        "rows": [],
                    }
                )
            )

            with self.assertRaisesRegex(
                charts.ChartError, "publication_candidate"
            ):
                charts.load_mtp_6bit_summary(summary_path)

    def test_mtp_approximate_chart_is_labeled_as_diagnostic(self) -> None:
        rows = [
            {
                "model": "Qwen3.6 35B-A3B",
                "suite_id": "long_code",
                "ax_direct_decode_tok_s": 121.0,
                "ax_mtp_decode_tok_s": 121.6,
            }
        ]

        chart = charts.render_mtp_6bit_ax_acceleration_chart(
            rows,
            Path("2026-07-11-run/summary.json"),
            approximate_diagnostic=True,
        )

        self.assertIn("AX approximate MTP diagnostic", chart)
        self.assertIn("not publication eligible", chart)
        self.assertNotIn("Higher is better", chart)

    def test_mtp_chart_source_label_is_repo_relative(self) -> None:
        summary_path = (
            charts.REPO_ROOT
            / "benchmarks/results/speculative/mtp-6bit/2026-07-13-exact-mtp-sampled-flappy-clean/summary.json"
        )
        chart = charts.render_mtp_6bit_ax_acceleration_chart(
            self.exact_mtp_chart_summary()["rows"], summary_path
        )

        self.assertIn("Source: benchmarks/results/speculative/mtp-6bit/", chart)
        self.assertNotIn(str(charts.REPO_ROOT), chart)

    def test_mtp_exact_chart_accepts_complete_matrix_with_a_loss(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            summary_path = Path(root_name) / "summary.json"
            summary = self.exact_mtp_chart_summary()
            summary_path.write_text(json.dumps(summary))

            loaded = charts.load_mtp_6bit_summary(summary_path)

            self.assertEqual(len(loaded["rows"]), 15)

            summary["rows"][0].update(
                ax_mtp_decode_tok_s=40.0,
                ax_mtp_speedup_x=0.8,
            )
            summary_path.write_text(json.dumps(summary))
            loaded = charts.load_mtp_6bit_summary(summary_path)
            self.assertEqual(loaded["rows"][0]["ax_mtp_speedup_x"], 0.8)

            summary["rows"][0]["ax_mtp_speedup_x"] = 2.0
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(charts.ChartError, "inconsistent"):
                charts.load_mtp_6bit_summary(summary_path)

    def test_mtp_exact_chart_rejects_partial_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            summary_path = Path(root_name) / "summary.json"
            summary = self.exact_mtp_chart_summary()
            summary["rows"].pop()
            summary_path.write_text(json.dumps(summary))

            with self.assertRaisesRegex(charts.ChartError, "complete supported matrix"):
                charts.load_mtp_6bit_summary(summary_path)

    def test_mtp_exact_chart_requires_full_measured_commit(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            summary_path = Path(root_name) / "summary.json"
            summary = self.exact_mtp_chart_summary()
            summary["build_commit"] = "abcdef12"
            summary_path.write_text(json.dumps(summary))

            with self.assertRaisesRegex(charts.ChartError, "measured build_commit"):
                charts.load_mtp_6bit_summary(summary_path)

    def test_embedding_scale_charts_use_embedding_results_tree(self) -> None:
        self.assertIn(
            "benchmarks/results/embedding/embedding-scale/",
            charts.EMBEDDING_SCALE_REFERENCE_ARTIFACT.as_posix(),
        )
        self.assertIn(
            "benchmarks/results/embedding/embedding-scale/",
            charts.EMBEDDING_SCALE_AX_ARTIFACT.as_posix(),
        )
        self.assertIn(
            "benchmarks/results/embedding/embedding-scale/",
            charts.EMBEDDINGGEMMA_SCALE_REFERENCE_ARTIFACT.as_posix(),
        )
        self.assertIn(
            "benchmarks/results/embedding/embedding-scale/",
            charts.EMBEDDINGGEMMA_SCALE_AX_ARTIFACT.as_posix(),
        )
        self.assertEqual(
            charts.embedding_artifact_engine_version(
                charts.REPO_ROOT, charts.EMBEDDING_SCALE_AX_ARTIFACT
            ),
            "6.11.1",
        )
        self.assertEqual(
            charts.embedding_artifact_engine_version(
                charts.REPO_ROOT, charts.EMBEDDINGGEMMA_SCALE_AX_ARTIFACT
            ),
            "6.11.1",
        )


if __name__ == "__main__":
    unittest.main()
