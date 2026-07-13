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
            "claim_type": "exact_mtp_acceleration",
            "rows": rows,
        }

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

            self.assertEqual(charts.find_mtp_6bit_summary(readme), summary_path)

    def test_mtp_6bit_refresh_defaults_to_speculative_results_tree(self) -> None:
        self.assertTrue(
            mtp_refresh.DEFAULT_OUTPUT_BASE.as_posix().endswith(
                "/benchmarks/results/speculative/mtp-6bit"
            )
        )

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

    def test_mtp_exact_chart_requires_complete_all_winning_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            summary_path = Path(root_name) / "summary.json"
            summary = self.exact_mtp_chart_summary()
            summary_path.write_text(json.dumps(summary))

            loaded = charts.load_mtp_6bit_summary(summary_path)

            self.assertEqual(len(loaded["rows"]), 15)

            summary["rows"][0].update(
                ax_mtp_decode_tok_s=50.0,
                ax_mtp_speedup_x=1.0,
            )
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(charts.ChartError, "does not win decode"):
                charts.load_mtp_6bit_summary(summary_path)

    def test_mtp_exact_chart_rejects_partial_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            summary_path = Path(root_name) / "summary.json"
            summary = self.exact_mtp_chart_summary()
            summary["rows"].pop()
            summary_path.write_text(json.dumps(summary))

            with self.assertRaisesRegex(charts.ChartError, "complete supported matrix"):
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


if __name__ == "__main__":
    unittest.main()
