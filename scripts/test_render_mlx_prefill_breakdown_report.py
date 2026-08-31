#!/usr/bin/env python3
"""Unit tests for render_mlx_prefill_breakdown_report.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("render_mlx_prefill_breakdown_report.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_mlx_prefill_breakdown_report",
    SCRIPT_PATH,
)
assert MODULE_SPEC and MODULE_SPEC.loader
reporter = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = reporter
MODULE_SPEC.loader.exec_module(reporter)


def metric(value: float) -> dict[str, float]:
    return {"median": value, "mean": value, "min": value, "max": value}


class PrefillBreakdownReportTests(unittest.TestCase):
    def write_artifact(self, path: Path, *, model: str = "test_model") -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "ax.mlx_inference_stack.v2",
                    "model": model,
                    "results": [
                        {
                            "engine": "mlx_lm",
                            "prompt_tokens": 128,
                            "generation_tokens": 128,
                            "prefill_tok_s": metric(500.0),
                        },
                        {
                            "engine": "ax_engine_mlx",
                            "prompt_tokens": 128,
                            "generation_tokens": 128,
                            "prefill_tok_s": metric(750.0),
                            "ax_mlx_telemetry": {
                                "ax_mlx_prefill_wall_us": 200_000,
                                "ax_mlx_prefill_forward_wall_us": 160_000,
                                "ax_mlx_prefill_prefix_cache_wall_us": 20_000,
                                "ax_mlx_prefill_generation_state_wall_us": 10_000,
                                "ax_mlx_prefill_eval_barriers": 1,
                                "ax_mlx_prefill_drain_async_evals": 2,
                            },
                        },
                    ],
                },
                indent=2,
            )
            + "\n"
        )

    def write_llama_artifact(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "ax.mlx_inference_stack.v2",
                    "results": [
                        {
                            "engine": "llama_cpp_metal",
                            "prompt_tokens": 128,
                            "generation_tokens": 128,
                            "prefill_tok_s": metric(1000.0),
                        }
                    ],
                },
                indent=2,
            )
            + "\n"
        )

    def test_directory_inputs_skip_diagnostic_artifacts_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public_artifact = root / "model.json"
            diagnostic_artifact = root / "model-linear-profile.json"
            self.write_artifact(public_artifact, model="public")
            self.write_artifact(diagnostic_artifact, model="diagnostic")

            paths = reporter.artifact_paths([root])
            diagnostic_paths = reporter.artifact_paths([root], include_diagnostics=True)

        self.assertEqual(paths, [public_artifact])
        self.assertCountEqual(diagnostic_paths, [public_artifact, diagnostic_artifact])

    def test_cli_writes_report_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.json"
            output = root / "report.md"
            self.write_artifact(artifact)

            exit_code = reporter.main_with_args_for_test(
                [str(root), "--output", str(output)]
            )

            self.assertEqual(exit_code, 0)
            self.assertIn("AX MLX Prefill Breakdown Report", output.read_text())

    def test_render_report_calculates_prefill_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.json"
            self.write_artifact(artifact)

            rows = reporter.build_rows(artifact)
            rendered = reporter.render_report(rows, title="Test Report")

        self.assertIn("# Test Report", rendered)
        self.assertIn("| test_model | 128 | 750.0 | 1.500x | n/a |", rendered)
        self.assertIn("| 200.0 | 160.0 | 20.0 | 10.0 | 10.0 | 80.0% | 1 | 2 |", rendered)

    def test_render_report_includes_llama_ratio_when_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.json"
            llama_dir = root / "llama"
            llama_dir.mkdir()
            self.write_artifact(artifact)
            self.write_llama_artifact(llama_dir / "model.json")

            rows = reporter.build_rows(artifact, llama_dir=llama_dir)
            rendered = reporter.render_report(rows, title="Prefill Slice")

        self.assertIn("test_model | 128 | 750.0 | 1.500x | 0.750x", rendered)
        self.assertIn("Worst AX/llama.cpp row", rendered)

    def test_missing_prefill_telemetry_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "bad.json"
            artifact.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "engine": "ax_engine_mlx",
                                "prompt_tokens": 128,
                                "generation_tokens": 128,
                                "prefill_tok_s": metric(1000.0),
                            }
                        ]
                    }
                )
                + "\n"
            )

            with self.assertRaisesRegex(
                reporter.PrefillBreakdownReportError,
                "lacks ax_mlx_telemetry",
            ):
                reporter.build_rows(artifact)


    def _row(self, *, model: str, ax_prefill_tok_s: float, mlx_lm_prefill_tok_s: float | None) -> "reporter.PrefillBreakdownRow":
        return reporter.PrefillBreakdownRow(
            model=model,
            artifact=Path("artifact.json"),
            prompt_tokens=128,
            ax_prefill_tok_s=ax_prefill_tok_s,
            mlx_lm_prefill_tok_s=mlx_lm_prefill_tok_s,
            mlx_swift_lm_prefill_tok_s=None,
            llama_cpp_prefill_tok_s=None,
            prefill_wall_ms=1.0,
            forward_ms=1.0,
            prefix_cache_ms=0.0,
            generation_state_ms=0.0,
            other_ms=0.0,
            forward_share=1.0,
            eval_barriers=0,
            drain_async_evals=0,
        )

    def test_sort_rows_does_not_hide_a_genuine_zero_ratio(self) -> None:
        # Regression test: with no llama_cpp comparison available, sort_rows
        # fell back to `row.ax_to_mlx_lm or 999.0`. A genuine 0.0
        # ax_to_mlx_lm ratio (AX prefill measured 0 tok/s against a valid
        # mlx_lm baseline — a catastrophic result) is falsy, so it was
        # sorted as if no comparison data existed (999.0) instead of as the
        # worst possible ratio.
        worst_but_hidden = self._row(
            model="broken", ax_prefill_tok_s=0.0, mlx_lm_prefill_tok_s=500.0
        )
        middling = self._row(
            model="middling", ax_prefill_tok_s=400.0, mlx_lm_prefill_tok_s=500.0
        )

        sorted_rows = reporter.sort_rows([middling, worst_but_hidden])

        self.assertEqual(sorted_rows[0].model, "broken")


if __name__ == "__main__":
    unittest.main()
