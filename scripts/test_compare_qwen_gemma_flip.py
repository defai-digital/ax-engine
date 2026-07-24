#!/usr/bin/env python3
"""Tests for Qwen3+Gemma4 flip comparator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import compare_qwen_gemma_flip as compare
import test_check_ax_multimodel_serving_artifact as fixtures


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload))


class FlipCompareTests(unittest.TestCase):
    def test_pass_when_candidate_beats_provisional_gates(self) -> None:
        candidate = fixtures._artifact()
        baseline = fixtures._artifact()
        baseline["summary"]["output_token_throughput_tok_s"] = 10.0  # type: ignore[index]
        candidate["summary"]["output_token_throughput_tok_s"] = 20.0  # type: ignore[index]
        baseline["summary"]["ttft_ms"] = fixtures._dist(200.0)  # type: ignore[index]
        candidate["summary"]["ttft_ms"] = fixtures._dist(100.0)  # type: ignore[index]
        baseline["interactive_stream_gap_ms"] = fixtures._dist(40.0)  # type: ignore[index]
        candidate["interactive_stream_gap_ms"] = fixtures._dist(20.0)  # type: ignore[index]

        with tempfile.TemporaryDirectory() as temp_dir:
            cand_path = Path(temp_dir) / "cand.json"
            base_path = Path(temp_dir) / "base.json"
            _write(cand_path, candidate)
            _write(base_path, baseline)
            code = compare.main(
                [
                    "--candidate",
                    str(cand_path),
                    "--baseline",
                    str(base_path),
                    "--min-throughput-ratio",
                    "1.15",
                    "--max-ttft-p95-ratio",
                    "0.90",
                    "--max-stream-gap-p95-ratio",
                    "0.90",
                ]
            )
        self.assertEqual(code, 0)

    def test_fail_when_throughput_below_gate(self) -> None:
        candidate = fixtures._artifact()
        baseline = fixtures._artifact()
        candidate["summary"]["output_token_throughput_tok_s"] = 10.0  # type: ignore[index]
        baseline["summary"]["output_token_throughput_tok_s"] = 20.0  # type: ignore[index]
        # Keep latency ratios green so only throughput fails.
        candidate["summary"]["ttft_ms"] = fixtures._dist(80.0)  # type: ignore[index]
        baseline["summary"]["ttft_ms"] = fixtures._dist(100.0)  # type: ignore[index]
        candidate["interactive_stream_gap_ms"] = fixtures._dist(10.0)  # type: ignore[index]
        baseline["interactive_stream_gap_ms"] = fixtures._dist(20.0)  # type: ignore[index]

        with tempfile.TemporaryDirectory() as temp_dir:
            cand_path = Path(temp_dir) / "cand.json"
            base_path = Path(temp_dir) / "base.json"
            out_path = Path(temp_dir) / "report.json"
            _write(cand_path, candidate)
            _write(base_path, baseline)
            code = compare.main(
                [
                    "--candidate",
                    str(cand_path),
                    "--baseline",
                    str(base_path),
                    "--output",
                    str(out_path),
                ]
            )
            report = json.loads(out_path.read_text())

        self.assertEqual(code, 1)
        self.assertIn("throughput_ratio", report["failed_required_gates"])

    def test_report_only_exits_zero_on_fail(self) -> None:
        candidate = fixtures._artifact()
        baseline = fixtures._artifact()
        candidate["summary"]["output_token_throughput_tok_s"] = 1.0  # type: ignore[index]
        baseline["summary"]["output_token_throughput_tok_s"] = 100.0  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temp_dir:
            cand_path = Path(temp_dir) / "cand.json"
            base_path = Path(temp_dir) / "base.json"
            _write(cand_path, candidate)
            _write(base_path, baseline)
            code = compare.main(
                [
                    "--candidate",
                    str(cand_path),
                    "--baseline",
                    str(base_path),
                    "--report-only",
                ]
            )
        self.assertEqual(code, 0)

    def test_rejects_mismatched_scenario_contract(self) -> None:
        candidate = fixtures._artifact()
        baseline = fixtures._artifact()
        baseline["scenario"]["sha256"] = "b" * 64  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temp_dir:
            cand_path = Path(temp_dir) / "cand.json"
            base_path = Path(temp_dir) / "base.json"
            out_path = Path(temp_dir) / "report.json"
            _write(cand_path, candidate)
            _write(base_path, baseline)
            code = compare.main(
                [
                    "--candidate",
                    str(cand_path),
                    "--baseline",
                    str(base_path),
                    "--output",
                    str(out_path),
                ]
            )
            report = json.loads(out_path.read_text())

        self.assertEqual(code, 1)
        self.assertIn("comparison_contract", report["failed_required_gates"])
        self.assertIn(
            "scenario.sha256",
            "\n".join(report["comparison_contract"]["mismatches"]),
        )

    def test_rejects_different_model_package_path(self) -> None:
        candidate = fixtures._artifact()
        baseline = fixtures._artifact()
        baseline["target"]["model_packages"]["qwen3.5-9b"]["path"] = "/models/other"  # type: ignore[index]

        contract = compare.evaluate_comparison_contract(candidate, baseline)

        self.assertFalse(contract["passed"])
        self.assertIn(
            "model package identity differs for qwen3.5-9b",
            contract["mismatches"],
        )

    def test_rejects_different_prompt_tokenization(self) -> None:
        candidate = fixtures._artifact()
        baseline = fixtures._artifact()
        baseline["observations"][0]["input_tokens"] = 31  # type: ignore[index]

        contract = compare.evaluate_comparison_contract(candidate, baseline)

        self.assertFalse(contract["passed"])
        self.assertIn(
            "per-event input token counts differ",
            "\n".join(contract["mismatches"]),
        )


if __name__ == "__main__":
    unittest.main()
