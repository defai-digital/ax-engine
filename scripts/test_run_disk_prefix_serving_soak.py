#!/usr/bin/env python3
"""Unit tests for the disk-prefix serving soak runner."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("run_disk_prefix_serving_soak.py")
MODULE_SPEC = importlib.util.spec_from_file_location("run_disk_prefix_serving_soak", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
runner = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runner
MODULE_SPEC.loader.exec_module(runner)


class DiskPrefixServingSoakRunnerTests(unittest.TestCase):
    def test_dry_run_writes_auditable_command_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = runner.main_with_args_for_test(
                    [
                        "--model-id",
                        "qwen3_dense",
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        "unit-soak",
                        "--prompts",
                        "3",
                        "--prefix-tokens",
                        "32",
                        "--suffix-tokens",
                        "4",
                        "--requests",
                        "6",
                        "--concurrency",
                        "2",
                        "--route-decision-min",
                        "2",
                        "--dry-run",
                    ]
                )

            command_log = output_root / "unit-soak" / "commands.json"
            payload = json.loads(command_log.read_text())
            rendered_commands = [entry["shell"] for entry in payload["commands"]]

            self.assertEqual(code, 0)
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["artifact"], str(output_root / "unit-soak" / "artifact.json"))
            self.assertEqual(len(payload["commands"]), 4)
            self.assertIn("--warmup-requests 3", rendered_commands[1])
            self.assertIn("--min-input-tokens-p95 32", rendered_commands[2])
            self.assertIn("--require-route-decision-min ax_mlx_prefix_cache_disk_hits=2", rendered_commands[2])
            self.assertIn("render_ax_serving_benchmark_report.py", rendered_commands[3])
            self.assertIn("dry run written", stdout.getvalue())

    def test_plan_honors_explicit_goodput_and_request_rate_gates(self) -> None:
        parser = runner.build_parser()
        args = parser.parse_args(
            [
                "--model-id",
                "qwen3_dense",
                "--run-id",
                "unit-soak",
                "--request-rate-rps",
                "1.5",
                "--min-goodput-ratio",
                "0.95",
                "--min-input-tokens-p95",
                "4096",
            ]
        )

        plan = runner.build_plan(args)
        bench = " ".join(plan.commands[1])
        check = " ".join(plan.commands[2])
        render = " ".join(plan.commands[3])

        self.assertIn("--request-rate-rps 1.5", bench)
        self.assertIn("--min-goodput-ratio 0.95", check)
        self.assertIn("--min-goodput-ratio 0.95", render)
        self.assertIn("--min-input-tokens-p95 4096", check)

    def test_goodput_ratio_rejects_values_above_one(self) -> None:
        with self.assertRaisesRegex(Exception, "between 0 and 1"):
            runner.ratio_arg("1.5")


if __name__ == "__main__":
    unittest.main()
