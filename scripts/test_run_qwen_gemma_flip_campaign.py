#!/usr/bin/env python3
"""Tests for the repeated Qwen/Gemma flip campaign runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_qwen_gemma_flip_campaign as campaign


class QwenGemmaFlipCampaignTests(unittest.TestCase):
    def test_parse_scenario_derives_request_focus_families(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scenario.jsonl"
            path.write_text(
                '{"id":"q","kind":"request","at_ms":0,"model":"qwen3.5-9b",'
                '"input_text":"x","max_output_tokens":1}\n'
                '{"id":"g","kind":"request","at_ms":1,"model":"gemma-4-12b-it",'
                '"input_text":"y","max_output_tokens":1}\n'
            )

            parsed = campaign.parse_scenario(f"custom={path}")

        self.assertEqual(parsed.scenario_id, "custom")
        self.assertEqual(parsed.required_focus_families, ("qwen3", "gemma4"))

    def test_build_run_command_is_cache_isolated_and_report_only(self) -> None:
        scenario = campaign.ScenarioSpec("s0", Path("/tmp/s0.jsonl"), ("qwen3",))

        with mock.patch.object(campaign.sys, "executable", "/usr/bin/python3"):
            command = campaign.build_run_command(
                target_path=Path("/tmp/target.json"),
                scenario=scenario,
                artifact_path=Path("/tmp/artifact.json"),
                log_dir=Path("/tmp/logs"),
                workers=8,
                timeout=900.0,
            )

        self.assertEqual(command[0], "/usr/bin/python3")
        self.assertIn("--report-only", command)
        self.assertIn("/tmp/artifact.json", command)
        self.assertIn("/tmp/logs", command)

    def test_rejects_fewer_than_three_repetitions_without_smoke_override(self) -> None:
        with self.assertRaisesRegex(SystemExit, "at least 3"):
            campaign.main(
                [
                    "--target",
                    "/tmp/missing-target.json",
                    "--output-dir",
                    "/tmp/output",
                    "--repetitions",
                    "2",
                ]
            )


if __name__ == "__main__":
    unittest.main()
