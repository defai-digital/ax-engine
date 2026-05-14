#!/usr/bin/env python3
"""Unit tests for the MLX prefill claim-cycle aggregate gate."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("check_mlx_prefill_claim_cycle.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mlx_prefill_claim_cycle", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


class MlxPrefillClaimCycleTests(unittest.TestCase):
    def test_default_checks_cover_w0_to_w4_boundaries(self) -> None:
        repo_root = Path("/repo")

        checks = checker.default_checks(repo_root)

        self.assertEqual(
            [check.name for check in checks],
            [
                "W0-W3 README table and narrative claim gate",
                "W1 long-context prefill boundary",
                "W3 concurrent-prefill boundary",
                "W4 forward-profile diagnostic boundary",
            ],
        )
        commands = [" ".join(check.command) for check in checks]
        self.assertIn("check_readme_performance_artifacts.py", commands[0])
        self.assertIn("check_mlx_prefill_scaling_artifact.py", commands[1])
        self.assertIn("check_mlx_concurrent_prefill_artifact.py", commands[2])
        self.assertIn("--allow-missing-scheduler-evidence", commands[2])
        self.assertIn("check_mlx_forward_profile_artifact.py", commands[3])

    def test_main_returns_zero_when_all_checks_pass(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="ok\n",
            stderr="",
        )

        with patch.object(checker.subprocess, "run", return_value=completed):
            with redirect_stdout(io.StringIO()):
                exit_code = checker.main_with_args_for_test(["--repo-root", "/repo"])

        self.assertEqual(exit_code, 0)

    def test_main_returns_failure_when_any_check_fails(self) -> None:
        calls = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="bad\n"),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr=""),
        ]

        with patch.object(checker.subprocess, "run", side_effect=calls):
            with redirect_stdout(io.StringIO()):
                exit_code = checker.main_with_args_for_test(["--repo-root", "/repo"])

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
