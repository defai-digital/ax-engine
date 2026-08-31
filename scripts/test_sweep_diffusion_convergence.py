#!/usr/bin/env python3
"""Unit tests for sweep_diffusion_convergence.py.

No companion test file previously existed; scope limited to the specific
bug fixed here rather than full coverage of the script.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("sweep_diffusion_convergence.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "sweep_diffusion_convergence", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


def _row(entropy_threshold: float, denoise_steps_median: float, convergence_rate: float) -> dict:
    return {
        "entropy_threshold": entropy_threshold,
        "acceptance_rate_threshold": 0.9,
        "entropy_plateau_delta": 0.01,
        "denoise_steps": {"median": denoise_steps_median},
        "convergence_rate": convergence_rate,
    }


class SelectBestConfigTests(unittest.TestCase):
    def test_prefers_reliable_convergence_over_fewer_steps(self) -> None:
        # Regression test: a loose entropy_threshold can report spuriously
        # low denoise_steps (premature "convergence") with a low
        # convergence_rate — exactly the failure mode convergence_rate
        # exists to catch. Selecting by step count alone picks the
        # least-reliable config.
        loose = _row(entropy_threshold=0.5, denoise_steps_median=3.0, convergence_rate=0.2)
        strict = _row(entropy_threshold=0.01, denoise_steps_median=12.0, convergence_rate=1.0)

        best = mod.select_best_config([loose, strict])

        self.assertEqual(best["entropy_threshold"], 0.01)

    def test_breaks_ties_by_fewer_steps_among_equally_reliable_configs(self) -> None:
        fast = _row(entropy_threshold=0.05, denoise_steps_median=8.0, convergence_rate=1.0)
        slow = _row(entropy_threshold=0.01, denoise_steps_median=12.0, convergence_rate=1.0)

        best = mod.select_best_config([fast, slow])

        self.assertEqual(best["entropy_threshold"], 0.05)


if __name__ == "__main__":
    unittest.main()
