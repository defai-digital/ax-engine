#!/usr/bin/env python3
"""Unit tests for bench_diffusion_gemma_direct.py.

No companion test file previously existed; scope limited to the specific
bug fixed here rather than full coverage of the script.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("bench_diffusion_gemma_direct.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "bench_diffusion_gemma_direct", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


def _row(block_decode_tok_s: float, denoise_steps: float = 3.0) -> dict:
    return {
        "diffusion_denoise_steps": {"median": denoise_steps},
        "block_decode_tok_s": {"median": block_decode_tok_s},
    }


class AddBandwidthEstimatesTests(unittest.TestCase):
    def test_block_wall_s_uses_actual_canvas_size(self) -> None:
        # Regression test: block_wall_s inverts
        # block_decode_tok_s = canvas_size * 1e6 / block_wall_us, which
        # requires the *same* canvas_size used to compute block_decode_tok_s
        # in parse_trial_response. A hardcoded 256.0 silently corrupted
        # every derived bandwidth figure whenever --canvas-size != 256.
        canvas_size = 128
        block_decode_tok_s = 500.0
        rows = [_row(block_decode_tok_s)]

        mod.add_bandwidth_estimates(
            rows, model_weight_bytes=1_000_000_000, canvas_size=canvas_size
        )

        expected_block_wall_s = canvas_size / block_decode_tok_s
        expected_gb_per_block = 1.0 * (3.0 + 1.0)
        expected_gb_s = expected_gb_per_block / expected_block_wall_s
        self.assertAlmostEqual(
            rows[0]["effective_bandwidth_gb_s"]["median"], expected_gb_s, places=6
        )

    def test_canvas_size_256_matches_prior_hardcoded_behavior(self) -> None:
        rows = [_row(500.0)]
        mod.add_bandwidth_estimates(
            rows, model_weight_bytes=1_000_000_000, canvas_size=256
        )
        expected_block_wall_s = 256.0 / 500.0
        expected_gb_s = 4.0 / expected_block_wall_s
        self.assertAlmostEqual(
            rows[0]["effective_bandwidth_gb_s"]["median"], expected_gb_s, places=6
        )


if __name__ == "__main__":
    unittest.main()
