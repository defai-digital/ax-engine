#!/usr/bin/env python3
"""Unit tests for build_long_context_decode_at_depth_artifact.py.

No companion test file previously existed; scope limited to the specific
bug fixed here rather than full coverage of the script.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("build_long_context_decode_at_depth_artifact.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "build_long_context_decode_at_depth_artifact", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
builder = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = builder
MODULE_SPEC.loader.exec_module(builder)


def _metric(median: float) -> dict[str, float]:
    return {"mean": median, "median": median, "min": median, "max": median}


def _row(
    engine: str, context_depth_tokens: int, generation_tokens: int, decode_tok_s: float
) -> dict:
    return {
        "engine": engine,
        "context_depth_tokens": context_depth_tokens,
        "generation_tokens": generation_tokens,
        "decode_tok_s": _metric(decode_tok_s),
    }


class AttachRatiosTests(unittest.TestCase):
    def test_duplicate_mlx_lm_baseline_fails_closed(self) -> None:
        # Regression test: two mlx_lm rows sharing the same
        # (context_depth_tokens, generation_tokens) key used to be silently
        # collapsed by a dict comprehension, keeping only whichever row
        # came last and discarding the other measurement with no error.
        rows = [
            _row("mlx_lm", 8192, 128, 10.0),
            _row("mlx_lm", 8192, 128, 40.0),
            _row("ax_engine_mlx", 8192, 128, 20.0),
        ]
        with self.assertRaisesRegex(
            builder.LongContextDecodeAtDepthBuildError, "duplicate mlx_lm baseline"
        ):
            builder.attach_ratios(rows)

    def test_unique_baselines_compute_ratios(self) -> None:
        rows = [
            _row("mlx_lm", 8192, 128, 10.0),
            _row("ax_engine_mlx", 8192, 128, 20.0),
        ]
        builder.attach_ratios(rows)
        ax_row = next(r for r in rows if r["engine"] == "ax_engine_mlx")
        self.assertEqual(ax_row["ratios_to_mlx_lm"]["decode_tok_s"], 2.0)


if __name__ == "__main__":
    unittest.main()
