#!/usr/bin/env python3
"""Tests for the AX serving benchmark artifact checker.

This file previously did not exist; scope limited to the specific bug fixed
here rather than achieving full coverage of check_ax_serving_benchmark_artifact.py.
"""

from __future__ import annotations

import unittest

import check_ax_serving_benchmark_artifact as checker


def _dist(value: float = 10.0, count: int = 2) -> dict[str, float | int]:
    return {
        "count": count,
        "min": value,
        "mean": value,
        "p50": value,
        "p75": value,
        "p90": value,
        "p95": value,
        "p99": value,
        "max": value,
    }


class ServingBenchmarkArtifactCheckerTests(unittest.TestCase):
    def test_validate_distribution_rejects_non_monotonic_middle_percentiles(self) -> None:
        # Regression test: the monotonicity check only compared
        # min<=p50<=p95<=max, silently accepting p75/p90/p99 values that
        # violate a genuine percentile ordering as long as those three
        # outer checkpoints still held. This fixture deliberately keeps
        # min<=p50<=p95<=max true (0.0<=10.0<=15.0<=30.0) while p75 and p99
        # both violate their neighbors, so it only fails under the fixed,
        # full-chain check.
        dist = _dist(10.0)
        dist.update(min=0.0, p50=10.0, p75=5.0, p90=20.0, p95=15.0, p99=25.0, max=30.0)
        with self.assertRaisesRegex(checker.ArtifactCheckError, "not monotonic"):
            checker.validate_distribution(dist, "test")

    def test_validate_distribution_accepts_genuinely_monotonic_percentiles(self) -> None:
        dist = _dist(10.0)
        dist.update(min=1.0, p50=10.0, p75=20.0, p90=30.0, p95=40.0, p99=50.0, max=60.0)
        self.assertEqual(checker.validate_distribution(dist, "test"), dist)


if __name__ == "__main__":
    unittest.main()
