"""Protect the evidence binding used when comparing MLX admission probes."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.probe_mlx_runtime_upgrade import compare_reports


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "baseline.json"
        self.output = {"qmm": np.array([1.0, 2.0], dtype=np.float32)}
        self.baseline = {
            "schema": "ax.mlx.runtime_upgrade_probe.v2",
            "probe_sha256": "source-a",
            "host": {"chip": "test", "memory_bytes": 1024, "macos": "test"},
            "input_sha256": {"x": "input-a"},
            "checks": [{"name": "qmm", "tolerance": 0.004, "passed": True}],
            "runtime": {"mlx": "0.32.3", "python": "3.12", "numpy": "2.5"},
        }
        self.save()
        self.candidate = copy.deepcopy(self.baseline)
        self.candidate["runtime"]["mlx"] = "0.32.4"

    def save(self):
        np.savez(self.path.with_suffix(".npz"), **self.output)
        self.baseline["outputs_sha256"] = hashlib.sha256(
            self.path.with_suffix(".npz").read_bytes()).hexdigest()
        self.path.write_text(json.dumps(self.baseline))

    def test_numerical_drift_is_reported_separately_from_reference_accuracy(self):
        actual = {"qmm": np.array([1.0, 2.001], dtype=np.float32)}
        result = compare_reports(self.path, self.candidate, actual)
        drift = result["outputs"][0]
        self.assertFalse(drift["bit_identical"])
        self.assertEqual(drift["different_elements"], 1)
        self.assertGreater(drift["max_abs_difference"], 0)
        self.assertEqual(result["baseline_runtime"], self.baseline["runtime"])

    def test_python_or_numpy_changes_are_rejected(self):
        for field in ("python", "numpy"):
            with self.subTest(field=field):
                candidate = copy.deepcopy(self.candidate)
                candidate["runtime"][field] = "different"
                with self.assertRaisesRegex(ValueError, f"runtime {field}"):
                    compare_reports(self.path, candidate, self.output)

    def test_experiment_identity_mismatches_are_rejected(self):
        for field in ("schema", "probe_sha256", "host", "input_sha256"):
            with self.subTest(field=field):
                candidate = copy.deepcopy(self.candidate)
                candidate[field] = "different"
                with self.assertRaisesRegex(ValueError, field):
                    compare_reports(self.path, candidate, self.output)

    def test_changed_checks_or_tolerances_are_rejected(self):
        self.candidate["checks"][0]["tolerance"] = 1.0
        with self.assertRaisesRegex(ValueError, "checks and tolerances"):
            compare_reports(self.path, self.candidate, self.output)

    def test_modified_baseline_arrays_are_rejected(self):
        np.savez(self.path.with_suffix(".npz"), qmm=np.array([9.0, 9.0]))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            compare_reports(self.path, self.candidate, self.output)

    def test_missing_outputs_and_shape_changes_are_rejected(self):
        for actual in ({}, {"qmm": np.ones((1, 2), dtype=np.float32)}):
            with self.subTest(actual=actual):
                with self.assertRaisesRegex(ValueError, "matching"):
                    compare_reports(self.path, self.candidate, actual)

    def test_nonfinite_outputs_are_not_reported_as_valid_drift(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                actual = {"qmm": np.array([1.0, value], dtype=np.float32)}
                with self.assertRaisesRegex(ValueError, "finite outputs"):
                    compare_reports(self.path, self.candidate, actual)

    def test_signed_zero_is_not_bit_identical(self):
        self.output = {"qmm": np.array([0.0], dtype=np.float32)}
        self.save()
        actual = {"qmm": np.array([-0.0], dtype=np.float32)}
        result = compare_reports(self.path, self.candidate, actual)["outputs"][0]
        self.assertFalse(result["bit_identical"])
        self.assertEqual(result["different_elements"], 0)


if __name__ == "__main__":
    unittest.main()
