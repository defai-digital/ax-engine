"""Regression checks for portable serving CPU traces."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts import observe_serving_cpu as probe


class ObserveServingCpuTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.trace = Path(directory.name) / "cpu.jsonl"
        self.metadata = {"kind": "metadata", "version": 1, "host_ticks_per_second": 100}
        self.first = {
            "kind": "start", "perf_counter_ns": 10_000_000_000, "read_duration_ns": 200,
            "host_ticks": {"user": probe.UINT32_MASK - 9, "system": 10, "idle": 45, "nice": 0},
            "server_cpu_ns": 50_000_000, "client_cpu_ns": 150_000_000,
        }
        self.last = {
            "kind": "end", "perf_counter_ns": 11_000_000_000, "read_duration_ns": 100,
            "host_ticks": {"user": 60, "system": 30, "idle": 145, "nice": 10},
            "server_cpu_ns": 450_000_000, "client_cpu_ns": 400_000_000,
        }

    def summarize(self, *rows):
        self.trace.write_text("".join(json.dumps(row) + "\n" for row in (self.metadata, *rows)))
        return probe.summarize(self.trace)

    def test_wraps_host_ticks_and_accounts_for_both_processes(self):
        result = self.summarize(self.first, self.last)
        self.assertEqual(result["client_window_seconds"], 1.0)
        self.assertEqual(result["host_idle_percent"], 50.0)
        self.assertEqual(result["host_busy_core_seconds"], 1.0)
        self.assertEqual(result["server_cpu_seconds"], 0.4)
        self.assertEqual(result["client_cpu_seconds"], 0.25)
        self.assertAlmostEqual(result["unassigned_busy_core_seconds"], 0.35)

    def test_missing_terminal_sample_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "completed client run"):
            self.summarize(self.first, self.last | {"kind": "sample"})

    def test_decreasing_server_counter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-monotonic"):
            self.summarize(self.first, self.last | {"server_cpu_ns": 1})


if __name__ == "__main__":
    unittest.main()
