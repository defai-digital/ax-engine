#!/usr/bin/env python3
"""Unit tests for the native generation fault-soak runner."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCRIPT_PATH = Path(__file__).with_name("run_native_generation_fault_soak.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "run_native_generation_fault_soak", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
runner = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runner
MODULE_SPEC.loader.exec_module(runner)


class NativeGenerationFaultSoakTests(unittest.TestCase):
    def test_parse_prometheus_metrics_ignores_comments_and_labels(self):
        metrics = runner.parse_prometheus_metrics(
            """
            # HELP ignored comment
            ax_engine_jobs_in_flight 0
            labelled_metric{route="mlx"} 3
            ax_engine_generation_saturated_commands_total 2
            """
        )

        self.assertEqual(
            metrics,
            {
                "ax_engine_jobs_in_flight": 0.0,
                "ax_engine_generation_saturated_commands_total": 2.0,
            },
        )

    def test_disconnect_request_stops_after_first_output_event(self):
        closed = []

        def stream_func(_url, _payload, _timeout):
            try:
                yield "__http_status__", {"status": 200}, 0.0
                yield "step", {"delta_tokens": [7]}, 0.01
                yield "response", {"response": {"output_tokens": [7]}}, 0.02
            finally:
                closed.append(True)

        spec = runner.RequestSpec(
            request_id="disconnect-1",
            kind="disconnect",
            input_tokens=[1, 2],
            max_output_tokens=8,
        )
        outcome = runner.run_fault_request(
            spec,
            base_url="http://127.0.0.1:1",
            model_id="model",
            timeout=1.0,
            disconnect_after_output_events=1,
            slow_delay_s=0.0,
            stream_func=stream_func,
        )

        self.assertEqual(outcome["outcome"], "expected_disconnect")
        self.assertEqual(outcome["output_tokens"], 1)
        self.assertEqual(closed, [True])

    def test_fault_payload_forces_bounded_length_instead_of_early_eos(self):
        spec = runner.RequestSpec(
            request_id="slow-1",
            kind="slow",
            input_tokens=[1, 2],
            max_output_tokens=512,
        )

        payload = runner.build_payload(spec, "model")

        self.assertTrue(payload["sampling"]["ignore_eos"])
        self.assertEqual(payload["max_output_tokens"], 512)

    def test_evaluate_run_requires_quiescent_lifecycle(self):
        outcomes = [
            {"request_id": "normal-1", "kind": "normal", "outcome": "completed"},
            {
                "request_id": "disconnect-1",
                "kind": "disconnect",
                "outcome": "expected_disconnect",
            },
            {"request_id": "slow-1", "kind": "slow", "outcome": "completed"},
        ]
        deltas = {
            runner.SATURATION_COUNTER: 0.0,
            runner.BACKLOG_OVERFLOW_COUNTER: 0.0,
        }

        verdict, reasons = runner.evaluate_run(
            outcomes,
            quiescent=False,
            counter_deltas=deltas,
        )

        self.assertEqual(verdict, "fail")
        self.assertIn("generation lifecycle gauges did not return to zero", reasons)

    def test_disconnect_completion_before_window_is_allowed_when_another_disconnects(self):
        outcomes = [
            {
                "request_id": "disconnect-1",
                "kind": "disconnect",
                "outcome": "completed_before_disconnect_window",
            },
            {
                "request_id": "disconnect-2",
                "kind": "disconnect",
                "outcome": "expected_disconnect",
            },
        ]
        deltas = {
            runner.SATURATION_COUNTER: 0.0,
            runner.BACKLOG_OVERFLOW_COUNTER: 0.0,
        }

        verdict, reasons = runner.evaluate_run(
            outcomes,
            quiescent=True,
            counter_deltas=deltas,
        )

        self.assertEqual(verdict, "pass")
        self.assertEqual(reasons, [])

    def test_metric_peaks_ignore_failed_samples(self):
        samples = [
            {"elapsed_seconds": 0.1, "error": "transient"},
            {
                "elapsed_seconds": 0.2,
                "values": {
                    name: float(index)
                    for index, name in enumerate(
                        (
                            *runner.QUIESCENT_GAUGES,
                            runner.SATURATION_COUNTER,
                            runner.BACKLOG_OVERFLOW_COUNTER,
                        )
                    )
                },
            },
        ]

        peaks = runner.summarize_metric_peaks(samples)

        self.assertEqual(
            peaks["ax_engine_generation_active_streams"],
            3.0,
        )

    def test_slow_stream_error_requires_overflow_counter(self):
        outcomes = [
            {"request_id": "slow-1", "kind": "slow", "outcome": "stream_error"}
        ]
        deltas = {
            runner.SATURATION_COUNTER: 0.0,
            runner.BACKLOG_OVERFLOW_COUNTER: 0.0,
        }

        verdict, reasons = runner.evaluate_run(
            outcomes,
            quiescent=True,
            counter_deltas=deltas,
        )

        self.assertEqual(verdict, "fail")
        self.assertTrue(any("backlog-overflow" in reason for reason in reasons))

    def test_required_backpressure_must_have_buffer_or_overflow_evidence(self):
        outcomes = [
            {"request_id": "slow-1", "kind": "slow", "outcome": "completed"}
        ]
        deltas = {
            runner.SATURATION_COUNTER: 0.0,
            runner.BACKLOG_OVERFLOW_COUNTER: 0.0,
        }

        verdict, reasons = runner.evaluate_run(
            outcomes,
            quiescent=True,
            counter_deltas=deltas,
            metric_peaks={"ax_engine_generation_buffered_stream_events": 0.0},
            require_backpressure=True,
            min_buffered_events=1,
        )

        self.assertEqual(verdict, "fail")
        self.assertTrue(any("required backpressure" in reason for reason in reasons))

        verdict, reasons = runner.evaluate_run(
            outcomes,
            quiescent=True,
            counter_deltas=deltas,
            metric_peaks={"ax_engine_generation_buffered_stream_events": 1.0},
            require_backpressure=True,
            min_buffered_events=1,
        )
        self.assertEqual(verdict, "pass")
        self.assertEqual(reasons, [])

    def test_request_specs_keep_the_frozen_prompt(self):
        rounds = runner.build_request_specs(
            rounds=1,
            normal_per_round=1,
            disconnect_per_round=1,
            slow_per_round=1,
            stalled_per_round=1,
            base_tokens=[7, 8, 9],
            normal_output_tokens=16,
            fault_output_tokens=512,
        )

        self.assertEqual(len(rounds[0]), 4)
        self.assertTrue(all(spec.input_tokens == [7, 8, 9] for spec in rounds[0]))


if __name__ == "__main__":
    unittest.main()
