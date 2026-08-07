#!/usr/bin/env python3
"""Focused unit tests for the AXQ long-duration endurance runner."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "run_axq_endurance.py"
MODULE_SPEC = importlib.util.spec_from_file_location("run_axq_endurance", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
runner = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runner
MODULE_SPEC.loader.exec_module(runner)


def request_record(
    *,
    shape: str,
    ttft_ms: float = 100.0,
    decode_tok_s: float = 50.0,
    prefill_tok_s: float | None = None,
    ok: bool = True,
) -> dict:
    """Make a small synthetic record matching the persisted request shape."""
    return {
        "shape": shape,
        "request": {
            "ok": ok,
            "ttft_ms": ttft_ms if ok else None,
            "client_tpot_ms": 1000.0 / decode_tok_s if ok else None,
            "client_decode_tok_s": decode_tok_s if ok else None,
            "effective_prefill_tok_s": prefill_tok_s if ok else None,
            "e2e_latency_ms": 300.0 if ok else 50.0,
            "output_tokens": 10 if ok else None,
            "route_decisions": {"accepted": 1} if ok else {},
        },
    }


class AxqEnduranceTests(unittest.TestCase):
    def test_workload_mix_is_bounded_and_interleaved(self) -> None:
        shapes = [runner.select_shape(index).name for index in range(1, 21)]

        self.assertEqual(shapes.count("short_unique"), 14)
        self.assertEqual(shapes.count("medium_unique"), 3)
        self.assertEqual(shapes.count("shared_prefix"), 2)
        self.assertEqual(shapes.count("long_unique"), 1)
        self.assertEqual(shapes[3], "medium_unique")
        self.assertEqual(shapes[6], "shared_prefix")
        self.assertEqual(shapes[13], "long_unique")

    def test_prompt_unique_and_shared_prefix_modes_have_intended_cache_shape(self) -> None:
        unique = runner.WORKLOAD_SHAPES["medium_unique"]
        shared = runner.WORKLOAD_SHAPES["shared_prefix"]

        unique_first = runner.deterministic_prompt(unique, 1)
        unique_second = runner.deterministic_prompt(unique, 2)
        shared_first = runner.deterministic_prompt(shared, 1)
        shared_second = runner.deterministic_prompt(shared, 2)

        self.assertNotEqual(unique_first, unique_second)
        self.assertTrue(unique_first.startswith("unique_nonce_00000001"))
        self.assertNotEqual(shared_first, shared_second)
        self.assertTrue(shared_first.startswith("Shared AX Engine endurance prefix follows."))
        self.assertIn("unique_tail_nonce=00000001", shared_first)
        self.assertGreaterEqual(len(shared_first.split()), shared.nominal_input_words)

    def test_warmup_prompt_index_can_never_replay_the_measured_sequence(self) -> None:
        warmup = runner.make_prompt_item(runner.select_shape(1), 1_000_001)
        measured = runner.make_prompt_item(runner.select_shape(1), 1)

        self.assertNotEqual(warmup.id, measured.id)
        self.assertNotEqual(warmup.input_text, measured.input_text)

    def test_stream_request_uses_ignore_eos_and_requires_terminal_response(self) -> None:
        prompt = runner.make_prompt_item(runner.WORKLOAD_SHAPES["short_unique"], 1)
        captured = {}

        def stream_func(_url, payload, _timeout):
            captured.update(payload)
            yield "__http_status__", {"status": 200}, 0.0
            yield "request", {"request": {"prompt_len": 321}}, 0.001
            yield "step", {"delta_tokens": [7]}, 0.01
            yield "response", {"response": {"output_token_count": 2}}, 0.02

        observation = runner.run_stream_request(
            prompt=prompt,
            model_id="test-model",
            base_url="http://127.0.0.1:1",
            timeout_s=1.0,
            stream_func=stream_func,
        )

        self.assertTrue(observation["ok"])
        self.assertTrue(captured["sampling"]["ignore_eos"])
        self.assertEqual(observation["output_tokens"], 2)
        self.assertEqual(observation["prompt_token_count"], 321)
        self.assertIsNotNone(observation["effective_prefill_tok_s"])

    def test_prometheus_parser_selects_target_model_labels(self) -> None:
        samples = runner.parse_prometheus_samples(
            """
            # HELP ignored
            ax_engine_jobs_in_flight 0
            ax_engine_model_memory_kv_physical_bytes{model="qwen-target"} 4096
            ax_engine_model_memory_kv_physical_bytes{model="other"} 9999
            ax_runtime_ttft_p95_ms 42
            """
        )
        selected = runner.select_ax_metrics(samples, "qwen-target")

        self.assertEqual(selected["ax_engine_jobs_in_flight"], 0.0)
        self.assertEqual(selected["ax_engine_model_memory_kv_physical_bytes"], 4096.0)
        self.assertEqual(selected["ax_runtime_ttft_p95_ms"], 42.0)
        self.assertEqual(
            runner.parse_prometheus_metrics('labelled_metric{model="qwen"} 4\nunlabelled 1'),
            {"unlabelled": 1.0},
        )

    def test_lifecycle_state_requires_all_gauges_and_detects_busy(self) -> None:
        metrics = {
            "ok": True,
            "values": {name: 0.0 for name in runner.LIFECYCLE_METRICS},
        }
        self.assertEqual(runner.lifecycle_state(metrics)["state"], "drained")
        metrics["values"]["ax_engine_generation_active_streams"] = 1.0
        self.assertEqual(runner.lifecycle_state(metrics)["state"], "busy")
        del metrics["values"]["ax_engine_generation_active_streams"]
        self.assertEqual(runner.lifecycle_state(metrics)["state"], "inconclusive")

    def test_lifecycle_guard_requires_native_kv_memory_report(self) -> None:
        state = runner.RunState(
            started_wall="2026-08-06T00:00:00+00:00",
            started_monotonic=1.0,
            server_pid=42,
        )
        drain = {
            "state": "drained",
            "metrics": {
                "values": {
                    "ax_engine_model_memory_kv_report_available": 0.0,
                    "ax_engine_model_memory_kv_logical_bytes": 0.0,
                }
            },
        }

        messages = runner.guard_lifecycle(
            state=state,
            drain=drain,
            max_quiescent_kv_logical_mib=1024.0,
        )

        self.assertEqual(state.kv_report_unavailable, 1)
        self.assertEqual(
            messages,
            ["post-request native model KV memory report is unavailable"],
        )

    def test_vm_stat_swap_and_iogpu_parsers_calculate_memory_inputs(self) -> None:
        vm = runner.parse_vm_stat(
            """
            Mach Virtual Memory Statistics: (page size of 16384 bytes)
            Pages wired down: 12,345.
            Pages active: 22.
            Pages occupied by compressor: 33.
            """
        )
        swap = runner.parse_swap_usage("total = 1024.00M  used = 128.50M  free = 895.50M")
        iogpu = runner.parse_iogpu_memory(
            '"PerformanceStatistics" = {"Alloc system memory"=745930752,'
            '"In use system memory"=391020544,'
            '"In use system memory (driver)"=0}'
        )

        self.assertEqual(vm["wired_pages"], 12_345)
        self.assertEqual(vm["compressor_pages"], 33)
        self.assertEqual(swap["total_bytes"], 1024 * 1024 * 1024)
        self.assertEqual(swap["used_bytes"], int(128.5 * 1024 * 1024))
        self.assertEqual(iogpu["alloc_system_memory_bytes"], 745930752)
        self.assertEqual(iogpu["in_use_system_memory_bytes"], 391020544)

    def test_linear_slope_and_memory_growth_are_computed(self) -> None:
        samples = [
            {
                "elapsed_seconds": float(hour * 3600),
                "process": {"rss_bytes": (1_000 + hour * 300) * runner.MEBIBYTE},
                "host": {},
                "metrics": {
                    "values": {
                        "ax_engine_model_memory_kv_logical_bytes": (20 + hour * 10)
                        * runner.MEBIBYTE,
                    }
                },
            }
            for hour in range(4)
        ]
        analysis = runner.memory_analysis(
            samples=samples,
            resource_baseline={
                "server_rss_bytes": 1_000 * runner.MEBIBYTE,
                "model_kv_logical_bytes": 20 * runner.MEBIBYTE,
            },
            window_start_elapsed_s=0.0,
        )
        rss = analysis["series"]["server_rss_bytes"]

        self.assertAlmostEqual(rss["growth_mib"], 900.0)
        self.assertAlmostEqual(rss["lifetime_slope_mib_per_hour"], 300.0)
        self.assertAlmostEqual(
            analysis["series"]["model_kv_logical_bytes"]["growth_mib"],
            30.0,
        )
        alerts = runner.evaluate_memory_alerts(
            analysis=analysis, max_growth_mib=500.0, max_slope_mib_per_hour=200.0
        )
        self.assertEqual(len(alerts), 1)

    def test_baseline_stability_rejects_a_still_climbing_reference(self) -> None:
        samples = [
            {
                "elapsed_seconds": float(hour * 3600),
                "process": {"rss_bytes": (1_000 + hour * 400) * runner.MEBIBYTE},
                "host": {},
                "metrics": {"values": {}},
            }
            for hour in range(8)
        ]

        alerts = runner.evaluate_baseline_stability(
            samples=samples,
            baseline_s=7 * 3600.0,
            baseline_growth_mib=1_024.0,
            max_slope_mib_per_hour=256.0,
            max_swap_growth_mib=512.0,
        )

        self.assertEqual(len(alerts), 1)
        self.assertIn("baseline did not settle", alerts[0])

    def test_baseline_stability_allows_an_early_warm_cache_step_that_settles(self) -> None:
        samples = [
            {
                "elapsed_seconds": float(hour * 3600),
                "process": {
                    "rss_bytes": (1_000 if hour < 2 else 2_500) * runner.MEBIBYTE,
                },
                "host": {},
                "metrics": {"values": {}},
            }
            for hour in range(8)
        ]

        alerts = runner.evaluate_baseline_stability(
            samples=samples,
            baseline_s=7 * 3600.0,
            baseline_growth_mib=1_024.0,
            max_slope_mib_per_hour=256.0,
            max_swap_growth_mib=512.0,
        )

        self.assertEqual(alerts, [])

    def test_swap_growth_has_a_tighter_host_safety_guardrail(self) -> None:
        analysis = {
            "series": {
                "host_swap_used_bytes": {
                    "growth_mib": 5_000.0,
                    "lifetime_slope_mib_per_hour": 300.0,
                }
            }
        }

        alerts = runner.evaluate_memory_alerts(
            analysis=analysis,
            max_growth_mib=4_096.0,
            max_slope_mib_per_hour=256.0,
            max_swap_growth_mib=512.0,
        )

        self.assertEqual(len(alerts), 1)
        self.assertIn("swap", alerts[0])

    def test_performance_regression_compares_same_shape_tail_metrics(self) -> None:
        baseline = runner.summarize_requests(
            [
                request_record(shape="short_unique", ttft_ms=100.0, decode_tok_s=100.0)
                for _ in range(10)
            ],
            elapsed_s=10.0,
        )
        window = runner.summarize_requests(
            [
                request_record(shape="short_unique", ttft_ms=180.0, decode_tok_s=60.0)
                for _ in range(10)
            ],
            elapsed_s=10.0,
        )

        alerts = runner.evaluate_performance_regression(
            baseline=baseline,
            window=window,
            min_samples=8,
            max_ttft_p95_ratio=1.5,
            min_decode_p05_ratio=0.75,
        )

        self.assertEqual(len(alerts), 2)
        self.assertIn("p95 TTFT", alerts[0])
        self.assertIn("p05 decode", alerts[1])

    def test_prefill_regression_uses_exact_native_prompt_metric(self) -> None:
        baseline = runner.summarize_requests(
            [
                request_record(
                    shape="medium_unique",
                    ttft_ms=100.0,
                    decode_tok_s=100.0,
                    prefill_tok_s=1_000.0,
                )
                for _ in range(10)
            ],
            elapsed_s=10.0,
        )
        window = runner.summarize_requests(
            [
                request_record(
                    shape="medium_unique",
                    ttft_ms=100.0,
                    decode_tok_s=100.0,
                    prefill_tok_s=600.0,
                )
                for _ in range(10)
            ],
            elapsed_s=10.0,
        )

        alerts = runner.evaluate_performance_regression(
            baseline=baseline,
            window=window,
            min_samples=8,
            max_ttft_p95_ratio=1.5,
            min_decode_p05_ratio=0.75,
            min_prefill_p05_ratio=0.75,
        )

        self.assertEqual(len(alerts), 1)
        self.assertIn("effective prefill", alerts[0])

    def test_baseline_coverage_requires_all_client_measurements(self) -> None:
        records = [
            request_record(
                shape=shape,
                ttft_ms=100.0,
                decode_tok_s=100.0,
                prefill_tok_s=(None if shape == "long_unique" else 1_000.0),
            )
            for shape in runner.WORKLOAD_SHAPES
            for _ in range(8)
        ]
        baseline = runner.summarize_requests(records, elapsed_s=10.0)

        concerns = runner.evaluate_baseline_coverage(baseline, min_samples=8)

        self.assertEqual(len(concerns), 1)
        self.assertIn("long_unique", concerns[0])
        self.assertIn("effective prefill", concerns[0])

    def test_performance_regression_requires_enough_baseline_samples(self) -> None:
        baseline = runner.summarize_requests(
            [
                request_record(
                    shape="short_unique",
                    ttft_ms=100.0,
                    decode_tok_s=100.0,
                )
            ],
            elapsed_s=10.0,
        )
        window = runner.summarize_requests(
            [
                request_record(
                    shape="short_unique",
                    ttft_ms=180.0,
                    decode_tok_s=60.0,
                )
                for _ in range(10)
            ],
            elapsed_s=10.0,
        )

        self.assertEqual(
            runner.evaluate_performance_regression(
                baseline=baseline,
                window=window,
                min_samples=8,
                max_ttft_p95_ratio=1.5,
                min_decode_p05_ratio=0.75,
            ),
            [],
        )

    def test_terminal_window_guardrails_use_latest_partial_window(self) -> None:
        state = runner.RunState(
            started_wall="2026-08-06T00:00:00+00:00",
            started_monotonic=1.0,
            server_pid=42,
            requests_attempted=20,
            requests_ok=20,
            baseline=runner.summarize_requests(
                [
                    request_record(
                        shape="short_unique",
                        ttft_ms=100.0,
                        decode_tok_s=100.0,
                    )
                    for _ in range(10)
                ],
                elapsed_s=10.0,
            ),
            counter_baseline={
                "ax_engine_http_status_5xx_total": 0.0,
            },
        )
        records = [
            request_record(
                shape="short_unique",
                ttft_ms=180.0,
                decode_tok_s=60.0,
            )
            for _ in range(10)
        ]

        performance, memory = runner.evaluate_window_guardrails(
            state=state,
            records=records,
            window_elapsed_s=600.0,
            latest_metrics={
                "values": {
                    "ax_engine_http_status_5xx_total": 1.0,
                }
            },
            resource_samples=[],
            window_start_elapsed_s=0.0,
            min_performance_samples=8,
            max_ttft_p95_ratio=1.5,
            min_decode_p05_ratio=0.75,
            max_client_error_rate=0.0,
            memory_growth_mib=500.0,
            memory_slope_mib_per_hour=200.0,
        )

        self.assertEqual(len(performance), 3)
        self.assertTrue(any("p95 TTFT" in alert for alert in performance))
        self.assertTrue(any("p05 decode" in alert for alert in performance))
        self.assertTrue(any("status_5xx" in alert for alert in performance))
        self.assertEqual(memory, [])

    def test_interrupted_run_assessment_never_passes(self) -> None:
        state = runner.RunState(
            started_wall="2026-08-06T00:00:00+00:00",
            started_monotonic=1.0,
            server_pid=42,
            baseline={"by_shape": {}},
        )

        verdict, concerns = runner.assessment(
            state=state,
            terminal_status="interrupted",
        )

        self.assertEqual(verdict, "watch")
        self.assertTrue(any("interrupted" in concern for concern in concerns))

    def test_unsettled_or_incomplete_baseline_never_passes(self) -> None:
        state = runner.RunState(
            started_wall="2026-08-06T00:00:00+00:00",
            started_monotonic=1.0,
            server_pid=42,
            baseline={"by_shape": {}},
            baseline_coverage_concerns=["baseline missing long-prompt TTFT evidence"],
            baseline_stability_alerts=["baseline did not settle: server RSS rose"],
        )

        verdict, concerns = runner.assessment(state=state, terminal_status="completed")

        self.assertEqual(verdict, "watch")
        self.assertEqual(len(concerns), 2)

    def test_summary_reports_latency_and_failure_counts(self) -> None:
        summary = runner.summarize_window(
            [
                request_record(shape="short_unique", ttft_ms=100.0, decode_tok_s=50.0),
                request_record(shape="short_unique", ok=False),
            ],
            elapsed_s=10.0,
        )

        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["successful_requests"], 1)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["overall"]["ttft_ms"]["p50"], 100.0)
        self.assertEqual(summary["overall"]["route_decisions"], {"accepted": 1})

    def test_prepare_output_dir_refuses_existing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            runner.prepare_output_dir(output)
            (output / "manifest.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "not empty"):
                runner.prepare_output_dir(output)

    def test_write_checkpoint_creates_immutable_json_markdown_and_current_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "checkpoints").mkdir()
            (output / "reports").mkdir()
            state = runner.RunState(
                started_wall="2026-08-06T00:00:00+00:00",
                started_monotonic=1.0,
                server_pid=42,
            )
            with mock.patch.object(runner, "utc_now", return_value="2026-08-06T04:00:00+00:00"):
                summary = runner.write_checkpoint(
                    output_dir=output,
                    reason="periodic",
                    state=state,
                    status="running",
                    elapsed_s=14_400.0,
                    target_duration_s=259_200.0,
                    records=[],
                    window_elapsed_s=14_400.0,
                    latest_server={"alive": True, "pid": 42},
                    latest_host={},
                    latest_metrics={"values": {}},
                    resource_samples=[],
                    window_start_elapsed_s=0.0,
                    performance_alerts=[],
                    memory_alerts=[],
                    alerts=[],
                )

            self.assertEqual(summary["status"], "running")
            self.assertEqual(summary["verdict"], "watch")
            self.assertTrue((output / "summary.json").is_file())
            checkpoints = list((output / "checkpoints").glob("*.json"))
            reports = list((output / "reports").glob("*.md"))
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(len(reports), 1)
            self.assertEqual(json.loads(checkpoints[0].read_text())["reason"], "periodic")

    def test_build_server_command_pins_single_request_limits(self) -> None:
        args = runner.build_parser().parse_args(
            [
                "--server",
                "/tmp/server",
                "--model-dir",
                "/tmp/model",
                "--output-dir",
                "/tmp/output",
                "--server-extra-arg=--some-flag",
            ]
        )

        command = runner.build_server_command(args)

        self.assertIn("--max-concurrent-requests", command)
        self.assertIn("--max-concurrent-requests-per-model", command)
        self.assertIn("--some-flag", command)

    def test_cli_allows_zero_client_error_rate(self) -> None:
        args = runner.build_parser().parse_args(
            [
                "--server",
                "/tmp/server",
                "--model-dir",
                "/tmp/model",
                "--output-dir",
                "/tmp/output",
                "--max-client-error-rate",
                "0",
            ]
        )

        self.assertEqual(args.max_client_error_rate, 0.0)

    def test_run_endurance_installs_and_restores_sigterm_handler(self) -> None:
        installed = []

        def record_signal(signum, handler):
            installed.append((signum, handler))

        with (
            mock.patch.object(runner.signal, "getsignal", return_value="previous"),
            mock.patch.object(runner.signal, "signal", side_effect=record_signal),
            mock.patch.object(runner, "_run_endurance", return_value=7),
        ):
            result = runner.run_endurance(mock.sentinel.args)

        self.assertEqual(result, 7)
        self.assertEqual(installed[0][0], runner.signal.SIGTERM)
        with self.assertRaises(KeyboardInterrupt):
            installed[0][1](runner.signal.SIGTERM, None)
        self.assertEqual(installed[-1], (runner.signal.SIGTERM, "previous"))


if __name__ == "__main__":
    unittest.main()
