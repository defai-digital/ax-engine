#!/usr/bin/env python3
"""Unit tests for the AXQ long-duration endurance runner."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).with_name("run_axq_endurance.py")
MODULE_SPEC = importlib.util.spec_from_file_location("run_axq_endurance", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
runner = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runner
MODULE_SPEC.loader.exec_module(runner)


class AxqEnduranceTests(unittest.TestCase):
    def test_workload_mix_is_bounded_and_deterministic(self) -> None:
        shapes = [runner.select_shape(index).name for index in range(1, 25)]

        self.assertEqual(shapes.count("short"), 21)
        self.assertEqual(shapes.count("medium"), 2)
        self.assertEqual(shapes.count("long"), 1)
        self.assertEqual(shapes[-1], "long")

    def test_measured_mix_after_warmups_does_not_replay_prompt_ids(self) -> None:
        warmups = 2
        measured_indices = range(warmups + 1, warmups + 25)
        shapes = [runner.select_shape(index).name for index in measured_indices]

        self.assertEqual(shapes.count("short"), 21)
        self.assertEqual(shapes.count("medium"), 2)
        self.assertEqual(shapes.count("long"), 1)
        warmup_ids = {
            runner.make_prompt_item(runner.select_shape(index), index).id
            for index in range(1, warmups + 1)
        }
        first_measured = runner.make_prompt_item(
            runner.select_shape(warmups + 1), warmups + 1
        )
        self.assertNotIn(first_measured.id, warmup_ids)

    def test_prompt_is_unique_and_uses_requested_nominal_length(self) -> None:
        shape = runner.WORKLOAD_SHAPES["medium"]
        first = runner.deterministic_prompt(shape, 1)
        second = runner.deterministic_prompt(shape, 2)

        self.assertNotEqual(first, second)
        self.assertIn("Endurance request 1", first)
        self.assertGreaterEqual(len(first.split()), shape.nominal_input_words)

    def test_stream_request_uses_ignore_eos_and_requires_terminal_response(self) -> None:
        prompt = runner.make_prompt_item(runner.WORKLOAD_SHAPES["short"], 1)
        captured = {}

        def stream_func(_url, payload, _timeout):
            captured.update(payload)
            yield "__http_status__", {"status": 200}, 0.0
            yield "step", {"delta_tokens": [7]}, 0.01
            yield "response", {"response": {"output_token_count": 1}}, 0.02

        observation = runner.run_stream_request(
            prompt=prompt,
            model_id="test-model",
            base_url="http://127.0.0.1:1",
            timeout_s=1.0,
            stream_func=stream_func,
        )

        self.assertTrue(observation["ok"])
        self.assertTrue(captured["sampling"]["ignore_eos"])
        self.assertEqual(observation["output_tokens"], 1)

    def test_parse_prometheus_metrics_skips_labels_and_comments(self) -> None:
        parsed = runner.parse_prometheus_metrics(
            """
            # HELP ignored
            ax_engine_jobs_in_flight 0
            labelled_metric{model=\"qwen\"} 4
            ax_engine_generation_active_streams 1
            """
        )

        self.assertEqual(
            parsed,
            {
                "ax_engine_jobs_in_flight": 0.0,
                "ax_engine_generation_active_streams": 1.0,
            },
        )

    def test_summary_reports_latency_and_failure_counts(self) -> None:
        records = [
            {
                "request": {
                    "ok": True,
                    "ttft_ms": 100.0,
                    "client_tpot_ms": 20.0,
                    "e2e_latency_ms": 300.0,
                    "output_tokens": 10,
                    "route_decisions": {"accepted": 1},
                }
            },
            {
                "request": {
                    "ok": False,
                    "ttft_ms": None,
                    "client_tpot_ms": None,
                    "e2e_latency_ms": 50.0,
                    "output_tokens": None,
                    "route_decisions": {},
                }
            },
        ]

        summary = runner.summarize_window(records, elapsed_s=10.0)

        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["successful_requests"], 1)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["ttft_ms"]["p50"], 100.0)
        self.assertEqual(summary["route_decisions"], {"accepted": 1})

    def test_prepare_output_dir_refuses_existing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            runner.prepare_output_dir(output)
            (output / "manifest.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "not empty"):
                runner.prepare_output_dir(output)

    def test_write_checkpoint_creates_immutable_and_current_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "checkpoints").mkdir()
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
                    window_elapsed_s=14_400.0,
                    target_duration_s=259_200.0,
                    records=[],
                    latest_server={"alive": True, "pid": 42},
                    latest_host={},
                    latest_metrics={},
                    alerts=[],
                )

            self.assertEqual(summary["status"], "running")
            self.assertTrue((output / "summary.json").is_file())
            checkpoints = list((output / "checkpoints").glob("*.json"))
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(json.loads(checkpoints[0].read_text())["reason"], "periodic")

    def test_checkpoint_uses_window_duration_for_throughput(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "checkpoints").mkdir()
            state = runner.RunState(
                started_wall="2026-08-06T00:00:00+00:00",
                started_monotonic=1.0,
                server_pid=42,
                requests_attempted=1,
                requests_ok=1,
            )
            records = [
                {
                    "request": {
                        "ok": True,
                        "ttft_ms": 100.0,
                        "client_tpot_ms": 20.0,
                        "e2e_latency_ms": 300.0,
                        "output_tokens": 10,
                        "route_decisions": {},
                    }
                }
            ]

            summary = runner.write_checkpoint(
                output_dir=output,
                reason="periodic",
                state=state,
                status="running",
                elapsed_s=28_800.0,
                window_elapsed_s=14_400.0,
                target_duration_s=259_200.0,
                records=records,
                latest_server={"alive": True, "pid": 42},
                latest_host={},
                latest_metrics={},
                alerts=[],
            )

            self.assertAlmostEqual(
                summary["latest_window"]["request_throughput_rps"],
                1 / 14_400.0,
            )
            self.assertAlmostEqual(
                summary["latest_window"]["output_token_throughput_tok_s"],
                10 / 14_400.0,
            )

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

    def test_measured_clock_starts_after_warmups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = root / "ax-engine-server"
            server.write_text("#!/bin/sh\n", encoding="utf-8")
            server.chmod(0o755)
            model = root / "model"
            model.mkdir()
            for name in ("config.json", "model-manifest.json", "tokenizer.json"):
                (model / name).write_text("{}\n", encoding="utf-8")
            (model / "weights.safetensors").write_bytes(b"weights")
            output = root / "output"
            args = runner.build_parser().parse_args(
                [
                    "--server",
                    str(server),
                    "--model-dir",
                    str(model),
                    "--output-dir",
                    str(output),
                    "--duration-hours",
                    "0.000001",
                    "--warmup-requests",
                    "2",
                ]
            )
            process = mock.Mock(pid=42)
            process.poll.return_value = None
            process.returncode = 0
            monotonic_values = iter(
                [
                    10.0,  # Initial RunState timestamp.
                    100.0,  # Measured clock, after both warmups.
                    100.1,  # First loop duration check.
                    100.2,  # Final elapsed time.
                    100.3,  # Final checkpoint-window duration.
                ]
            )

            with (
                mock.patch.object(runner, "runtime_metadata", return_value={}),
                mock.patch.object(runner.subprocess, "Popen", return_value=process),
                mock.patch.object(runner, "wait_for_server", return_value={"ok": True}),
                mock.patch.object(
                    runner,
                    "run_stream_request",
                    return_value={"ok": True, "error": None},
                ),
                mock.patch.object(
                    runner,
                    "process_snapshot",
                    return_value={"alive": True, "pid": 42, "rss_kb": 100},
                ),
                mock.patch.object(runner, "collect_host_snapshot", return_value={}),
                mock.patch.object(
                    runner,
                    "collect_metrics",
                    return_value={"ok": True, "values": {}},
                ),
                mock.patch.object(runner, "write_checkpoint") as checkpoint,
                mock.patch.object(runner, "append_jsonl"),
                mock.patch.object(runner, "stop_server"),
                mock.patch.object(runner.time, "monotonic", side_effect=monotonic_values),
            ):
                result = runner._run_endurance(args)

            self.assertEqual(result, 0)
            self.assertEqual(checkpoint.call_args_list[0].kwargs["elapsed_s"], 0.0)
            self.assertAlmostEqual(
                checkpoint.call_args_list[-1].kwargs["elapsed_s"],
                0.2,
            )


if __name__ == "__main__":
    unittest.main()
