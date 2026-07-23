#!/usr/bin/env python3
"""Tests for the shared AX/mlxcel Qwen+Gemma flip target runner."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import bench_ax_multimodel_serving as multimodel
import bench_ax_serving as serving
import bench_qwen_gemma_flip_target as flip


class FlipTargetBenchmarkTests(unittest.TestCase):
    def test_loads_unmanaged_ax_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "target.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": flip.TARGET_SCHEMA_VERSION,
                        "name": "ax-candidate",
                        "runtime": "ax-engine",
                        "runtime_revision": "deadbeef",
                        "managed_processes": False,
                        "models": {
                            "qwen3.5-9b": {
                                "served_model": "qwen3.5-9b",
                                "base_url": "http://127.0.0.1:31418",
                                "memory_cap_bytes": 1024,
                            }
                        },
                        "comparison_contract": {
                            "id": "test",
                            "total_memory_cap_bytes": 1024,
                        },
                    }
                )
            )

            target = flip.load_target(path)

        self.assertEqual(target.runtime, "ax-engine")
        self.assertFalse(target.managed_processes)
        self.assertEqual(
            target.models["qwen3.5-9b"].base_url,
            "http://127.0.0.1:31418",
        )

    def test_managed_mlxcel_command_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            target_path = root / "target.json"
            target_path.write_text(
                json.dumps(
                    {
                        "schema_version": flip.TARGET_SCHEMA_VERSION,
                        "name": "mlxcel-peer",
                        "runtime": "mlxcel",
                        "runtime_revision": "v0.4.2@1b9a0018",
                        "managed_processes": True,
                        "binary": "/usr/bin/true",
                        "common_args": ["--parallel", "4"],
                        "models": {
                            "qwen3.5-9b": {
                                "model_path": str(model_dir),
                                "served_model": "qwen3.5-9b",
                                "port": 31801,
                                "args": ["--max-batch-prefill", "4"],
                                "memory_cap_bytes": 1024,
                            }
                        },
                        "comparison_contract": {
                            "id": "test",
                            "total_memory_cap_bytes": 1024,
                        },
                    }
                )
            )
            target = flip.load_target(target_path)
            supervisor = flip.ProcessSupervisor(target, root / "logs")

            command = supervisor.command_for(target.models["qwen3.5-9b"])

        self.assertEqual(command[0], "/usr/bin/true")
        self.assertIn("--alias", command)
        self.assertIn("qwen3.5-9b", command)
        self.assertEqual(command[-4:], ["--parallel", "4", "--max-batch-prefill", "4"])

    def test_managed_ax_command_uses_one_primary_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            qwen_dir = root / "qwen"
            gemma_dir = root / "gemma"
            qwen_dir.mkdir()
            gemma_dir.mkdir()
            target_path = root / "target.json"
            target_path.write_text(
                json.dumps(
                    {
                        "schema_version": flip.TARGET_SCHEMA_VERSION,
                        "name": "ax-candidate",
                        "runtime": "ax-engine",
                        "runtime_revision": "deadbeef",
                        "managed_single_process": True,
                        "primary_model_id": "qwen3.5-9b",
                        "binary": "/usr/bin/true",
                        "common_args": ["--port", "31418"],
                        "models": {
                            "qwen3.5-9b": {
                                "model_path": str(qwen_dir),
                                "base_url": "http://127.0.0.1:31418",
                                "memory_cap_bytes": 1024,
                            },
                            "gemma-4-12b-it": {
                                "model_path": str(gemma_dir),
                                "base_url": "http://127.0.0.1:31418",
                                "memory_cap_bytes": 1024,
                            },
                        },
                        "comparison_contract": {
                            "id": "test",
                            "total_memory_cap_bytes": 1024,
                        },
                    }
                )
            )
            target = flip.load_target(target_path)
            supervisor = flip.SingleProcessSupervisor(target, root / "logs")

            command = supervisor.command()

        self.assertTrue(target.managed_single_process)
        self.assertEqual(command[0], "/usr/bin/true")
        self.assertEqual(
            command[1:7],
            [
                "--model-id",
                "qwen3.5-9b",
                "--mlx",
                "--mlx-model-artifacts-dir",
                str(qwen_dir.resolve()),
                "--port",
            ],
        )
        self.assertNotIn(str(gemma_dir.resolve()), command)

    def test_openai_stream_uses_authoritative_usage(self) -> None:
        prompt = serving.PromptItem(
            id="request",
            category="interactive_decode",
            input_text="hello",
            input_tokens=None,
            input_tokens_count=None,
            max_output_tokens=2,
            metadata={},
        )
        events = [
            ("__http_status__", {"status": 200}, 0.0),
            (None, {"choices": [{"delta": {"role": "assistant"}}]}, 0.01),
            (None, {"choices": [{"delta": {"reasoning_content": "r"}}]}, 0.10),
            (None, {"choices": [{"delta": {"content": "x"}}]}, 0.20),
            (
                None,
                {
                    "choices": [{"delta": {}, "finish_reason": "length"}],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
                0.21,
            ),
            (None, {"done": True}, 0.22),
        ]

        observed = flip.observe_openai_stream(
            events,
            prompt=prompt,
            scheduled_at_s=0.0,
            started_at_s=0.0,
            completed_at_s=0.30,
        )

        self.assertTrue(observed["ok"])
        self.assertEqual(observed["input_tokens"], 3)
        self.assertEqual(observed["output_tokens"], 2)
        self.assertEqual(observed["ttft_ms"], 100.0)
        self.assertEqual(observed["stream_step_interval_ms"], [100.0])
        self.assertEqual(observed["output_identity"]["kind"], "text_utf8")

    def test_openai_stream_fails_closed_without_usage(self) -> None:
        prompt = serving.PromptItem(
            id="request",
            category="interactive_decode",
            input_text="hello",
            input_tokens=None,
            input_tokens_count=None,
            max_output_tokens=2,
            metadata={},
        )

        observed = flip.observe_openai_stream(
            [(None, {"choices": [{"delta": {"content": "x"}}]}, 0.1), (None, {"done": True}, 0.2)],
            prompt=prompt,
            scheduled_at_s=0.0,
            started_at_s=0.0,
            completed_at_s=0.3,
        )

        self.assertFalse(observed["ok"])
        self.assertIn("completion_tokens", observed["error"])

    def test_managed_control_maps_unload_and_load(self) -> None:
        event = multimodel.ScenarioEvent(
            id="unload",
            kind="unload",
            at_s=0.0,
            model_id="gemma-4-12b-it",
            category="sibling_unload",
            raw={},
        )
        supervisor = mock.Mock()
        supervisor.stop.return_value = (True, {"exit_code": -15})

        observed = flip.run_managed_control_event(
            event,
            supervisor=supervisor,
            benchmark_started=time.perf_counter(),
        )

        self.assertTrue(observed["ok"])
        supervisor.stop.assert_called_once_with("gemma-4-12b-it")


if __name__ == "__main__":
    unittest.main()
