#!/usr/bin/env python3
"""Unit tests for the P2 latency artifact runner helpers."""

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

SCRIPT_PATH = Path(__file__).with_name("run_mlx_p2_latency_artifacts.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "run_mlx_p2_latency_artifacts", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
runner = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runner
MODULE_SPEC.loader.exec_module(runner)


def prompt(index: int = 0) -> object:
    return runner.PromptDoc(
        prompt_tokens=8192,
        generation_tokens=1,
        vocab_size=1000,
        token_ids=[1 + index, 2 + index, 3 + index],
        token_ids_sha256=f"{index + 1:064x}"[-64:],
        token_ids_path=f"/tmp/prompt-{index}.json",
    )


class P2LatencyRunnerTests(unittest.TestCase):
    def test_build_prompt_docs_can_reuse_an_exact_shared_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(runner.bench, "model_vocab_size", return_value=100),
                mock.patch.object(
                    runner.bench,
                    "mlx_lm_reference_prompt_tokens",
                    return_value=[1, 2, 3],
                ),
            ):
                docs = runner.build_prompt_docs(
                    model_dir=Path(tmp),
                    artifact_root=Path(tmp) / "prompts",
                    prompt_tokens=3,
                    generation_tokens=1,
                    request_count=4,
                    shared_prefix=True,
                )

        self.assertEqual([doc.token_ids for doc in docs], [[1, 2, 3]] * 4)
        self.assertEqual(len({doc.token_ids_sha256 for doc in docs}), 1)

    def test_run_one_request_uses_client_observed_ttft(self) -> None:
        result = {
            "ttft_ms": 0.0,
            "client_wall_ttft_ms": 125.0,
            "decode_tok_s": 80.0,
        }
        with mock.patch.object(
            runner.bench, "axengine_one_run", return_value=result
        ) as one_run:
            observation = runner.run_one_request(19091, prompt(0), None)

        self.assertEqual(observation["ttft_ms"], 125.0)
        self.assertEqual(observation["ttft_source"], "client_wall_first_output")
        one_run.assert_called_once_with(
            19091,
            prompt(0).token_ids,
            prompt(0).generation_tokens,
            capture_scheduler_step_telemetry=True,
            server_pid=None,
        )

    def test_start_direct_server_passes_model_identity(self) -> None:
        process = mock.Mock()
        with (
            mock.patch.object(runner.bench, "start_axengine", return_value=process) as start,
            mock.patch.object(runner.bench, "wait_for_server", return_value=True),
        ):
            returned, _spawn_ms, _ready_ms = runner.start_direct_server(
                Path("/tmp/model"),
                "published/model-id",
                19091,
            )

        self.assertIs(returned, process)
        start.assert_called_once_with(
            runner.bench.AX_ENGINE_SERVER,
            Path("/tmp/model"),
            19091,
            model_id="published/model-id",
            direct_mode=True,
        )

    def test_parse_concurrency_levels_adds_single_request_baseline(self) -> None:
        self.assertEqual(runner.parse_concurrency_levels("2,4"), [1, 2, 4])

    def test_concurrent_row_computes_ratios_and_overlap_classification(self) -> None:
        single_row = runner.concurrent_row(
            prompts=[prompt(0)],
            trials=[
                {
                    "request_ttft_ms": 100.0,
                    "total_wall_ms": 120.0,
                    "queue_delay_ms": 20.0,
                    "failure_count": 0.0,
                    "peak_memory_gb": 20.0,
                    "observations": [
                        {
                            "scheduler_telemetry": {
                                "ax_scheduler_scheduled_prefill_tokens": 8192,
                                "ax_scheduler_scheduled_decode_tokens": 0,
                                "ax_scheduler_skipped_prefill_tokens": 0,
                                "ax_scheduler_skipped_decode_tokens": 0,
                                "ax_scheduler_mixed_prefill_decode_batches": 0,
                            }
                        }
                    ],
                },
                {
                    "request_ttft_ms": 110.0,
                    "total_wall_ms": 130.0,
                    "queue_delay_ms": 20.0,
                    "failure_count": 0.0,
                    "peak_memory_gb": 21.0,
                    "observations": [],
                },
                {
                    "request_ttft_ms": 120.0,
                    "total_wall_ms": 140.0,
                    "queue_delay_ms": 20.0,
                    "failure_count": 0.0,
                    "peak_memory_gb": 22.0,
                    "observations": [],
                },
            ],
            single_row=None,
        )
        multi_row = runner.concurrent_row(
            prompts=[prompt(0), prompt(1), prompt(2), prompt(3)],
            trials=[
                {
                    "request_ttft_ms": 300.0,
                    "total_wall_ms": 260.0,
                    "queue_delay_ms": 80.0,
                    "failure_count": 0.0,
                    "peak_memory_gb": 30.0,
                    "observations": [
                        {
                            "scheduler_telemetry": {
                                "ax_scheduler_scheduled_prefill_tokens": 2047,
                                "ax_scheduler_scheduled_decode_tokens": 1,
                                "ax_scheduler_skipped_prefill_tokens": 4096,
                                "ax_scheduler_skipped_decode_tokens": 0,
                                "ax_scheduler_mixed_prefill_decode_batches": 1,
                            }
                        }
                    ],
                },
                {
                    "request_ttft_ms": 330.0,
                    "total_wall_ms": 280.0,
                    "queue_delay_ms": 90.0,
                    "failure_count": 0.0,
                    "peak_memory_gb": 31.0,
                    "observations": [],
                },
                {
                    "request_ttft_ms": 360.0,
                    "total_wall_ms": 300.0,
                    "queue_delay_ms": 100.0,
                    "failure_count": 0.0,
                    "peak_memory_gb": 32.0,
                    "observations": [],
                },
            ],
            single_row=single_row,
        )

        self.assertEqual(multi_row["concurrent_requests"], 4)
        self.assertAlmostEqual(
            multi_row["ratios_to_single_request"]["request_ttft_ms"],
            3.0,
        )
        self.assertEqual(
            multi_row["prefill_overlap"]["classification"],
            "partial_overlap",
        )
        self.assertEqual(
            multi_row["scheduler_evidence"]["scheduled_decode_tokens"],
            1,
        )
        self.assertEqual(
            multi_row["scheduler_evidence"]["mixed_prefill_decode_batches"],
            1,
        )

    def test_scheduler_evidence_deduplicates_shared_step_ids(self) -> None:
        shared_step = {
            "step_id": 10,
            "ax_scheduler_scheduled_decode_tokens": 2,
        }
        evidence = runner.summarize_scheduler_evidence(
            [
                {
                    "observations": [
                        {
                            "scheduler_telemetry": {
                                "ax_scheduler_scheduled_decode_tokens": 2,
                            },
                            "scheduler_step_telemetry": [shared_step],
                        },
                        {
                            "scheduler_telemetry": {
                                "ax_scheduler_scheduled_decode_tokens": 3,
                            },
                            "scheduler_step_telemetry": [
                                shared_step,
                                {
                                    "step_id": 11,
                                    "ax_scheduler_scheduled_decode_tokens": 1,
                                },
                            ],
                        },
                    ]
                }
            ]
        )

        self.assertEqual(evidence["scheduled_decode_tokens"], 3)

    def test_shared_step_evidence_reports_engine_step_fanout(self) -> None:
        evidence = runner.summarize_shared_step_evidence(
            [
                {
                    "observations": [
                        {
                            "scheduler_step_telemetry": [
                                {"step_id": 10},
                                {"step_id": 11},
                            ]
                        },
                        {
                            "scheduler_step_telemetry": [
                                {"step_id": 10},
                                {"step_id": 11},
                            ]
                        },
                    ]
                }
            ]
        )

        self.assertTrue(evidence["available"])
        self.assertEqual(evidence["trials"][0]["stream_step_records"], 4)
        self.assertEqual(evidence["trials"][0]["unique_engine_steps"], 2)
        self.assertEqual(evidence["trials"][0]["shared_engine_steps"], 2)
        self.assertEqual(evidence["trials"][0]["max_step_fanout"], 2)

    def test_startup_rows_keep_warm_load_metrics_out_of_benchmark_warm(self) -> None:
        row = runner.startup_phase_row(
            phase="benchmark_warm",
            prompt=prompt(0),
            observations=[
                {
                    "ttft_ms": 100.0,
                    "decode_tok_s": 50.0,
                    "wall_ms": 150.0,
                    "peak_memory_gb": 20.0,
                }
            ],
        )

        self.assertNotIn("model_load_ms", row)
        self.assertNotIn("server_ready_ms", row)

    def test_concurrent_capture_warms_each_shape_before_measurement(self) -> None:
        process = mock.Mock(pid=1234)
        trial = {
            "request_ttft_ms": 10.0,
            "total_wall_ms": 20.0,
            "queue_delay_ms": 10.0,
            "failure_count": 0,
            "peak_memory_gb": 1.0,
            "observations": [],
        }
        prompt_groups = {1: [prompt(0)], 4: [prompt(index) for index in range(4)]}
        with (
            mock.patch.object(
                runner,
                "start_direct_server",
                return_value=(process, 0.0, 0.0),
            ),
            mock.patch.object(
                runner,
                "run_concurrent_trial",
                return_value=trial,
            ) as run_trial,
            mock.patch.object(runner.bench, "kill_proc"),
        ):
            artifact = runner.capture_concurrent_artifact(
                model_dir=Path("/tmp/model"),
                model_id="test-model",
                model_metadata={},
                host_label="test-host",
                port=19091,
                prompt_groups=prompt_groups,
                warmup_repetitions=2,
                repetitions=3,
                cooldown=0.0,
            )

        self.assertEqual(run_trial.call_count, 10)
        self.assertEqual(artifact["warmup_repetitions"], 2)
        self.assertEqual([row["repetitions"] for row in artifact["rows"]], [3, 3])

    def test_concurrent_trial_samples_peak_rss_while_requests_run(self) -> None:
        observation = {
            "ttft_ms": 10.0,
            "wall_ms": 20.0,
            "decode_tok_s": 1.0,
        }
        with (
            mock.patch.object(runner, "run_one_request", return_value=observation),
            mock.patch.object(
                runner.bench,
                "process_rss_gb",
                side_effect=[1.0, 4.0, 2.0],
            ),
        ):
            trial = runner.run_concurrent_trial(
                port=19091,
                prompts=[prompt(0)],
                server_pid=1234,
            )

        self.assertEqual(trial["peak_memory_gb"], 4.0)

    def test_concurrent_trial_preserves_request_errors(self) -> None:
        with (
            mock.patch.object(
                runner,
                "run_one_request",
                side_effect=RuntimeError("pool exhausted"),
            ),
            mock.patch.object(runner.bench, "process_rss_gb", return_value=1.0),
        ):
            trial = runner.run_concurrent_trial(
                port=19091,
                prompts=[prompt(0)],
                server_pid=1234,
            )

        self.assertEqual(trial["failure_count"], 1)
        self.assertEqual(
            trial["request_errors"],
            ["RuntimeError: pool exhausted"],
        )

    def test_concurrent_capture_records_explicit_shared_prefix_mode(self) -> None:
        process = mock.Mock(pid=1234)
        trial = {
            "request_ttft_ms": 10.0,
            "total_wall_ms": 20.0,
            "queue_delay_ms": 10.0,
            "failure_count": 0,
            "peak_memory_gb": 1.0,
            "observations": [],
        }
        with (
            mock.patch.object(
                runner,
                "start_direct_server",
                return_value=(process, 0.0, 0.0),
            ) as start,
            mock.patch.object(
                runner,
                "run_concurrent_trial",
                return_value=trial,
            ) as run_trial,
            mock.patch.object(runner.bench, "process_rss_gb", return_value=0.5),
            mock.patch.object(runner.bench, "kill_proc"),
        ):
            artifact = runner.capture_concurrent_artifact(
                model_dir=Path("/tmp/model"),
                model_id="test-model",
                model_metadata={},
                host_label="test-host",
                port=19091,
                prompt_groups={1: [prompt(0)]},
                warmup_repetitions=1,
                repetitions=3,
                cooldown=0.0,
                prefix_cache_enabled=True,
                shared_prefix=True,
                capture_output_token_ids=True,
            )

        start.assert_called_once_with(
            Path("/tmp/model"),
            "test-model",
            19091,
            prefix_cache_enabled=True,
        )
        self.assertTrue(artifact["prefix_cache_enabled"])
        self.assertTrue(artifact["shared_prefix"])
        self.assertTrue(artifact["capture_output_token_ids"])
        self.assertTrue(
            all(
                call.kwargs["capture_output_token_ids"]
                for call in run_trial.call_args_list
            )
        )

    def test_dry_run_does_not_require_real_model_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen"}))
            output_dir = root / "out"

            code = runner.main_with_args_for_test(
                [
                    "--model-dir",
                    str(model_dir),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ]
            )

            self.assertEqual(code, 0)

    def test_dry_run_reports_markdown_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen"}))
            output_dir = root / "out"

            code = runner.main_with_args_for_test(
                [
                    "--model-dir",
                    str(model_dir),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
