#!/usr/bin/env python3
"""Unit tests for the MLX inference-stack benchmark contract."""

from __future__ import annotations

import argparse
import io
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("bench_mlx_inference_stack.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "bench_mlx_inference_stack", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
bench = importlib.util.module_from_spec(MODULE_SPEC)
# Register in sys.modules before exec so @dataclass decorators can resolve
# their owning module via `cls.__module__` (Python 3.14 requirement).
sys.modules["bench_mlx_inference_stack"] = bench
MODULE_SPEC.loader.exec_module(bench)

CHECKER_PATH = Path(__file__).with_name("check_turboquant_quality_artifact.py")
CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_turboquant_quality_artifact",
    CHECKER_PATH,
)
assert CHECKER_SPEC and CHECKER_SPEC.loader
turboquant_checker = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(turboquant_checker)


def write_gateddelta_model(
    root: Path,
    *,
    model_family: str = "qwen3_5",
    key_head_dim: int = 64,
) -> Path:
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (model_dir / "model-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ax.native_model_manifest.v1",
                "model_family": model_family,
                "linear_attention": {
                    "num_value_heads": 4,
                    "num_key_heads": 4,
                    "key_head_dim": key_head_dim,
                    "value_head_dim": 128,
                    "conv_kernel_dim": 4,
                },
            }
        )
    )
    return model_dir


def write_sparse_file(path: Path, size: int) -> None:
    with path.open("wb") as handle:
        handle.truncate(size)


class MlxInferenceStackBenchTests(unittest.TestCase):
    def test_default_repetition_and_cooldown_contract_matches_docs(self) -> None:
        self.assertEqual(bench.DEFAULT_REPETITIONS, 5)
        self.assertEqual(bench.DEFAULT_COOLDOWN, 15.0)

    def test_hf_cache_roots_follow_hugging_face_env_order(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HF_HUB_CACHE": "/tmp/hf-hub",
                "HF_HOME": "/tmp/hf-home",
                "XDG_CACHE_HOME": "/tmp/xdg-cache",
            },
            clear=True,
        ):
            roots = bench.hf_cache_roots()

        self.assertEqual(
            roots[:4],
            [
                Path("/tmp/hf-hub"),
                Path("/tmp/hf-home/hub"),
                Path("/tmp/xdg-cache/huggingface/hub"),
                Path.home() / ".cache" / "huggingface" / "hub",
            ],
        )

    def test_collect_performance_condition_metadata_parses_pmset(self) -> None:
        def fake_check_output(
            cmd: list[str],
            *,
            text: bool,
            stderr: int | None = None,
        ) -> str:
            del text, stderr
            if cmd == ["pmset", "-g", "batt"]:
                return (
                    "Now drawing from 'AC Power'\n"
                    " -InternalBattery-0\t80%; AC attached; not charging present: true\n"
                )
            if cmd == ["pmset", "-g", "therm"]:
                return (
                    "Note: No thermal warning level has been recorded\n"
                    "Note: Performance warning level: 1\n"
                    "Note: No CPU power status has been recorded\n"
                )
            raise AssertionError(cmd)

        with (
            patch.object(subprocess, "check_output", side_effect=fake_check_output),
            patch.object(bench.os, "getloadavg", return_value=(1.23456, 2.0, 3.0)),
        ):
            metadata = bench.collect_performance_condition_metadata()

        self.assertEqual(
            metadata["load_average"],
            {"one_minute": 1.235, "five_minutes": 2.0, "fifteen_minutes": 3.0},
        )
        self.assertEqual(metadata["power_source"], "AC Power")
        self.assertIn("80%", metadata["battery_status"])
        self.assertFalse(metadata["thermal_warning_recorded"])
        self.assertTrue(metadata["performance_warning_recorded"])
        self.assertFalse(metadata["cpu_power_status_recorded"])

    def test_collect_host_metadata_uses_supplied_performance_conditions(self) -> None:
        supplied = {"load_average": {"one_minute": 1.0}}

        with patch.object(
            bench,
            "collect_performance_condition_metadata",
            side_effect=AssertionError("should not resample"),
        ):
            metadata = bench.collect_host_metadata(supplied)

        self.assertIs(metadata["performance_conditions"], supplied)

    def test_resolve_model_dir_uses_hugging_face_cache_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "models--mlx-community--Qwen3.5-9B-MLX-4bit"
            snapshot = repo_cache / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (repo_cache / "refs").mkdir()
            (repo_cache / "refs" / "main").write_text("abc123")
            (snapshot / "config.json").write_text("{}")
            (snapshot / "model-manifest.json").write_text("{}")
            (snapshot / "model.safetensors").write_bytes(b"")

            resolved = bench.resolve_model_dir(
                None,
                "mlx-community/Qwen3.5-9B-MLX-4bit",
                root,
            )

        self.assertEqual(resolved, snapshot)

    def test_resolve_model_dir_rejects_unprepared_hugging_face_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "models--mlx-community--Qwen3.5-9B-MLX-4bit"
            snapshot = repo_cache / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (repo_cache / "refs").mkdir()
            (repo_cache / "refs" / "main").write_text("abc123")

            with self.assertRaisesRegex(RuntimeError, "not AX-ready"):
                bench.resolve_model_dir(
                    None,
                    "mlx-community/Qwen3.5-9B-MLX-4bit",
                    root,
                )

    def test_resolve_model_dir_prefers_explicit_model_dir(self) -> None:
        explicit = Path("/tmp/ax-model")

        self.assertEqual(
            bench.resolve_model_dir(
                explicit,
                "mlx-community/Qwen3.5-9B-MLX-4bit",
                Path("/missing-cache"),
            ),
            explicit,
        )

    def test_explicit_model_repo_id_model_resolves_same_cache_repo(self) -> None:
        self.assertEqual(
            bench.normalize_model_repo_id_for_cache(
                model="mlx-community/gemma-4-e2b-it-4bit",
                model_repo_id=bench.DEFAULT_MODEL_REPO_ID,
                model_arg_explicit=True,
                model_repo_id_arg_explicit=False,
                model_dir_explicit=False,
            ),
            "mlx-community/gemma-4-e2b-it-4bit",
        )

    def test_explicit_model_keeps_explicit_model_repo_id(self) -> None:
        self.assertEqual(
            bench.normalize_model_repo_id_for_cache(
                model="mlx-community/gemma-4-e2b-it-4bit",
                model_repo_id="mlx-community/Qwen3.5-9B-MLX-4bit",
                model_arg_explicit=True,
                model_repo_id_arg_explicit=True,
                model_dir_explicit=False,
            ),
            "mlx-community/Qwen3.5-9B-MLX-4bit",
        )

    def test_explicit_non_repo_model_requires_repo_id_or_model_dir(self) -> None:
        with self.assertRaisesRegex(ValueError, "--model was provided"):
            bench.normalize_model_repo_id_for_cache(
                model="./models/local",
                model_repo_id=bench.DEFAULT_MODEL_REPO_ID,
                model_arg_explicit=True,
                model_repo_id_arg_explicit=False,
                model_dir_explicit=False,
            )

    def test_kv_compression_blocker_keys_match_quality_gate_contract(self) -> None:
        self.assertEqual(
            bench.KV_COMPRESSION_FUSED_DECODE_BLOCKED_COUNTERS,
            turboquant_checker.FUSED_DECODE_BLOCKED_COUNTERS,
        )
        self.assertEqual(
            bench.KV_COMPRESSION_FUSED_DECODE_BLOCKED_ATTENTION_KIND_COUNTERS,
            turboquant_checker.FUSED_DECODE_BLOCKED_ATTENTION_KIND_COUNTERS,
        )

    def test_parse_mlx_lm_trials_and_reported_averages(self) -> None:
        parsed = bench.parse_mlx_lm_benchmark_output(
            "\n".join(
                [
                    "Trial 1:  prompt_tps=10.000, generation_tps=20.000, "
                    "peak_memory=3.000, total_time=4.000",
                    "Trial 2:  prompt_tps=14.000, generation_tps=30.000, "
                    "peak_memory=5.000, total_time=6.000",
                    "Averages: prompt_tps=12.000, generation_tps=25.000, "
                    "peak_memory=4.000",
                ]
            )
        )

        self.assertEqual(parsed["prefill_tok_s"]["mean"], 12.0)
        self.assertEqual(parsed["prefill_tok_s"]["median"], 12.0)
        self.assertEqual(parsed["decode_tok_s"]["median"], 25.0)
        self.assertEqual(parsed["peak_memory_gb"]["max"], 5.0)
        self.assertEqual(parsed["reported_averages"]["decode_tok_s"], 25.0)

    def test_mlx_lm_row_attaches_derived_ttft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt = bench.write_prompt_tokens(
                Path(tmp),
                prompt_tokens=4,
                generation_tokens=2,
                vocab_size=100,
                tokens=[1, 2, 3, 4],
            )
            prompt["token_ids"] = [1, 2, 3, 4]
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="\n".join(
                    [
                        "Trial 1:  prompt_tps=10.000, generation_tps=20.000, "
                        "peak_memory=3.000, total_time=4.000",
                        "Trial 2:  prompt_tps=20.000, generation_tps=30.000, "
                        "peak_memory=5.000, total_time=6.000",
                        "Averages: prompt_tps=15.000, generation_tps=25.000, "
                        "peak_memory=4.000",
                    ]
                ),
                stderr="",
            )
            with patch.object(bench.subprocess, "run", return_value=completed):
                row = bench.run_mlx_lm_benchmark(
                    "model",
                    4,
                    2,
                    2,
                    0.0,
                    2048,
                    prompt,
                )

        self.assertEqual(row["ttft_source"], "derived_from_mlx_lm_prefill_tok_s")
        self.assertEqual(row["ttft_ms"]["median"], 300.0)

    def test_wait_for_server_returns_when_process_exits(self) -> None:
        class ExitedProcess:
            def poll(self) -> int:
                return 1

        with patch.object(bench.urllib.request, "urlopen") as urlopen:
            ready = bench.wait_for_server(
                "http://127.0.0.1:1/health",
                timeout=60.0,
                proc=ExitedProcess(),
            )

        self.assertFalse(ready)
        urlopen.assert_not_called()

    def test_ensure_port_available_rejects_existing_listener(self) -> None:
        class ConnectedSocket:
            def __enter__(self) -> "ConnectedSocket":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def settimeout(self, _timeout: float) -> None:
                return None

            def connect_ex(self, _address: tuple[str, int]) -> int:
                return 0

        with patch.object(bench.socket, "socket", return_value=ConnectedSocket()):
            with self.assertRaisesRegex(RuntimeError, "--axengine-port"):
                bench.ensure_port_available(19091)

    def test_ensure_port_available_allows_closed_listener_port(self) -> None:
        class DisconnectedSocket:
            def __enter__(self) -> "DisconnectedSocket":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def settimeout(self, _timeout: float) -> None:
                return None

            def connect_ex(self, _address: tuple[str, int]) -> int:
                return 61

        with patch.object(bench.socket, "socket", return_value=DisconnectedSocket()):
            bench.ensure_port_available(19091)

    def test_ensure_ax_engine_server_binary_builds_before_checking_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "ax-engine-server"
            binary.write_text("#!/bin/sh\n")
            with (
                patch.object(bench, "AX_ENGINE_SERVER", binary),
                patch.object(bench.subprocess, "run") as run,
            ):
                bench.ensure_ax_engine_server_binary(build=True)

        run.assert_called_once_with(
            ["cargo", "build", "-p", "ax-engine-server", "--release"],
            cwd=bench.REPO_ROOT,
            check=True,
        )

    def test_ensure_ax_engine_server_binary_reports_missing_without_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "missing-server"
            with patch.object(bench, "AX_ENGINE_SERVER", binary):
                with self.assertRaisesRegex(RuntimeError, "ax-engine-server not found"):
                    bench.ensure_ax_engine_server_binary(build=False)

    def test_prompt_artifact_records_hash_and_validates_inline_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt = bench.write_prompt_tokens(
                Path(tmp),
                prompt_tokens=4,
                generation_tokens=2,
                vocab_size=100,
                tokens=[1, 2, 3, 4],
            )
            prompt["token_ids"] = [1, 2, 3, 4]

            bench.validate_prompt_doc(
                prompt,
                prompt_tokens=4,
                generation_tokens=2,
            )
            payload = json.loads(Path(prompt["token_ids_path"]).read_text())
            self.assertEqual(payload["schema_version"], "ax.mlx_reference_prompt.v1")
            self.assertEqual(payload["sha256"], prompt["token_ids_sha256"])

            prompt["token_ids"] = [1, 2, 3, 5]
            with self.assertRaisesRegex(RuntimeError, "inline token hash mismatch"):
                bench.validate_prompt_doc(
                    prompt,
                    prompt_tokens=4,
                    generation_tokens=2,
                )

    def test_axengine_summary_includes_ttft_and_memory(self) -> None:
        runs = [
            {
                "prefill_s": 0.3,
                "decode_s": 0.1,
                "ttft_ms": 300.0,
                "client_wall_ttft_ms": 330.0,
                "client_wall_total_ms": 520.0,
                "prefill_tok_s": 10.0,
                "decode_tok_s": 20.0,
                "output_tokens": 3.0,
                "peak_memory_gb": 11.0,
            },
            {
                "prefill_s": 0.2,
                "decode_s": 0.1,
                "ttft_ms": 200.0,
                "client_wall_ttft_ms": 230.0,
                "client_wall_total_ms": 420.0,
                "prefill_tok_s": 15.0,
                "decode_tok_s": 20.0,
                "output_tokens": 3.0,
                "peak_memory_gb": 12.0,
            },
            {
                "prefill_s": 0.4,
                "decode_s": 0.1,
                "ttft_ms": 400.0,
                "client_wall_ttft_ms": 430.0,
                "client_wall_total_ms": 620.0,
                "prefill_tok_s": 8.0,
                "decode_tok_s": 20.0,
                "output_tokens": 3.0,
                "peak_memory_gb": 13.0,
            },
        ]
        with patch.object(bench, "axengine_one_run", side_effect=[runs[0], *runs]):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                3,
                0.0,
                model_metadata={},
                direct_mode=True,
                server_pid=123,
            )

        self.assertEqual(row["ttft_ms"]["median"], 300.0)
        self.assertEqual(row["ttft_source"], "ax_engine_runner_prefill_time")
        self.assertEqual(row["client_wall_ttft_ms"]["median"], 330.0)
        self.assertEqual(row["client_wall_total_ms"]["median"], 520.0)
        self.assertEqual(
            row["client_wall_ttft_source"],
            "http_sse_first_output_token_observed_by_client",
        )
        self.assertEqual(
            row["prefill_work_contract"],
            "historical_full_logits_prefill_or_sampler_required",
        )
        self.assertEqual(row["runtime_identity"]["selected_backend"], "mlx")
        self.assertEqual(row["runtime_identity"]["route_identity"], "repo_owned_mlx")
        self.assertEqual(row["peak_memory_gb"]["max"], 13.0)
        self.assertEqual(row["memory_source"], "server_process_rss_after_stream")

    def test_bench_axengine_records_configured_seed(self) -> None:
        run = {
            "prefill_s": 0.3,
            "decode_s": 0.1,
            "ttft_ms": 300.0,
            "client_wall_total_ms": 420.0,
            "prefill_tok_s": 10.0,
            "decode_tok_s": 20.0,
            "output_tokens": 3.0,
        }
        with patch.object(bench, "axengine_one_run", side_effect=[dict(run), dict(run)]):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                1,
                1,
                0.0,
                model_metadata={},
                direct_mode=True,
                seed=44,
                prompt_source="real",
            )

        self.assertEqual(row["random_seed"], 44)
        self.assertEqual(row["seed"], 44)
        self.assertEqual(row["prompt_contract"], "real_prompt_tokenized")
        self.assertEqual(row["ax_decode_row_identity"]["seed"], 44)
        self.assertEqual(row["trials"][0]["random_seed"], 44)
        self.assertEqual(row["trials"][0]["seed"], 44)

    def test_bench_axengine_reports_interim_summary_before_cooldown(self) -> None:
        runs = [
            {
                "prefill_s": 0.3,
                "decode_s": 0.1,
                "ttft_ms": 300.0,
                "prefill_tok_s": 10.0,
                "decode_tok_s": 20.0,
                "output_tokens": 3.0,
            },
            {
                "prefill_s": 0.2,
                "decode_s": 0.1,
                "ttft_ms": 200.0,
                "prefill_tok_s": 15.0,
                "decode_tok_s": 30.0,
                "output_tokens": 3.0,
            },
        ]
        stderr = io.StringIO()
        with (
            patch.object(bench, "axengine_one_run", side_effect=runs),
            patch.object(bench.time, "sleep") as sleep,
            patch.object(sys, "stderr", stderr),
        ):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                2,
                0,
                5.0,
                model_metadata={},
                direct_mode=True,
            )

        sleep.assert_called_once_with(5.0)
        self.assertEqual(row["decode_tok_s"]["median"], 25.0)
        output = stderr.getvalue()
        self.assertIn("interim after 1/2:", output)
        self.assertIn("median prefill=10.0 tok/s", output)
        self.assertIn("decode=20.0 tok/s", output)
        self.assertIn("range=20.0-20.0", output)
        self.assertIn("cooldown 5s", output)

    def test_bench_axengine_records_tail_regression_stability(self) -> None:
        runs = [
            {
                "prefill_s": 0.3,
                "decode_s": 0.1,
                "ttft_ms": 300.0,
                "prefill_tok_s": 10.0,
                "decode_tok_s": 20.0,
                "output_tokens": 3.0,
            },
            {
                "prefill_s": 0.3,
                "decode_s": 0.1,
                "ttft_ms": 300.0,
                "prefill_tok_s": 10.0,
                "decode_tok_s": 17.0,
                "output_tokens": 3.0,
            },
        ]
        stderr = io.StringIO()
        with (
            patch.object(bench, "axengine_one_run", side_effect=runs),
            patch.object(sys, "stderr", stderr),
        ):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                2,
                0,
                0.0,
                model_metadata={},
                direct_mode=True,
            )

        self.assertEqual(
            row["run_stability"]["schema_version"],
            "ax.benchmark_run_stability.v1",
        )
        self.assertEqual(row["run_stability"]["metric"], "decode_tok_s")
        self.assertEqual(row["run_stability"]["classification"], "tail_regression")
        self.assertEqual(row["run_stability"]["trial_count"], 2)
        self.assertEqual(row["run_stability"]["first"], 20.0)
        self.assertEqual(row["run_stability"]["last"], 17.0)
        self.assertAlmostEqual(row["run_stability"]["last_vs_first_pct"], -15.0)

    def test_print_summary_includes_run_stability(self) -> None:
        stdout = io.StringIO()
        with patch.object(sys, "stdout", stdout):
            bench.print_summary(
                {
                    "results": [
                        {
                            "engine": "ax_engine_mlx",
                            "prompt_tokens": 128,
                            "prefill_tok_s": {"median": 100.0},
                            "decode_tok_s": {"median": 17.0},
                            "baseline": {"decode_ratio_to_mlx_lm": 0.9},
                            "method": "server_sse_runner_time_us",
                            "run_stability": {
                                "classification": "tail_regression",
                                "last_vs_first_pct": -15.0,
                            },
                        }
                    ]
                }
            )

        output = stdout.getvalue()
        self.assertIn("Stability", output)
        self.assertIn("tail -15.0%", output)

    def test_ax_prefill_work_contract_labels_long_greedy_cache_only_boundary(
        self,
    ) -> None:
        self.assertEqual(
            bench.ax_prefill_work_contract(2048, sampler=None),
            "mlx_lm_style_cache_only_prefix_plus_final_prompt_token",
        )
        self.assertEqual(
            bench.ax_prefill_work_contract(512, sampler=None),
            "historical_full_logits_prefill_or_sampler_required",
        )
        self.assertEqual(
            bench.ax_prefill_work_contract(2048, sampler={"temperature": 0.7}),
            "historical_full_logits_prefill_or_sampler_required",
        )

    def test_axengine_one_run_records_client_wall_ttft_from_first_output(self) -> None:
        class FakeResponse:
            status = 200

            def __iter__(self):
                frames = [
                    {
                        "step": {"runner_time_us": 100_000, "scheduled_tokens": 4},
                        "request": {"output_len": 1},
                    },
                    {
                        "response": {
                            "output_tokens": [42],
                        },
                    },
                ]
                for event_name, payload in (
                    ("step", frames[0]),
                    ("response", frames[1]),
                ):
                    yield f"event: {event_name}\n".encode()
                    yield b"data: " + json.dumps(payload).encode() + b"\n"
                    yield b"\n"

        class FakeConnection:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def request(self, *_args, **_kwargs) -> None:
                pass

            def getresponse(self) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with patch.object(bench.http.client, "HTTPConnection", FakeConnection):
            with patch.object(
                bench.time, "perf_counter", side_effect=[10.0, 10.123, 10.456]
            ):
                run = bench.axengine_one_run(19091, [1, 2, 3, 4], 1)

        self.assertAlmostEqual(run["client_wall_ttft_ms"], 123.0, places=6)
        self.assertAlmostEqual(run["client_wall_total_ms"], 456.0, places=6)
        self.assertEqual(run["ttft_ms"], 100.0)

    def test_axengine_one_run_sends_configured_sampling_seed(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            status = 200

            def __iter__(self):
                frame = {
                    "response": {
                        "output_tokens": [42],
                    },
                }
                yield b"event: response\n"
                yield b"data: " + json.dumps(frame).encode() + b"\n"
                yield b"\n"

        class FakeConnection:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def request(self, *_args, **kwargs) -> None:
                captured["body"] = kwargs["body"]

            def getresponse(self) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with patch.object(bench.http.client, "HTTPConnection", FakeConnection):
            with patch.object(
                bench.time, "perf_counter", side_effect=[10.0, 10.1, 10.456]
            ):
                bench.axengine_one_run(
                    19091,
                    [1, 2, 3, 4],
                    1,
                    sampler={"temperature": 0.6, "seed": 999},
                    seed=44,
                )

        body = json.loads(captured["body"])
        self.assertEqual(body["sampling"]["temperature"], 0.6)
        self.assertEqual(body["sampling"]["seed"], 44)

    def test_axengine_one_run_decode_rate_matches_mlx_lm_generation_contract(
        self,
    ) -> None:
        class FakeResponse:
            status = 200

            def __iter__(self):
                frames = [
                    {
                        "step": {"runner_time_us": 200_000, "scheduled_tokens": 4},
                        "request": {"output_len": 1},
                    },
                    {
                        "step": {"runner_time_us": 500_000, "scheduled_tokens": 1},
                        "request": {"output_len": 2},
                    },
                    {
                        "step": {"runner_time_us": 500_000, "scheduled_tokens": 1},
                        "request": {"output_len": 3},
                    },
                    {
                        "response": {
                            "output_tokens": [42, 43, 44],
                        },
                    },
                ]
                for event_name, payload in (
                    ("step", frames[0]),
                    ("step", frames[1]),
                    ("step", frames[2]),
                    ("response", frames[3]),
                ):
                    yield f"event: {event_name}\n".encode()
                    yield b"data: " + json.dumps(payload).encode() + b"\n"
                    yield b"\n"

        class FakeConnection:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def request(self, *_args, **_kwargs) -> None:
                pass

            def getresponse(self) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with patch.object(bench.http.client, "HTTPConnection", FakeConnection):
            with patch.object(
                bench.time, "perf_counter", side_effect=[10.0, 10.1, 11.0]
            ):
                run = bench.axengine_one_run(19091, [1, 2, 3, 4], 3)

        self.assertEqual(run["output_tokens"], 3.0)
        self.assertEqual(run["decode_s"], 1.0)
        self.assertEqual(run["decode_tok_s"], 3.0)

    def test_axengine_summary_exposes_kv_compression_blocker_row_fields(self) -> None:
        run = {
            "prefill_s": 0.3,
            "decode_s": 0.1,
            "ttft_ms": 300.0,
            "prefill_tok_s": 10.0,
            "decode_tok_s": 20.0,
            "output_tokens": 3.0,
            "kv_compression_telemetry": {
                "ax_mlx_kv_compression_decode_path": 1,
                "ax_mlx_kv_compression_fused_decode_fallback_reason": 1,
                "ax_mlx_kv_compression_fused_decode_blocked_attention_kind": 2,
                "ax_mlx_kv_compression_fused_decode_blocked_linear_attention": 2,
                "ax_mlx_kv_compression_fused_decode_blocked_missing_storage": 1,
            },
        }
        with patch.object(bench, "axengine_one_run", side_effect=[run, run]):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                1,
                0.0,
                model_metadata={},
                direct_mode=True,
                kv_compression="turboquant-fused-experimental",
            )

        self.assertEqual(
            row["kv_compression_claim_status"],
            "telemetry_only_full_precision_generation",
        )
        self.assertEqual(
            row["kv_compression_fused_decode_fallback_reason_label"],
            "shadow_only",
        )
        self.assertEqual(row["kv_compression_fused_decode_blocked_total"], 3)
        self.assertEqual(
            row["kv_compression_fused_decode_blocked_reasons"],
            ["attention_kind", "missing_storage"],
        )
        self.assertEqual(
            row["kv_compression_fused_decode_blocked_attention_kind_total"],
            2,
        )
        self.assertEqual(
            row["kv_compression_fused_decode_blocked_attention_kind_reasons"],
            ["linear_attention"],
        )

    def test_axengine_summary_can_label_linear_attention_pack_row(self) -> None:
        runs = [
            {
                "prefill_s": 0.2,
                "decode_s": 0.1,
                "ttft_ms": 200.0,
                "prefill_tok_s": 15.0,
                "decode_tok_s": 20.0,
                "output_tokens": 3.0,
            },
            {
                "prefill_s": 0.2,
                "decode_s": 0.1,
                "ttft_ms": 200.0,
                "prefill_tok_s": 15.0,
                "decode_tok_s": 20.0,
                "output_tokens": 3.0,
            },
        ]
        with patch.object(bench, "axengine_one_run", side_effect=runs):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                1,
                0.0,
                model_metadata={},
                direct_mode=True,
                engine_key_override=bench.AX_ENGINE_LINEAR_ATTENTION_PACK_KEY,
            )

        self.assertEqual(row["engine"], bench.AX_ENGINE_LINEAR_ATTENTION_PACK_KEY)
        self.assertEqual(row["ax_decode_policy"], "direct_no_ngram_acceleration")

    def test_axengine_summary_exposes_direct_cpp_linear_attention_route(self) -> None:
        run = {
            "prefill_s": 0.2,
            "decode_s": 0.1,
            "ttft_ms": 200.0,
            "prefill_tok_s": 15.0,
            "decode_tok_s": 20.0,
            "output_tokens": 3.0,
            "ax_mlx_telemetry": {
                "ax_mlx_direct_cpp_linear_attention_inputs_attempts": 4,
                "ax_mlx_direct_cpp_linear_attention_inputs_hits": 4,
                "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks": 0,
                "ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked": 0,
            },
        }
        with patch.object(bench, "axengine_one_run", side_effect=[run, run]):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                1,
                0.0,
                model_metadata={},
                direct_mode=True,
            )

        route = row["ax_mlx_direct_cpp_linear_attention_inputs"]
        self.assertEqual(
            route["schema_version"], "ax.mlx_direct_cpp_linear_attention_inputs.v1"
        )
        self.assertEqual(route["classification"], "all_hits")
        self.assertEqual(route["attempts"], 4)
        self.assertEqual(route["hits"], 4)

    def test_axengine_summary_exposes_direct_cpp_linear_attention_post_input_route(
        self,
    ) -> None:
        run = {
            "prefill_s": 0.2,
            "decode_s": 0.1,
            "ttft_ms": 200.0,
            "prefill_tok_s": 15.0,
            "decode_tok_s": 20.0,
            "output_tokens": 3.0,
            "ax_mlx_telemetry": {
                "ax_mlx_direct_cpp_linear_attention_post_input_attempts": 4,
                "ax_mlx_direct_cpp_linear_attention_post_input_hits": 4,
                "ax_mlx_direct_cpp_linear_attention_post_input_fallbacks": 0,
                "ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked": 0,
            },
        }
        with patch.object(bench, "axengine_one_run", side_effect=[run, run]):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                1,
                0.0,
                model_metadata={},
                direct_mode=True,
            )

        route = row["ax_mlx_direct_cpp_linear_attention_post_input"]
        self.assertEqual(
            route["schema_version"],
            "ax.mlx_direct_cpp_linear_attention_post_input.v1",
        )
        self.assertEqual(route["classification"], "all_hits")
        self.assertEqual(route["attempts"], 4)
        self.assertEqual(route["hits"], 4)

    def test_axengine_direct_summary_rejects_ngram_telemetry(self) -> None:
        contaminated = {
            "prefill_s": 0.2,
            "decode_s": 0.1,
            "ttft_ms": 200.0,
            "prefill_tok_s": 15.0,
            "decode_tok_s": 20.0,
            "output_tokens": 3.0,
            "ngram_acceleration_telemetry": {
                "ax_ngram_draft_attempts": 1,
                "ax_ngram_draft_tokens": 2,
                "ax_ngram_accepted_tokens": 1,
            },
            "ax_mlx_telemetry": {
                "ax_mlx_decode_steps": 2,
                "ax_mlx_ngram_decode_steps": 2,
            },
        }
        with patch.object(
            bench, "axengine_one_run", side_effect=[contaminated, contaminated]
        ):
            with self.assertRaisesRegex(RuntimeError, "direct AX benchmark row"):
                bench.bench_axengine(
                    19091,
                    [1, 2, 3],
                    3,
                    1,
                    0.0,
                    model_metadata={},
                    direct_mode=True,
                )

    def test_axengine_ngram_summary_allows_ngram_telemetry(self) -> None:
        ngram_run = {
            "prefill_s": 0.2,
            "decode_s": 0.1,
            "ttft_ms": 200.0,
            "prefill_tok_s": 15.0,
            "decode_tok_s": 20.0,
            "output_tokens": 3.0,
            "ngram_acceleration_telemetry": {
                "ax_ngram_draft_attempts": 1,
                "ax_ngram_draft_tokens": 2,
                "ax_ngram_accepted_tokens": 1,
            },
            "ax_mlx_telemetry": {
                "ax_mlx_decode_steps": 2,
                "ax_mlx_ngram_decode_steps": 2,
            },
        }
        with patch.object(
            bench, "axengine_one_run", side_effect=[ngram_run, ngram_run]
        ):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                1,
                0.0,
                model_metadata={},
                direct_mode=False,
            )

        self.assertEqual(row["ax_mlx_decode_route"]["classification"], "ngram")

    def test_axengine_pure_mtp_row_records_head_only_policy(self) -> None:
        mtp_run = {
            "prefill_s": 0.2,
            "decode_s": 0.1,
            "ttft_ms": 200.0,
            "prefill_tok_s": 15.0,
            "decode_tok_s": 20.0,
            "output_tokens": 3.0,
            "ngram_acceleration_telemetry": {
                "ax_mtp_draft_tokens": 6,
                "ax_mtp_accepted_tokens": 4,
                "ax_mtp_ngram_hit_steps": 0,
            },
            "ax_mlx_telemetry": {
                "ax_mlx_decode_steps": 2,
                "ax_mlx_bonus_tokens": 4,
            },
        }
        with patch.object(bench, "axengine_one_run", side_effect=[mtp_run, mtp_run]):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                1,
                0.0,
                model_metadata={},
                direct_mode=False,
                engine_key_override=bench.AX_ENGINE_PURE_MTP_KEY,
                mtp_disable_ngram_stacking=True,
            )

        self.assertEqual(row["engine"], "ax_engine_mlx_pure_mtp")
        self.assertEqual(row["ax_decode_policy"], "mtp_head_only_no_ngram_stacking")
        self.assertEqual(row["ax_decode_claim_status"], "mtp_head_only_effective")
        self.assertEqual(row["ax_decode_claim_mode"], "mtp_greedy_exact_candidate")
        self.assertEqual(row["ax_decode_effective_route"], "mtp_head_only_verify_loop")
        self.assertEqual(row["ax_mtp_draft_source"], "mtp_head_only")

    def test_axengine_gemma4_assistant_mtp_row_summarizes_route_metadata(self) -> None:
        run = {
            "prefill_s": 0.2,
            "decode_s": 0.1,
            "ttft_ms": 200.0,
            "prefill_tok_s": 15.0,
            "decode_tok_s": 20.0,
            "output_tokens": 3.0,
            "ngram_acceleration_telemetry": {
                "ax_mtp_draft_tokens": 2,
                "ax_mtp_accepted_tokens": 1,
            },
            "ax_mlx_gemma4_assistant_mtp": {
                "ax_mlx_gemma4_assistant_mtp_configured": 1,
                "ax_mlx_gemma4_assistant_mtp_validated": 1,
                "ax_mlx_gemma4_assistant_mtp_enabled": 1,
                "ax_mlx_gemma4_assistant_mtp_attach_failed": 0,
                "ax_mlx_gemma4_assistant_mtp_disable_reason": 0,
                "ax_mlx_gemma4_assistant_mtp_depth": 1,
                "ax_mlx_gemma4_assistant_mtp_draft_tokens": 2,
                "ax_mlx_gemma4_assistant_mtp_accepted_tokens": 1,
                "ax_mlx_gemma4_assistant_mtp_rejected_tokens": 1,
                "ax_mlx_gemma4_assistant_mtp_corrections": 1,
                "ax_mlx_gemma4_assistant_mtp_accept_rate_x1000": 500,
                "ax_mlx_gemma4_assistant_mtp_verify_forward_wall_us": 40,
                "ax_mlx_gemma4_assistant_mtp_verify_eval_wall_us": 10,
                "ax_mlx_gemma4_assistant_mtp_draft_forward_wall_us": 30,
            },
        }
        with patch.object(bench, "axengine_one_run", side_effect=[run, run]):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                1,
                0.0,
                model_metadata={},
                direct_mode=False,
                engine_key_override=bench.AX_ENGINE_GEMMA4_ASSISTANT_MTP_KEY,
                mtp_disable_ngram_stacking=True,
                gemma4_assistant_mtp=True,
            )

        self.assertEqual(row["engine"], bench.AX_ENGINE_GEMMA4_ASSISTANT_MTP_KEY)
        self.assertEqual(
            row["ax_decode_policy"], "gemma4_assistant_mtp_no_ngram_stacking"
        )
        self.assertEqual(row["ax_mtp_draft_source"], "gemma4_assistant_head_only")
        assistant = row["ax_mlx_gemma4_assistant_mtp"]
        self.assertEqual(assistant["ax_mlx_gemma4_assistant_mtp_enabled"], 1)
        self.assertEqual(assistant["ax_mlx_gemma4_assistant_mtp_draft_tokens"], 2)
        self.assertEqual(assistant["ax_mlx_gemma4_assistant_mtp_accepted_tokens"], 1)
        self.assertEqual(assistant["ax_mlx_gemma4_assistant_mtp_rejected_tokens"], 1)
        self.assertEqual(
            assistant["ax_mlx_gemma4_assistant_mtp_accept_rate_x1000"], 500
        )

    def test_axengine_gemma4_assistant_mtp_ngram_row_has_distinct_identity(
        self,
    ) -> None:
        run = {
            "prefill_s": 0.2,
            "decode_s": 0.1,
            "ttft_ms": 200.0,
            "prefill_tok_s": 15.0,
            "decode_tok_s": 20.0,
            "output_tokens": 3.0,
            "ngram_acceleration_telemetry": {
                "ax_ngram_draft_attempts": 2,
                "ax_ngram_draft_tokens": 2,
                "ax_ngram_accepted_tokens": 1,
                "ax_mtp_draft_tokens": 2,
                "ax_mtp_accepted_tokens": 1,
                "ax_mtp_ngram_hit_steps": 1,
            },
            "ax_mlx_gemma4_assistant_mtp": {
                "ax_mlx_gemma4_assistant_mtp_configured": 1,
                "ax_mlx_gemma4_assistant_mtp_validated": 1,
                "ax_mlx_gemma4_assistant_mtp_enabled": 1,
                "ax_mlx_gemma4_assistant_mtp_draft_tokens": 2,
                "ax_mlx_gemma4_assistant_mtp_accepted_tokens": 1,
            },
            "ax_mlx_telemetry": {
                "ax_mlx_ngram_decode_steps": 1,
                "ax_mlx_ngram_decode_wall_us": 10,
            },
        }
        with patch.object(bench, "axengine_one_run", side_effect=[run, run]):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                1,
                0.0,
                model_metadata={},
                direct_mode=False,
                engine_key_override=bench.AX_ENGINE_GEMMA4_ASSISTANT_MTP_NGRAM_KEY,
                gemma4_assistant_mtp=True,
            )

        self.assertEqual(row["engine"], bench.AX_ENGINE_GEMMA4_ASSISTANT_MTP_NGRAM_KEY)
        self.assertEqual(row["ax_decode_policy"], "gemma4_assistant_mtp_ngram_stacking")
        self.assertEqual(
            row["ax_mtp_draft_source"], "gemma4_assistant_head_or_ngram_stacked"
        )

    def test_ax_decode_claim_status_reports_throughput_policy_only(self) -> None:
        ngram_telemetry = {
            "ax_ngram_draft_attempts": 10,
            "ax_ngram_accepted_tokens": 8,
        }
        self.assertEqual(
            bench.ax_decode_claim_status(False, ngram_telemetry),
            "ngram_acceleration_effective_throughput",
        )
        self.assertEqual(
            bench.ax_decode_claim_status(True, {}),
            "direct_same_policy_baseline",
        )
        self.assertEqual(
            bench.ax_decode_claim_status(
                False,
                {"ax_mtp_draft_tokens": 4, "ax_mtp_ngram_hit_steps": 0},
                mtp_disable_ngram_stacking=True,
            ),
            "mtp_head_only_effective",
        )
        self.assertEqual(
            bench.ax_decode_claim_status(
                False,
                {"ax_mtp_draft_tokens": 4, "ax_mtp_ngram_hit_steps": 1},
                mtp_disable_ngram_stacking=True,
            ),
            "mtp_head_only_contract_violation",
        )
        self.assertEqual(
            bench.ax_decode_claim_mode(
                False,
                sampler={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                mtp_disable_ngram_stacking=True,
            ),
            "mtp_sampling_distribution_corrected",
        )

    def test_axengine_row_captures_output_token_ids_only_when_requested(self) -> None:
        ngram_run = {
            "prefill_s": 0.2,
            "decode_s": 0.1,
            "ttft_ms": 200.0,
            "prefill_tok_s": 15.0,
            "decode_tok_s": 20.0,
            "output_tokens": 3.0,
            "ngram_acceleration_telemetry": {
                "ax_ngram_draft_attempts": 4,
                "ax_ngram_draft_tokens": 8,
                "ax_ngram_accepted_tokens": 6,
            },
            "ax_mlx_telemetry": {
                "ax_mlx_decode_steps": 4,
                "ax_mlx_ngram_decode_steps": 4,
            },
        }
        with patch.object(
            bench, "axengine_one_run", side_effect=[ngram_run, ngram_run]
        ):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                1,
                0.0,
                model_metadata={},
                direct_mode=False,
            )
        self.assertFalse(any(key.startswith("decode_de") for key in row))
        self.assertNotIn("output_token_ids", row["trials"][0])

    def test_axengine_row_preserves_output_token_ids_when_requested(self) -> None:
        ngram_run = {
            "prefill_s": 0.2,
            "decode_s": 0.1,
            "ttft_ms": 200.0,
            "prefill_tok_s": 15.0,
            "decode_tok_s": 20.0,
            "output_tokens": 3.0,
            "output_token_ids": [101, 102, 103],
            "ngram_acceleration_telemetry": {
                "ax_ngram_draft_attempts": 1,
                "ax_ngram_draft_tokens": 2,
                "ax_ngram_accepted_tokens": 1,
            },
            "ax_mlx_telemetry": {
                "ax_mlx_decode_steps": 2,
                "ax_mlx_ngram_decode_steps": 2,
            },
        }
        with patch.object(
            bench, "axengine_one_run", side_effect=[ngram_run, ngram_run]
        ):
            row = bench.bench_axengine(
                19091,
                [1, 2, 3],
                3,
                1,
                0.0,
                model_metadata={},
                direct_mode=False,
                capture_output_token_ids=True,
            )
        self.assertEqual(
            row["ax_decode_claim_status"],
            "ngram_acceleration_effective_throughput",
        )
        self.assertEqual(row["trials"][0]["output_token_ids"], [101, 102, 103])

    def test_load_real_prompt_suite_parses_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "coding.jsonl"
            suite.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "case_a",
                                "category": "coding",
                                "prompt": "Write a Python lru cache",
                                "max_tokens": 64,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "case_b",
                                "category": "coding",
                                "prompt": "Refactor this Rust function",
                            }
                        ),
                    ]
                )
                + "\n"
            )
            cases = bench.load_real_prompt_suite(suite)
        self.assertEqual([c.id for c in cases], ["case_a", "case_b"])
        self.assertEqual(cases[0].max_tokens, 64)
        # Default max_tokens when omitted should land on the harness default.
        self.assertEqual(cases[1].max_tokens, 128)

    def test_load_real_prompt_suite_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "dup.jsonl"
            suite.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "same", "category": "x", "prompt": "a"}),
                        json.dumps({"id": "same", "category": "x", "prompt": "b"}),
                    ]
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate prompt id"):
                bench.load_real_prompt_suite(suite)

    def test_load_real_prompt_suite_rejects_missing_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "bad.jsonl"
            suite.write_text(
                json.dumps({"id": "no_prompt", "category": "coding"}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "missing required key 'prompt'"):
                bench.load_real_prompt_suite(suite)

    def test_load_real_prompt_suite_skips_blank_and_comment_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "with_comments.jsonl"
            suite.write_text(
                "\n".join(
                    [
                        "# this is a comment",
                        "",
                        json.dumps({"id": "only_one", "category": "x", "prompt": "p"}),
                        "   ",
                    ]
                )
                + "\n"
            )
            cases = bench.load_real_prompt_suite(suite)
        self.assertEqual([c.id for c in cases], ["only_one"])

    def test_load_real_prompt_suite_rejects_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "empty.jsonl"
            suite.write_text("# only comments\n\n")
            with self.assertRaisesRegex(ValueError, "no prompt cases"):
                bench.load_real_prompt_suite(suite)

    def test_load_real_prompt_suite_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "syntax_err.jsonl"
            suite.write_text("{not json}\n")
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                bench.load_real_prompt_suite(suite)

    def test_write_real_prompt_tokens_emits_artifact_with_source_real(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "prompts"
            case = bench.RealPromptCase(
                id="case_a",
                category="coding",
                prompt="Hello world",
                max_tokens=32,
            )
            doc = bench.write_real_prompt_tokens(
                artifact_root,
                suite_id="coding",
                case=case,
                tokens=[10, 20, 30, 40, 50],
                generation_tokens=128,
                vocab_size=32_000,
            )
            self.assertEqual(doc["prompt_source"], "real")
            self.assertEqual(doc["prompt_suite_id"], "coding")
            self.assertEqual(doc["prompt_case_id"], "case_a")
            self.assertEqual(doc["prompt_category"], "coding")
            self.assertEqual(doc["prompt_tokens"], 5)
            self.assertEqual(doc["case_max_tokens"], 32)
            self.assertEqual(doc["token_ids"], [10, 20, 30, 40, 50])
            # The persisted artifact must round-trip the same fields,
            # including the prompt_text_sha256 so downstream tools can
            # match runs by the text of the prompt (not its tokens).
            path = Path(doc["token_ids_path"])
            self.assertTrue(path.is_file())
            payload = json.loads(path.read_text())
            self.assertEqual(
                payload["schema_version"], bench.REAL_PROMPT_SCHEMA_VERSION
            )
            self.assertEqual(payload["prompt_source"], "real")
            self.assertEqual(payload["prompt_text_sha256"], doc["prompt_text_sha256"])

    def test_build_real_prompts_attaches_suite_metadata(self) -> None:
        # Mock the tokenizer + vocab lookup so the test does not need an
        # actual model directory. The bench wires these through
        # `load_model_tokenizer` and `model_vocab_size`.
        class StubTokenizer:
            def encode(self, text: str) -> list[int]:
                return [ord(ch) for ch in text]

            def apply_chat_template(
                self,
                messages: list[dict[str, str]],
                *,
                tokenize: bool,
                add_generation_prompt: bool,
            ) -> list[int]:
                # Pretend the chat template prepends one BOS-equivalent token.
                content = messages[0]["content"]
                return [1] + [ord(ch) for ch in content]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(bench, "load_model_tokenizer", return_value=StubTokenizer()),
            patch.object(bench, "model_vocab_size", return_value=128),
        ):
            suite = Path(tmp) / "coding.jsonl"
            suite.write_text(
                json.dumps({"id": "one", "category": "coding", "prompt": "AB"})
                + "\n"
                + json.dumps({"id": "two", "category": "coding", "prompt": "CDE"})
                + "\n"
            )
            artifact_root = Path(tmp) / "artifacts"
            # Default: chat template applied. Token counts include the
            # synthetic BOS, so AB -> 3 and CDE -> 4.
            prompts = bench.build_real_prompts(
                suite,
                generation_tokens=128,
                model_dir=Path(tmp),
                artifact_root=artifact_root,
            )
            self.assertEqual(len(prompts), 2)
            self.assertEqual(
                [(p["prompt_case_id"], p["prompt_tokens"]) for p in prompts],
                [("one", 3), ("two", 4)],
            )
            for prompt in prompts:
                self.assertEqual(prompt["prompt_source"], "real")
                self.assertEqual(prompt["prompt_suite_id"], "coding")
                self.assertTrue(prompt["chat_template_applied"])
                self.assertTrue(Path(prompt["token_ids_path"]).is_file())

    def test_build_real_prompts_raw_encoding_skips_chat_template(self) -> None:
        # When the caller opts out of the chat template (base / non-IT
        # models), the harness must encode raw text and stamp the
        # artifact with chat_template_applied=false.
        class StubTokenizer:
            def encode(self, text: str) -> list[int]:
                return [ord(ch) for ch in text]

            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("chat template must not be applied")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(bench, "load_model_tokenizer", return_value=StubTokenizer()),
            patch.object(bench, "model_vocab_size", return_value=128),
        ):
            suite = Path(tmp) / "coding.jsonl"
            suite.write_text(
                json.dumps({"id": "one", "category": "coding", "prompt": "AB"}) + "\n"
            )
            prompts = bench.build_real_prompts(
                suite,
                generation_tokens=128,
                model_dir=Path(tmp),
                artifact_root=Path(tmp) / "artifacts",
                chat_template=False,
            )
            self.assertEqual(prompts[0]["prompt_tokens"], 2)
            self.assertFalse(prompts[0]["chat_template_applied"])

    def test_build_reference_prompts_tags_random_source(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(bench, "model_vocab_size", return_value=64),
            patch.object(
                bench, "mlx_lm_reference_prompt_tokens", return_value=[1, 2, 3, 4]
            ),
        ):
            prompts = bench.build_reference_prompts(
                [4],
                generation_tokens=8,
                model_dir=Path(tmp),
                artifact_root=Path(tmp) / "artifacts",
            )
        self.assertEqual(prompts[0]["prompt_source"], "random")

    def test_parse_llama_cpp_bench_json_combines_pp_and_tg_rows(self) -> None:
        parsed = bench.parse_llama_cpp_bench_json(
            json.dumps(
                [
                    {
                        "build_commit": "abc123",
                        "build_number": 6000,
                        "cpu_info": "Apple M5",
                        "gpu_info": "Apple M5 Max",
                        "backends": "Metal",
                        "model_filename": "/models/qwen.gguf",
                        "model_type": "qwen",
                        "model_size": 123,
                        "model_n_params": 456,
                        "n_batch": 2048,
                        "n_ubatch": 512,
                        "type_k": "f16",
                        "type_v": "f16",
                        "n_gpu_layers": 99,
                        "flash_attn": False,
                        "devices": "Metal",
                        "n_prompt": 512,
                        "n_gen": 0,
                        "avg_ts": 1100.0,
                        "stddev_ts": 10.0,
                        "samples_ns": [100, 200],
                        "samples_ts": [1000.0, 1200.0],
                    },
                    {
                        "backends": "Metal",
                        "n_prompt": 0,
                        "n_gen": 128,
                        "avg_ts": 55.0,
                        "stddev_ts": 5.0,
                        "samples_ns": [300, 400],
                        "samples_ts": [50.0, 60.0],
                    },
                ]
            ),
            prompt_tokens=512,
            generation_tokens=128,
        )

        self.assertEqual(parsed["prefill_tok_s"]["median"], 1100.0)
        self.assertEqual(parsed["decode_tok_s"]["median"], 55.0)
        self.assertEqual(parsed["llama_cpp"]["backends"], "Metal")
        self.assertEqual(parsed["llama_cpp"]["build_commit"], "abc123")
        self.assertEqual(parsed["prefill_trials"][0]["prefill_tok_s"], 1000.0)
        self.assertEqual(parsed["prefill_trials"][1]["prefill_tok_s"], 1200.0)
        self.assertEqual(parsed["decode_trials"][0]["decode_tok_s"], 50.0)
        self.assertEqual(parsed["decode_trials"][1]["decode_tok_s"], 60.0)
        self.assertIn("independent test", parsed["trials_pairing_note"])
        self.assertEqual(len(parsed["raw_rows"]), 2)
        self.assertEqual(parsed["raw_rows"][0]["n_prompt"], 512)
        self.assertEqual(parsed["raw_rows"][1]["n_gen"], 128)

    def test_run_llama_cpp_metal_benchmark_rounds_float_cooldown_to_int(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "llama-bench"
            gguf = root / "model.gguf"
            binary.write_text("#!/bin/sh\n")
            gguf.write_text("gguf")
            prompt = bench.write_prompt_tokens(
                root,
                prompt_tokens=4,
                generation_tokens=2,
                vocab_size=100,
                tokens=[1, 2, 3, 4],
            )
            prompt["token_ids"] = [1, 2, 3, 4]
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "backends": "Metal",
                            "n_prompt": 4,
                            "n_gen": 0,
                            "avg_ts": 1.0,
                            "samples_ts": [1.0],
                        },
                        {
                            "backends": "Metal",
                            "n_prompt": 0,
                            "n_gen": 2,
                            "avg_ts": 1.0,
                            "samples_ts": [1.0],
                        },
                    ]
                ),
                stderr="",
            )
            with (
                patch.object(bench.subprocess, "run", return_value=completed) as run,
                patch.object(
                    bench, "collect_llama_cpp_device_evidence", return_value=None
                ),
            ):
                bench.run_llama_cpp_metal_benchmark(
                    binary,
                    gguf,
                    prompt_tokens=4,
                    generation_tokens=2,
                    repetitions=1,
                    cooldown=2.6,
                    n_gpu_layers=99,
                    prompt_doc=prompt,
                    extra_args=None,
                )
        command_args = run.call_args.args[0]
        delay_value = command_args[command_args.index("--delay") + 1]
        self.assertEqual(delay_value, "3")

    def test_run_llama_cpp_metal_benchmark_sets_ubatch_to_match_prompt(self) -> None:
        # Regression guard: llama-bench defaults `-ub` to 512, which silently
        # caps p=2048 prefill throughput at ~7k tok/s. The bench harness must
        # set `-ub` to match the prompt length (capped at 2048) so the
        # comparison against mlx-lm is apples-to-apples instead of
        # llama.cpp-fighting-with-one-hand.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "llama-bench"
            gguf = root / "model.gguf"
            binary.write_text("#!/bin/sh\n")
            gguf.write_text("gguf")
            prompt = bench.write_prompt_tokens(
                root,
                prompt_tokens=2048,
                generation_tokens=128,
                vocab_size=100,
                tokens=list(range(2048)),
            )
            prompt["token_ids"] = list(range(2048))
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "backends": "Metal",
                            "n_prompt": 2048,
                            "n_gen": 0,
                            "avg_ts": 1.0,
                            "samples_ts": [1.0],
                        },
                        {
                            "backends": "Metal",
                            "n_prompt": 0,
                            "n_gen": 128,
                            "avg_ts": 1.0,
                            "samples_ts": [1.0],
                        },
                    ]
                ),
                stderr="",
            )
            with (
                patch.object(bench.subprocess, "run", return_value=completed) as run,
                patch.object(
                    bench, "collect_llama_cpp_device_evidence", return_value=None
                ),
            ):
                bench.run_llama_cpp_metal_benchmark(
                    binary,
                    gguf,
                    prompt_tokens=2048,
                    generation_tokens=128,
                    repetitions=1,
                    cooldown=1,
                    n_gpu_layers=99,
                    prompt_doc=prompt,
                    extra_args=None,
                )
        command_args = run.call_args.args[0]
        ub_value = command_args[command_args.index("-ub") + 1]
        b_value = command_args[command_args.index("-b") + 1]
        self.assertEqual(ub_value, "2048")
        self.assertEqual(b_value, "2048")

    def test_parse_llama_cpp_bench_json_requires_metal_backend(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Metal/MTL backend"):
            bench.parse_llama_cpp_bench_json(
                json.dumps(
                    [
                        {"backends": "CPU", "n_prompt": 4, "n_gen": 0, "avg_ts": 10.0},
                        {"backends": "CPU", "n_prompt": 0, "n_gen": 2, "avg_ts": 1.0},
                    ]
                ),
                prompt_tokens=4,
                generation_tokens=2,
            )

    def test_parse_llama_cpp_bench_json_accepts_mtl_token(self) -> None:
        # Real llama-bench emits comma-separated backend tokens like "BLAS,MTL".
        # The guard must accept the MTL token (Metal backend label).
        parsed = bench.parse_llama_cpp_bench_json(
            json.dumps(
                [
                    {
                        "backends": "BLAS,MTL",
                        "n_prompt": 4,
                        "n_gen": 0,
                        "avg_ts": 10.0,
                        "samples_ts": [10.0],
                    },
                    {
                        "backends": "BLAS,MTL",
                        "n_prompt": 0,
                        "n_gen": 2,
                        "avg_ts": 1.0,
                        "samples_ts": [1.0],
                    },
                ]
            ),
            prompt_tokens=4,
            generation_tokens=2,
        )
        self.assertEqual(parsed["llama_cpp"]["backends"], "BLAS,MTL")

    def test_parse_llama_cpp_decode_depth_json_requires_matching_depth(self) -> None:
        parsed = bench.parse_llama_cpp_decode_depth_json(
            json.dumps(
                [
                    {
                        "build_commit": "abc123",
                        "backends": "Metal",
                        "n_prompt": 0,
                        "n_gen": 128,
                        "n_depth": 8192,
                        "avg_ts": 42.0,
                        "samples_ts": [40.0, 44.0],
                    },
                    {
                        "backends": "Metal",
                        "n_prompt": 0,
                        "n_gen": 128,
                        "n_depth": 4096,
                        "avg_ts": 99.0,
                        "samples_ts": [99.0],
                    },
                ]
            ),
            context_depth_tokens=8192,
            generation_tokens=128,
        )

        self.assertEqual(parsed["decode_at_depth_tok_s"]["median"], 42.0)
        self.assertEqual(
            parsed["decode_at_depth_trials"][0]["decode_at_depth_tok_s"], 40.0
        )
        self.assertEqual(parsed["llama_cpp_depth"]["n_depth"], 8192)

    def test_attach_llama_cpp_decode_at_depth_benchmark_records_depth_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "llama-bench"
            gguf = root / "model.gguf"
            binary.write_text("#!/bin/sh\n")
            gguf.write_text("gguf")
            row = {"engine": "llama_cpp_metal"}
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "build_commit": "abc123",
                            "backends": "Metal",
                            "n_prompt": 0,
                            "n_gen": 2,
                            "n_depth": 4096,
                            "avg_ts": 20.0,
                            "samples_ts": [18.0, 22.0],
                        },
                    ]
                ),
                stderr="",
            )
            with patch.object(bench.subprocess, "run", return_value=completed) as run:
                bench.attach_llama_cpp_decode_at_depth_benchmark(
                    row,
                    binary,
                    gguf,
                    context_depth_tokens=4096,
                    generation_tokens=2,
                    repetitions=2,
                    cooldown=0.0,
                    n_gpu_layers=99,
                    extra_args="-fa 1",
                )

        command_args = run.call_args.args[0]
        self.assertEqual(command_args[command_args.index("-p") + 1], "0")
        self.assertEqual(command_args[command_args.index("-d") + 1], "4096")
        self.assertIn("-fa", command_args)
        self.assertEqual(row["context_depth_tokens"], 4096)
        self.assertEqual(row["decode_at_depth_contract"], "llama_bench_n_depth")
        self.assertEqual(row["decode_at_depth_tok_s"]["median"], 20.0)
        self.assertIn("not prompt-hash parity", row["decode_at_depth_claim_boundary"])

    def test_llama_cpp_metal_row_records_external_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "llama-bench"
            gguf = root / "model.gguf"
            binary.write_text("#!/bin/sh\n")
            gguf.write_text("gguf")
            prompt = bench.write_prompt_tokens(
                root,
                prompt_tokens=4,
                generation_tokens=2,
                vocab_size=100,
                tokens=[1, 2, 3, 4],
            )
            prompt["token_ids"] = [1, 2, 3, 4]
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "build_commit": "abc123",
                            "backends": "Metal",
                            "gpu_info": "Apple GPU",
                            "n_prompt": 4,
                            "n_gen": 0,
                            "n_gpu_layers": 99,
                            "avg_ts": 100.0,
                            "samples_ts": [90.0, 110.0],
                        },
                        {
                            "backends": "Metal",
                            "n_prompt": 0,
                            "n_gen": 2,
                            "avg_ts": 20.0,
                            "samples_ts": [18.0, 22.0],
                        },
                    ]
                ),
                stderr="",
            )
            with (
                patch.object(bench.subprocess, "run", return_value=completed) as run,
                patch.object(
                    bench,
                    "collect_llama_cpp_device_evidence",
                    return_value="Metal device",
                ),
            ):
                row = bench.run_llama_cpp_metal_benchmark(
                    binary,
                    gguf,
                    prompt_tokens=4,
                    generation_tokens=2,
                    repetitions=2,
                    cooldown=0.4,
                    n_gpu_layers=99,
                    prompt_doc=prompt,
                    extra_args="-fa 1",
                )

        command_args = run.call_args.args[0]
        self.assertIn("-o", command_args)
        self.assertIn("json", command_args)
        self.assertIn("-fa", command_args)
        self.assertIn("--delay", command_args)
        delay_value = command_args[command_args.index("--delay") + 1]
        self.assertEqual(delay_value, "0")
        self.assertEqual(row["engine"], "llama_cpp_metal")
        self.assertEqual(
            row["runtime_identity"]["route_identity"], "external_llama_cpp_metal"
        )
        self.assertEqual(
            row["prompt_contract"], "shape_compatible_llama_bench_internal_tokens"
        )
        self.assertIn("not prompt-hash parity", row["claim_boundary"])
        self.assertEqual(row["ttft_source"], "derived_from_llama_cpp_pp_tok_s")
        self.assertEqual(row["llama_cpp_device_evidence"], "Metal device")
        self.assertEqual(row["prefill_trials"][0]["prefill_tok_s"], 90.0)
        self.assertEqual(row["decode_trials"][1]["decode_tok_s"], 22.0)
        self.assertIn("ttft_ms", row["prefill_trials"][0])
        self.assertIn("independent test", row["trials_pairing_note"])
        self.assertNotIn("trials", row)

    def test_collect_model_metadata_detects_linear_attention_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5", "vocab_size": 10})
            )
            (root / "model-manifest.json").write_text(
                json.dumps(
                    {
                        "model_family": "qwen3_5",
                        "linear_attention": {
                            "full_attention_interval": None,
                            "num_value_heads": 4,
                            "num_key_heads": 1,
                            "key_head_dim": 128,
                            "value_head_dim": 128,
                            "conv_kernel_dim": 4,
                        },
                        "tensors": [
                            {
                                "role": "linear_attention_in_proj_qkv",
                                "layer_index": 0,
                            },
                            {
                                "role": "linear_attention_in_proj_z",
                                "layer_index": 0,
                            },
                            {
                                "role": "linear_attention_in_proj_a",
                                "layer_index": 0,
                            },
                            {
                                "role": "linear_attention_in_proj_b",
                                "layer_index": 0,
                            },
                        ],
                    }
                )
            )

            metadata = bench.collect_model_metadata(root)

        self.assertTrue(metadata["linear_attention_enabled"])
        self.assertEqual(
            bench.ax_decode_policy(metadata, direct_mode=False),
            "ngram_acceleration_linear_attention_branch_recompute",
        )
        self.assertEqual(
            bench.ax_decode_policy(metadata, direct_mode=True),
            "direct_no_ngram_acceleration",
        )
        self.assertEqual(
            bench.ax_decode_policy(
                metadata,
                direct_mode=False,
                mtp_disable_ngram_stacking=True,
            ),
            "mtp_head_only_no_ngram_stacking",
        )
        self.assertEqual(
            metadata["linear_attention_projection_layout"]["layout"],
            "split_qkv_z_a_b",
        )
        self.assertTrue(
            metadata["linear_attention_projection_layout"]["offline_pack_candidate"]
        )

    def test_linear_attention_projection_layout_detects_packed_manifest(self) -> None:
        layout = bench.linear_attention_projection_layout(
            {
                "tensors": [
                    {"role": "linear_attention_in_proj_qkvz", "layer_index": 0},
                    {"role": "linear_attention_in_proj_ba", "layer_index": 0},
                    {"role": "linear_attention_in_proj_qkvz", "layer_index": 1},
                    {"role": "linear_attention_in_proj_ba", "layer_index": 1},
                ]
            }
        )

        self.assertEqual(layout["layout"], "packed_qkvz_ba")
        self.assertEqual(layout["linear_layers"], 2)
        self.assertEqual(layout["packed_layers"], 2)
        self.assertFalse(layout["offline_pack_candidate"])

    def test_gateddelta_prefill_profile_defaults_to_long_prompt_matrix(self) -> None:
        self.assertEqual(
            bench.normalize_gateddelta_prefill_profile_prompt_lengths(
                bench.DEFAULT_PROMPT_TOKENS
            ),
            [512, 2048, 8192, 32768],
        )
        self.assertEqual(
            bench.normalize_gateddelta_prefill_profile_prompt_lengths(
                "512,2048,8192,32768"
            ),
            [512, 2048, 8192, 32768],
        )
        with self.assertRaisesRegex(ValueError, "requires --prompt-tokens"):
            bench.normalize_gateddelta_prefill_profile_prompt_lengths("512,8192")

    def test_gateddelta_prefill_profile_requires_linear_attention_metadata(
        self,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "linear-attention MLX manifest"):
            bench.build_gateddelta_prefill_profile_contract(
                {"linear_attention_enabled": False},
                [512, 2048, 8192, 32768],
            )

        contract = bench.build_gateddelta_prefill_profile_contract(
            {
                "linear_attention_enabled": True,
                "model_family": "qwen3_next",
                "model_preflight_schema_version": "ax.gateddelta_prefill_model_preflight.v1",
                "model_type": "qwen3_5",
                "linear_attention": {
                    "num_value_heads": 4,
                    "num_key_heads": 4,
                    "key_head_dim": 64,
                    "value_head_dim": 128,
                    "conv_kernel_dim": 4,
                },
            },
            [512, 2048, 8192, 32768],
        )

        self.assertEqual(
            contract["schema_version"],
            "ax.gateddelta_prefill_profile.v1",
        )
        self.assertEqual(contract["model_family"], "qwen3_next")
        self.assertEqual(contract["direct_ax_row_required"], True)
        self.assertEqual(contract["ngram_policy_allowed"], False)
        self.assertEqual(contract["required_prompt_tokens"], [512, 2048, 8192, 32768])
        self.assertEqual(
            contract["runtime_profile_env"], "AX_MLX_LINEAR_ATTENTION_PROFILE=1"
        )
        self.assertEqual(
            contract["model_preflight"]["schema_version"],
            "ax.gateddelta_prefill_model_preflight.v1",
        )
        self.assertEqual(contract["model_preflight"]["status"], "passed")
        self.assertEqual(
            contract["model_preflight"]["linear_attention"]["key_head_dim"],
            64,
        )
        self.assertIn(
            "ax_mlx_linear_attention_profile_recurrent_wall_us",
            contract["primary_metrics"],
        )

    def test_gateddelta_prefill_profile_model_preflight_requires_qwen_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            valid_dir = write_gateddelta_model(Path(tmp), model_family="qwen3_next")

            metadata = bench.validate_gateddelta_prefill_profile_model(valid_dir)

        self.assertEqual(metadata["model_family"], "qwen3_next")
        self.assertEqual(metadata["linear_attention"]["key_head_dim"], 64)

        with tempfile.TemporaryDirectory() as tmp:
            invalid_family_dir = write_gateddelta_model(
                Path(tmp), model_family="gemma4"
            )

            with self.assertRaisesRegex(RuntimeError, "model_family must be"):
                bench.validate_gateddelta_prefill_profile_model(invalid_family_dir)

        with tempfile.TemporaryDirectory() as tmp:
            invalid_dim_dir = write_gateddelta_model(Path(tmp), key_head_dim=48)

            with self.assertRaisesRegex(RuntimeError, "divisible by 32"):
                bench.validate_gateddelta_prefill_profile_model(invalid_dim_dir)

    def test_gateddelta_prefill_profile_cli_reports_model_preflight_before_server_check(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = write_gateddelta_model(Path(tmp), model_family="gemma4")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--model-dir",
                    str(model_dir),
                    "--gateddelta-prefill-profile",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("model_family must be", result.stderr)
        self.assertNotIn("ax-engine-server not found", result.stderr)

    def test_validate_gateddelta_prefill_profile_output_runs_checker(self) -> None:
        generation_tokens = 128
        results = []
        for prompt_tokens in [512, 2048, 8192, 32768]:
            prompt_hash = f"{prompt_tokens:064x}"[-64:]
            results.append(
                {
                    "engine": "mlx_lm",
                    "method": "mlx_lm.benchmark",
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": generation_tokens,
                    "prompt_token_ids_sha256": prompt_hash,
                    "prefill_tok_s": {"median": 2000.0},
                    "decode_tok_s": {"median": 100.0},
                    "baseline": {"role": "primary_reference"},
                }
            )
            results.append(
                {
                    "engine": "ax_engine_mlx",
                    "method": "server_sse_runner_time_us",
                    "ax_decode_policy": "direct_no_ngram_acceleration",
                    "ax_decode_claim_status": "direct_same_policy_baseline",
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": generation_tokens,
                    "prompt_token_ids_sha256": prompt_hash,
                    "prefill_tok_s": {"median": 1800.0},
                    "decode_tok_s": {"median": 90.0},
                    "ax_mlx_telemetry": {"ax_mlx_prefill_wall_us": prompt_tokens * 10},
                    "ax_mlx_linear_attention_profile": {
                        "ax_mlx_linear_attention_profile_enabled": 1,
                        "ax_mlx_linear_attention_profile_layers": 28,
                        "ax_mlx_linear_attention_profile_tokens": prompt_tokens * 28,
                        "ax_mlx_linear_attention_profile_projection_wall_us": 100,
                        "ax_mlx_linear_attention_profile_conv_wall_us": 80,
                        "ax_mlx_linear_attention_profile_qk_norm_wall_us": 30,
                        "ax_mlx_linear_attention_profile_recurrent_wall_us": 90,
                        "ax_mlx_linear_attention_profile_output_wall_us": 20,
                    },
                    "baseline": {
                        "engine": "mlx_lm",
                        "prefill_ratio_to_mlx_lm": 0.9,
                    },
                }
            )
        artifact = {
            "schema_version": bench.MLX_INFERENCE_STACK_SCHEMA_VERSION,
            "model_config": {"linear_attention_enabled": True},
            "prompt_tokens": [512, 2048, 8192, 32768],
            "generation_tokens": generation_tokens,
            "ax_linear_attention_profile": True,
            "gateddelta_prefill_profile": {
                "schema_version": "ax.gateddelta_prefill_profile.v1",
                "direct_ax_row_required": True,
                "ngram_policy_allowed": False,
                "kv_compression_allowed": False,
                "prompt_tokens": [512, 2048, 8192, 32768],
                "required_prompt_tokens": [512, 2048, 8192, 32768],
                "runtime_profile_env": "AX_MLX_LINEAR_ATTENTION_PROFILE=1",
                "model_preflight": {
                    "schema_version": "ax.gateddelta_prefill_model_preflight.v1",
                    "status": "passed",
                    "checker": "scripts/check_gateddelta_prefill_model.py",
                    "model_family": "qwen3_next",
                    "model_type": "qwen3_5",
                    "linear_attention": {
                        "num_value_heads": 4,
                        "num_key_heads": 4,
                        "key_head_dim": 64,
                        "value_head_dim": 128,
                        "conv_kernel_dim": 4,
                    },
                },
            },
            "results": results,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gateddelta-profile.json"
            path.write_text(json.dumps(artifact))

            checked = bench.validate_gateddelta_prefill_profile_output(path)

        self.assertEqual(len(checked), 4)

    def test_render_gateddelta_prefill_profile_output_writes_markdown(self) -> None:
        generation_tokens = 128
        results = []
        for prompt_tokens in [512, 2048, 8192, 32768]:
            prompt_hash = f"{prompt_tokens:064x}"[-64:]
            results.append(
                {
                    "engine": "mlx_lm",
                    "method": "mlx_lm.benchmark",
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": generation_tokens,
                    "prompt_token_ids_sha256": prompt_hash,
                    "prefill_tok_s": {"median": 2000.0},
                    "decode_tok_s": {"median": 100.0},
                    "baseline": {"role": "primary_reference"},
                }
            )
            results.append(
                {
                    "engine": "ax_engine_mlx",
                    "method": "server_sse_runner_time_us",
                    "ax_decode_policy": "direct_no_ngram_acceleration",
                    "ax_decode_claim_status": "direct_same_policy_baseline",
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": generation_tokens,
                    "prompt_token_ids_sha256": prompt_hash,
                    "prefill_tok_s": {"median": 1800.0},
                    "decode_tok_s": {"median": 90.0},
                    "ax_mlx_telemetry": {"ax_mlx_prefill_wall_us": prompt_tokens * 10},
                    "ax_mlx_linear_attention_profile": {
                        "ax_mlx_linear_attention_profile_enabled": 1,
                        "ax_mlx_linear_attention_profile_layers": 28,
                        "ax_mlx_linear_attention_profile_tokens": prompt_tokens * 28,
                        "ax_mlx_linear_attention_profile_projection_wall_us": 100,
                        "ax_mlx_linear_attention_profile_conv_wall_us": 80,
                        "ax_mlx_linear_attention_profile_qk_norm_wall_us": 30,
                        "ax_mlx_linear_attention_profile_recurrent_wall_us": 900,
                        "ax_mlx_linear_attention_profile_output_wall_us": 20,
                    },
                    "baseline": {
                        "engine": "mlx_lm",
                        "prefill_ratio_to_mlx_lm": 0.9,
                    },
                }
            )
        artifact = {
            "schema_version": bench.MLX_INFERENCE_STACK_SCHEMA_VERSION,
            "model_config": {"linear_attention_enabled": True},
            "prompt_tokens": [512, 2048, 8192, 32768],
            "generation_tokens": generation_tokens,
            "ax_linear_attention_profile": True,
            "gateddelta_prefill_profile": {
                "schema_version": "ax.gateddelta_prefill_profile.v1",
                "direct_ax_row_required": True,
                "ngram_policy_allowed": False,
                "kv_compression_allowed": False,
                "prompt_tokens": [512, 2048, 8192, 32768],
                "required_prompt_tokens": [512, 2048, 8192, 32768],
                "runtime_profile_env": "AX_MLX_LINEAR_ATTENTION_PROFILE=1",
                "model_preflight": {
                    "schema_version": "ax.gateddelta_prefill_model_preflight.v1",
                    "status": "passed",
                    "checker": "scripts/check_gateddelta_prefill_model.py",
                    "model_family": "qwen3_next",
                    "model_type": "qwen3_5",
                    "linear_attention": {
                        "num_value_heads": 4,
                        "num_key_heads": 4,
                        "key_head_dim": 64,
                        "value_head_dim": 128,
                        "conv_kernel_dim": 4,
                    },
                },
            },
            "results": results,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gateddelta-profile.json"
            output = Path(tmp) / "gateddelta-profile.md"
            path.write_text(json.dumps(artifact))

            bench.render_gateddelta_prefill_profile_output(path, output)

            report = output.read_text()

        self.assertIn("GatedDelta Prefill Profile Report", report)
        self.assertIn("prioritize recurrent scan", report)

    def test_ax_decode_policy_defaults_to_kv_trim(self) -> None:
        self.assertEqual(
            bench.ax_decode_policy(
                {"linear_attention_enabled": False}, direct_mode=False
            ),
            "ngram_acceleration_kv_trim",
        )

    def test_ax_ngram_telemetry_is_extracted_from_route(self) -> None:
        telemetry = bench.extract_ax_ngram_telemetry(
            {
                "crossover_decisions": {
                    "ax_ngram_draft_attempts": 3,
                    "ax_ngram_draft_tokens": 12,
                    "ax_ngram_accepted_tokens": 9,
                    "ax_ngram_request_disable_events": 1,
                    "ax_ngram_request_disabled_steps": 4,
                    "ax_ngram_fallback_confidence_filtered_steps": 2,
                    "ax_ngram_policy_variant": 1,
                    "ax_ngram_adaptive_draft_len_steps": 3,
                    "ax_ngram_adaptive_draft_len_total": 14,
                    "unrelated": 99,
                }
            }
        )

        self.assertEqual(telemetry["ax_ngram_draft_attempts"], 3)
        self.assertEqual(telemetry["ax_ngram_draft_tokens"], 12)
        self.assertEqual(telemetry["ax_ngram_accepted_tokens"], 9)
        # Keys absent from the route are not present in the result (prefix-based
        # collection does not fill in zero defaults for missing keys).
        self.assertEqual(telemetry.get("ax_ngram_complete_misses", 0), 0)
        self.assertEqual(telemetry.get("ax_ngram_cooldown_steps", 0), 0)
        self.assertEqual(telemetry.get("ax_ngram_cooldown_events", 0), 0)
        self.assertEqual(telemetry["ax_ngram_request_disable_events"], 1)
        self.assertEqual(telemetry["ax_ngram_request_disabled_steps"], 4)
        self.assertEqual(telemetry["ax_ngram_fallback_confidence_filtered_steps"], 2)
        self.assertEqual(telemetry["ax_ngram_policy_variant"], 1)
        self.assertEqual(telemetry["ax_ngram_adaptive_draft_len_steps"], 3)
        self.assertEqual(telemetry["ax_ngram_adaptive_draft_len_total"], 14)
        self.assertEqual(telemetry["ax_ngram_accept_rate_micros"], 750000)
        self.assertNotIn("unrelated", telemetry)

    def test_ax_ngram_telemetry_summarizes_trials(self) -> None:
        summary = bench.summarize_telemetry(
            [
                {
                    "ngram_acceleration_telemetry": {
                        "ax_ngram_draft_tokens": 8,
                        "ax_ngram_accepted_tokens": 4,
                        "ax_ngram_request_disabled_steps": 2,
                        "ax_ngram_fallback_short_output_steps": 1,
                        "ax_ngram_policy_variant": 1,
                        "ax_ngram_adaptive_draft_len_total": 6,
                        "ax_ngram_accept_rate_micros": 500000,
                    }
                },
                {
                    "ngram_acceleration_telemetry": {
                        "ax_ngram_draft_tokens": 12,
                        "ax_ngram_accepted_tokens": 11,
                        "ax_ngram_request_disabled_steps": 3,
                        "ax_ngram_fallback_short_output_steps": 2,
                        "ax_ngram_policy_variant": 1,
                        "ax_ngram_adaptive_draft_len_total": 8,
                        "ax_ngram_accept_rate_micros": 916667,
                    }
                },
            ]
        )

        self.assertEqual(summary["ax_ngram_draft_tokens"], 20)
        self.assertEqual(summary["ax_ngram_accepted_tokens"], 15)
        self.assertEqual(summary["ax_ngram_request_disabled_steps"], 5)
        self.assertEqual(summary["ax_ngram_fallback_short_output_steps"], 3)
        self.assertEqual(summary["ax_ngram_policy_variant"], 1)
        self.assertEqual(summary["ax_ngram_adaptive_draft_len_total"], 14)
        self.assertEqual(summary["ax_ngram_accept_rate_micros"], 750000)

    def test_ax_decode_claim_status_distinguishes_no_draft_fallback(self) -> None:
        self.assertEqual(
            bench.ax_decode_claim_status(
                True,
                {
                    "ax_ngram_draft_attempts": 0,
                    "ax_ngram_no_draft_steps": 17,
                },
            ),
            "direct_same_policy_baseline",
        )
        self.assertEqual(
            bench.ax_decode_claim_status(
                False,
                {
                    "ax_ngram_draft_attempts": 0,
                    "ax_ngram_no_draft_steps": 17,
                    "ax_ngram_request_disabled_steps": 111,
                },
            ),
            "ngram_no_draft_direct_fallback",
        )
        self.assertEqual(
            bench.ax_decode_claim_status(
                False,
                {
                    "ax_ngram_draft_attempts": 0,
                    "ax_ngram_no_draft_steps": 0,
                    "ax_ngram_request_disabled_steps": 0,
                },
            ),
            "ngram_no_observed_draft_path",
        )
        self.assertEqual(
            bench.ax_decode_claim_status(
                False,
                {
                    "ax_ngram_draft_attempts": 3,
                    "ax_ngram_draft_tokens": 12,
                    "ax_ngram_accepted_tokens": 0,
                },
            ),
            "ngram_no_accept_fallback",
        )
        self.assertEqual(
            bench.ax_decode_claim_status(
                False,
                {
                    "ax_ngram_draft_attempts": 3,
                    "ax_ngram_draft_tokens": 12,
                    "ax_ngram_accepted_tokens": 12,
                },
            ),
            "ngram_acceleration_effective_throughput",
        )

    def test_ax_decode_effective_route_identifies_linear_no_draft_fallback(
        self,
    ) -> None:
        telemetry = {
            "ax_ngram_draft_attempts": 0,
            "ax_ngram_accepted_tokens": 0,
            "ax_ngram_request_disabled_steps": 381,
            "ax_ngram_fallback_linear_no_draft_steps": 381,
        }
        ax_mlx_telemetry = {
            "ax_mlx_direct_pipeline_steps": 381,
            "ax_mlx_single_decode_steps": 0,
        }

        self.assertEqual(
            bench.ax_decode_effective_route(
                direct_mode=False,
                model_metadata={"linear_attention_enabled": True},
                telemetry=telemetry,
                ax_mlx_telemetry=ax_mlx_telemetry,
            ),
            "linear_no_draft_direct_pipeline_fallback",
        )
        self.assertEqual(
            bench.ax_decode_effective_route(
                direct_mode=True,
                model_metadata={"linear_attention_enabled": True},
                telemetry={},
                ax_mlx_telemetry=ax_mlx_telemetry,
            ),
            "direct_pipeline_baseline",
        )

    def test_summarize_ngram_accept_at_depth_yields_histogram_or_empty(self) -> None:
        # No depth keys at all => empty (do not annotate older runtime rows).
        self.assertEqual(bench.summarize_ngram_accept_at_depth({}), {})

        # Single bucket present => full 8-bucket histogram emitted.
        result = bench.summarize_ngram_accept_at_depth(
            {"ax_ngram_accept_at_depth_2": 5}
        )
        self.assertEqual(result["schema"], "ax.ngram_accept_at_depth.v1")
        self.assertEqual(result["bucket_count"], 8)
        self.assertEqual(len(result["buckets"]), 8)
        self.assertEqual(result["buckets"][2]["attempts"], 5)
        self.assertEqual(result["total_attempts"], 5)
        # 2 * 5 accepted tokens.
        self.assertEqual(result["weighted_accepted_tokens_lower_bound"], 10)

    def test_canonical_prompt_hash_is_stable_and_order_sensitive(self) -> None:
        a = bench.canonical_prompt_hash([1, 2, 3])
        b = bench.canonical_prompt_hash([1, 2, 3])
        c = bench.canonical_prompt_hash([3, 2, 1])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 16)
        int(a, 16)  # raises if non-hex

    def test_canonical_sampler_signature_collapses_greedy_equivalents(self) -> None:
        # Greedy-equivalent dicts must all produce "greedy", otherwise
        # a direct-vs-n-gram pair with the same effective sampling would
        # be rejected by the promotion gate.
        self.assertEqual(bench.canonical_sampler_signature(None), "greedy")
        self.assertEqual(bench.canonical_sampler_signature({}), "greedy")
        self.assertEqual(
            bench.canonical_sampler_signature(
                {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repetition_penalty": 1.0,
                }
            ),
            "greedy",
        )
        sig = bench.canonical_sampler_signature({"temperature": 0.7, "top_p": 0.95})
        self.assertTrue(sig.startswith("sampling["))
        self.assertIn("temperature=0.7", sig)
        self.assertIn("top_p=0.95", sig)

    def test_build_row_identity_matches_rust_gate_schema(self) -> None:
        ident = bench.build_row_identity(
            model_id="qwen3",
            tokens=[1, 2, 3, 4],
            seed=0,
            max_output_tokens=64,
            sampler=None,
        )
        # Schema name and field set must match the Rust harness
        # RowIdentity struct ordering so a Python aggregator can hand a
        # JSON-deserialized row to the gate without renaming fields.
        self.assertEqual(ident["schema"], "ax.row_identity.v1")
        for key in (
            "model_id",
            "prompt_hash",
            "seed",
            "max_output_tokens",
            "sampler_signature",
        ):
            self.assertIn(key, ident)
        self.assertEqual(ident["model_id"], "qwen3")
        self.assertEqual(ident["seed"], 0)
        self.assertEqual(ident["max_output_tokens"], 64)
        self.assertEqual(ident["sampler_signature"], "greedy")

    def test_absent_ax_mlx_telemetry_stays_silent(self) -> None:
        self.assertEqual(
            bench.extract_ax_mlx_telemetry({"crossover_decisions": {"unrelated": 99}}),
            {},
        )

    def test_ax_mlx_telemetry_is_extracted_and_summarized(self) -> None:
        telemetry = bench.extract_ax_mlx_telemetry(
            {
                "crossover_decisions": {
                    "ax_mlx_prefill_steps": 1,
                    "ax_mlx_prefill_wall_us": 100,
                    "ax_mlx_prefill_forward_wall_us": 70,
                    "ax_mlx_prefill_prefix_cache_wall_us": 20,
                    "ax_mlx_prefill_generation_state_wall_us": 5,
                    "ax_mlx_prefill_eval_barriers": 1,
                    "ax_mlx_prefill_drain_async_evals": 2,
                    "ax_mlx_decode_steps": 2,
                    "ax_mlx_decode_wall_us": 80,
                    "ax_mlx_direct_pipeline_steps": 2,
                    "ax_mlx_direct_pipeline_wall_us": 70,
                    "ax_mlx_direct_pipeline_forward_wall_us": 40,
                    "ax_mlx_direct_pipeline_forward_layer_loop_wall_us": 35,
                    "ax_mlx_direct_pipeline_forward_head_wall_us": 4,
                    "ax_mlx_direct_pipeline_argmax_wall_us": 2,
                    "ax_mlx_direct_pipeline_async_eval_wall_us": 5,
                    "ax_mlx_direct_pipeline_next_complete_wall_us": 9,
                    "ax_mlx_direct_pipeline_pending_eval_wall_us": 20,
                    "ax_mlx_direct_pipeline_pending_read_wall_us": 3,
                    "ax_mlx_direct_pipeline_op_count": 84,
                    "ax_mlx_direct_pipeline_linear_attention_layer_ops": 44,
                    "ax_mlx_direct_pipeline_linear_attention_layer_count": 2,
                    "ax_mlx_direct_pipeline_full_attention_layer_ops": 40,
                    "ax_mlx_direct_pipeline_full_attention_layer_count": 4,
                    "ax_mlx_prefix_cache_hits": 1,
                    "ax_mlx_prefix_cache_blocked_policy_disabled": 2,
                    "ax_mlx_prefix_cache_reused_tokens": 16,
                    "ax_mlx_dense_attention_qkv_packed_layers": 16,
                    "ax_mlx_dense_attention_split_qkv_layers": 0,
                    "ax_mlx_dense_ffn_gate_up_packed_layers": 0,
                    "ax_mlx_dense_ffn_split_gate_up_layers": 64,
                    "ax_mlx_linear_attention_qkvz_ba_packed_layers": 48,
                    "ax_mlx_linear_attention_split_qkvba_layers": 0,
                    "ax_mlx_direct_cpp_linear_attention_inputs_attempts": 4,
                    "ax_mlx_direct_cpp_linear_attention_inputs_hits": 4,
                    "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks": 0,
                    "ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked": 0,
                    "ax_mlx_direct_cpp_linear_attention_post_input_attempts": 2,
                    "ax_mlx_direct_cpp_linear_attention_post_input_hits": 2,
                    "ax_mlx_direct_cpp_linear_attention_post_input_fallbacks": 0,
                    "ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked": 0,
                    "ax_mlx_qwen_linear_attention_decode_post_input_metal_attempts": 2,
                    "ax_mlx_qwen_linear_attention_decode_post_input_metal_hits": 2,
                    "ax_mlx_qwen_linear_attention_decode_post_input_metal_fallbacks": 0,
                    "ax_mlx_qwen_linear_attention_decode_post_input_metal_profile_blocked": 0,
                    "ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_attempts": 64,
                    "ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_hits": 64,
                    "ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_fallbacks": 0,
                    "unrelated": 99,
                }
            }
        )

        self.assertEqual(telemetry["ax_mlx_prefill_steps"], 1)
        self.assertEqual(telemetry["ax_mlx_prefill_wall_us"], 100)
        self.assertEqual(telemetry["ax_mlx_prefill_forward_wall_us"], 70)
        self.assertEqual(telemetry["ax_mlx_prefill_prefix_cache_wall_us"], 20)
        self.assertEqual(telemetry["ax_mlx_prefill_generation_state_wall_us"], 5)
        self.assertEqual(telemetry["ax_mlx_prefill_eval_barriers"], 1)
        self.assertEqual(telemetry["ax_mlx_prefill_drain_async_evals"], 2)
        self.assertEqual(telemetry["ax_mlx_decode_steps"], 2)
        self.assertEqual(telemetry["ax_mlx_decode_wall_us"], 80)
        self.assertEqual(telemetry["ax_mlx_direct_pipeline_steps"], 2)
        self.assertEqual(telemetry["ax_mlx_direct_pipeline_wall_us"], 70)
        self.assertEqual(telemetry["ax_mlx_direct_pipeline_forward_wall_us"], 40)
        self.assertEqual(
            telemetry["ax_mlx_direct_pipeline_forward_layer_loop_wall_us"], 35
        )
        self.assertEqual(telemetry["ax_mlx_direct_pipeline_forward_head_wall_us"], 4)
        self.assertEqual(telemetry["ax_mlx_direct_pipeline_argmax_wall_us"], 2)
        self.assertEqual(telemetry["ax_mlx_direct_pipeline_async_eval_wall_us"], 5)
        self.assertEqual(telemetry["ax_mlx_direct_pipeline_next_complete_wall_us"], 9)
        self.assertEqual(telemetry["ax_mlx_direct_pipeline_pending_eval_wall_us"], 20)
        self.assertEqual(telemetry["ax_mlx_direct_pipeline_pending_read_wall_us"], 3)
        self.assertEqual(telemetry["ax_mlx_direct_pipeline_op_count"], 84)
        self.assertEqual(
            telemetry["ax_mlx_direct_pipeline_linear_attention_layer_ops"], 44
        )
        self.assertEqual(
            telemetry["ax_mlx_direct_pipeline_linear_attention_layer_count"], 2
        )
        self.assertEqual(
            telemetry["ax_mlx_direct_pipeline_full_attention_layer_ops"], 40
        )
        self.assertEqual(
            telemetry["ax_mlx_direct_pipeline_full_attention_layer_count"], 4
        )
        self.assertEqual(telemetry["ax_mlx_prefix_cache_hits"], 1)
        self.assertEqual(telemetry["ax_mlx_prefix_cache_blocked_policy_disabled"], 2)
        self.assertEqual(telemetry["ax_mlx_prefix_cache_reused_tokens"], 16)
        self.assertEqual(telemetry["ax_mlx_dense_attention_qkv_packed_layers"], 16)
        self.assertEqual(telemetry["ax_mlx_dense_attention_split_qkv_layers"], 0)
        self.assertEqual(telemetry["ax_mlx_dense_ffn_gate_up_packed_layers"], 0)
        self.assertEqual(telemetry["ax_mlx_dense_ffn_split_gate_up_layers"], 64)
        self.assertEqual(telemetry["ax_mlx_linear_attention_qkvz_ba_packed_layers"], 48)
        self.assertEqual(telemetry["ax_mlx_linear_attention_split_qkvba_layers"], 0)
        self.assertEqual(telemetry["ax_mlx_prefix_cache_evictions"], 0)
        self.assertEqual(telemetry["ax_mlx_prefix_cache_blocked_unsupported_layout"], 0)
        self.assertEqual(telemetry["ax_mlx_prefix_cache_blocked_trim_failure"], 0)
        self.assertEqual(
            telemetry["ax_mlx_direct_cpp_linear_attention_inputs_attempts"],
            4,
        )
        self.assertEqual(telemetry["ax_mlx_direct_cpp_linear_attention_inputs_hits"], 4)
        self.assertEqual(
            telemetry["ax_mlx_direct_cpp_linear_attention_post_input_attempts"],
            2,
        )
        self.assertEqual(
            telemetry["ax_mlx_direct_cpp_linear_attention_post_input_hits"], 2
        )
        self.assertEqual(
            telemetry["ax_mlx_qwen_linear_attention_decode_post_input_metal_hits"], 2
        )
        self.assertEqual(
            telemetry["ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_attempts"], 64
        )
        self.assertEqual(
            telemetry["ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_hits"], 64
        )
        self.assertEqual(telemetry["ax_mlx_single_decode_steps"], 0)
        self.assertEqual(telemetry["ax_mlx_bonus_tokens"], 0)
        self.assertNotIn("unrelated", telemetry)

        partial_telemetry = bench.extract_ax_mlx_telemetry(
            {"crossover_decisions": {"ax_mlx_decode_steps": 2}}
        )
        self.assertEqual(partial_telemetry["ax_mlx_decode_steps"], 2)
        self.assertEqual(partial_telemetry["ax_mlx_direct_pipeline_argmax_wall_us"], 0)

        summary = bench.summarize_ax_mlx_telemetry(
            [
                {"ax_mlx_telemetry": telemetry},
                {
                    "ax_mlx_telemetry": {
                        "ax_mlx_prefill_steps": 1,
                        "ax_mlx_prefill_forward_wall_us": 30,
                        "ax_mlx_prefill_prefix_cache_wall_us": 10,
                        "ax_mlx_prefill_generation_state_wall_us": 1,
                        "ax_mlx_prefill_eval_barriers": 1,
                        "ax_mlx_prefill_drain_async_evals": 3,
                        "ax_mlx_decode_steps": 3,
                        "ax_mlx_decode_wall_us": 120,
                    }
                },
            ]
        )
        self.assertEqual(summary["ax_mlx_prefill_steps"], 2)
        self.assertEqual(summary["ax_mlx_prefill_forward_wall_us"], 100)
        self.assertEqual(summary["ax_mlx_prefill_prefix_cache_wall_us"], 30)
        self.assertEqual(summary["ax_mlx_prefill_generation_state_wall_us"], 6)
        self.assertEqual(summary["ax_mlx_prefill_eval_barriers"], 2)
        self.assertEqual(summary["ax_mlx_prefill_drain_async_evals"], 5)
        self.assertEqual(summary["ax_mlx_decode_steps"], 5)
        self.assertEqual(
            summary["ax_mlx_direct_cpp_linear_attention_inputs_attempts"],
            4,
        )
        self.assertEqual(
            summary["ax_mlx_direct_cpp_linear_attention_post_input_attempts"],
            2,
        )
        self.assertEqual(summary["ax_mlx_dense_attention_qkv_packed_layers"], 16)
        self.assertEqual(summary["ax_mlx_dense_attention_split_qkv_layers"], 0)
        self.assertEqual(summary["ax_mlx_dense_ffn_gate_up_packed_layers"], 0)
        self.assertEqual(summary["ax_mlx_dense_ffn_split_gate_up_layers"], 64)
        self.assertEqual(
            summary["ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_attempts"], 64
        )
        self.assertEqual(
            summary["ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_hits"], 64
        )
        self.assertEqual(summary["ax_mlx_linear_attention_qkvz_ba_packed_layers"], 48)
        self.assertEqual(summary["ax_mlx_linear_attention_split_qkvba_layers"], 0)

        direct_cpp_summary = bench.summarize_ax_mlx_direct_cpp_linear_attention_inputs(
            telemetry
        )
        self.assertEqual(direct_cpp_summary["classification"], "all_hits")
        self.assertEqual(direct_cpp_summary["hit_rate_micros"], 1_000_000)
        direct_cpp_post_input_summary = (
            bench.summarize_ax_mlx_direct_cpp_linear_attention_post_input(telemetry)
        )
        self.assertEqual(direct_cpp_post_input_summary["classification"], "all_hits")
        self.assertEqual(direct_cpp_post_input_summary["hit_rate_micros"], 1_000_000)
        qwen_post_input_metal_summary = (
            bench.summarize_ax_mlx_qwen_linear_attention_decode_post_input_metal(
                telemetry
            )
        )
        self.assertEqual(qwen_post_input_metal_summary["classification"], "all_hits")
        self.assertEqual(
            qwen_post_input_metal_summary["hit_rate_micros"], 1_000_000
        )
        qwen_dense_ffn_matvec_summary = (
            bench.summarize_ax_mlx_qwen_dense_ffn_gate_up_matvec_metal(
                telemetry
            )
        )
        self.assertEqual(qwen_dense_ffn_matvec_summary["classification"], "all_hits")
        self.assertEqual(
            qwen_dense_ffn_matvec_summary["hit_rate_micros"], 1_000_000
        )
        effective_routes = bench.summarize_ax_mlx_effective_routes(telemetry)
        self.assertEqual(
            effective_routes["dense_attention_qkv"]["status"],
            "packed",
        )
        self.assertEqual(
            effective_routes["dense_ffn_gate_up"]["status"],
            "split",
        )
        self.assertEqual(
            effective_routes["dense_ffn_gate_up"]["qwen_gate_up_matvec_metal"][
                "classification"
            ],
            "all_hits",
        )
        self.assertEqual(
            effective_routes["linear_attention_qkvz_ba"]["status"],
            "packed",
        )
        self.assertEqual(
            effective_routes["linear_attention_direct_cpp_inputs"]["classification"],
            "all_hits",
        )
        self.assertEqual(
            effective_routes["linear_attention_direct_cpp_post_input"][
                "classification"
            ],
            "all_hits",
        )
        self.assertEqual(
            effective_routes["qwen_linear_attention_decode_post_input_metal"][
                "classification"
            ],
            "all_hits",
        )
        self.assertEqual(summary["ax_mlx_decode_wall_us"], 200)
        self.assertEqual(summary["ax_mlx_direct_pipeline_forward_wall_us"], 40)
        self.assertEqual(
            summary["ax_mlx_direct_pipeline_forward_layer_loop_wall_us"], 35
        )
        self.assertEqual(summary["ax_mlx_direct_pipeline_forward_head_wall_us"], 4)
        self.assertEqual(summary["ax_mlx_direct_pipeline_argmax_wall_us"], 2)
        self.assertEqual(summary["ax_mlx_direct_pipeline_async_eval_wall_us"], 5)
        self.assertEqual(summary["ax_mlx_direct_pipeline_next_complete_wall_us"], 9)
        self.assertEqual(summary["ax_mlx_direct_pipeline_pending_eval_wall_us"], 20)
        self.assertEqual(summary["ax_mlx_direct_pipeline_pending_read_wall_us"], 3)
        self.assertEqual(summary["ax_mlx_direct_pipeline_op_count"], 84)
        self.assertEqual(
            summary["ax_mlx_direct_pipeline_linear_attention_layer_ops"], 44
        )
        self.assertEqual(
            summary["ax_mlx_direct_pipeline_linear_attention_layer_count"], 2
        )
        self.assertEqual(summary["ax_mlx_direct_pipeline_full_attention_layer_ops"], 40)
        self.assertEqual(
            summary["ax_mlx_direct_pipeline_full_attention_layer_count"], 4
        )
        self.assertEqual(summary["ax_mlx_prefix_cache_hits"], 1)
        self.assertEqual(summary["ax_mlx_prefix_cache_blocked_policy_disabled"], 2)
        self.assertEqual(summary["ax_mlx_prefix_cache_reused_tokens"], 16)

    def test_attempted_fastpath_summary_counts_profile_blocked_attempts(self) -> None:
        telemetry = {
            "ax_mlx_direct_cpp_linear_attention_inputs_attempts": 3,
            "ax_mlx_direct_cpp_linear_attention_inputs_hits": 0,
            "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks": 0,
            "ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked": 3,
            "ax_mlx_qwen_linear_attention_decode_post_input_metal_attempts": 4,
            "ax_mlx_qwen_linear_attention_decode_post_input_metal_hits": 1,
            "ax_mlx_qwen_linear_attention_decode_post_input_metal_fallbacks": 0,
            "ax_mlx_qwen_linear_attention_decode_post_input_metal_profile_blocked": 3,
        }

        direct_cpp_summary = bench.summarize_ax_mlx_direct_cpp_linear_attention_inputs(
            telemetry
        )
        self.assertEqual(
            direct_cpp_summary["classification"], "profile_blocked_fallback"
        )
        self.assertEqual(direct_cpp_summary["profile_blocked"], 3)

        qwen_post_input_summary = (
            bench.summarize_ax_mlx_qwen_linear_attention_decode_post_input_metal(
                telemetry
            )
        )
        self.assertEqual(
            qwen_post_input_summary["classification"],
            "mixed_hit_profile_blocked",
        )
        self.assertEqual(qwen_post_input_summary["profile_blocked"], 3)

    def test_ax_mlx_decode_route_summary_classifies_pipeline_and_mixed_rows(
        self,
    ) -> None:
        direct = bench.summarize_ax_mlx_decode_route(
            {
                "ax_mlx_decode_steps": 10,
                "ax_mlx_decode_wall_us": 1000,
                "ax_mlx_direct_pipeline_steps": 10,
                "ax_mlx_direct_pipeline_wall_us": 900,
                "ax_mlx_direct_pipeline_forward_wall_us": 300,
                "ax_mlx_direct_pipeline_forward_layer_loop_wall_us": 240,
                "ax_mlx_direct_pipeline_forward_head_wall_us": 30,
                "ax_mlx_direct_pipeline_argmax_wall_us": 100,
                "ax_mlx_direct_pipeline_async_eval_wall_us": 400,
                "ax_mlx_direct_pipeline_next_complete_wall_us": 600,
                "ax_mlx_direct_pipeline_pending_eval_wall_us": 80,
                "ax_mlx_direct_pipeline_pending_read_wall_us": 20,
                "ax_mlx_direct_pipeline_op_count": 350,
                "ax_mlx_direct_pipeline_linear_attention_layer_ops": 220,
                "ax_mlx_direct_pipeline_linear_attention_layer_count": 10,
                "ax_mlx_direct_pipeline_full_attention_layer_ops": 130,
                "ax_mlx_direct_pipeline_full_attention_layer_count": 5,
            }
        )
        self.assertEqual(direct["classification"], "direct_pipeline")
        self.assertEqual(direct["direct_pipeline_step_share_micros"], 1_000_000)
        self.assertEqual(direct["direct_pipeline_wall_share_micros"], 900_000)
        self.assertEqual(direct["direct_pipeline_forward_wall_us"], 300)
        self.assertEqual(direct["direct_pipeline_forward_wall_share_micros"], 333_333)
        self.assertEqual(direct["direct_pipeline_forward_layer_loop_wall_us"], 240)
        self.assertEqual(
            direct["direct_pipeline_forward_layer_loop_wall_share_micros"], 800_000
        )
        self.assertEqual(direct["direct_pipeline_forward_head_wall_us"], 30)
        self.assertEqual(
            direct["direct_pipeline_forward_head_wall_share_micros"], 100_000
        )
        self.assertEqual(direct["direct_pipeline_argmax_wall_us"], 100)
        self.assertEqual(direct["direct_pipeline_argmax_wall_share_micros"], 111_111)
        self.assertEqual(direct["direct_pipeline_async_eval_wall_us"], 400)
        self.assertEqual(
            direct["direct_pipeline_async_eval_wall_share_micros"],
            444_444,
        )
        self.assertEqual(direct["direct_pipeline_next_complete_wall_us"], 600)
        self.assertEqual(
            direct["direct_pipeline_next_complete_wall_share_micros"],
            666_667,
        )
        self.assertEqual(direct["direct_pipeline_pending_eval_wall_us"], 80)
        self.assertEqual(
            direct["direct_pipeline_pending_eval_wall_share_micros"],
            88_889,
        )
        self.assertEqual(direct["direct_pipeline_pending_read_wall_us"], 20)
        self.assertEqual(
            direct["direct_pipeline_pending_read_wall_share_micros"],
            22_222,
        )
        self.assertEqual(direct["direct_pipeline_op_count"], 350)
        self.assertEqual(direct["direct_pipeline_op_count_per_step"], 35)
        self.assertEqual(direct["direct_pipeline_linear_attention_layer_ops"], 220)
        self.assertEqual(direct["direct_pipeline_linear_attention_layer_count"], 10)
        self.assertEqual(direct["direct_pipeline_linear_attention_ops_per_layer"], 22)
        self.assertEqual(direct["direct_pipeline_full_attention_layer_ops"], 130)
        self.assertEqual(direct["direct_pipeline_full_attention_layer_count"], 5)
        self.assertEqual(direct["direct_pipeline_full_attention_ops_per_layer"], 26)

        mixed = bench.summarize_ax_mlx_decode_route(
            {
                "ax_mlx_decode_steps": 10,
                "ax_mlx_decode_wall_us": 1000,
                "ax_mlx_direct_pipeline_steps": 7,
                "ax_mlx_direct_pipeline_wall_us": 700,
                "ax_mlx_single_decode_steps": 3,
                "ax_mlx_single_decode_wall_us": 300,
            }
        )
        self.assertEqual(mixed["classification"], "mixed")
        self.assertEqual(mixed["direct_pipeline_step_share_micros"], 700_000)
        self.assertEqual(mixed["single_decode_step_share_micros"], 300_000)

    def test_ax_mlx_decode_route_summary_classifies_diffusion(self) -> None:
        diffusion = bench.summarize_ax_mlx_decode_route(
            {
                "ax_mlx_decode_steps": 20,
                "ax_mlx_decode_wall_us": 2000,
                "ax_mlx_diffusion_blocks": 4,
                "ax_mlx_diffusion_denoise_steps": 16,
                "ax_mlx_diffusion_converged_blocks": 3,
                "ax_mlx_diffusion_denoise_wall_us": 1200,
                "ax_mlx_diffusion_commit_wall_us": 400,
                "ax_mlx_diffusion_block_wall_us": 1800,
            }
        )
        self.assertEqual(diffusion["classification"], "diffusion")
        self.assertEqual(diffusion["diffusion_blocks"], 4)
        self.assertEqual(diffusion["diffusion_denoise_steps"], 16)
        self.assertEqual(diffusion["diffusion_converged_blocks"], 3)
        self.assertEqual(diffusion["diffusion_denoise_wall_us"], 1200)
        self.assertEqual(diffusion["diffusion_commit_wall_us"], 400)
        self.assertEqual(diffusion["diffusion_block_wall_us"], 1800)
        self.assertEqual(diffusion["diffusion_denoise_step_share_micros"], 800_000)
        self.assertEqual(diffusion["diffusion_block_wall_share_micros"], 900_000)

    def test_scheduler_telemetry_is_extracted_and_summarized(self) -> None:
        telemetry = bench.extract_scheduler_telemetry(
            {
                "crossover_decisions": {
                    "ax_scheduler_scheduled_prefill_tokens": 2047,
                    "ax_scheduler_scheduled_decode_tokens": 1,
                    "ax_scheduler_skipped_prefill_tokens": 4096,
                    "ax_scheduler_mixed_prefill_decode_batches": 1,
                    "unrelated": 99,
                }
            }
        )

        self.assertEqual(telemetry["ax_scheduler_scheduled_prefill_tokens"], 2047)
        self.assertEqual(telemetry["ax_scheduler_scheduled_decode_tokens"], 1)
        self.assertEqual(telemetry["ax_scheduler_skipped_prefill_tokens"], 4096)
        self.assertEqual(telemetry["ax_scheduler_skipped_decode_tokens"], 0)
        self.assertEqual(telemetry["ax_scheduler_mixed_prefill_decode_batches"], 1)
        self.assertNotIn("unrelated", telemetry)

        summary = bench.summarize_scheduler_telemetry(
            [
                {"scheduler_telemetry": telemetry},
                {
                    "scheduler_telemetry": {
                        "ax_scheduler_scheduled_prefill_tokens": 3,
                        "ax_scheduler_scheduled_decode_tokens": 2,
                    }
                },
            ]
        )
        self.assertEqual(summary["ax_scheduler_scheduled_prefill_tokens"], 2050)
        self.assertEqual(summary["ax_scheduler_scheduled_decode_tokens"], 3)
        self.assertEqual(summary["ax_scheduler_skipped_prefill_tokens"], 4096)

    def test_prefix_reuse_evidence_summarizes_ax_rows(self) -> None:
        evidence = bench.summarize_prefix_reuse_evidence(
            [
                {
                    "engine": "mlx_lm",
                    "ax_mlx_telemetry": {"ax_mlx_prefix_cache_hits": 99},
                },
                {
                    "engine": "ax_engine_mlx",
                    "ax_mlx_telemetry": {
                        "ax_mlx_prefix_cache_hits": 1,
                        "ax_mlx_prefix_cache_misses": 2,
                        "ax_mlx_prefix_cache_blocked": 3,
                        "ax_mlx_prefix_cache_blocked_policy_disabled": 1,
                        "ax_mlx_prefix_cache_blocked_unsupported_layout": 2,
                        "ax_mlx_prefix_cache_blocked_trim_failure": 0,
                        "ax_mlx_prefix_cache_stores": 4,
                        "ax_mlx_prefix_cache_evictions": 5,
                        "ax_mlx_prefix_cache_reused_tokens": 16,
                        "ax_mlx_prefix_cache_warmup_tokens": 8,
                        "ax_mlx_prefix_cache_entries": 6,
                        "ax_mlx_prefix_cache_bytes_kib": 128,
                    },
                },
                {
                    "engine": "ax_engine_mlx_ngram_accel",
                    "ax_mlx_telemetry": {
                        "ax_mlx_prefix_cache_hits": 1,
                        "ax_mlx_prefix_cache_blocked": 1,
                        "ax_mlx_prefix_cache_blocked_trim_failure": 1,
                        "ax_mlx_prefix_cache_entries": 2,
                        "ax_mlx_prefix_cache_bytes_kib": 64,
                    },
                },
            ]
        )

        self.assertEqual(evidence["hit_count"], 2)
        self.assertEqual(evidence["miss_count"], 2)
        self.assertEqual(evidence["blocked_count"], 4)
        self.assertEqual(evidence["blocked_policy_disabled_count"], 1)
        self.assertEqual(evidence["blocked_unsupported_layout_count"], 2)
        self.assertEqual(evidence["blocked_trim_failure_count"], 1)
        self.assertEqual(evidence["stored_prefix_count"], 4)
        self.assertEqual(evidence["eviction_count"], 5)
        self.assertEqual(evidence["reused_token_count"], 16)
        self.assertEqual(evidence["warmup_token_count"], 8)
        self.assertEqual(evidence["cache_entry_count"], 6)
        self.assertEqual(evidence["cache_bytes_kib"], 128)
        self.assertTrue(evidence["physical_snapshot_hit_observed"])
        self.assertTrue(evidence["physical_snapshot_miss_warmup_observed"])
        self.assertTrue(evidence["physical_snapshot_blocked_observed"])
        self.assertEqual(evidence["physical_snapshot_coverage"], "hit_and_miss_warmup")
        self.assertEqual(evidence["blocked_reason_count"], 4)
        self.assertEqual(evidence["blocked_reason_accounting_gap_count"], 0)

    def test_artifact_run_stability_summarizes_ax_rows(self) -> None:
        summary = bench.summarize_artifact_run_stability(
            [
                {"engine": "mlx_lm"},
                {
                    "engine": "ax_engine_mlx",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "run_stability": {
                        "schema_version": "ax.benchmark_run_stability.v1",
                        "metric": "decode_tok_s",
                        "classification": "stable_enough",
                    },
                },
                {
                    "engine": "ax_engine_mlx_ngram_accel",
                    "prompt_tokens": 512,
                    "generation_tokens": 128,
                    "run_stability": {
                        "schema_version": "ax.benchmark_run_stability.v1",
                        "metric": "decode_tok_s",
                        "classification": "tail_regression",
                        "last_vs_first_pct": -12.5,
                    },
                },
                {
                    "engine": "ax_engine_gemma4_assistant_mtp",
                    "prompt_tokens": 2048,
                    "generation_tokens": 128,
                },
            ]
        )

        self.assertEqual(
            summary["schema_version"], "ax.benchmark_run_stability_summary.v1"
        )
        self.assertEqual(summary["scope"], "ax_engine_rows")
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["stable_enough_count"], 1)
        self.assertEqual(summary["unstable_count"], 1)
        self.assertEqual(summary["missing_count"], 1)
        self.assertFalse(summary["publication_candidate"])
        self.assertEqual(summary["classification_counts"]["stable_enough"], 1)
        self.assertEqual(summary["classification_counts"]["tail_regression"], 1)
        self.assertEqual(
            summary["unstable_rows"],
            [
                {
                    "engine": "ax_engine_mlx_ngram_accel",
                    "prompt_tokens": 512,
                    "generation_tokens": 128,
                    "classification": "tail_regression",
                    "last_vs_first_pct": -12.5,
                }
            ],
        )

    def test_artifact_run_stability_rejects_empty_ax_scope(self) -> None:
        summary = bench.summarize_artifact_run_stability(
            [
                {
                    "engine": "mlx_lm",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                }
            ]
        )

        self.assertEqual(summary["row_count"], 0)
        self.assertFalse(summary["publication_candidate"])

    def test_artifact_run_stability_rejects_stale_row_schema(self) -> None:
        summary = bench.summarize_artifact_run_stability(
            [
                {
                    "engine": "ax_engine_mlx",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "run_stability": {
                        "schema_version": "ax.benchmark_run_stability.v0",
                        "metric": "decode_tok_s",
                        "classification": "stable_enough",
                    },
                }
            ]
        )

        self.assertEqual(summary["stable_enough_count"], 0)
        self.assertEqual(summary["unstable_count"], 1)
        self.assertFalse(summary["publication_candidate"])
        self.assertEqual(
            summary["classification_counts"]["invalid_run_stability_schema"],
            1,
        )
        self.assertEqual(
            summary["unstable_rows"],
            [
                {
                    "engine": "ax_engine_mlx",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "classification": "invalid_run_stability_schema",
                }
            ],
        )

    def test_artifact_run_stability_ignores_non_object_rows(self) -> None:
        summary = bench.summarize_artifact_run_stability([None, "bad-row"])

        self.assertEqual(summary["row_count"], 0)
        self.assertFalse(summary["publication_candidate"])

    def test_ax_only_refresh_regression_blocks_slower_ax_rows(self) -> None:
        results = [
            {
                "engine": "ax_engine_mlx",
                "prompt_tokens": 128,
                "generation_tokens": 128,
                "prefill_tok_s": {"median": 410.0},
                "decode_tok_s": {"median": 18.8},
                "ttft_ms": {"median": 312.0},
            }
        ]
        reference_doc = {
            "results": [
                {
                    "engine": "ax_engine_mlx",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "prefill_tok_s": {"median": 430.0},
                    "decode_tok_s": {"median": 20.2},
                    "ttft_ms": {"median": 297.0},
                }
            ]
        }

        summary = bench.summarize_ax_only_refresh_regression(
            results=results,
            reference_doc=reference_doc,
        )

        self.assertEqual(
            summary["schema_version"], "ax.ax_only_refresh_regression.v1"
        )
        self.assertEqual(summary["matched_count"], 1)
        self.assertEqual(summary["decode_regression_count"], 1)
        self.assertFalse(summary["publication_candidate"])
        self.assertEqual(
            summary["classification_counts"],
            {"decode_regression": 1},
        )
        self.assertEqual(summary["rows"][0]["classification"], "decode_regression")
        self.assertAlmostEqual(
            summary["rows"][0]["decode_ratio_to_reference"],
            18.8 / 20.2,
        )
        self.assertEqual(
            bench.ax_only_refresh_failure_reasons(summary),
            ["decode_regression=1"],
        )

    def test_ax_only_refresh_passing_summary_has_no_failure_reasons(self) -> None:
        row = {
            "engine": "ax_engine_mlx",
            "prompt_tokens": 128,
            "generation_tokens": 128,
            "decode_tok_s": {"median": 20.0},
        }
        summary = bench.summarize_ax_only_refresh_regression(
            results=[row],
            reference_doc={"results": [row]},
        )

        self.assertTrue(summary["publication_candidate"])
        self.assertEqual(bench.ax_only_refresh_failure_reasons(summary), [])

    def test_ax_only_refresh_non_publication_candidate_exits_nonzero(self) -> None:
        summary = {
            "publication_candidate": False,
            "row_count": 1,
            "decode_regression_count": 1,
            "classification_counts": {"missing_decode_metric": 1},
        }
        stderr = io.StringIO()

        with (
            patch.object(sys, "stderr", stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            bench.fail_if_ax_only_refresh_not_publication_candidate(summary)

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("not a publication candidate", stderr.getvalue())
        self.assertIn("decode_regression=1", stderr.getvalue())
        self.assertIn("missing_decode_metric=1", stderr.getvalue())

    def test_ax_only_refresh_regression_blocks_missing_reference_rows(self) -> None:
        summary = bench.summarize_ax_only_refresh_regression(
            results=[
                {
                    "engine": "ax_engine_mlx",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "decode_tok_s": {"median": 20.0},
                }
            ],
            reference_doc={"results": []},
        )

        self.assertEqual(summary["row_count"], 1)
        self.assertEqual(summary["matched_count"], 0)
        self.assertEqual(summary["missing_reference_count"], 1)
        self.assertEqual(summary["classification_counts"], {"missing_reference": 1})
        self.assertEqual(
            summary["missing_reference_rows"],
            [
                {
                    "engine": "ax_engine_mlx",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "classification": "missing_reference",
                }
            ],
        )
        self.assertFalse(summary["publication_candidate"])

    def test_ax_only_refresh_regression_blocks_duplicate_reference_rows(self) -> None:
        row = {
            "engine": "ax_engine_mlx",
            "prompt_tokens": 128,
            "generation_tokens": 128,
            "decode_tok_s": {"median": 20.0},
        }
        summary = bench.summarize_ax_only_refresh_regression(
            results=[row],
            reference_doc={"results": [row, dict(row)]},
        )

        self.assertEqual(summary["duplicate_reference_count"], 1)
        self.assertEqual(
            summary["classification_counts"],
            {"duplicate_reference": 1, "within_tolerance": 1},
        )
        self.assertEqual(
            summary["duplicate_reference_rows"],
            [
                {
                    "engine": "ax_engine_mlx",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "classification": "duplicate_reference",
                }
            ],
        )
        self.assertFalse(summary["publication_candidate"])

    def test_ax_only_refresh_regression_blocks_duplicate_current_rows(self) -> None:
        row = {
            "engine": "ax_engine_mlx",
            "prompt_tokens": 128,
            "generation_tokens": 128,
            "decode_tok_s": {"median": 20.0},
        }
        summary = bench.summarize_ax_only_refresh_regression(
            results=[row, dict(row)],
            reference_doc={"results": [row]},
        )

        self.assertEqual(summary["duplicate_current_count"], 1)
        self.assertEqual(
            summary["classification_counts"],
            {"duplicate_current": 1, "within_tolerance": 1},
        )
        self.assertEqual(
            summary["duplicate_current_rows"],
            [
                {
                    "engine": "ax_engine_mlx",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "classification": "duplicate_current",
                }
            ],
        )
        self.assertFalse(summary["publication_candidate"])

    def test_prefix_reuse_evidence_classifies_absent_and_partial_coverage(self) -> None:
        self.assertEqual(
            bench.summarize_prefix_reuse_evidence([])["physical_snapshot_coverage"],
            "none_observed",
        )
        self.assertEqual(
            bench.summarize_prefix_reuse_evidence(
                [
                    {
                        "engine": "ax_engine_mlx",
                        "ax_mlx_telemetry": {
                            "ax_mlx_prefix_cache_hits": 1,
                        },
                    }
                ]
            )["physical_snapshot_coverage"],
            "hit_only",
        )
        miss_warmup = bench.summarize_prefix_reuse_evidence(
            [
                {
                    "engine": "ax_engine_mlx",
                    "ax_mlx_telemetry": {
                        "ax_mlx_prefix_cache_misses": 1,
                        "ax_mlx_prefix_cache_warmup_tokens": 16,
                    },
                }
            ]
        )
        self.assertEqual(miss_warmup["physical_snapshot_coverage"], "miss_warmup_only")
        self.assertTrue(miss_warmup["physical_snapshot_miss_warmup_observed"])
        blocked = bench.summarize_prefix_reuse_evidence(
            [
                {
                    "engine": "ax_engine_mlx",
                    "ax_mlx_telemetry": {
                        "ax_mlx_prefix_cache_blocked": 2,
                        "ax_mlx_prefix_cache_blocked_unsupported_layout": 1,
                    },
                }
            ]
        )
        self.assertEqual(blocked["physical_snapshot_coverage"], "blocked_only")
        self.assertEqual(blocked["blocked_reason_count"], 1)
        self.assertEqual(blocked["blocked_reason_accounting_gap_count"], 1)

    def test_ax_mlx_gemma4_moe_profile_is_extracted_and_summarized(self) -> None:
        profile = bench.extract_ax_mlx_gemma4_moe_profile(
            {
                "crossover_decisions": {
                    "ax_mlx_gemma4_moe_profile_enabled": 1,
                    "ax_mlx_gemma4_moe_profile_decode_layers": 2,
                    "ax_mlx_gemma4_moe_profile_topk_selections": 16,
                    "ax_mlx_gemma4_moe_profile_unsorted_gather_layers": 2,
                    "ax_mlx_gemma4_moe_profile_attention_wall_us": 100,
                    "ax_mlx_gemma4_moe_profile_dense_wall_us": 80,
                    "ax_mlx_gemma4_moe_profile_router_wall_us": 30,
                    "ax_mlx_gemma4_moe_profile_expert_wall_us": 90,
                    "ax_mlx_gemma4_moe_profile_post_wall_us": 20,
                    "unrelated": 99,
                }
            }
        )

        self.assertEqual(profile["ax_mlx_gemma4_moe_profile_enabled"], 1)
        self.assertEqual(profile["ax_mlx_gemma4_moe_profile_decode_layers"], 2)
        self.assertEqual(profile["ax_mlx_gemma4_moe_profile_topk_selections"], 16)
        self.assertEqual(profile["ax_mlx_gemma4_moe_profile_sorted_gather_layers"], 0)
        self.assertEqual(profile["ax_mlx_gemma4_moe_profile_unsorted_gather_layers"], 2)
        self.assertNotIn("unrelated", profile)

        summary = bench.summarize_ax_mlx_gemma4_moe_profile(
            [
                {"ax_mlx_gemma4_moe_profile": profile},
                {
                    "ax_mlx_gemma4_moe_profile": {
                        "ax_mlx_gemma4_moe_profile_enabled": 1,
                        "ax_mlx_gemma4_moe_profile_decode_layers": 3,
                        "ax_mlx_gemma4_moe_profile_topk_selections": 24,
                        "ax_mlx_gemma4_moe_profile_attention_wall_us": 150,
                    }
                },
            ]
        )
        self.assertEqual(summary["ax_mlx_gemma4_moe_profile_enabled"], 1)
        self.assertEqual(summary["ax_mlx_gemma4_moe_profile_decode_layers"], 5)
        self.assertEqual(summary["ax_mlx_gemma4_moe_profile_topk_selections"], 40)
        self.assertEqual(summary["ax_mlx_gemma4_moe_profile_attention_wall_us"], 250)

    def test_ax_mlx_gemma4_assistant_mtp_is_extracted_and_summarized(self) -> None:
        telemetry = bench.extract_ax_mlx_gemma4_assistant_mtp(
            {
                "crossover_decisions": {
                    "ax_mlx_gemma4_assistant_mtp_configured": 1,
                    "ax_mlx_gemma4_assistant_mtp_validated": 1,
                    "ax_mlx_gemma4_assistant_mtp_enabled": 1,
                    "ax_mlx_gemma4_assistant_mtp_attach_failed": 0,
                    "ax_mlx_gemma4_assistant_mtp_disable_reason": 0,
                    "ax_mlx_gemma4_assistant_mtp_depth": 1,
                    "ax_mlx_gemma4_assistant_mtp_confidence_mode": 1,
                    "ax_mlx_gemma4_assistant_mtp_draft_tokens": 4,
                    "ax_mlx_gemma4_assistant_mtp_accepted_tokens": 3,
                    "ax_mlx_gemma4_assistant_mtp_rejected_tokens": 1,
                    "ax_mlx_gemma4_assistant_mtp_corrections": 1,
                    "ax_mlx_gemma4_assistant_mtp_accept_rate_x1000": 750,
                    "ax_mlx_gemma4_assistant_mtp_verify_forward_wall_us": 100,
                    "ax_mlx_gemma4_assistant_mtp_verify_eval_wall_us": 40,
                    "ax_mlx_gemma4_assistant_mtp_draft_forward_wall_us": 80,
                    "unrelated": 99,
                }
            }
        )

        self.assertEqual(telemetry["ax_mlx_gemma4_assistant_mtp_enabled"], 1)
        self.assertEqual(telemetry["ax_mlx_gemma4_assistant_mtp_confidence_mode"], 1)
        self.assertEqual(telemetry["ax_mlx_gemma4_assistant_mtp_draft_tokens"], 4)
        self.assertNotIn("unrelated", telemetry)

        summary = bench.summarize_ax_mlx_gemma4_assistant_mtp(
            [
                {"ax_mlx_gemma4_assistant_mtp": telemetry},
                {
                    "ax_mlx_gemma4_assistant_mtp": {
                        "ax_mlx_gemma4_assistant_mtp_enabled": 1,
                        "ax_mlx_gemma4_assistant_mtp_depth": 1,
                        "ax_mlx_gemma4_assistant_mtp_confidence_mode": 1,
                        "ax_mlx_gemma4_assistant_mtp_draft_tokens": 2,
                        "ax_mlx_gemma4_assistant_mtp_accepted_tokens": 1,
                        "ax_mlx_gemma4_assistant_mtp_rejected_tokens": 1,
                        "ax_mlx_gemma4_assistant_mtp_corrections": 1,
                        "ax_mlx_gemma4_assistant_mtp_verify_forward_wall_us": 50,
                        "ax_mlx_gemma4_assistant_mtp_verify_eval_wall_us": 20,
                        "ax_mlx_gemma4_assistant_mtp_draft_forward_wall_us": 60,
                    }
                },
            ]
        )
        self.assertEqual(summary["ax_mlx_gemma4_assistant_mtp_enabled"], 1)
        self.assertEqual(summary["ax_mlx_gemma4_assistant_mtp_depth"], 1)
        self.assertEqual(summary["ax_mlx_gemma4_assistant_mtp_confidence_mode"], 1)
        self.assertEqual(summary["ax_mlx_gemma4_assistant_mtp_draft_tokens"], 6)
        self.assertEqual(summary["ax_mlx_gemma4_assistant_mtp_accepted_tokens"], 4)
        self.assertEqual(summary["ax_mlx_gemma4_assistant_mtp_rejected_tokens"], 2)
        self.assertEqual(summary["ax_mlx_gemma4_assistant_mtp_corrections"], 2)
        self.assertEqual(summary["ax_mlx_gemma4_assistant_mtp_accept_rate_x1000"], 666)

    def test_ax_mlx_linear_attention_profile_is_extracted_and_summarized(self) -> None:
        profile = bench.extract_ax_mlx_linear_attention_profile(
            {
                "crossover_decisions": {
                    "ax_mlx_linear_attention_profile_enabled": 1,
                    "ax_mlx_linear_attention_profile_layers": 2,
                    "ax_mlx_linear_attention_profile_tokens": 1024,
                    "ax_mlx_linear_attention_profile_projection_wall_us": 100,
                    "ax_mlx_linear_attention_profile_conv_wall_us": 80,
                    "ax_mlx_linear_attention_profile_qk_norm_wall_us": 30,
                    "ax_mlx_linear_attention_profile_recurrent_wall_us": 90,
                    "ax_mlx_linear_attention_profile_output_wall_us": 20,
                    "unrelated": 99,
                }
            }
        )

        self.assertEqual(profile["ax_mlx_linear_attention_profile_enabled"], 1)
        self.assertEqual(profile["ax_mlx_linear_attention_profile_layers"], 2)
        self.assertEqual(profile["ax_mlx_linear_attention_profile_tokens"], 1024)
        self.assertEqual(
            profile["ax_mlx_linear_attention_profile_recurrent_wall_us"],
            90,
        )
        self.assertNotIn("unrelated", profile)

        summary = bench.summarize_ax_mlx_linear_attention_profile(
            [
                {"ax_mlx_linear_attention_profile": profile},
                {
                    "ax_mlx_linear_attention_profile": {
                        "ax_mlx_linear_attention_profile_enabled": 1,
                        "ax_mlx_linear_attention_profile_layers": 3,
                        "ax_mlx_linear_attention_profile_tokens": 2048,
                        "ax_mlx_linear_attention_profile_recurrent_wall_us": 135,
                    }
                },
            ]
        )
        self.assertEqual(summary["ax_mlx_linear_attention_profile_enabled"], 1)
        self.assertEqual(summary["ax_mlx_linear_attention_profile_layers"], 5)
        self.assertEqual(summary["ax_mlx_linear_attention_profile_tokens"], 3072)
        self.assertEqual(
            summary["ax_mlx_linear_attention_profile_recurrent_wall_us"],
            225,
        )

    def test_ax_mlx_decode_profile_is_extracted_and_summarized(self) -> None:
        profile = bench.extract_ax_mlx_decode_profile(
            {
                "crossover_decisions": {
                    "ax_mlx_decode_profile_enabled": 1,
                    "ax_mlx_decode_profile_decode_steps": 2,
                    "ax_mlx_decode_profile_layers": 84,
                    "ax_mlx_decode_profile_per_layer_input_wall_us": 100,
                    "ax_mlx_decode_profile_pre_sdpa_wall_us": 200,
                    "ax_mlx_decode_profile_pre_sdpa_qkv_proj_wall_us": 80,
                    "ax_mlx_decode_profile_pre_sdpa_qk_norm_wall_us": 30,
                    "ax_mlx_decode_profile_pre_sdpa_rope_kv_wall_us": 70,
                    "ax_mlx_decode_profile_sdpa_wall_us": 300,
                    "ax_mlx_decode_profile_post_attn_wall_us": 400,
                    "ax_mlx_decode_profile_post_attn_ffn_wall_us": 250,
                    "ax_mlx_decode_profile_post_attn_output_proj_wall_us": 60,
                    "ax_mlx_decode_profile_post_attn_residual_norm_wall_us": 40,
                    "ax_mlx_decode_profile_post_attn_residual_gate_wall_us": 50,
                    "ax_mlx_decode_profile_lm_head_wall_us": 50,
                    "unrelated": 99,
                }
            }
        )

        self.assertEqual(profile["ax_mlx_decode_profile_enabled"], 1)
        self.assertEqual(profile["ax_mlx_decode_profile_decode_steps"], 2)
        self.assertEqual(profile["ax_mlx_decode_profile_layers"], 84)
        self.assertEqual(
            profile["ax_mlx_decode_profile_per_layer_input_wall_us"],
            100,
        )
        self.assertNotIn("unrelated", profile)

        summary = bench.summarize_ax_mlx_decode_profile(
            [
                {"ax_mlx_decode_profile": profile},
                {
                    "ax_mlx_decode_profile": {
                        "ax_mlx_decode_profile_enabled": 1,
                        "ax_mlx_decode_profile_decode_steps": 3,
                        "ax_mlx_decode_profile_layers": 126,
                        "ax_mlx_decode_profile_per_layer_input_wall_us": 150,
                        "ax_mlx_decode_profile_post_attn_wall_us": 600,
                        "ax_mlx_decode_profile_post_attn_residual_gate_wall_us": 75,
                    }
                },
            ]
        )
        self.assertEqual(summary["ax_mlx_decode_profile_enabled"], 1)
        self.assertEqual(summary["ax_mlx_decode_profile_decode_steps"], 5)
        self.assertEqual(summary["ax_mlx_decode_profile_layers"], 210)
        self.assertEqual(
            summary["ax_mlx_decode_profile_per_layer_input_wall_us"],
            250,
        )
        self.assertEqual(summary["ax_mlx_decode_profile_post_attn_wall_us"], 1000)
        self.assertEqual(
            summary["ax_mlx_decode_profile_post_attn_residual_gate_wall_us"],
            125,
        )

    def test_ax_mlx_kv_compression_telemetry_is_extracted_and_summarized(self) -> None:
        telemetry = bench.extract_ax_mlx_kv_compression_telemetry(
            {
                "crossover_decisions": {
                    "ax_mlx_kv_compression_request_snapshots": 1,
                    "ax_mlx_kv_compression_status": 2,
                    "ax_mlx_kv_compression_preset": 1,
                    "ax_mlx_kv_compression_key_bits": 8,
                    "ax_mlx_kv_compression_value_bits": 4,
                    "ax_mlx_kv_compression_candidate_token_layers": 100,
                    "ax_mlx_kv_compression_estimated_saved_kib": 20,
                    "ax_mlx_kv_compression_route_metadata_schema": 1,
                    "ax_mlx_kv_compression_production_ready": 0,
                    "ax_mlx_kv_compression_production_blockers": 1,
                    "ax_mlx_kv_compression_runtime_storage_written_slots": 50,
                    "ax_mlx_kv_compression_shadow_sync_calls": 1,
                    "ax_mlx_kv_compression_shadow_sync_wall_us": 1234,
                    "ax_mlx_kv_compression_decode_path": 1,
                    "ax_mlx_kv_compression_fused_decode_candidates": 1,
                    "ax_mlx_kv_compression_fused_decode_attempts": 0,
                    "ax_mlx_kv_compression_fused_decode_successes": 0,
                    "ax_mlx_kv_compression_fused_decode_metal_successes": 0,
                    "ax_mlx_kv_compression_fused_decode_fallbacks": 0,
                    "ax_mlx_kv_compression_fused_decode_fallback_reason": 1,
                    "ax_mlx_kv_compression_fused_decode_blocked_attention_kind": 2,
                    "ax_mlx_kv_compression_fused_decode_blocked_linear_attention": 2,
                    "ax_mlx_kv_compression_fused_decode_blocked_missing_storage": 1,
                    "unrelated": 99,
                }
            }
        )

        self.assertEqual(telemetry["ax_mlx_kv_compression_preset"], 1)
        self.assertEqual(telemetry["ax_mlx_kv_compression_key_bits"], 8)
        self.assertEqual(
            telemetry["ax_mlx_kv_compression_runtime_storage_written_slots"],
            50,
        )
        self.assertEqual(telemetry["ax_mlx_kv_compression_shadow_sync_calls"], 1)
        self.assertEqual(telemetry["ax_mlx_kv_compression_shadow_sync_wall_us"], 1234)
        self.assertEqual(telemetry["ax_mlx_kv_compression_decode_path"], 1)
        self.assertEqual(telemetry["ax_mlx_kv_compression_fused_decode_candidates"], 1)
        self.assertEqual(telemetry["ax_mlx_kv_compression_fused_decode_attempts"], 0)
        self.assertEqual(telemetry["ax_mlx_kv_compression_fused_decode_fallbacks"], 0)
        self.assertEqual(
            telemetry["ax_mlx_kv_compression_fused_decode_fallback_reason"], 1
        )
        self.assertEqual(
            telemetry["ax_mlx_kv_compression_fused_decode_blocked_attention_kind"],
            2,
        )
        self.assertEqual(
            telemetry["ax_mlx_kv_compression_fused_decode_blocked_linear_attention"],
            2,
        )
        self.assertEqual(
            telemetry["ax_mlx_kv_compression_fused_decode_blocked_missing_storage"],
            1,
        )
        self.assertNotIn("unrelated", telemetry)

        summary = bench.summarize_ax_mlx_kv_compression_telemetry(
            [
                {"kv_compression_telemetry": telemetry},
                {
                    "kv_compression_telemetry": {
                        "ax_mlx_kv_compression_request_snapshots": 1,
                        "ax_mlx_kv_compression_status": 2,
                        "ax_mlx_kv_compression_preset": 1,
                        "ax_mlx_kv_compression_runtime_storage_written_slots": 75,
                        "ax_mlx_kv_compression_shadow_sync_calls": 1,
                        "ax_mlx_kv_compression_shadow_sync_wall_us": 4321,
                        "ax_mlx_kv_compression_decode_path": 1,
                        "ax_mlx_kv_compression_fused_decode_candidates": 1,
                        "ax_mlx_kv_compression_fused_decode_attempts": 0,
                        "ax_mlx_kv_compression_fused_decode_successes": 0,
                        "ax_mlx_kv_compression_fused_decode_metal_successes": 0,
                        "ax_mlx_kv_compression_fused_decode_fallbacks": 0,
                        "ax_mlx_kv_compression_fused_decode_fallback_reason": 1,
                        "ax_mlx_kv_compression_fused_decode_blocked_attention_kind": 3,
                        "ax_mlx_kv_compression_fused_decode_blocked_linear_attention": 1,
                        "ax_mlx_kv_compression_fused_decode_blocked_sliding_window": 2,
                        "ax_mlx_kv_compression_fused_decode_blocked_unsupported_head_dim": 4,
                    }
                },
            ]
        )

        self.assertEqual(summary["ax_mlx_kv_compression_request_snapshots"], 2)
        self.assertEqual(summary["ax_mlx_kv_compression_status"], 2)
        self.assertEqual(summary["ax_mlx_kv_compression_preset"], 1)
        self.assertEqual(
            summary["ax_mlx_kv_compression_runtime_storage_written_slots"],
            125,
        )
        self.assertEqual(summary["ax_mlx_kv_compression_shadow_sync_calls"], 2)
        self.assertEqual(summary["ax_mlx_kv_compression_shadow_sync_wall_us"], 5555)
        self.assertEqual(summary["ax_mlx_kv_compression_decode_path"], 1)
        self.assertEqual(summary["ax_mlx_kv_compression_fused_decode_candidates"], 2)
        self.assertEqual(summary["ax_mlx_kv_compression_fused_decode_attempts"], 0)
        self.assertEqual(summary["ax_mlx_kv_compression_fused_decode_successes"], 0)
        self.assertEqual(
            summary["ax_mlx_kv_compression_fused_decode_metal_successes"], 0
        )
        self.assertEqual(summary["ax_mlx_kv_compression_fused_decode_fallbacks"], 0)
        self.assertEqual(
            summary["ax_mlx_kv_compression_fused_decode_fallback_reason"], 1
        )
        self.assertEqual(
            summary["ax_mlx_kv_compression_fused_decode_blocked_attention_kind"],
            5,
        )
        self.assertEqual(
            summary["ax_mlx_kv_compression_fused_decode_blocked_linear_attention"],
            3,
        )
        self.assertEqual(
            summary["ax_mlx_kv_compression_fused_decode_blocked_sliding_window"],
            2,
        )
        self.assertEqual(
            summary["ax_mlx_kv_compression_fused_decode_blocked_missing_storage"],
            1,
        )
        self.assertEqual(
            summary["ax_mlx_kv_compression_fused_decode_blocked_unsupported_head_dim"],
            4,
        )
        self.assertEqual(
            bench.kv_compression_fused_decode_blocked_summary(summary),
            {
                "total": 10,
                "reasons": [
                    "attention_kind",
                    "unsupported_head_dim",
                    "missing_storage",
                ],
                "counters": {
                    "prefill_only": 0,
                    "attention_kind": 5,
                    "ineligible_layer": 0,
                    "unsupported_preset": 0,
                    "unsupported_head_dim": 4,
                    "gqa": 0,
                    "missing_storage": 1,
                },
            },
        )
        self.assertEqual(
            bench.kv_compression_fused_decode_blocked_attention_kind_summary(summary),
            {
                "total": 5,
                "reasons": [
                    "linear_attention",
                    "sliding_window",
                ],
                "counters": {
                    "linear_attention": 3,
                    "sliding_window": 2,
                    "kv_shared": 0,
                },
            },
        )
        self.assertEqual(
            bench.kv_compression_decode_path_label(summary),
            "full_precision_shadow",
        )
        self.assertEqual(
            bench.kv_compression_fused_decode_fallback_reason_label(summary),
            "shadow_only",
        )

        summary["ax_mlx_kv_compression_decode_path"] = 2
        self.assertEqual(
            bench.kv_compression_decode_path_label(summary),
            "fused_compressed_decode",
        )
        summary["ax_mlx_kv_compression_decode_path"] = 3
        self.assertEqual(
            bench.kv_compression_decode_path_label(summary),
            "cpu_oracle_compressed_decode",
        )
        summary["ax_mlx_kv_compression_fused_decode_fallback_reason"] = 4
        self.assertEqual(
            bench.kv_compression_fused_decode_fallback_reason_label(summary),
            "runner_not_integrated",
        )
        summary["ax_mlx_kv_compression_fused_decode_fallback_reason"] = 5
        self.assertEqual(
            bench.kv_compression_fused_decode_fallback_reason_label(summary),
            "cpu_oracle_unavailable",
        )

    def test_absent_ax_mlx_kv_compression_telemetry_stays_silent(self) -> None:
        self.assertEqual(
            bench.extract_ax_mlx_kv_compression_telemetry(
                {"crossover_decisions": {"ax_mlx_decode_steps": 2}}
            ),
            {},
        )

    def test_absent_ax_mlx_gemma4_moe_profile_stays_silent(self) -> None:
        self.assertEqual(
            bench.extract_ax_mlx_gemma4_moe_profile(
                {"crossover_decisions": {"ax_mlx_decode_steps": 2}}
            ),
            {},
        )

    def test_absent_ax_mlx_gemma4_assistant_mtp_stays_silent(self) -> None:
        self.assertEqual(
            bench.extract_ax_mlx_gemma4_assistant_mtp(
                {"crossover_decisions": {"ax_mlx_decode_steps": 2}}
            ),
            {},
        )

    def test_absent_ax_mlx_linear_attention_profile_stays_silent(self) -> None:
        self.assertEqual(
            bench.extract_ax_mlx_linear_attention_profile(
                {"crossover_decisions": {"ax_mlx_decode_steps": 2}}
            ),
            {},
        )

    def test_absent_ax_mlx_decode_profile_stays_silent(self) -> None:
        self.assertEqual(
            bench.extract_ax_mlx_decode_profile(
                {"crossover_decisions": {"ax_mlx_decode_steps": 2}}
            ),
            {},
        )

    def test_ax_step_timing_classifies_chunked_prefill_by_scheduled_tokens(
        self,
    ) -> None:
        self.assertTrue(
            bench.is_ax_prefill_step({"scheduled_tokens": 1}, seen_prefill=False)
        )
        self.assertTrue(
            bench.is_ax_prefill_step({"scheduled_tokens": 2048}, seen_prefill=True)
        )
        self.assertFalse(
            bench.is_ax_prefill_step({"scheduled_tokens": 1}, seen_prefill=True)
        )

    def test_ax_step_timing_prefers_route_metadata_over_token_heuristic(self) -> None:
        self.assertTrue(
            bench.is_ax_prefill_step(
                {
                    "scheduled_tokens": 1,
                    "route": {"execution_plan": "phase1.qwen3.dense_prefill"},
                },
                seen_prefill=True,
            )
        )
        self.assertFalse(
            bench.is_ax_prefill_step(
                {
                    "scheduled_tokens": 2048,
                    "route": {"attention_route": "qwen3_paged_decode"},
                },
                seen_prefill=False,
            )
        )

    def test_ax_sse_parser_resets_event_after_each_frame(self) -> None:
        events = list(
            bench.iter_sse_json_events_from_lines(
                [
                    "event: step\n",
                    'data: {"step":{"runner_time_us":100,"scheduled_tokens":1}}\n',
                    "\n",
                    'data: {"response":{"output_tokens":[42]}}\n',
                    "\n",
                ]
            )
        )

        self.assertEqual(events[0][0], "step")
        self.assertEqual(events[1][0], "")
        self.assertIn("response", events[1][1])

    def test_ax_sse_parser_supports_multiline_data_without_stale_event(self) -> None:
        events = list(
            bench.iter_sse_json_events_from_lines(
                [
                    "event: response\n",
                    'data: {"response":\n',
                    'data: {"output_tokens":[7]}}\n',
                    "\n",
                    "data: [DONE]\n",
                    "\n",
                ]
            )
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "response")
        self.assertEqual(events[0][1]["response"]["output_tokens"], [7])

    def test_axengine_command_can_enable_experimental_kv_compression(self) -> None:
        with (
            patch.object(bench, "ensure_port_available") as ensure_port_available,
            patch.object(bench.subprocess, "Popen") as popen,
        ):
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                model_id="test-model",
                direct_mode=True,
                kv_compression="turboquant-shadow",
                kv_compression_hot_window_tokens=128,
                kv_compression_min_context_tokens=1024,
            )

        ensure_port_available.assert_called_once_with(19091)
        command = popen.call_args.args[0]
        self.assertIn("--disable-ngram-acceleration", command)
        self.assertIn("--experimental-mlx-kv-compression", command)
        self.assertIn("turboquant-shadow", command)
        self.assertIn("--experimental-mlx-kv-compression-hot-window-tokens", command)
        self.assertIn("--experimental-mlx-kv-compression-min-context-tokens", command)

    def test_axengine_command_can_request_fused_experimental_kv_compression(
        self,
    ) -> None:
        with (
            patch.object(bench, "ensure_port_available"),
            patch.object(bench.subprocess, "Popen") as popen,
        ):
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                model_id="test-model",
                direct_mode=True,
                kv_compression="turboquant-fused-experimental",
            )

        command = popen.call_args.args[0]
        self.assertIn("--experimental-mlx-kv-compression", command)
        self.assertIn("turboquant-fused-experimental", command)

    def test_axengine_command_can_enable_gemma4_moe_profile(self) -> None:
        with (
            patch.object(bench, "ensure_port_available"),
            patch.object(bench.subprocess, "Popen") as popen,
        ):
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                model_id="test-model",
                direct_mode=True,
                gemma4_moe_profile=True,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_GEMMA4_MOE_PROFILE"], "1")

    def test_axengine_command_can_enable_linear_attention_profile(self) -> None:
        with (
            patch.object(bench, "ensure_port_available"),
            patch.object(bench.subprocess, "Popen") as popen,
        ):
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                model_id="test-model",
                direct_mode=True,
                linear_attention_profile=True,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_LINEAR_ATTENTION_PROFILE"], "1")

    def test_axengine_command_can_enable_decode_profile(self) -> None:
        with (
            patch.object(bench, "ensure_port_available"),
            patch.object(bench.subprocess, "Popen") as popen,
        ):
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                model_id="test-model",
                direct_mode=True,
                decode_profile=True,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_DECODE_PROFILE"], "1")

    def test_axengine_command_can_enable_linear_attention_projection_pack(self) -> None:
        with (
            patch.object(bench, "ensure_port_available"),
            patch.object(bench.subprocess, "Popen") as popen,
        ):
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                model_id="test-model",
                direct_mode=True,
                pack_linear_attention_projections=True,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS"], "1")

    def test_axengine_command_can_enable_qwen_dense_ffn_matvec_metal(self) -> None:
        with (
            patch.object(bench, "ensure_port_available"),
            patch.object(bench.subprocess, "Popen") as popen,
        ):
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                model_id="test-model",
                direct_mode=True,
                qwen_dense_ffn_gate_up_matvec_metal=True,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL"], "1")

    def test_axengine_command_can_enable_gemma4_assistant_mtp(self) -> None:
        with (
            patch.object(bench, "ensure_port_available"),
            patch.object(bench.subprocess, "Popen") as popen,
        ):
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                model_id="test-model",
                direct_mode=False,
                gemma4_assistant_mtp=True,
                mtp_max_depth=1,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_GEMMA4_ASSISTANT_MTP"], "1")
        self.assertEqual(env["AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH"], "1")

    def test_axengine_command_can_enable_direct_linear_attention_routes(self) -> None:
        with (
            patch.object(bench, "ensure_port_available"),
            patch.object(bench.subprocess, "Popen") as popen,
        ):
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                model_id="test-model",
                direct_mode=True,
                direct_linear_attention_inputs_route=True,
                direct_linear_attention_post_input_route=True,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS"], "1")
        self.assertEqual(env["AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT"], "1")

    def direct_linear_attention_post_input_route_compare_args(
        self, **overrides: bool
    ) -> argparse.Namespace:
        values = {
            "ax_compare_direct_linear_attention_post_input_route": True,
            "skip_ax_engine": False,
            "ax_ngram_accel": False,
            "ax_compare_policies": False,
            "ax_compare_linear_attention_projection_pack": False,
            "ax_compare_dense_ffn_gate_up_pack": False,
            "gateddelta_prefill_profile": False,
            "ax_linear_attention_profile": False,
            "ax_prefill_profile": False,
            "ax_decode_profile": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_direct_linear_attention_post_input_compare_accepts_direct_ax_rows(
        self,
    ) -> None:
        bench.validate_direct_linear_attention_post_input_route_compare_args(
            self.direct_linear_attention_post_input_route_compare_args()
        )

    def test_direct_linear_attention_post_input_compare_rejects_missing_ax_rows(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "requires AX rows"):
            bench.validate_direct_linear_attention_post_input_route_compare_args(
                self.direct_linear_attention_post_input_route_compare_args(
                    skip_ax_engine=True
                )
            )

    def test_direct_linear_attention_post_input_compare_rejects_non_direct_ax_modes(
        self,
    ) -> None:
        for option in ("ax_ngram_accel", "ax_compare_policies"):
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "requires direct AX rows"):
                    bench.validate_direct_linear_attention_post_input_route_compare_args(
                        self.direct_linear_attention_post_input_route_compare_args(
                            **{option: True}
                        )
                    )

    def test_direct_linear_attention_post_input_compare_rejects_other_paired_modes(
        self,
    ) -> None:
        for option in (
            "ax_compare_linear_attention_projection_pack",
            "ax_compare_dense_ffn_gate_up_pack",
        ):
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "run one comparison at a time"):
                    bench.validate_direct_linear_attention_post_input_route_compare_args(
                        self.direct_linear_attention_post_input_route_compare_args(
                            **{option: True}
                        )
                    )

    def test_direct_linear_attention_post_input_compare_rejects_profile_modes(
        self,
    ) -> None:
        for option in (
            "gateddelta_prefill_profile",
            "ax_linear_attention_profile",
            "ax_prefill_profile",
            "ax_decode_profile",
        ):
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "profil"):
                    bench.validate_direct_linear_attention_post_input_route_compare_args(
                        self.direct_linear_attention_post_input_route_compare_args(
                            **{option: True}
                        )
                    )

    def test_linear_attention_pack_compare_enables_profile_gate(self) -> None:
        args = argparse.Namespace(
            gateddelta_prefill_profile=False,
            ax_linear_attention_profile=False,
            ax_compare_linear_attention_projection_pack=True,
        )

        self.assertTrue(bench.ax_linear_attention_profile_enabled(args))

    def test_axengine_command_disables_prefix_cache_by_default(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "AX_MLX_PREFIX_CACHE_MAX_BYTES": "268435456",
                    "AX_MLX_PREFIX_CACHE_MAX_ENTRIES": "64",
                    "AX_MLX_PREFIX_CACHE_DISK_DISABLED": "0",
                },
                clear=True,
            ),
            patch.object(bench, "ensure_port_available"),
            patch.object(bench.subprocess, "Popen") as popen,
        ):
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                model_id="test-model",
                direct_mode=True,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_NATIVE_CONFIRM"], "1")
        self.assertEqual(env["AX_MLX_PREFIX_CACHE_MAX_BYTES"], "0")
        self.assertEqual(env["AX_MLX_PREFIX_CACHE_MAX_ENTRIES"], "0")
        self.assertEqual(env["AX_MLX_PREFIX_CACHE_DISK_DISABLED"], "1")

    def test_axengine_command_can_keep_prefix_cache_enabled(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(bench, "ensure_port_available"),
            patch.object(bench.subprocess, "Popen") as popen,
        ):
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                model_id="test-model",
                direct_mode=True,
                prefix_cache_enabled=True,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_NATIVE_CONFIRM"], "1")
        self.assertNotIn("AX_MLX_PREFIX_CACHE_MAX_BYTES", env)
        self.assertNotIn("AX_MLX_PREFIX_CACHE_MAX_ENTRIES", env)
        self.assertNotIn("AX_MLX_PREFIX_CACHE_DISK_DISABLED", env)

    def test_summarize_runs_ignores_invalid_none_values(self) -> None:
        partial = bench.summarize_runs(
            [
                {"prefill_tok_s": None},
                {"prefill_tok_s": 100.0},
                {"prefill_tok_s": 120.0},
            ],
            "prefill_tok_s",
        )
        self.assertEqual(partial["median"], 110.0)

        empty = bench.summarize_runs(
            [{"prefill_tok_s": None}, {"prefill_tok_s": None}],
            "prefill_tok_s",
        )
        self.assertEqual(
            empty,
            {"mean": None, "median": None, "min": None, "max": None},
        )

    def test_route_with_more_decisions_keeps_step_telemetry_over_response_route(
        self,
    ) -> None:
        step_route = {
            "attention_route": "qwen_paged_decode",
            "crossover_decisions": {
                "ax_ngram_draft_attempts": 3,
                "ax_ngram_accepted_tokens": 6,
            },
        }
        response_route = {
            "attention_route": "qwen_paged_decode",
            "crossover_decisions": {},
        }

        self.assertIs(bench.route_with_more_decisions(step_route, None), step_route)
        self.assertIs(
            bench.route_with_more_decisions(response_route, step_route),
            step_route,
        )

    def test_route_with_more_decisions_prefers_nonzero_counters_on_equal_keys(
        self,
    ) -> None:
        prefill_route = {
            "crossover_decisions": {
                "ax_ngram_draft_attempts": 0,
                "ax_ngram_accepted_tokens": 0,
            },
        }
        decode_route = {
            "crossover_decisions": {
                "ax_ngram_draft_attempts": 2,
                "ax_ngram_accepted_tokens": 7,
            },
        }

        self.assertIs(
            bench.route_with_more_decisions(decode_route, prefill_route),
            decode_route,
        )

    def test_route_with_more_decisions_prefers_decode_signals_over_prefill_totals(
        self,
    ) -> None:
        prefill_route = {
            "crossover_decisions": {
                "ax_scheduler_scheduled_prefill_tokens": 512,
                "ax_mlx_prefill_steps": 1,
                "ax_mlx_prefix_cache_blocked": 1,
                "ax_ngram_draft_attempts": 0,
                "ax_ngram_policy_variant": 0,
            },
        }
        decode_route = {
            "crossover_decisions": {
                "ax_scheduler_scheduled_prefill_tokens": 0,
                "ax_mlx_ngram_decode_steps": 18,
                "ax_ngram_draft_attempts": 18,
                "ax_ngram_accepted_tokens": 108,
                "ax_ngram_policy_variant": 1,
            },
        }

        self.assertIs(
            bench.route_with_more_decisions(decode_route, prefill_route),
            decode_route,
        )

    def test_step_local_route_decisions_are_merged_into_selected_decode_route(
        self,
    ) -> None:
        step_local: dict[str, int] = {}
        bench.merge_step_local_route_decisions(
            step_local,
            {
                "crossover_decisions": {
                    "ax_scheduler_scheduled_prefill_tokens": 512,
                    "ax_mlx_prefix_cache_misses": 1,
                    "ax_mlx_prefix_cache_warmup_tokens": 128,
                    "ax_mlx_prefix_cache_entries": 2,
                    "ax_mlx_prefix_cache_bytes_kib": 64,
                    "ax_ngram_draft_attempts": 99,
                }
            },
        )
        bench.merge_step_local_route_decisions(
            step_local,
            {
                "crossover_decisions": {
                    "ax_scheduler_scheduled_decode_tokens": 8,
                    "ax_mlx_prefix_cache_hits": 1,
                    "ax_mlx_prefix_cache_entries": 1,
                    "ax_mlx_prefix_cache_bytes_kib": 32,
                }
            },
        )
        selected = bench.route_with_step_local_decisions(
            {
                "attention_route": "qwen_paged_decode",
                "crossover_decisions": {
                    "ax_ngram_draft_attempts": 18,
                    "ax_ngram_accepted_tokens": 108,
                    "ax_mlx_prefix_cache_misses": 0,
                },
            },
            step_local,
        )

        decisions = selected["crossover_decisions"]
        self.assertEqual(decisions["ax_ngram_draft_attempts"], 18)
        self.assertEqual(decisions["ax_ngram_accepted_tokens"], 108)
        self.assertEqual(decisions["ax_scheduler_scheduled_prefill_tokens"], 512)
        self.assertEqual(decisions["ax_scheduler_scheduled_decode_tokens"], 8)
        self.assertEqual(decisions["ax_mlx_prefix_cache_misses"], 1)
        self.assertEqual(decisions["ax_mlx_prefix_cache_hits"], 1)
        self.assertEqual(decisions["ax_mlx_prefix_cache_warmup_tokens"], 128)
        self.assertEqual(decisions["ax_mlx_prefix_cache_entries"], 2)
        self.assertEqual(decisions["ax_mlx_prefix_cache_bytes_kib"], 64)

    def test_linear_attention_profile_prefers_prefill_route(self) -> None:
        prefill_route = {
            "crossover_decisions": {
                "ax_mlx_prefill_steps": 1,
                "ax_mlx_linear_attention_profile_enabled": 1,
                "ax_mlx_linear_attention_profile_layers": 30,
                "ax_mlx_linear_attention_profile_tokens": 3840,
                "ax_mlx_linear_attention_profile_recurrent_wall_us": 5000,
            },
        }
        final_route = {
            "crossover_decisions": {
                "ax_mlx_decode_steps": 128,
                "ax_mlx_linear_attention_profile_enabled": 1,
                "ax_mlx_linear_attention_profile_layers": 30,
                "ax_mlx_linear_attention_profile_tokens": 30,
                "ax_mlx_linear_attention_profile_recurrent_wall_us": 90,
            },
        }

        selected = bench.route_for_linear_attention_profile(prefill_route, final_route)
        profile = bench.extract_ax_mlx_linear_attention_profile(selected)

        self.assertIs(selected, prefill_route)
        self.assertEqual(profile["ax_mlx_linear_attention_profile_tokens"], 3840)
        self.assertEqual(
            profile["ax_mlx_linear_attention_profile_recurrent_wall_us"],
            5000,
        )

    def test_attach_baseline_requires_matching_mlx_lm_row(self) -> None:
        results = [
            {
                "engine": "mlx_lm",
                "method": "mlx_lm.benchmark",
                "prompt_tokens": 4,
                "generation_tokens": 2,
                "prompt_contract": "mlx_lm_random_tokens_seed_0",
                "timing_scope": "upstream_mlx_lm_response_stats",
                "prefill_tok_s": {"median": 100.0},
                "decode_tok_s": {"median": 50.0},
            },
            {
                "engine": "ax_engine_mlx",
                "method": "server_sse_runner_time_us",
                "prompt_tokens": 4,
                "generation_tokens": 2,
                "prefill_tok_s": {"median": 80.0},
                "decode_tok_s": {"median": 40.0},
            },
        ]

        bench.attach_mlx_lm_baselines(results)

        self.assertEqual(results[0]["baseline"]["role"], "primary_reference")
        self.assertEqual(results[1]["baseline"]["decode_ratio_to_mlx_lm"], 0.8)

        # When some mlx_lm rows exist but none match a non-mlx_lm row's (pt, gt),
        # that's a bug -> raise.
        with self.assertRaisesRegex(RuntimeError, "missing mlx_lm.benchmark baseline"):
            bench.attach_mlx_lm_baselines(
                [
                    {
                        "engine": "mlx_lm",
                        "prompt_tokens": 4,
                        "generation_tokens": 2,
                        "prefill_tok_s": {"median": 100.0},
                        "decode_tok_s": {"median": 50.0},
                    },
                    {
                        "engine": "ax_engine_mlx",
                        "prompt_tokens": 8,
                        "generation_tokens": 2,
                        "prefill_tok_s": {"median": 80.0},
                        "decode_tok_s": {"median": 40.0},
                    },
                ]
            )

        # When NO mlx_lm rows exist (e.g. --skip-mlx-lm), do not raise;
        # mark the non-mlx_lm row's baseline as explicitly absent.
        results_no_mlx_lm = [
            {
                "engine": "llama_cpp_metal",
                "prompt_tokens": 8,
                "generation_tokens": 2,
                "prefill_tok_s": {"median": 80.0},
                "decode_tok_s": {"median": 40.0},
            }
        ]
        bench.attach_mlx_lm_baselines(results_no_mlx_lm)
        self.assertEqual(
            results_no_mlx_lm[0]["baseline"]["role"], "absent_skipped_via_cli"
        )

    def test_load_reused_reference_rows_filters_and_requires_mlx_lm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": bench.MLX_INFERENCE_STACK_SCHEMA_VERSION,
                        "results": [
                            {
                                "engine": "mlx_lm",
                                "prompt_tokens": 4,
                                "generation_tokens": 2,
                                "prefill_tok_s": {"median": 100.0},
                                "decode_tok_s": {"median": 50.0},
                            },
                            {
                                "engine": "ax_engine_mlx",
                                "prompt_tokens": 4,
                                "generation_tokens": 2,
                            },
                        ],
                    }
                )
            )

            rows, doc = bench.load_reused_reference_rows(
                path,
                prompt_lengths=[4],
                generation_tokens=2,
            )

            self.assertEqual(
                doc["schema_version"], bench.MLX_INFERENCE_STACK_SCHEMA_VERSION
            )
            self.assertEqual([row["engine"] for row in rows], ["mlx_lm"])

            with self.assertRaisesRegex(RuntimeError, "missing mlx_lm reference rows"):
                bench.load_reused_reference_rows(
                    path,
                    prompt_lengths=[8],
                    generation_tokens=2,
                )

    def test_load_reused_reference_rows_rejects_duplicate_mlx_lm_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": bench.MLX_INFERENCE_STACK_SCHEMA_VERSION,
                        "results": [
                            {
                                "engine": "mlx_lm",
                                "prompt_tokens": 4,
                                "generation_tokens": 2,
                                "prefill_tok_s": {"median": 100.0},
                                "decode_tok_s": {"median": 50.0},
                            },
                            {
                                "engine": "mlx_lm",
                                "prompt_tokens": 4,
                                "generation_tokens": 2,
                                "prefill_tok_s": {"median": 90.0},
                                "decode_tok_s": {"median": 45.0},
                            },
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(
                RuntimeError, "duplicate mlx_lm reference rows"
            ):
                bench.load_reused_reference_rows(
                    path,
                    prompt_lengths=[4],
                    generation_tokens=2,
                )

    def test_load_reused_reference_rows_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(json.dumps({"schema_version": "wrong", "results": []}))

            with self.assertRaisesRegex(
                RuntimeError, "reused reference artifact has unexpected schema_version"
            ):
                bench.load_reused_reference_rows(
                    path,
                    prompt_lengths=[4],
                    generation_tokens=2,
                )

    def test_load_reused_reference_rows_rejects_missing_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(
                json.dumps(
                    {"schema_version": bench.MLX_INFERENCE_STACK_SCHEMA_VERSION}
                )
            )

            with self.assertRaisesRegex(
                RuntimeError, "reused reference artifact lacks results list"
            ):
                bench.load_reused_reference_rows(
                    path,
                    prompt_lengths=[4],
                    generation_tokens=2,
                )

    def test_validate_reused_reference_prompt_hashes_rejects_mismatches(self) -> None:
        rows = [
            {
                "engine": "mlx_lm",
                "prompt_tokens": 4,
                "generation_tokens": 2,
                "prompt_token_ids_sha256": "old",
            }
        ]
        prompts = [
            {
                "prompt_tokens": 4,
                "generation_tokens": 2,
                "token_ids_sha256": "new",
            }
        ]

        with self.assertRaisesRegex(RuntimeError, "prompt hash mismatch"):
            bench.validate_reused_reference_prompt_hashes(rows, prompts)

    def test_bandwidth_accounting_dense_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_sparse_file(model_dir / "model-00001.safetensors", 5_000_000_000)
            write_sparse_file(model_dir / "model-00002.safetensors", 5_000_000_000)
            results = [
                {
                    "engine": "ax_engine_mlx",
                    "method": "server_sse_runner_time_us",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "decode_tok_s": {"median": 35.0},
                },
                {
                    "engine": "ax_engine_mlx_ngram_accel",
                    "method": "server_sse_runner_time_us",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "decode_tok_s": {"median": 70.0},
                },
                {
                    "engine": "mlx_lm",
                    "method": "mlx_lm.benchmark",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "decode_tok_s": {"median": 30.0},
                },
            ]
            accounting = bench.build_bandwidth_accounting(model_dir, results)

        self.assertEqual(accounting["safetensor_bytes"], 10_000_000_000)
        self.assertEqual(accounting["estimate_kind"], "dense_safetensor_total")
        self.assertEqual(accounting["bytes_used_for_estimate"], 10_000_000_000)
        self.assertIsNone(accounting["moe_block"])
        self.assertIsNone(accounting["moe_active_bytes"])
        self.assertEqual(len(accounting["per_row"]), 1)
        row = accounting["per_row"][0]
        self.assertEqual(row["engine"], "ax_engine_mlx")
        self.assertEqual(row["ax_effective_weight_bytes_per_token"], 10_000_000_000)
        self.assertAlmostEqual(row["ax_effective_bandwidth_gb_s"], 350.0, places=1)
        self.assertEqual(row["ax_bandwidth_estimate_kind"], "dense_safetensor_total")

    def test_bandwidth_accounting_moe_model_with_active_experts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "model.safetensors").write_bytes(b"\x00" * 10_000_000)
            (model_dir / "model-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ax.native_model_manifest.v1",
                        "model_family": "gemma4",
                        "moe": {"expert_count": 8, "experts_per_token": 2},
                        "tensors": [
                            {"role": "self_attn.q_proj", "length_bytes": 3_000_000},
                            {
                                "role": "block_sparse_moe._exps.weight",
                                "length_bytes": 6_000_000,
                            },
                            {
                                "role": "block_sparse_moe.shared_expert.weight",
                                "length_bytes": 1_000_000,
                            },
                        ],
                    }
                )
            )
            results = [
                {
                    "engine": "ax_engine_mlx",
                    "method": "server_sse_runner_time_us",
                    "prompt_tokens": 512,
                    "generation_tokens": 128,
                    "decode_tok_s": {"median": 40.0},
                },
            ]
            accounting = bench.build_bandwidth_accounting(model_dir, results)

        self.assertEqual(accounting["estimate_kind"], "moe_active_estimate")
        self.assertIsNotNone(accounting["moe_block"])
        non_routed = 3_000_000 + 1_000_000
        active_routed = int(6_000_000 * (2 / 8))
        self.assertEqual(accounting["moe_active_bytes"], non_routed + active_routed)
        self.assertEqual(
            accounting["bytes_used_for_estimate"], non_routed + active_routed
        )
        row = accounting["per_row"][0]
        self.assertEqual(row["ax_bandwidth_estimate_kind"], "moe_active_estimate")

    def test_bandwidth_accounting_with_peak_bandwidth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_sparse_file(model_dir / "w.safetensors", 1_000_000_000)
            results = [
                {
                    "engine": "ax_engine_mlx",
                    "method": "server_sse_runner_time_us",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "decode_tok_s": {"median": 50.0},
                },
            ]
            accounting = bench.build_bandwidth_accounting(
                model_dir,
                results,
                peak_bandwidth_gb_s=800.0,
                peak_bandwidth_source="mlx_read_calibration",
            )

        self.assertEqual(accounting["peak_bandwidth_gb_s"], 800.0)
        self.assertEqual(accounting["peak_bandwidth_source"], "mlx_read_calibration")
        self.assertEqual(
            accounting["ax_bandwidth_peak_source"], "mlx_read_calibration"
        )
        row = accounting["per_row"][0]
        self.assertIn("ax_effective_bandwidth_percent_of_peak", row)
        self.assertEqual(row["ax_bandwidth_peak_source"], "mlx_read_calibration")
        expected_pct = (50.0 / 800.0) * 100
        self.assertAlmostEqual(
            row["ax_effective_bandwidth_percent_of_peak"], expected_pct, places=1
        )

    def test_bandwidth_accounting_skips_zero_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "w.safetensors").write_bytes(b"\x00" * 1_000_000)
            results = [
                {
                    "engine": "ax_engine_mlx",
                    "method": "server_sse_runner_time_us",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "decode_tok_s": {"median": 0.0},
                },
            ]
            accounting = bench.build_bandwidth_accounting(model_dir, results)

        self.assertEqual(len(accounting["per_row"]), 0)

    def test_bandwidth_accounting_blocks_moe_without_active_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "model.safetensors").write_bytes(b"\x00" * 1_000_000)
            (model_dir / "model-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ax.native_model_manifest.v1",
                        "moe": {"expert_count": 8, "experts_per_token": 2},
                        "tensors": [
                            {"role": "self_attn.q_proj", "length_bytes": 1_000_000}
                        ],
                    }
                )
            )
            results = [
                {
                    "engine": "ax_engine_mlx",
                    "method": "server_sse_runner_time_us",
                    "prompt_tokens": 128,
                    "generation_tokens": 128,
                    "decode_tok_s": {"median": 50.0},
                }
            ]
            accounting = bench.build_bandwidth_accounting(model_dir, results)

        self.assertEqual(accounting["estimate_kind"], "not_comparable")
        self.assertIsNone(accounting["bytes_used_for_estimate"])
        row = accounting["per_row"][0]
        self.assertEqual(row["ax_bandwidth_estimate_kind"], "not_comparable")
        self.assertNotIn("ax_effective_weight_bytes_per_token", row)
        self.assertNotIn("ax_effective_bandwidth_gb_s", row)


if __name__ == "__main__":
    unittest.main()
