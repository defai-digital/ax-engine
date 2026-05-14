#!/usr/bin/env python3
"""Unit tests for the MLX inference-stack benchmark contract."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("bench_mlx_inference_stack.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_mlx_inference_stack", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
bench = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(bench)


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


class MlxInferenceStackBenchTests(unittest.TestCase):
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

    def test_swift_adapter_trial_json_is_summarized(self) -> None:
        parsed = bench.parse_swift_adapter_json(
            json.dumps(
                {
                    "trials": [
                        {"prefill_tok_s": 100.0, "decode_tok_s": 50.0},
                        {"prompt_tps": 120.0, "generation_tps": 70.0},
                    ],
                    "peak_memory_gb": 12.0,
                }
            )
        )

        self.assertEqual(parsed["prefill_tok_s"]["mean"], 110.0)
        self.assertEqual(parsed["decode_tok_s"]["median"], 60.0)
        self.assertEqual(len(parsed["trials"]), 2)

    def test_axengine_summary_includes_ttft_and_memory(self) -> None:
        runs = [
            {"prefill_s": 0.3, "decode_s": 0.1, "ttft_ms": 300.0, "prefill_tok_s": 10.0, "decode_tok_s": 20.0, "output_tokens": 3.0, "peak_memory_gb": 11.0},
            {"prefill_s": 0.2, "decode_s": 0.1, "ttft_ms": 200.0, "prefill_tok_s": 15.0, "decode_tok_s": 20.0, "output_tokens": 3.0, "peak_memory_gb": 12.0},
            {"prefill_s": 0.4, "decode_s": 0.1, "ttft_ms": 400.0, "prefill_tok_s": 8.0, "decode_tok_s": 20.0, "output_tokens": 3.0, "peak_memory_gb": 13.0},
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
        self.assertEqual(row["runtime_identity"]["selected_backend"], "mlx")
        self.assertEqual(row["runtime_identity"]["route_identity"], "repo_owned_mlx")
        self.assertEqual(row["peak_memory_gb"]["max"], 13.0)
        self.assertEqual(row["memory_source"], "server_process_rss_after_stream")

    def test_axengine_summary_can_label_linear_attention_pack_row(self) -> None:
        runs = [
            {"prefill_s": 0.2, "decode_s": 0.1, "ttft_ms": 200.0, "prefill_tok_s": 15.0, "decode_tok_s": 20.0, "output_tokens": 3.0},
            {"prefill_s": 0.2, "decode_s": 0.1, "ttft_ms": 200.0, "prefill_tok_s": 15.0, "decode_tok_s": 20.0, "output_tokens": 3.0},
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

    def test_swift_adapter_command_gets_prompt_artifact_placeholders(self) -> None:
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
                stdout=json.dumps({"prefill_tok_s": 100.0, "decode_tok_s": 50.0}),
                stderr="",
            )
            with patch.object(bench.subprocess, "run", return_value=completed) as run:
                row = bench.run_mlx_swift_lm_adapter(
                    "swift-bench --prompt {prompt_token_ids_path} --hash {prompt_token_ids_sha256}",
                    "model",
                    4,
                    2,
                    3,
                    0.0,
                    2048,
                    prompt,
                )

            command_args = run.call_args.args[0]
            self.assertIn(prompt["token_ids_path"], command_args)
            self.assertIn(prompt["token_ids_sha256"], command_args)
            self.assertEqual(row["method"], "mlx_swift_lm_benchmark_adapter")
            self.assertEqual(row["prompt_token_ids_sha256"], prompt["token_ids_sha256"])

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
                        {"backends": "Metal", "n_prompt": 4, "n_gen": 0, "avg_ts": 1.0, "samples_ts": [1.0]},
                        {"backends": "Metal", "n_prompt": 0, "n_gen": 2, "avg_ts": 1.0, "samples_ts": [1.0]},
                    ]
                ),
                stderr="",
            )
            with (
                patch.object(bench.subprocess, "run", return_value=completed) as run,
                patch.object(bench, "collect_llama_cpp_device_evidence", return_value=None),
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
                    {"backends": "BLAS,MTL", "n_prompt": 4, "n_gen": 0, "avg_ts": 10.0, "samples_ts": [10.0]},
                    {"backends": "BLAS,MTL", "n_prompt": 0, "n_gen": 2, "avg_ts": 1.0, "samples_ts": [1.0]},
                ]
            ),
            prompt_tokens=4,
            generation_tokens=2,
        )
        self.assertEqual(parsed["llama_cpp"]["backends"], "BLAS,MTL")

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
                patch.object(bench, "collect_llama_cpp_device_evidence", return_value="Metal device"),
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
        self.assertEqual(row["runtime_identity"]["route_identity"], "external_llama_cpp_metal")
        self.assertEqual(row["prompt_contract"], "shape_compatible_llama_bench_internal_tokens")
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

    def test_gateddelta_prefill_profile_requires_linear_attention_metadata(self) -> None:
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
        self.assertEqual(contract["runtime_profile_env"], "AX_MLX_LINEAR_ATTENTION_PROFILE=1")
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
            invalid_family_dir = write_gateddelta_model(Path(tmp), model_family="gemma4")

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
            "schema_version": "ax.mlx_inference_stack.v2",
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
            "schema_version": "ax.mlx_inference_stack.v2",
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
        self.assertEqual(telemetry["ax_ngram_complete_misses"], 0)
        self.assertEqual(telemetry["ax_ngram_cooldown_steps"], 0)
        self.assertEqual(telemetry["ax_ngram_cooldown_events"], 0)
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
                    "ax_mlx_prefix_cache_hits": 1,
                    "ax_mlx_prefix_cache_blocked_policy_disabled": 2,
                    "ax_mlx_prefix_cache_reused_tokens": 16,
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
        self.assertEqual(telemetry["ax_mlx_prefix_cache_hits"], 1)
        self.assertEqual(telemetry["ax_mlx_prefix_cache_blocked_policy_disabled"], 2)
        self.assertEqual(telemetry["ax_mlx_prefix_cache_reused_tokens"], 16)
        self.assertEqual(telemetry["ax_mlx_prefix_cache_evictions"], 0)
        self.assertEqual(telemetry["ax_mlx_prefix_cache_blocked_unsupported_layout"], 0)
        self.assertEqual(telemetry["ax_mlx_prefix_cache_blocked_trim_failure"], 0)
        self.assertEqual(telemetry["ax_mlx_single_decode_steps"], 0)
        self.assertEqual(telemetry["ax_mlx_bonus_tokens"], 0)
        self.assertNotIn("unrelated", telemetry)

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
        self.assertEqual(summary["ax_mlx_decode_wall_us"], 200)
        self.assertEqual(summary["ax_mlx_prefix_cache_hits"], 1)
        self.assertEqual(summary["ax_mlx_prefix_cache_blocked_policy_disabled"], 2)
        self.assertEqual(summary["ax_mlx_prefix_cache_reused_tokens"], 16)

    def test_ax_mlx_decode_route_summary_classifies_pipeline_and_mixed_rows(self) -> None:
        direct = bench.summarize_ax_mlx_decode_route(
            {
                "ax_mlx_decode_steps": 10,
                "ax_mlx_decode_wall_us": 1000,
                "ax_mlx_direct_pipeline_steps": 10,
                "ax_mlx_direct_pipeline_wall_us": 900,
            }
        )
        self.assertEqual(direct["classification"], "direct_pipeline")
        self.assertEqual(direct["direct_pipeline_step_share_micros"], 1_000_000)
        self.assertEqual(direct["direct_pipeline_wall_share_micros"], 900_000)

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
                {"engine": "mlx_lm", "ax_mlx_telemetry": {"ax_mlx_prefix_cache_hits": 99}},
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
                    "ax_mlx_kv_compression_fused_decode_fallbacks": 0,
                    "ax_mlx_kv_compression_fused_decode_fallback_reason": 1,
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
        self.assertEqual(telemetry["ax_mlx_kv_compression_fused_decode_fallback_reason"], 1)
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
                        "ax_mlx_kv_compression_fused_decode_fallbacks": 0,
                        "ax_mlx_kv_compression_fused_decode_fallback_reason": 1,
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
        self.assertEqual(summary["ax_mlx_kv_compression_fused_decode_fallbacks"], 0)
        self.assertEqual(summary["ax_mlx_kv_compression_fused_decode_fallback_reason"], 1)
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

    def test_ax_step_timing_classifies_chunked_prefill_by_scheduled_tokens(self) -> None:
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

    def test_axengine_command_can_enable_experimental_kv_compression(self) -> None:
        with patch.object(bench.subprocess, "Popen") as popen:
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                direct_mode=True,
                kv_compression="turboquant-shadow",
                kv_compression_hot_window_tokens=128,
                kv_compression_min_context_tokens=1024,
            )

        command = popen.call_args.args[0]
        self.assertIn("--disable-ngram-acceleration", command)
        self.assertIn("--experimental-mlx-kv-compression", command)
        self.assertIn("turboquant-shadow", command)
        self.assertIn("--experimental-mlx-kv-compression-hot-window-tokens", command)
        self.assertIn("--experimental-mlx-kv-compression-min-context-tokens", command)

    def test_axengine_command_can_request_fused_experimental_kv_compression(self) -> None:
        with patch.object(bench.subprocess, "Popen") as popen:
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                direct_mode=True,
                kv_compression="turboquant-fused-experimental",
            )

        command = popen.call_args.args[0]
        self.assertIn("--experimental-mlx-kv-compression", command)
        self.assertIn("turboquant-fused-experimental", command)

    def test_axengine_command_can_enable_gemma4_moe_profile(self) -> None:
        with patch.object(bench.subprocess, "Popen") as popen:
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                direct_mode=True,
                gemma4_moe_profile=True,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_GEMMA4_MOE_PROFILE"], "1")

    def test_axengine_command_can_enable_linear_attention_profile(self) -> None:
        with patch.object(bench.subprocess, "Popen") as popen:
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                direct_mode=True,
                linear_attention_profile=True,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_LINEAR_ATTENTION_PROFILE"], "1")

    def test_axengine_command_can_enable_decode_profile(self) -> None:
        with patch.object(bench.subprocess, "Popen") as popen:
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                direct_mode=True,
                decode_profile=True,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_DECODE_PROFILE"], "1")

    def test_axengine_command_can_enable_linear_attention_projection_pack(self) -> None:
        with patch.object(bench.subprocess, "Popen") as popen:
            bench.start_axengine(
                Path("/tmp/ax-engine-server"),
                Path("/tmp/model"),
                19091,
                direct_mode=True,
                pack_linear_attention_projections=True,
            )

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS"], "1")

    def test_route_with_more_decisions_keeps_step_telemetry_over_response_route(self) -> None:
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

    def test_route_with_more_decisions_prefers_nonzero_counters_on_equal_keys(self) -> None:
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

    def test_route_with_more_decisions_prefers_decode_signals_over_prefill_totals(self) -> None:
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

    def test_step_local_route_decisions_are_merged_into_selected_decode_route(self) -> None:
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
                        "schema_version": "test",
                        "results": [
                            {
                                "engine": "mlx_lm",
                                "prompt_tokens": 4,
                                "generation_tokens": 2,
                                "prefill_tok_s": {"median": 100.0},
                                "decode_tok_s": {"median": 50.0},
                            },
                            {
                                "engine": "mlx_swift_lm",
                                "prompt_tokens": 4,
                                "generation_tokens": 2,
                                "prefill_tok_s": {"median": 90.0},
                                "decode_tok_s": {"median": 45.0},
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

            self.assertEqual(doc["schema_version"], "test")
            self.assertEqual([row["engine"] for row in rows], ["mlx_lm", "mlx_swift_lm"])

            with self.assertRaisesRegex(RuntimeError, "missing mlx_lm reference rows"):
                bench.load_reused_reference_rows(
                    path,
                    prompt_lengths=[8],
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


if __name__ == "__main__":
    unittest.main()
