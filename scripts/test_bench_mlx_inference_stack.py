#!/usr/bin/env python3
"""Unit tests for the MLX inference-stack benchmark contract."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("bench_mlx_inference_stack.py")
SPEC = importlib.util.spec_from_file_location("bench_mlx_inference_stack", SCRIPT_PATH)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


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
                    }
                )
            )

            metadata = bench.collect_model_metadata(root)

        self.assertTrue(metadata["linear_attention_enabled"])
        self.assertEqual(
            bench.ax_speculative_decode_policy(metadata, no_speculative=False),
            "ngram_linear_attention_support_gated_branch_recompute",
        )
        self.assertEqual(
            bench.ax_speculative_decode_policy(metadata, no_speculative=True),
            "greedy_no_speculative_decode",
        )

    def test_ax_speculative_decode_policy_defaults_to_kv_trim(self) -> None:
        self.assertEqual(
            bench.ax_speculative_decode_policy(
                {"linear_attention_enabled": False}, no_speculative=False
            ),
            "ngram_kv_trim",
        )

    def test_ax_speculative_telemetry_is_extracted_from_route(self) -> None:
        telemetry = bench.extract_ax_speculative_telemetry(
            {
                "crossover_decisions": {
                    "ax_spec_draft_attempts": 3,
                    "ax_spec_draft_tokens": 12,
                    "ax_spec_accepted_tokens": 9,
                    "unrelated": 99,
                }
            }
        )

        self.assertEqual(telemetry["ax_spec_draft_attempts"], 3)
        self.assertEqual(telemetry["ax_spec_draft_tokens"], 12)
        self.assertEqual(telemetry["ax_spec_accepted_tokens"], 9)
        self.assertEqual(telemetry["ax_spec_complete_misses"], 0)
        self.assertEqual(telemetry["ax_spec_cooldown_steps"], 0)
        self.assertEqual(telemetry["ax_spec_cooldown_events"], 0)
        self.assertEqual(telemetry["ax_spec_accept_rate_micros"], 750000)
        self.assertNotIn("unrelated", telemetry)

    def test_ax_speculative_telemetry_summarizes_trials(self) -> None:
        summary = bench.summarize_telemetry(
            [
                {
                    "speculative_telemetry": {
                        "ax_spec_draft_tokens": 8,
                        "ax_spec_accepted_tokens": 4,
                        "ax_spec_accept_rate_micros": 500000,
                    }
                },
                {
                    "speculative_telemetry": {
                        "ax_spec_draft_tokens": 12,
                        "ax_spec_accepted_tokens": 11,
                        "ax_spec_accept_rate_micros": 916667,
                    }
                },
            ]
        )

        self.assertEqual(summary["ax_spec_draft_tokens"], 20)
        self.assertEqual(summary["ax_spec_accepted_tokens"], 15)
        self.assertEqual(summary["ax_spec_accept_rate_micros"], 750000)

    def test_ax_mlx_telemetry_is_extracted_and_summarized(self) -> None:
        telemetry = bench.extract_ax_mlx_telemetry(
            {
                "crossover_decisions": {
                    "ax_mlx_prefill_steps": 1,
                    "ax_mlx_prefill_wall_us": 100,
                    "ax_mlx_decode_steps": 2,
                    "ax_mlx_decode_wall_us": 80,
                    "ax_mlx_greedy_pipeline_steps": 2,
                    "ax_mlx_greedy_pipeline_wall_us": 70,
                    "unrelated": 99,
                }
            }
        )

        self.assertEqual(telemetry["ax_mlx_prefill_steps"], 1)
        self.assertEqual(telemetry["ax_mlx_prefill_wall_us"], 100)
        self.assertEqual(telemetry["ax_mlx_decode_steps"], 2)
        self.assertEqual(telemetry["ax_mlx_decode_wall_us"], 80)
        self.assertEqual(telemetry["ax_mlx_greedy_pipeline_steps"], 2)
        self.assertEqual(telemetry["ax_mlx_greedy_pipeline_wall_us"], 70)
        self.assertEqual(telemetry["ax_mlx_single_decode_steps"], 0)
        self.assertEqual(telemetry["ax_mlx_bonus_tokens"], 0)
        self.assertNotIn("unrelated", telemetry)

        summary = bench.summarize_ax_mlx_telemetry(
            [
                {"ax_mlx_telemetry": telemetry},
                {
                    "ax_mlx_telemetry": {
                        "ax_mlx_prefill_steps": 1,
                        "ax_mlx_decode_steps": 3,
                        "ax_mlx_decode_wall_us": 120,
                    }
                },
            ]
        )
        self.assertEqual(summary["ax_mlx_prefill_steps"], 2)
        self.assertEqual(summary["ax_mlx_decode_steps"], 5)
        self.assertEqual(summary["ax_mlx_decode_wall_us"], 200)

    def test_route_with_more_decisions_keeps_step_telemetry_over_response_route(self) -> None:
        step_route = {
            "attention_route": "qwen_paged_decode",
            "crossover_decisions": {
                "ax_spec_draft_attempts": 3,
                "ax_spec_accepted_tokens": 6,
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
                "engine": "ax_engine_mlx_greedy",
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

        with self.assertRaisesRegex(RuntimeError, "missing mlx_lm.benchmark baseline"):
            bench.attach_mlx_lm_baselines(
                [
                    {
                        "engine": "ax_engine_mlx_greedy",
                        "prompt_tokens": 8,
                        "generation_tokens": 2,
                        "prefill_tok_s": {"median": 80.0},
                        "decode_tok_s": {"median": 40.0},
                    }
                ]
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
                                "engine": "ax_engine_mlx_greedy",
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
