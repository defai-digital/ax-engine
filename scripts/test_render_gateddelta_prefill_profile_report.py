#!/usr/bin/env python3
"""Unit tests for rendering GatedDelta prefill profile reports."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "render_gateddelta_prefill_profile_report.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_gateddelta_prefill_profile_report", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
renderer = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = renderer
MODULE_SPEC.loader.exec_module(renderer)


def metric(median: float) -> dict[str, float]:
    return {
        "mean": median,
        "median": median,
        "min": median,
        "max": median,
    }


def prompt_hash(prompt_tokens: int) -> str:
    return f"{prompt_tokens:064x}"[-64:]


def mlx_lm_row(prompt_tokens: int, generation_tokens: int) -> dict[str, object]:
    return {
        "engine": "mlx_lm",
        "method": "mlx_lm.benchmark",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prompt_token_ids_sha256": prompt_hash(prompt_tokens),
        "prefill_tok_s": metric(2000.0),
        "decode_tok_s": metric(100.0),
        "baseline": {"role": "primary_reference"},
    }


def ax_row(
    prompt_tokens: int,
    generation_tokens: int,
    *,
    recurrent_wall_us: int,
    projection_wall_us: int = 100,
) -> dict[str, object]:
    return {
        "engine": "ax_engine_mlx",
        "method": "server_sse_runner_time_us",
        "ax_decode_policy": "direct_no_ngram_acceleration",
        "ax_decode_claim_status": "direct_same_policy_baseline",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prompt_token_ids_sha256": prompt_hash(prompt_tokens),
        "prefill_tok_s": metric(1800.0),
        "decode_tok_s": metric(90.0),
        "ax_mlx_telemetry": {
            "ax_mlx_prefill_wall_us": prompt_tokens * 10,
        },
        "ax_mlx_linear_attention_profile": {
            "ax_mlx_linear_attention_profile_enabled": 1,
            "ax_mlx_linear_attention_profile_layers": 28,
            "ax_mlx_linear_attention_profile_tokens": prompt_tokens * 28,
            "ax_mlx_linear_attention_profile_projection_wall_us": projection_wall_us,
            "ax_mlx_linear_attention_profile_conv_wall_us": 80,
            "ax_mlx_linear_attention_profile_qk_norm_wall_us": 30,
            "ax_mlx_linear_attention_profile_recurrent_wall_us": recurrent_wall_us,
            "ax_mlx_linear_attention_profile_output_wall_us": 20,
        },
        "baseline": {
            "engine": "mlx_lm",
            "prefill_ratio_to_mlx_lm": 0.9,
        },
    }


def artifact() -> dict[str, object]:
    generation_tokens = 128
    results: list[dict[str, object]] = []
    for prompt_tokens in [512, 2048, 8192, 32768]:
        results.append(mlx_lm_row(prompt_tokens, generation_tokens))
        results.append(
            ax_row(
                prompt_tokens,
                generation_tokens,
                recurrent_wall_us=90 if prompt_tokens < 32768 else 900,
            )
        )
    return {
        "schema_version": "ax.mlx_inference_stack.v2",
        "host": {"chip": "Apple M5 Max", "memory_gb": 128},
        "model": "mlx-community/Qwen3.5-9B-4bit",
        "model_config": {
            "model_family": "qwen3_next",
            "linear_attention_enabled": True,
        },
        "prompt_tokens": [512, 2048, 8192, 32768],
        "generation_tokens": generation_tokens,
        "repetitions": 3,
        "prefill_step_size": 2048,
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


class GatedDeltaPrefillProfileReportTests(unittest.TestCase):
    def write_artifact(self) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "gateddelta-profile.json"
        path.write_text(json.dumps(artifact(), indent=2) + "\n")
        return path

    def test_renders_stage_table_and_decision_hint(self) -> None:
        path = self.write_artifact()

        report = renderer.render_report(path)

        self.assertIn("# GatedDelta Prefill Profile Report", report)
        self.assertIn("Model preflight: `passed`", report)
        self.assertIn("ax.gateddelta_prefill_model_preflight.v1", report)
        self.assertIn("key_head_dim=64", report)
        self.assertIn("| 32,768 | 1,800.0 | 0.900x | 327.7 |", report)
        self.assertIn("recurrent | prioritize recurrent scan", report)
        self.assertIn("Highest recurrent share", report)
        self.assertIn("timing barriers perturb latency", report)

    def test_cli_writes_report(self) -> None:
        path = self.write_artifact()
        output = path.with_suffix(".md")

        exit_code = renderer.main_with_args_for_test([str(path), "--output", str(output)])

        self.assertEqual(exit_code, 0)
        self.assertIn("GatedDelta Prefill Profile Report", output.read_text())


if __name__ == "__main__":
    unittest.main()
