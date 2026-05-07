#!/usr/bin/env python3
"""Unit tests for multi-model prefill scaling campaign checks."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "check_mlx_prefill_scaling_campaign.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mlx_prefill_scaling_campaign", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
campaign = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = campaign
MODULE_SPEC.loader.exec_module(campaign)


def metric(median: float, *, max_value: float | None = None) -> dict[str, float]:
    return {
        "mean": median,
        "median": median,
        "min": median,
        "max": max_value if max_value is not None else median,
    }


def row(
    *,
    engine: str,
    context_tokens: int,
    prompt_hash: str,
    prefill_tok_s: float,
    ttft_ms: float,
    peak_memory_gb: float,
    ratios: dict[str, float] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "engine": engine,
        "context_tokens": context_tokens,
        "generation_tokens": 1,
        "prompt_token_ids_sha256": prompt_hash,
        "repetitions": 3,
        "prefill_tok_s": metric(prefill_tok_s),
        "ttft_ms": metric(ttft_ms),
        "peak_memory_gb": metric(peak_memory_gb, max_value=peak_memory_gb),
    }
    if engine == "mlx_lm":
        payload["baseline"] = {"role": "primary_reference", "method": "mlx_lm.benchmark"}
    else:
        payload["ax_decode_policy"] = "direct_no_ngram_acceleration"
        payload["route"] = {"selected_backend": "mlx"}
        payload["ratios_to_mlx_lm"] = ratios or {
            "prefill_tok_s": 1.0,
            "ttft_ms": 1.0,
        }
    return payload


def artifact(model_id: str, hash_prefix: str, *, chip: str = "Apple M5 Max") -> dict[str, object]:
    hash_1k = hash_prefix * 64
    hash_8k = chr(ord(hash_prefix) + 1) * 64
    return {
        "schema_version": "ax.mlx_prefill_scaling.v1",
        "model": {"id": model_id},
        "host": {"chip": chip, "memory_gb": 128, "os_version": "26.4.1"},
        "benchmark": {
            "batch_size": 1,
            "temperature": 0.0,
            "prefill_step_size": 2048,
            "repetitions": 3,
        },
        "rows": [
            row(
                engine="mlx_lm",
                context_tokens=1024,
                prompt_hash=hash_1k,
                prefill_tok_s=3000.0,
                ttft_ms=340.0,
                peak_memory_gb=20.0,
            ),
            row(
                engine="ax_engine_mlx",
                context_tokens=1024,
                prompt_hash=hash_1k,
                prefill_tok_s=3300.0,
                ttft_ms=310.0,
                peak_memory_gb=21.0,
                ratios={"prefill_tok_s": 1.1, "ttft_ms": 310.0 / 340.0},
            ),
            row(
                engine="mlx_lm",
                context_tokens=8192,
                prompt_hash=hash_8k,
                prefill_tok_s=2200.0,
                ttft_ms=3700.0,
                peak_memory_gb=32.0,
            ),
            row(
                engine="ax_engine_mlx",
                context_tokens=8192,
                prompt_hash=hash_8k,
                prefill_tok_s=2420.0,
                ttft_ms=3400.0,
                peak_memory_gb=33.0,
                ratios={"prefill_tok_s": 1.1, "ttft_ms": 3400.0 / 3700.0},
            ),
        ],
    }


class PrefillScalingCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_artifact(self, name: str, payload: dict[str, object]) -> Path:
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def test_campaign_with_required_families_passes_and_renders(self) -> None:
        paths = [
            self.write_artifact("gemma", artifact("mlx-community/Gemma-4-E2B", "a")),
            self.write_artifact("qwen", artifact("mlx-community/Qwen3.5-9B", "c")),
            self.write_artifact("glm", artifact("mlx-community/GLM-4.7-Flash", "e")),
        ]

        summaries = campaign.validate_campaign(
            paths,
            required_families=["gemma", "qwen", "glm"],
            allow_mixed_host=False,
            bend_drop_ratio=0.75,
            min_context_count=2,
            min_largest_context_tokens=8192,
        )
        report = campaign.render_campaign_report(summaries)

        self.assertEqual({summary.family for summary in summaries}, {"gemma", "qwen", "glm"})
        self.assertIn("# MLX Prefill Scaling Campaign", report)
        self.assertIn("8,192", report)

    def test_missing_required_family_fails(self) -> None:
        paths = [
            self.write_artifact("gemma", artifact("mlx-community/Gemma-4-E2B", "a")),
            self.write_artifact("qwen", artifact("mlx-community/Qwen3.5-9B", "c")),
        ]

        with self.assertRaisesRegex(campaign.PrefillScalingCampaignError, "lacks required"):
            campaign.validate_campaign(
                paths,
                required_families=["gemma", "qwen", "glm"],
                allow_mixed_host=False,
                bend_drop_ratio=0.75,
                min_context_count=2,
                min_largest_context_tokens=8192,
            )

    def test_mixed_host_fails_by_default(self) -> None:
        paths = [
            self.write_artifact("gemma", artifact("mlx-community/Gemma-4-E2B", "a")),
            self.write_artifact(
                "qwen",
                artifact("mlx-community/Qwen3.5-9B", "c", chip="Apple M4 Max"),
            ),
        ]

        with self.assertRaisesRegex(campaign.PrefillScalingCampaignError, "mixes host"):
            campaign.validate_campaign(
                paths,
                required_families=["gemma", "qwen"],
                allow_mixed_host=False,
                bend_drop_ratio=0.75,
                min_context_count=2,
                min_largest_context_tokens=8192,
            )

    def test_cli_writes_campaign_report(self) -> None:
        paths = [
            self.write_artifact("gemma", artifact("mlx-community/Gemma-4-E2B", "a")),
            self.write_artifact("qwen", artifact("mlx-community/Qwen3.5-9B", "c")),
        ]
        output = self.root / "campaign.md"

        exit_code = campaign.main_with_args_for_test(
            [
                str(paths[0]),
                str(paths[1]),
                "--required-family",
                "gemma",
                "--required-family",
                "qwen",
                "--output",
                str(output),
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("MLX Prefill Scaling Campaign", output.read_text())


if __name__ == "__main__":
    unittest.main()
