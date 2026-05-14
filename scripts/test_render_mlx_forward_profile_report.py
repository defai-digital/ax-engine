#!/usr/bin/env python3
"""Unit tests for rendering AX MLX forward-profile reports."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "render_mlx_forward_profile_report.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_mlx_forward_profile_report", SCRIPT_PATH
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


def mlx_lm_row(prompt_tokens: int, generation_tokens: int) -> dict[str, object]:
    return {
        "engine": "mlx_lm",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prefill_tok_s": metric(1200.0),
    }


def mlx_swift_lm_row(prompt_tokens: int, generation_tokens: int) -> dict[str, object]:
    return {
        "engine": "mlx_swift_lm",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prefill_tok_s": metric(1000.0),
    }


def ax_row(
    prompt_tokens: int,
    generation_tokens: int,
    *,
    projection_wall_us: int = 9000,
    recurrent_wall_us: int = 3000,
    profile_tokens: int | None = None,
    projection_split: bool = False,
    projection_pack: bool | None = None,
) -> dict[str, object]:
    profile = {
        "ax_mlx_linear_attention_profile_enabled": 1,
        "ax_mlx_linear_attention_profile_layers": 30,
        "ax_mlx_linear_attention_profile_tokens": (
            profile_tokens if profile_tokens is not None else prompt_tokens * 30
        ),
        "ax_mlx_linear_attention_profile_projection_wall_us": projection_wall_us,
        "ax_mlx_linear_attention_profile_conv_wall_us": 1000,
        "ax_mlx_linear_attention_profile_qk_norm_wall_us": 500,
        "ax_mlx_linear_attention_profile_recurrent_wall_us": recurrent_wall_us,
        "ax_mlx_linear_attention_profile_output_wall_us": 1000,
    }
    if projection_split:
        profile.update(
            {
                "ax_mlx_linear_attention_profile_projection_qkvz_wall_us": 0,
                "ax_mlx_linear_attention_profile_projection_ba_wall_us": 0,
                "ax_mlx_linear_attention_profile_projection_qkv_wall_us": 7000,
                "ax_mlx_linear_attention_profile_projection_z_wall_us": 800,
                "ax_mlx_linear_attention_profile_projection_a_wall_us": 700,
                "ax_mlx_linear_attention_profile_projection_b_wall_us": 500,
            }
        )
    row: dict[str, object] = {
        "engine": "ax_engine_mlx",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prefill_tok_s": metric(1800.0),
        "ax_mlx_telemetry": {
            "ax_mlx_prefill_wall_us": 128_000,
            "ax_mlx_prefill_forward_wall_us": 127_000,
        },
        "ax_mlx_linear_attention_profile": profile,
    }
    if projection_pack is not None:
        row["ax_linear_attention_projection_pack"] = projection_pack
    return row


def artifact(
    *,
    profile_tokens: int | None = None,
    projection_split: bool = False,
    projection_layout: bool = False,
    projection_pack: bool | None = None,
) -> dict[str, object]:
    prompt_tokens = 128
    generation_tokens = 128
    payload: dict[str, object] = {
        "schema_version": "ax.mlx_inference_stack.v2",
        "model": "qwen3_6_35b_a3b_8bit",
        "ax_linear_attention_profile": True,
        "results": [
            mlx_lm_row(prompt_tokens, generation_tokens),
            mlx_swift_lm_row(prompt_tokens, generation_tokens),
            ax_row(
                prompt_tokens,
                generation_tokens,
                profile_tokens=profile_tokens,
                projection_split=projection_split,
                projection_pack=projection_pack,
            ),
        ],
    }
    if projection_pack is not None:
        payload["ax_linear_attention_projection_pack"] = projection_pack
    if projection_layout:
        payload["model_config"] = {
            "linear_attention_projection_layout": {
                "schema_version": "ax.linear_attention_projection_layout.v1",
                "layout": "split_qkv_z_a_b",
                "linear_layers": 30,
                "packed_layers": 0,
                "split_layers": 30,
                "offline_pack_candidate": True,
            }
        }
    return payload


class MlxForwardProfileReportTests(unittest.TestCase):
    def write_artifact(
        self,
        payload: dict[str, object],
        *,
        name: str = "qwen-linear-profile.json",
    ) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / name
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def test_renders_stage_table_and_decision_hint(self) -> None:
        path = self.write_artifact(artifact())

        report = renderer.render_report(renderer.build_rows(path), title="Forward Profile")

        self.assertIn("# Forward Profile", report)
        self.assertIn("Timing barriers perturb latency", report)
        self.assertIn("| qwen3_6_35b_a3b_8bit | 128 | 1,800.0 | 1.500x | 1.800x |", report)
        self.assertIn("projection | 62.1% | inspect projection/layout fusion", report)
        self.assertIn("Keep barrier-profile artifacts out of README headline tables", report)

    def test_renders_projection_breakdown_and_pack_hint(self) -> None:
        path = self.write_artifact(
            artifact(projection_split=True, projection_layout=True)
        )

        report = renderer.render_report(renderer.build_rows(path), title="Forward Profile")

        self.assertIn("## Projection Breakdown", report)
        self.assertIn(
            "| qwen3_6_35b_a3b_8bit | 128 | split_qkv_z_a_b | n/a | yes | 9.0 | 0.0 | 0.0 | 7.0 | 0.8 | 0.7 | 0.5 | 77.8% | 22.2% |",
            report,
        )
        self.assertIn("evaluate offline packed qkvz/ba projection", report)
        self.assertIn("row-order equivalence tests", report)

    def test_renders_runtime_pack_marker_and_delta_hint(self) -> None:
        payload = artifact(projection_layout=True, projection_pack=True)
        profile = payload["results"][2]["ax_mlx_linear_attention_profile"]  # type: ignore[index]
        assert isinstance(profile, dict)
        profile.update(
            {
                "ax_mlx_linear_attention_profile_projection_qkvz_wall_us": 6000,
                "ax_mlx_linear_attention_profile_projection_ba_wall_us": 3000,
                "ax_mlx_linear_attention_profile_projection_qkv_wall_us": 0,
                "ax_mlx_linear_attention_profile_projection_z_wall_us": 0,
                "ax_mlx_linear_attention_profile_projection_a_wall_us": 0,
                "ax_mlx_linear_attention_profile_projection_b_wall_us": 0,
            }
        )
        path = self.write_artifact(payload)

        report = renderer.render_report(renderer.build_rows(path), title="Forward Profile")

        self.assertIn(
            "| qwen3_6_35b_a3b_8bit | 128 | split_qkv_z_a_b | yes | yes | 9.0 | 6.0 | 3.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% | 0.0% |",
            report,
        )
        self.assertIn("compare packed vs split projection delta", report)
        self.assertIn("AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS=1", report)

    def test_rejects_stale_profile_token_sentinel(self) -> None:
        path = self.write_artifact(artifact(profile_tokens=4_294_967_295))

        with self.assertRaisesRegex(
            renderer.MlxForwardProfileReportError,
            "stale profile token sentinel",
        ):
            renderer.build_rows(path)

    def test_directory_filter_selects_profile_artifacts(self) -> None:
        path = self.write_artifact(artifact())
        ignored = path.with_name("regular-result.json")
        ignored.write_text(json.dumps({"results": []}) + "\n")

        selected = renderer.artifact_paths([path.parent])

        self.assertEqual(selected, [path])

    def test_cli_writes_report(self) -> None:
        path = self.write_artifact(artifact())
        output = path.with_suffix(".md")

        exit_code = renderer.main_with_args_for_test([str(path), "--output", str(output)])

        self.assertEqual(exit_code, 0)
        self.assertIn("AX MLX Forward Profile Report", output.read_text())


if __name__ == "__main__":
    unittest.main()
