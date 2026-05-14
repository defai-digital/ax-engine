#!/usr/bin/env python3
"""Unit tests for rendering AX MLX decode-profile reports."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "render_mlx_decode_profile_report.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_mlx_decode_profile_report", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
renderer = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = renderer
MODULE_SPEC.loader.exec_module(renderer)


def artifact(*, include_new_fields: bool = True) -> dict[str, object]:
    profile = {
        "ax_mlx_decode_profile_enabled": 1,
        "ax_mlx_decode_profile_decode_steps": 127,
        "ax_mlx_decode_profile_layers": 4445,
        "ax_mlx_decode_profile_per_layer_input_wall_us": 100,
        "ax_mlx_decode_profile_pre_sdpa_wall_us": 1000,
        "ax_mlx_decode_profile_pre_sdpa_qkv_proj_wall_us": 300,
        "ax_mlx_decode_profile_sdpa_wall_us": 600,
        "ax_mlx_decode_profile_post_attn_wall_us": 2000,
        "ax_mlx_decode_profile_post_attn_ffn_wall_us": 900,
        "ax_mlx_decode_profile_lm_head_wall_us": 300,
    }
    if include_new_fields:
        profile.update(
            {
                "ax_mlx_decode_profile_pre_sdpa_qk_norm_wall_us": 200,
                "ax_mlx_decode_profile_pre_sdpa_rope_kv_wall_us": 400,
                "ax_mlx_decode_profile_post_attn_output_proj_wall_us": 500,
                "ax_mlx_decode_profile_post_attn_residual_norm_wall_us": 100,
                "ax_mlx_decode_profile_post_attn_residual_gate_wall_us": 300,
            }
        )
    return {
        "schema_version": "ax.mlx_inference_stack.v2",
        "model": "gemma_4_e2b_it_4bit",
        "ax_decode_profile": True,
        "results": [
            {
                "engine": "ax_engine_mlx",
                "prompt_tokens": 128,
                "generation_tokens": 128,
                "ax_mlx_decode_profile": profile,
            }
        ],
    }


class MlxDecodeProfileReportTests(unittest.TestCase):
    def write_artifact(
        self,
        payload: dict[str, object],
        *,
        name: str = "gemma-decode-profile.json",
    ) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / name
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def test_renders_stage_and_substage_tables(self) -> None:
        path = self.write_artifact(artifact())

        report = renderer.render_report(renderer.build_rows(path), title="Decode Profile")

        self.assertIn("# Decode Profile", report)
        self.assertIn("| post-attention | 2,000 | 50.0% |", report)
        self.assertIn("| QK norm | 200 | 20.0% | 5.0% |", report)
        self.assertIn("| RoPE + KV append | 400 | 40.0% | 10.0% |", report)
        self.assertIn("| Unsplit pre-SDPA tail | 100 | n/a | 2.5% |", report)
        self.assertIn("Dominant parent stage: post-attention", report)

    def test_old_artifact_missing_new_fields_renders_na(self) -> None:
        path = self.write_artifact(artifact(include_new_fields=False))

        report = renderer.render_report(renderer.build_rows(path), title="Decode Profile")

        self.assertIn("| QK norm | n/a | n/a | n/a |", report)
        self.assertIn("| Unsplit pre-SDPA tail | 700 | n/a | 17.5% |", report)
        self.assertIn("artifact predates that finer-grained counter", report)

    def test_directory_filter_selects_decode_profile_artifacts(self) -> None:
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
        self.assertIn("AX MLX Decode Profile Report", output.read_text())


if __name__ == "__main__":
    unittest.main()
