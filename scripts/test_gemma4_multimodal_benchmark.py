#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = load_script("bench_gemma4_multimodal")
checker = load_script("check_gemma4_multimodal_benchmark_artifact")
renderer = load_script("render_gemma4_multimodal_charts")


def metric(mean: float, median: float, min_value: float, max_value: float) -> dict:
    return {"mean": mean, "median": median, "min": min_value, "max": max_value}


def sample_artifact(*, tracked_dirty: bool = False, zero_metric: bool = False) -> dict:
    ttft = 0.0 if zero_metric else 120.0
    return {
        "schema": "ax.gemma4_multimodal_benchmark.v1",
        "benchmark": {
            "name": "gemma4_12b_multimodal",
            "model": "gemma-4-12B-it",
            "warmup": 1,
            "repetitions": 2,
            "max_output_tokens": 8,
        },
        "provenance": {
            "git": {"commit": "abc123", "tracked_dirty": tracked_dirty},
            "model_fingerprints": {"tokenizer.json": "sha"},
        },
        "rows": [
            {
                "engine": "ax_engine",
                "backend": "mlx",
                "layer": "native_runtime_prefill",
                "case_id": "image_single_256soft",
                "modalities": ["image"],
                "status": "measured",
                "prompt": {
                    "original_tokens": 20,
                    "expanded_tokens": 277,
                    "image_soft_tokens": [256],
                    "audio_soft_tokens": [],
                    "video_soft_tokens": [],
                    "video_frame_counts": [],
                },
                "runs": [
                    {"runner_prefill_ttft_ms": ttft, "prefill_tok_s": 2300.0},
                    {"runner_prefill_ttft_ms": 140.0, "prefill_tok_s": 2100.0},
                ],
                "summary": {
                    "runner_prefill_ttft_ms": metric(130.0, ttft, ttft, 140.0),
                    "client_wall_ttft_ms": metric(200.0, 200.0, 180.0, 220.0),
                    "client_wall_total_ms": metric(220.0, 220.0, 200.0, 240.0),
                    "prefill_tok_s": metric(2200.0, 2200.0, 2100.0, 2300.0),
                },
            },
            {
                "engine": "llama_cpp",
                "backend": "metal",
                "layer": "openai_chat_e2e",
                "case_id": "video_2frame_distinct",
                "modalities": ["video"],
                "status": "skipped",
                "skip_reason": "llama_cpp_video_not_supported",
                "skip_detail": "video is not supported by the peer server",
                "prompt": {
                    "original_tokens": 20,
                    "expanded_tokens": 180,
                    "image_soft_tokens": [],
                    "audio_soft_tokens": [],
                    "video_soft_tokens": [140],
                    "video_frame_counts": [2],
                },
                "runs": [],
                "summary": {},
            },
        ],
    }


class Gemma4MultimodalBenchmarkTests(unittest.TestCase):
    def test_summarize_ignores_none(self) -> None:
        self.assertEqual(
            bench.summarize([1.0, None, 3.0]),
            {"mean": 2.0, "median": 2.0, "min": 1.0, "max": 3.0},
        )

    @unittest.skipIf(bench.Image is None, "Pillow is required for multimodal fixtures")
    def test_select_cases_includes_distinct_multimodal_fixtures(self) -> None:
        cases = bench.select_cases("image_single_256soft,audio_0_5s,video_2frame_distinct")
        self.assertEqual([case.case_id for case in cases], [
            "image_single_256soft",
            "audio_0_5s",
            "video_2frame_distinct",
        ])
        self.assertEqual(cases[0].modalities, ["image"])
        self.assertEqual(cases[1].modalities, ["audio"])
        self.assertEqual(cases[2].modalities, ["video"])
        self.assertEqual(len(cases[2].videos[0]), 2)

    def test_video_timestamp_tokens_match_server_prefix_shape(self) -> None:
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return type("Encoded", (), {"ids": [ord(char) for char in text]})()

        tokens = bench.video_timestamp_token_ids(FakeTokenizer(), [[0.0, 2.0]])
        self.assertEqual(tokens, [[[48, 48, 58, 48, 48, 32], [32, 48, 48, 58, 48, 50, 32]]])

    def test_checker_accepts_valid_artifact_and_skip_row(self) -> None:
        errors = checker.validate_artifact(
            sample_artifact(),
            min_repetitions=2,
            require_modalities={"image"},
            require_build_provenance=True,
        )
        self.assertEqual(errors, [])

    def test_checker_rejects_readme_ready_dirty_artifact(self) -> None:
        errors = checker.validate_artifact(sample_artifact(tracked_dirty=True), readme_ready=True)
        self.assertIn(
            "readme-ready artifacts must not have provenance.git.tracked_dirty=true",
            errors,
        )

    def test_checker_rejects_zero_measured_timing(self) -> None:
        errors = checker.validate_artifact(sample_artifact(zero_metric=True))
        self.assertTrue(
            any("runner_prefill_ttft_ms" in error and "must be positive" in error for error in errors)
        )

    def test_renderer_accepts_matrix_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "artifact.json"
            assets_dir = root / "assets"
            artifact_path.write_text(__import__("json").dumps(sample_artifact()))
            outputs = renderer.render(artifact_path, assets_dir)
            self.assertEqual(len(outputs), 2)
            for path in outputs:
                text = path.read_text()
                self.assertIn("<svg", text)
                self.assertIn("image_single_256soft", text)


if __name__ == "__main__":
    unittest.main()
