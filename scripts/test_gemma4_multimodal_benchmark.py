#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
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


def prompt_block(
    fixture_ids: list[str],
    *,
    original_tokens: int = 20,
    expanded_tokens: int = 300,
    soft_tokens: dict | None = None,
) -> dict:
    soft_tokens = soft_tokens or {"image": 256, "audio": 0, "video": 0}
    return {
        "original_tokens": original_tokens,
        "expanded_tokens": expanded_tokens,
        "soft_tokens": soft_tokens,
        "span_order": [modality for modality, value in soft_tokens.items() if value],
        "fixture_ids": fixture_ids,
        "image_soft_tokens": [soft_tokens["image"]] if soft_tokens["image"] else [],
        "audio_soft_tokens": [soft_tokens["audio"]] if soft_tokens["audio"] else [],
        "video_soft_tokens": [soft_tokens["video"]] if soft_tokens["video"] else [],
        "video_frame_counts": [2] if soft_tokens["video"] else [],
        "video_timestamp_seconds": [[0.0, 2.0]] if soft_tokens["video"] else [],
    }


def soft_tokens_for_modalities(modalities: list[str]) -> dict:
    return {
        "image": 256 if "image" in modalities else 0,
        "audio": 80 if "audio" in modalities else 0,
        "video": 140 if "video" in modalities else 0,
    }


def modality_set_for_modalities(modalities: list[str]) -> list[str]:
    return [modality for modality in ("image", "audio", "video") if modality in set(modalities)]


def measured_row(case_id: str, fixture_ids: list[str], *, modalities: list[str]) -> dict:
    return {
        "row_id": f"ax_engine_mlx.native_runtime_prefill.{case_id}",
        "engine": "ax_engine_mlx",
        "backend": "mlx",
        "layer": "native_runtime_prefill",
        "endpoint": "/v1/generate/stream",
        "case_id": case_id,
        "description": "sample row",
        "modalities": modalities,
        "modality_set": modality_set_for_modalities(modalities),
        "fixture_ids": fixture_ids,
        "prompt": prompt_block(
            fixture_ids,
            expanded_tokens=320,
            soft_tokens=soft_tokens_for_modalities(modalities),
        ),
        "status": "measured",
        "sampling": {"temperature": 0.0, "ignore_eos": True},
        "max_output_tokens": 8,
        "warmup": 1,
        "repetitions": 2,
        "runs": [
            {"runner_prefill_ttft_ms": 120.0, "prefill_tok_s": 2300.0},
            {"runner_prefill_ttft_ms": 140.0, "prefill_tok_s": 2100.0},
        ],
        "summary": {
            "runner_prefill_ttft_ms": metric(130.0, 130.0, 120.0, 140.0),
            "client_wall_ttft_ms": metric(200.0, 200.0, 180.0, 220.0),
            "client_wall_total_ms": metric(220.0, 220.0, 200.0, 240.0),
            "prefill_tok_s": metric(2200.0, 2200.0, 2100.0, 2300.0),
        },
    }


def skipped_peer_row() -> dict:
    return {
        "row_id": "llama_cpp_metal.peer_comparison.video_2frame_distinct",
        "engine": "llama_cpp_metal",
        "backend": "metal",
        "layer": "peer_comparison",
        "endpoint": None,
        "case_id": "video_2frame_distinct",
        "description": "video peer skip",
        "modalities": ["video"],
        "modality_set": ["video"],
        "fixture_ids": ["video_2frame_red_green"],
        "prompt": prompt_block(
            ["video_2frame_red_green"],
            expanded_tokens=240,
            soft_tokens={"image": 0, "audio": 0, "video": 140},
        ),
        "status": "skipped",
        "skip_reason": "llama_cpp_video_not_supported",
        "skip_detail": "video is not supported by the peer server",
        "runs": [],
        "summary": {},
        "capability": {
            "url": None,
            "binary": None,
            "text_gguf": None,
            "text_gguf_sha256": None,
            "mmproj": None,
            "mmproj_sha256": None,
            "supports_image": True,
            "supports_audio": True,
            "supports_video": False,
            "prompt_contract": "openai_chat_completions",
            "proof": False,
        },
    }


def sample_artifact(*, tracked_dirty: bool = False, zero_metric: bool = False) -> dict:
    row = measured_row("image_single_256soft", ["image_red_64"], modalities=["image"])
    if zero_metric:
        row["summary"]["runner_prefill_ttft_ms"]["median"] = 0.0
    return {
        "schema": "ax.gemma4_multimodal_benchmark.v1",
        "created_at": "2026-06-09T00:00:00+00:00",
        "host": {
            "platform": "darwin",
            "platform_detail": "macOS",
            "machine": "arm64",
            "processor": "arm",
            "python": "3.14",
            "chip": "Apple",
            "memory_gb": None,
            "os_version": "macOS",
        },
        "build": {
            "commit": "abc123",
            "build_profile": "release",
            "git_tracked_dirty": tracked_dirty,
            "git_tracked_status": [],
            "git_tracked_dirty_accepted": False,
        },
        "server": {
            "url": "http://127.0.0.1:18080",
            "binary": "target/release/ax-engine-server",
            "command": ["target/release/ax-engine-server"],
            "command_source": "cli",
            "endpoint_layers": ["/v1/generate/stream", "/v1/chat/completions"],
            "request_timeout_s": 300,
        },
        "model": {
            "id": "gemma-4-12B-it",
            "model_dir": ".internal/models/gemma-4-12B-it-4bit",
            "model_type": "gemma4_unified",
            "model_manifest_sha256": "manifest-sha",
            "config_sha256": "config-sha",
            "processor_config_sha256": "processor-sha",
            "tokenizer_sha256": "tokenizer-sha",
        },
        "benchmark": {
            "name": "gemma4_12b_multimodal",
            "model": "gemma-4-12B-it",
            "model_dir": ".internal/models/gemma-4-12B-it-4bit",
            "layers": ["native_runtime_prefill"],
            "cases": ["image_single_256soft"],
            "warmup": 1,
            "repetitions": 2,
            "cooldown_s": 0.0,
            "max_output_tokens": 8,
            "timeout_s": 300,
        },
        "fixtures": [
            {
                "id": "image_red_64",
                "modality": "image",
                "source": "generated",
                "sha256": "image-sha",
                "mime": "image/png",
                "raw": {"width": 64, "height": 64, "generator": "solid_rgb"},
            },
            {
                "id": "video_2frame_red_green",
                "modality": "video",
                "source": "generated",
                "sha256": "video-sha",
                "mime": "image/gif",
                "raw": {
                    "width": 32,
                    "height": 32,
                    "source_frame_count": 2,
                    "timestamp_seconds": [0.0, 2.0],
                },
            },
            {
                "id": "audio_tone_0_5s",
                "modality": "audio",
                "source": "generated",
                "sha256": "audio-sha",
                "mime": "audio/wav",
                "raw": {
                    "duration_s": 0.5,
                    "sample_rate": 16000,
                    "sample_count": 8000,
                },
            },
        ],
        "rows": [row, skipped_peer_row()],
        "summary": {"measured_rows": 1, "skipped_rows": 1},
        "command": {"argv": ["bench"]},
    }


class Gemma4MultimodalBenchmarkTests(unittest.TestCase):
    def test_summarize_ignores_none(self) -> None:
        self.assertEqual(
            bench.summarize([1.0, None, 3.0]),
            {"mean": 2.0, "median": 2.0, "min": 1.0, "max": 3.0},
        )

    def test_modality_set_is_canonical_and_deduplicated(self) -> None:
        self.assertEqual(bench.modality_set(["video", "image", "image"]), ["image", "video"])

    @unittest.skipIf(bench.Image is None, "Pillow is required for multimodal fixtures")
    def test_all_cases_cover_required_matrix(self) -> None:
        names = {case.case_id for case in bench.select_cases("all")}
        required = {
            "image_single_256soft",
            "image_multi_2x256soft",
            "image_aspect_portrait",
            "image_aspect_landscape",
            "image_max_soft_tokens",
            "audio_0_5s",
            "audio_2s",
            "audio_10s",
            "audio_cap",
            "video_1frame",
            "video_2frame_distinct",
            "video_8frame",
            "video_32frame_cap",
            "image_audio",
            "image_video",
            "audio_video",
            "image_audio_video",
        }
        self.assertLessEqual(required, names)

    @unittest.skipIf(bench.Image is None, "Pillow is required for multimodal fixtures")
    def test_video_fixture_timestamps_match_gif_duration(self) -> None:
        fixtures = bench.build_fixture_registry()
        raw = fixtures["video_2frame_red_green"].raw
        self.assertEqual(raw["timestamp_seconds"], [0.0, 2.0])
        self.assertEqual(raw["duration_ms_per_frame"], 2000)

    def test_video_timestamp_tokens_match_server_prefix_shape(self) -> None:
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return type("Encoded", (), {"ids": [ord(char) for char in text]})()

        tokens = bench.video_timestamp_token_ids(FakeTokenizer(), [[0.0, 2.0]])
        self.assertEqual(tokens, [[[48, 48, 58, 48, 48, 32], [32, 48, 48, 58, 48, 50, 32]]])

    def test_peer_decision_fails_closed_without_mmproj(self) -> None:
        args = type(
            "Args",
            (),
            {
                "llama_url": "http://127.0.0.1:18081",
                "llama_binary": None,
                "llama_gguf": None,
                "llama_mmproj": None,
            },
        )()
        case = type("Case", (), {"modalities": ["image"]})()
        decision = bench.peer_capability(args, case)
        self.assertEqual(decision.status, "skipped")
        self.assertEqual(decision.reason, "missing_llama_cpp_gguf_for_gemma4_12b")

    def test_checker_accepts_valid_artifact_and_skip_row(self) -> None:
        errors = checker.validate_artifact(
            sample_artifact(),
            min_repetitions=2,
            require_modalities={"image"},
            require_build_provenance=True,
            readme_ready=True,
        )
        self.assertEqual(errors, [])

    def test_checker_rejects_readme_ready_dirty_artifact(self) -> None:
        errors = checker.validate_artifact(sample_artifact(tracked_dirty=True), readme_ready=True)
        self.assertIn("readme-ready artifacts must have build.git_tracked_dirty=false", errors)

    def test_checker_rejects_zero_measured_timing(self) -> None:
        errors = checker.validate_artifact(sample_artifact(zero_metric=True))
        self.assertTrue(
            any("runner_prefill_ttft_ms" in error and "must be positive" in error for error in errors)
        )

    def test_checker_rejects_missing_fixture_reference(self) -> None:
        artifact = sample_artifact()
        artifact["rows"][0]["prompt"]["fixture_ids"] = ["missing_fixture"]
        errors = checker.validate_artifact(artifact)
        self.assertTrue(any("missing fixtures" in error for error in errors))

    def test_checker_rejects_row_fixture_modality_mismatch(self) -> None:
        artifact = sample_artifact()
        artifact["rows"][0]["modalities"] = ["audio"]
        errors = checker.validate_artifact(artifact)
        self.assertTrue(any("missing modality fixtures" in error for error in errors))
        self.assertTrue(any("unrelated modalities" in error for error in errors))

    def test_checker_rejects_modality_set_mismatch(self) -> None:
        artifact = sample_artifact()
        artifact["rows"][0]["modality_set"] = ["audio"]
        errors = checker.validate_artifact(artifact)
        self.assertTrue(any("modality_set must match row modalities" in error for error in errors))

    def test_checker_rejects_uncanonical_modality_set_order(self) -> None:
        artifact = sample_artifact()
        artifact["rows"][0] = measured_row(
            "image_audio",
            ["image_red_64", "audio_tone_0_5s"],
            modalities=["image", "audio"],
        )
        artifact["rows"][0]["modality_set"] = ["audio", "image"]
        errors = checker.validate_artifact(artifact)
        self.assertTrue(any("modality_set must match row modalities" in error for error in errors))

    def test_checker_rejects_row_fixture_id_not_in_registry(self) -> None:
        artifact = sample_artifact()
        artifact["rows"][0]["fixture_ids"] = ["missing_fixture"]
        errors = checker.validate_artifact(artifact)
        self.assertTrue(any("fixture_ids missing fixtures" in error for error in errors))

    def test_checker_rejects_prompt_fixture_id_mismatch(self) -> None:
        artifact = sample_artifact()
        artifact["rows"][0]["prompt"]["fixture_ids"] = ["video_2frame_red_green"]
        errors = checker.validate_artifact(artifact)
        self.assertTrue(any("prompt.fixture_ids must match row fixture_ids" in error for error in errors))

    def test_checker_rejects_span_order_mismatch(self) -> None:
        artifact = sample_artifact()
        artifact["rows"][0]["prompt"]["span_order"] = ["audio"]
        errors = checker.validate_artifact(artifact)
        self.assertTrue(any("prompt.span_order modalities must match row modalities" in error for error in errors))

    def test_checker_rejects_repeated_span_order_loss(self) -> None:
        artifact = sample_artifact()
        artifact["rows"][0]["fixture_ids"] = ["image_red_64", "image_red_64"]
        artifact["rows"][0]["prompt"]["fixture_ids"] = ["image_red_64", "image_red_64"]
        artifact["rows"][0]["prompt"]["span_order"] = ["image"]
        errors = checker.validate_artifact(artifact)
        self.assertTrue(any("prompt.span_order must match fixture order" in error for error in errors))

    def test_checker_rejects_measured_peer_without_capability_proof(self) -> None:
        artifact = sample_artifact()
        peer = skipped_peer_row()
        peer["status"] = "measured"
        peer["runs"] = [{"client_wall_ms": 1.0}]
        peer["summary"] = {"client_wall_ms": metric(1.0, 1.0, 1.0, 1.0)}
        artifact["rows"] = [peer]
        errors = checker.validate_artifact(artifact)
        self.assertTrue(any("capability.proof=true" in error for error in errors))

    def test_renderer_accepts_matrix_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "artifact.json"
            assets_dir = root / "assets"
            artifact = sample_artifact()
            artifact["rows"].insert(
                1,
                measured_row("audio_0_5s", ["audio_tone_0_5s"], modalities=["audio"]),
            )
            artifact_path.write_text(json.dumps(artifact))
            outputs = renderer.render(artifact_path, assets_dir)
            self.assertEqual(len(outputs), 2)
            for path in outputs:
                text = path.read_text()
                self.assertIn("<svg", text)
                self.assertIn("image_single_256soft", text)
                self.assertIn("audio_0_5s", text)


if __name__ == "__main__":
    unittest.main()
