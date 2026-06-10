from __future__ import annotations

import base64
import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    Image = None


SOURCE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = SOURCE_ROOT / "ax_engine" / "gemma4_unified.py"


def load_module():
    module_name = "ax_engine_gemma4_unified_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Gemma4UnifiedImagePreprocessTests(unittest.TestCase):
    @unittest.skipIf(Image is None, "Pillow is required for Gemma4 image preprocessing")
    def test_prepare_image_request_patchifies_and_expands_placeholder(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_tiny_config(model_dir)
            image = tiny_rgb_image()

            request = module.prepare_gemma4_unified_image_request(
                model_dir,
                [7, 100, 8],
                [image],
            )

        self.assertEqual(request.input_tokens, [7, 101, 100, 100, 102, 8])
        self.assertEqual(request.soft_token_counts, [2])
        gemma4 = request.multimodal_inputs["gemma4_unified"]
        self.assertEqual(gemma4["audios"], [])
        self.assertEqual(gemma4["videos"], [])
        image_input = gemma4["images"][0]
        self.assertEqual(
            image_input["span"],
            {
                "modality": "image",
                "placeholder_index": 1,
                "replacement_start": 1,
                "soft_token_count": 2,
                "replacement_token_count": 4,
            },
        )
        self.assertEqual(image_input["pixel_position_ids"], [[0, 0], [1, 0]])
        self.assertEqual(len(image_input["pixel_values"]), 24)
        self.assertEqual(
            image_input["pixel_values"][:12],
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
            ],
        )

    @unittest.skipIf(Image is None, "Pillow is required for Gemma4 image preprocessing")
    def test_prepare_image_request_reads_preprocessor_config_json(self) -> None:
        # Most HF checkpoints ship the image/audio params in
        # `preprocessor_config.json`; the SDK must accept it like the AX server
        # does, not only `processor_config.json`.
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_tiny_config(model_dir, processor_filename="preprocessor_config.json")
            image = tiny_rgb_image()

            request = module.prepare_gemma4_unified_image_request(
                model_dir,
                [7, 100, 8],
                [image],
            )

        self.assertEqual(request.input_tokens, [7, 101, 100, 100, 102, 8])
        self.assertEqual(request.soft_token_counts, [2])

    @unittest.skipIf(Image is None, "Pillow is required for Gemma4 image preprocessing")
    def test_prepare_image_request_rejects_placeholder_mismatch(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_tiny_config(model_dir)
            image = tiny_rgb_image()

            with self.assertRaisesRegex(ValueError, "placeholder count mismatch"):
                module.prepare_gemma4_unified_image_request(model_dir, [7, 8], [image])

    @unittest.skipIf(Image is None, "Pillow is required for Gemma4 image preprocessing")
    def test_prepare_image_request_decodes_data_uri_source(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_tiny_config(model_dir)

            request = module.prepare_gemma4_unified_image_request(
                model_dir,
                [7, 100, 8],
                [{"url": tiny_png_data_uri()}],
            )

        self.assertEqual(request.input_tokens, [7, 101, 100, 100, 102, 8])
        image_input = request.multimodal_inputs["gemma4_unified"]["images"][0]
        self.assertEqual(image_input["span"]["soft_token_count"], 2)
        self.assertEqual(len(image_input["pixel_values"]), 24)

    def test_prepare_audio_request_chunks_waveform_and_expands_placeholder(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_tiny_config(model_dir)

            request = module.prepare_gemma4_unified_audio_request(
                model_dir,
                [7, 200, 8],
                [[0.5, -0.5, 1.0]],
                sampling_rates=[4],
            )

        self.assertEqual(request.input_tokens, [7, 201, 200, 200, 202, 8])
        self.assertEqual(request.soft_token_counts, [2])
        gemma4 = request.multimodal_inputs["gemma4_unified"]
        self.assertEqual(gemma4["images"], [])
        self.assertEqual(gemma4["videos"], [])
        audio_input = gemma4["audios"][0]
        self.assertEqual(
            audio_input["span"],
            {
                "modality": "audio",
                "placeholder_index": 1,
                "replacement_start": 1,
                "soft_token_count": 2,
                "replacement_token_count": 4,
            },
        )
        self.assertEqual(audio_input["input_features"], [0.5, -0.5, 1.0, 0.0])
        self.assertEqual(audio_input["frame_count"], 2)
        self.assertEqual(audio_input["feature_count"], 2)

    def test_prepare_audio_request_rejects_encoded_audio(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_tiny_config(model_dir)

            with self.assertRaisesRegex(ValueError, "expects waveform samples"):
                module.prepare_gemma4_unified_audio_request(
                    model_dir,
                    [7, 200, 8],
                    [b"RIFF"],
                )

    def test_prepare_audio_request_decodes_wav_bytes(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_tiny_config(model_dir)

            request = module.prepare_gemma4_unified_audio_request(
                model_dir,
                [7, 200, 8],
                [tiny_wav_bytes([0, 16384, -16384], 4)],
            )

        self.assertEqual(request.input_tokens, [7, 201, 200, 200, 202, 8])
        audio_input = request.multimodal_inputs["gemma4_unified"]["audios"][0]
        self.assertEqual(audio_input["input_features"], [0.0, 0.5, -0.5, 0.0])
        self.assertEqual(audio_input["frame_count"], 2)
        self.assertEqual(audio_input["feature_count"], 2)

    def test_prepare_audio_request_decodes_data_uri_source(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_tiny_config(model_dir)
            audio_uri = "data:audio/wav;base64," + base64.b64encode(
                tiny_wav_bytes([0, 16384, -16384], 4)
            ).decode("ascii")

            request = module.prepare_gemma4_unified_audio_request(
                model_dir,
                [7, 200, 8],
                [{"url": audio_uri}],
            )

        audio_input = request.multimodal_inputs["gemma4_unified"]["audios"][0]
        self.assertEqual(audio_input["input_features"], [0.0, 0.5, -0.5, 0.0])
        self.assertEqual(audio_input["frame_count"], 2)

    def test_prepare_audio_request_accepts_openai_input_audio_dict(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_tiny_config(model_dir)
            audio_data = base64.b64encode(tiny_wav_bytes([0, 16384, -16384], 4)).decode(
                "ascii"
            )

            request = module.prepare_gemma4_unified_audio_request(
                model_dir,
                [7, 200, 8],
                [{"input_audio": {"data": audio_data, "format": "wav"}}],
            )

        audio_input = request.multimodal_inputs["gemma4_unified"]["audios"][0]
        self.assertEqual(audio_input["input_features"], [0.0, 0.5, -0.5, 0.0])
        self.assertEqual(audio_input["frame_count"], 2)

    @unittest.skipIf(Image is None, "Pillow is required for Gemma4 video preprocessing")
    def test_prepare_video_request_uses_frame_ranges_and_timestamps(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_tiny_config(model_dir)
            request = module.prepare_gemma4_unified_video_request(
                model_dir,
                [9, 300, 10],
                [[tiny_rgb_image(), tiny_rgb_image()]],
                timestamp_token_ids=[[[400], [401, 402]]],
            )

        self.assertEqual(
            request.input_tokens,
            [9, 400, 101, 300, 300, 102, 401, 402, 101, 300, 300, 102, 10],
        )
        self.assertEqual(request.soft_token_counts, [4])
        self.assertEqual(request.frame_counts, [2])
        gemma4 = request.multimodal_inputs["gemma4_unified"]
        self.assertEqual(gemma4["images"], [])
        self.assertEqual(gemma4["audios"], [])
        video_input = gemma4["videos"][0]
        self.assertEqual(
            video_input["span"],
            {
                "modality": "video",
                "placeholder_index": 1,
                "replacement_start": 1,
                "soft_token_count": 4,
                "replacement_token_count": 11,
            },
        )
        self.assertEqual(
            video_input["soft_token_ranges"],
            [
                {"start": 3, "soft_token_count": 2},
                {"start": 9, "soft_token_count": 2},
            ],
        )
        self.assertEqual(video_input["frame_count"], 2)
        self.assertEqual(len(video_input["pixel_position_ids"]), 4)
        self.assertEqual(len(video_input["pixel_values"]), 48)

    @unittest.skipIf(Image is None, "Pillow is required for Gemma4 video preprocessing")
    def test_prepare_video_request_rejects_malformed_timestamp_tokens(self) -> None:
        module = load_module()
        bad_timestamp_cases = [
            ([[[400], 401]], "frame entry"),
            ([[[400.5], [401]]], "non-integer"),
            ([[[-1], [401]]], "non-negative"),
        ]
        for timestamp_token_ids, message in bad_timestamp_cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmp:
                    model_dir = Path(tmp)
                    write_tiny_config(model_dir)
                    with self.assertRaisesRegex(ValueError, message):
                        module.prepare_gemma4_unified_video_request(
                            model_dir,
                            [9, 300, 10],
                            [[tiny_rgb_image(), tiny_rgb_image()]],
                            timestamp_token_ids=timestamp_token_ids,
                        )


class Gemma4UnifiedConfigValidationTests(unittest.TestCase):
    def test_load_config_rejects_non_positive_image_std_when_normalizing(self) -> None:
        # Zero divides every pixel into inf/NaN, a negative channel silently
        # sign-flips pixels, a subnormal-for-f32 value (1e-40) passes a naive
        # > 0 check but overflows the division once values land in f32, and
        # NaN/inf are never valid; the checkpoint config must be rejected at
        # load time for all of them.
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_tiny_config(model_dir)
            processor_path = model_dir / "processor_config.json"
            processor = json.loads(processor_path.read_text())
            processor["image_processor"]["do_normalize"] = True

            for bad_channel in [0.0, -0.5, 1e-40, 1e40, float("nan"), float("inf")]:
                with self.subTest(bad_channel=bad_channel):
                    processor["image_processor"]["image_std"] = [0.5, bad_channel, 0.5]
                    processor_path.write_text(json.dumps(processor))
                    with self.assertRaisesRegex(ValueError, "image_std"):
                        module._load_config(model_dir)

            # Reset to a finite (still invalid-for-normalization) std so the
            # loadable-without-normalization check below writes standard JSON
            # rather than leaking the loop's last non-finite value.
            processor["image_processor"]["image_std"] = [0.5, -0.5, 0.5]

            # The same std values are unused (and loadable) when
            # normalization stays off.
            processor["image_processor"]["do_normalize"] = False
            processor_path.write_text(json.dumps(processor))
            config = module._load_config(model_dir)
            self.assertFalse(config.do_normalize)

    def test_resized_dimensions_extreme_aspect_ratios_stay_nonzero(self) -> None:
        # Regression pin for the resize-extreme-aspect-zero report: the
        # single-axis fallback only runs when the floored width/height ratio
        # is >= 1, so degenerate aspect ratios still produce a non-zero patch
        # grid (matches core's resize_target test).
        module = load_module()
        self.assertEqual(
            module._resized_dimensions(
                1, 10_000, patch_size=4, pooling_kernel_size=2, max_soft_tokens=4
            ),
            (8, 32),
        )
        self.assertEqual(
            module._resized_dimensions(
                10_000, 1, patch_size=4, pooling_kernel_size=2, max_soft_tokens=4
            ),
            (32, 8),
        )
        # Moderate-aspect vectors pinned to the same values as core's
        # resize_target test, plus the divisibility contract: both dimensions
        # must stay multiples of patch_size * pooling_kernel_size.
        self.assertEqual(
            module._resized_dimensions(
                100, 5, patch_size=4, pooling_kernel_size=2, max_soft_tokens=4
            ),
            (32, 8),
        )
        self.assertEqual(
            module._resized_dimensions(
                5, 100, patch_size=4, pooling_kernel_size=2, max_soft_tokens=4
            ),
            (8, 32),
        )
        # (33, 7) also hits the fallback branch (pre-clamp height floors to 0)
        # without being a degenerate ratio.
        self.assertEqual(
            module._resized_dimensions(
                33, 7, patch_size=4, pooling_kernel_size=2, max_soft_tokens=4
            ),
            (32, 8),
        )
        for width, height in [(100, 5), (5, 100), (1, 10_000), (33, 7)]:
            with self.subTest(width=width, height=height):
                target_w, target_h = module._resized_dimensions(
                    width, height, patch_size=4, pooling_kernel_size=2, max_soft_tokens=4
                )
                self.assertEqual(target_w % 8, 0)
                self.assertEqual(target_h % 8, 0)


def write_tiny_config(
    model_dir: Path, processor_filename: str = "processor_config.json"
) -> None:
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "image_token_id": 100,
                "audio_token_id": 200,
                "video_token_id": 300,
                "boi_token_id": 101,
                "eoi_token_id": 102,
                "boa_token_id": 201,
                "eoa_token_id": 202,
                "vision_config": {
                    "patch_size": 1,
                    "model_patch_size": 2,
                    "pooling_kernel_size": 2,
                    "default_output_length": 2,
                },
                "audio_config": {
                    "audio_samples_per_token": 2,
                    "audio_embed_dim": 2,
                },
            }
        )
    )
    (model_dir / processor_filename).write_text(
        json.dumps(
            {
                "image_processor": {
                    "image_processor_type": "Gemma4UnifiedImageProcessor",
                    "do_convert_rgb": True,
                    "do_resize": False,
                    "do_rescale": False,
                    "do_normalize": False,
                    "patch_size": 1,
                    "model_patch_size": 2,
                    "pooling_kernel_size": 2,
                    "max_soft_tokens": 2,
                    "image_mean": [0.0, 0.0, 0.0],
                    "image_std": [1.0, 1.0, 1.0],
                },
                "feature_extractor": {
                    "feature_extractor_type": "Gemma4UnifiedAudioFeatureExtractor",
                    "sampling_rate": 4,
                    "audio_samples_per_token": 2,
                },
                "audio_seq_length": 3,
                "audio_ms_per_token": 500,
                "processor_class": "Gemma4UnifiedProcessor",
            }
        )
    )


def tiny_rgb_image():
    assert Image is not None
    image = Image.new("RGB", (4, 2))
    for y in range(2):
        for x in range(4):
            image.putpixel((x, y), (x, y, x + y))
    return image


def tiny_png_data_uri() -> str:
    assert Image is not None
    buffer = BytesIO()
    tiny_rgb_image().save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def tiny_wav_bytes(samples: list[int], sampling_rate: int) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sampling_rate)
        wav.writeframes(
            b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)
        )
    return buffer.getvalue()


GOLDEN_DIR = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "ax-engine-server"
    / "src"
    / "tests"
    / "fixtures"
    / "gemma4_golden"
)


def write_golden_image_config(model_dir: Path, cfg: dict) -> None:
    """Config matching scripts/gen_gemma4_golden.py so the Python SDK output can
    be compared against the same HF reference vectors the Rust path uses."""
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "image_token_id": 100,
                "audio_token_id": 200,
                "video_token_id": 300,
                "boi_token_id": 101,
                "eoi_token_id": 102,
                "boa_token_id": 201,
                "eoa_token_id": 202,
                "vision_config": {
                    "patch_size": cfg["patch_size"],
                    "model_patch_size": cfg["model_patch_size"],
                    "pooling_kernel_size": cfg["pooling_kernel_size"],
                    "default_output_length": cfg["max_soft_tokens"],
                },
            }
        )
    )
    (model_dir / "processor_config.json").write_text(
        json.dumps(
            {
                "image_processor": {
                    "image_processor_type": "Gemma4UnifiedImageProcessor",
                    "do_convert_rgb": True,
                    "do_resize": True,
                    "do_rescale": True,
                    "rescale_factor": 1.0 / 255.0,
                    "do_normalize": False,
                    "patch_size": cfg["patch_size"],
                    "model_patch_size": cfg["model_patch_size"],
                    "pooling_kernel_size": cfg["pooling_kernel_size"],
                    "max_soft_tokens": cfg["max_soft_tokens"],
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                },
                "processor_class": "Gemma4UnifiedProcessor",
            }
        )
    )


class Gemma4UnifiedGoldenParityTests(unittest.TestCase):
    """Validate the Python SDK preprocessing against the same golden vectors the
    Rust server path uses, so the two parallel implementations cannot drift apart
    from the HF reference."""

    @unittest.skipIf(Image is None, "Pillow is required for Gemma4 image preprocessing")
    def test_image_preprocessing_matches_golden_reference(self) -> None:
        golden_path = GOLDEN_DIR / "golden_noresize.json"
        if not golden_path.is_file():
            self.skipTest("golden fixtures not present")
        module = load_module()
        golden = json.loads(golden_path.read_text())

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_golden_image_config(model_dir, golden["config"])
            image = Image.open(GOLDEN_DIR / "image_noresize.png").convert("RGB")
            request = module.prepare_gemma4_unified_image_request(model_dir, [100], [image])

        image_input = request.multimodal_inputs["gemma4_unified"]["images"][0]
        self.assertEqual(
            image_input["pixel_position_ids"],
            [list(pair) for pair in golden["positions"]],
        )
        expected = golden["pixel_values"]
        actual = image_input["pixel_values"]
        self.assertEqual(len(actual), len(expected))
        # No-resize fixture: patchify / normalize / positions must match exactly.
        max_diff = max(abs(a - b) for a, b in zip(actual, expected))
        self.assertLess(max_diff, 1e-6, f"pixel diff vs reference = {max_diff}")


if __name__ == "__main__":
    unittest.main()
