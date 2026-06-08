from __future__ import annotations

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
    def test_prepare_image_request_rejects_placeholder_mismatch(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_tiny_config(model_dir)
            image = tiny_rgb_image()

            with self.assertRaisesRegex(ValueError, "placeholder count mismatch"):
                module.prepare_gemma4_unified_image_request(model_dir, [7, 8], [image])

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


def write_tiny_config(model_dir: Path) -> None:
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
    (model_dir / "processor_config.json").write_text(
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


if __name__ == "__main__":
    unittest.main()
