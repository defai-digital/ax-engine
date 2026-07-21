from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - minimal dependency environments.
    Image = None


SOURCE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = SOURCE_ROOT / "ax_engine" / "unlimited_ocr.py"
IMAGE_TOKEN_ID = 128_815


def load_module():
    module_name = "ax_engine_unlimited_ocr_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def write_model_contract(model_dir: Path) -> None:
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "unlimited-ocr"}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {
                        "id": IMAGE_TOKEN_ID,
                        "content": "<image>",
                        "special": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


@unittest.skipIf(Image is None, "Pillow is required for Unlimited-OCR preprocessing")
class UnlimitedOcrRequestTests(unittest.TestCase):
    def test_expands_one_placeholder_and_preserves_rgb_bytes(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_model_contract(model_dir)
            image = Image.new("RGB", (2, 1))
            image.putdata([(1, 2, 3), (4, 5, 6)])

            request = module.prepare_unlimited_ocr_image_request(
                model_dir,
                [0, IMAGE_TOKEN_ID, 7],
                [image],
            )

        self.assertEqual(request.soft_token_count, 273)
        self.assertEqual(request.input_tokens[0], 0)
        self.assertEqual(request.input_tokens[-1], 7)
        self.assertEqual(request.input_tokens.count(IMAGE_TOKEN_ID), 273)
        runtime = request.multimodal_inputs["unlimited_ocr"]
        self.assertEqual(runtime["image_token_id"], IMAGE_TOKEN_ID)
        self.assertEqual(runtime["soft_token_count"], 273)
        self.assertTrue(runtime["cropping"])
        self.assertEqual(runtime["images"][0]["width"], 2)
        self.assertEqual(runtime["images"][0]["height"], 1)
        self.assertEqual(runtime["images"][0]["rgb_bytes"], bytes([1, 2, 3, 4, 5, 6]))

    def test_reads_local_image_path(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_model_contract(model_dir)
            image_path = model_dir / "page.png"
            Image.new("RGB", (1, 1), (10, 20, 30)).save(image_path)

            request = module.prepare_unlimited_ocr_image_request(
                model_dir,
                [IMAGE_TOKEN_ID, 7],
                [image_path],
            )

        image_input = request.multimodal_inputs["unlimited_ocr"]["images"][0]
        self.assertEqual(image_input["rgb_bytes"], bytes([10, 20, 30]))

    def test_rejects_missing_or_duplicate_placeholder(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_model_contract(model_dir)
            image = Image.new("RGB", (1, 1))
            for tokens in ([0, 7], [IMAGE_TOKEN_ID, IMAGE_TOKEN_ID]):
                with self.subTest(tokens=tokens):
                    with self.assertRaisesRegex(ValueError, "exactly one <image>"):
                        module.prepare_unlimited_ocr_image_request(
                            model_dir,
                            list(tokens),
                            [image],
                        )

    def test_rejects_multiple_images(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_model_contract(model_dir)
            image = Image.new("RGB", (1, 1))
            with self.assertRaisesRegex(ValueError, "exactly one source image"):
                module.prepare_unlimited_ocr_image_request(
                    model_dir,
                    [IMAGE_TOKEN_ID],
                    [image, image],
                )

    def test_dense_portrait_uses_reference_two_by_three_crop_grid(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_model_contract(model_dir)
            image = Image.new("RGB", (911, 1287))

            request = module.prepare_unlimited_ocr_image_request(
                model_dir,
                [0, IMAGE_TOKEN_ID, 7],
                [image],
            )
            global_only = module.prepare_unlimited_ocr_image_request(
                model_dir,
                [0, IMAGE_TOKEN_ID, 7],
                [image],
                cropping=False,
            )

        self.assertEqual(request.soft_token_count, 903)
        self.assertEqual(request.input_tokens.count(IMAGE_TOKEN_ID), 903)
        self.assertTrue(request.multimodal_inputs["unlimited_ocr"]["cropping"])
        self.assertEqual(global_only.soft_token_count, 273)
        self.assertFalse(global_only.multimodal_inputs["unlimited_ocr"]["cropping"])


if __name__ == "__main__":
    unittest.main()
