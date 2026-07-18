#!/usr/bin/env python3
"""Offline unit tests for multimodal QA helpers (no live server / GPU)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

from multimodal_probes import (  # noqa: E402
    package_looks_like_multimodal,
    png_data_url,
    probe_image_color_content,
    run_multimodal_probes,
    solid_png_rgb,
)
from surface_probes import (  # noqa: E402
    extract_sse_chat_text,
    model_advertises_image,
    normalize_answer_text,
    probe_multimodal_image,
    probe_remote_media_rejected,
    probe_video_rejected,
)


class MultimodalPackageTests(unittest.TestCase):
    def test_package_detection_vision_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plain = root / "plain"
            plain.mkdir()
            (plain / "config.json").write_text("{}")
            self.assertFalse(package_looks_like_multimodal(plain))

            mm = root / "mm"
            mm.mkdir()
            (mm / "config.json").write_text(
                json.dumps(
                    {
                        "image_token_id": 100,
                        "vision_config": {"patch_size": 4},
                    }
                )
            )
            self.assertTrue(package_looks_like_multimodal(mm))

    def test_solid_png_header(self) -> None:
        png = solid_png_rgb(8, 8, (0, 0, 255))
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"IHDR", png)
        self.assertTrue(png_data_url(png).startswith("data:image/png;base64,"))


class CapabilityAndPolicyTests(unittest.TestCase):
    def test_model_advertises_image_from_capabilities(self) -> None:
        self.assertFalse(model_advertises_image(None))
        self.assertFalse(model_advertises_image({"id": "x"}))
        self.assertTrue(
            model_advertises_image(
                {
                    "capabilities": {
                        "input": {"image": True, "text": True, "video": False}
                    }
                }
            )
        )
        self.assertTrue(
            model_advertises_image(
                {"ax_engine": {"native_multimodal_input_supported": True}}
            )
        )
        self.assertFalse(
            model_advertises_image(
                {"capabilities": {"input": {"image": False, "text": True}}}
            )
        )

    def test_multimodal_soft_skip_only_without_require(self) -> None:
        with mock.patch(
            "surface_probes._post_json", return_value=(400, {"error": "no vision"})
        ):
            soft = probe_multimodal_image(
                "http://127.0.0.1:9", "m", require_image=False
            )
            hard = probe_multimodal_image(
                "http://127.0.0.1:9", "m", require_image=True
            )
        self.assertTrue(soft.skipped)
        self.assertTrue(soft.passed)
        self.assertFalse(hard.passed)
        self.assertFalse(hard.skipped)

    def test_remote_media_must_reject(self) -> None:
        with mock.patch(
            "surface_probes._post_json", return_value=(400, {"error": "remote"})
        ):
            ok = probe_remote_media_rejected("http://127.0.0.1:9", "m")
        self.assertTrue(ok.passed)
        with mock.patch(
            "surface_probes._post_json",
            return_value=(200, {"choices": [{"message": {"content": "ok"}}]}),
        ):
            bad = probe_remote_media_rejected("http://127.0.0.1:9", "m")
        self.assertFalse(bad.passed)

    def test_video_must_reject(self) -> None:
        with mock.patch(
            "surface_probes._post_json",
            return_value=(400, {"error": "unsupported_modality"}),
        ):
            ok = probe_video_rejected("http://127.0.0.1:9", "m")
        self.assertTrue(ok.passed)
        with mock.patch(
            "surface_probes._post_json",
            return_value=(200, {"choices": [{"message": {"content": "3"}}]}),
        ):
            bad = probe_video_rejected("http://127.0.0.1:9", "m")
        self.assertFalse(bad.passed)

    def test_sse_extract_and_normalize(self) -> None:
        raw = (
            'data: {"choices":[{"delta":{"content":"7"}}]}\n\n'
            "data: [DONE]\n"
        )
        self.assertEqual(extract_sse_chat_text(raw), "7")
        self.assertEqual(normalize_answer_text("  7 \n"), "7")

    def test_image_color_content_match(self) -> None:
        with mock.patch(
            "multimodal_probes._post_json",
            return_value=(
                200,
                {"choices": [{"message": {"content": "Blue"}}]},
            ),
        ):
            result = probe_image_color_content("http://127.0.0.1:9", "m")
        self.assertTrue(result.passed)

    def test_run_multimodal_smoke_policy_only_with_mocks(self) -> None:
        def fake_post(url, payload, timeout=60.0):
            content = payload.get("messages", [{}])[0].get("content")
            if isinstance(content, list):
                types = {part.get("type") for part in content if isinstance(part, dict)}
                if "video_url" in types:
                    return 400, {"error": "unsupported_modality"}
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    img = part.get("image_url") or {}
                    url_s = str(img.get("url") or "")
                    if url_s.startswith("https://"):
                        return 400, {"error": "remote"}
                    if url_s.startswith("data:image"):
                        return 200, {
                            "choices": [{"message": {"content": "a blue square"}}]
                        }
            return 200, {"choices": [{"message": {"content": "ok"}}]}

        with mock.patch("surface_probes._post_json", side_effect=fake_post), mock.patch(
            "multimodal_probes._post_json", side_effect=fake_post
        ), mock.patch(
            "multimodal_probes.fetch_model_card",
            return_value={
                "capabilities": {"input": {"image": True, "text": True}}
            },
        ):
            report = run_multimodal_probes(
                "http://127.0.0.1:9",
                "gemma",
                tier="smoke",
                require_image=True,
            )
        self.assertTrue(report.hard_passed)
        names = {r.name for r in report.results}
        self.assertIn("remote_media_rejected", names)
        self.assertIn("video_rejected", names)
        self.assertIn("multimodal_image", names)


if __name__ == "__main__":
    unittest.main()
