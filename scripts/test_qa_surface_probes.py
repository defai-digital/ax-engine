#!/usr/bin/env python3
"""Offline unit tests for product-surface probe helpers (no live server)."""

from __future__ import annotations

import json
import base64
import struct
import zlib
import sys
import unittest
from pathlib import Path
import unittest.mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

from surface_probes import (  # noqa: E402
    SurfaceReport,
    chat_completion_payload,
    extract_chat_content,
    extract_sse_chat_text,
    model_advertises_image,
    model_advertises_video,
    normalize_answer_text,
    openclaw_sse_contract,
    probe_cancel_request,
    probe_concurrent_chat,
    probe_multimodal_image,
    probe_remote_media_rejected,
    probe_stream_and_nonstream,
    probe_tools_schema,
    probe_video_rejected,
    tiny_png_data_url,
)


class SurfaceProbeHelperTests(unittest.TestCase):
    def test_png_fixture_crc_and_pixel_data(self):
        data = base64.b64decode(tiny_png_data_url().split(",", 1)[1])
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        offset, image = 8, b""
        while offset < len(data):
            length = struct.unpack(">I", data[offset:offset + 4])[0]
            kind = data[offset + 4:offset + 8]
            payload = data[offset + 8:offset + 8 + length]
            crc = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])[0]
            self.assertEqual(crc, zlib.crc32(kind + payload))
            if kind == b"IDAT":
                image += payload
            offset += length + 12
        self.assertEqual(zlib.decompress(image), b"\x00\x00\x00\xff")

    def test_chat_completion_payload_with_tools(self) -> None:
        payload = chat_completion_payload(
            "m",
            "hi",
            tools=[{"type": "function", "function": {"name": "x"}}],
        )
        self.assertEqual(payload["model"], "m")
        self.assertEqual(payload["messages"][0]["content"], "hi")
        self.assertIn("tools", payload)
        self.assertEqual(payload["tool_choice"], "auto")

    def test_extract_chat_content(self) -> None:
        self.assertEqual(
            extract_chat_content(
                {"choices": [{"message": {"content": "ok"}}]}
            ),
            "ok",
        )
        self.assertIsNone(extract_chat_content({"choices": [{"message": {}}]}))
        self.assertIsNone(extract_chat_content("not-json-dict"))

    def test_tiny_png_data_url(self) -> None:
        url = tiny_png_data_url()
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertGreater(len(url), 40)

    def test_concurrent_chat_all_ok(self) -> None:
        def fake_post(url, payload, timeout=60.0):
            return 200, {"choices": [{"message": {"content": "ok"}}]}

        with unittest.mock.patch("surface_probes._post_json", side_effect=fake_post):
            result = probe_concurrent_chat("http://127.0.0.1:9", "m", workers=2)
        self.assertTrue(result.passed)
        self.assertEqual(result.name, "concurrent_chat")

    def test_tools_schema_soft_skip_on_422(self) -> None:
        with unittest.mock.patch(
            "surface_probes._post_json", return_value=(422, {"error": "no tools"})
        ):
            result = probe_tools_schema("http://127.0.0.1:9", "m")
        self.assertTrue(result.passed)
        self.assertTrue(result.skipped)
        self.assertFalse(result.hard)

    def test_tools_schema_fails_on_500(self) -> None:
        with unittest.mock.patch(
            "surface_probes._post_json", return_value=(500, {"error": "panic"})
        ):
            result = probe_tools_schema("http://127.0.0.1:9", "m")
        self.assertFalse(result.passed)
        self.assertTrue(result.hard)

    def test_multimodal_soft_skip_on_400_without_capability(self) -> None:
        with unittest.mock.patch(
            "surface_probes._post_json", return_value=(400, {"error": "no vision"})
        ):
            result = probe_multimodal_image(
                "http://127.0.0.1:9", "m", require_image=False
            )
        self.assertTrue(result.skipped)

    def test_multimodal_hard_fail_when_capability_claimed(self) -> None:
        with unittest.mock.patch(
            "surface_probes._post_json", return_value=(400, {"error": "no vision"})
        ):
            result = probe_multimodal_image(
                "http://127.0.0.1:9", "m", require_image=True
            )
        self.assertFalse(result.passed)
        self.assertTrue(result.hard)

    def test_stream_parity_ok(self) -> None:
        def fake_post(url, payload, timeout=60.0):
            return 200, {"choices": [{"message": {"content": "7"}}]}

        sse = (
            'data: {"choices":[{"delta":{"content":"7"}}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":4,'
            '"completion_tokens":1,"total_tokens":5}}\n\n'
            "data: [DONE]\n"
        )

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return sse.encode()

        with unittest.mock.patch("surface_probes._post_json", side_effect=fake_post), unittest.mock.patch(
            "surface_probes.urllib.request.urlopen", return_value=_Resp()
        ):
            result = probe_stream_and_nonstream(
                "http://127.0.0.1:9", "m", require_parity=True
            )
        self.assertTrue(result.passed)
        self.assertIn("parity=ok", result.detail)

    def test_sse_and_normalize_helpers(self) -> None:
        self.assertEqual(
            extract_sse_chat_text(
                'data: {"choices":[{"delta":{"content":"hi"}}]}\ndata: [DONE]\n'
            ),
            "hi",
        )
        self.assertEqual(normalize_answer_text("  A  B\n"), "a b")
        self.assertEqual(
            openclaw_sse_contract(
                'data: {"choices":[],"usage":{"total_tokens":5}}\n'
                "data: [DONE]\n"
            ),
            (True, True),
        )
        self.assertTrue(
            model_advertises_image(
                {"capabilities": {"input": {"image": True}}}
            )
        )

    def test_video_capability_requires_real_success(self):
        self.assertTrue(model_advertises_video({"capabilities": {"input": {"video": True}}}))
        self.assertFalse(model_advertises_video({"capabilities": {"input": {"image": True}}}))
        for response, expected in [((200, {"choices": [{"message": {"content": "One frame"}}], "usage": {"prompt_tokens": 128}}), True),
                                   ((200, {"choices": [{"message": {"content": ""}}]}), False),
                                   ((400, {"error": "unsupported"}), False), ((500, {}), False)]:
            baseline = (200, {"usage": {"prompt_tokens": 12}})
            with unittest.mock.patch("surface_probes._post_json", side_effect=[response, baseline]):
                self.assertEqual(probe_video_rejected("http://x", "m", require_video=True).passed, expected)
            if response[0] == 200:
                with unittest.mock.patch("surface_probes._post_json", return_value=response):
                    self.assertFalse(probe_video_rejected("http://x", "m").passed)
        ignored = (200, {"choices": [{"message": {"content": "I cannot see video"}}],
                         "usage": {"prompt_tokens": 12}})
        with unittest.mock.patch("surface_probes._post_json", side_effect=[ignored, baseline]):
            self.assertFalse(probe_video_rejected("http://x", "m", require_video=True).passed)
        for status in (401, 404, 429):
            with unittest.mock.patch("surface_probes._post_json", return_value=(status, {"error": "video unsupported"})):
                self.assertFalse(probe_video_rejected("http://x", "m").passed)

    def test_media_policy_probes(self) -> None:
        with unittest.mock.patch(
            "surface_probes._post_json", return_value=(422, {"error": "video unsupported; remote media disallowed"})
        ):
            self.assertTrue(probe_remote_media_rejected("http://x", "m").passed)
            self.assertTrue(probe_video_rejected("http://x", "m").passed)

    def test_cancel_skipped_on_404(self) -> None:
        with unittest.mock.patch(
            "surface_probes._post_json", return_value=(404, "missing")
        ):
            result = probe_cancel_request("http://127.0.0.1:9", "m")
        self.assertTrue(result.skipped)

    def test_cancel_success_path(self) -> None:
        responses = [
            (201, {"request_id": 7, "state": "waiting"}),
            (200, {"state": "cancelled", "cancel_requested": True}),
        ]

        def fake_post(url, payload, timeout=60.0):
            return responses.pop(0)

        with unittest.mock.patch("surface_probes._post_json", side_effect=fake_post):
            result = probe_cancel_request("http://127.0.0.1:9", "m")
        self.assertTrue(result.passed)
        self.assertFalse(result.skipped)

    def test_cancel_accepts_http_201_submit(self) -> None:
        """Server returns 201 Created for /v1/requests submit."""
        responses = [
            (201, {"request_id": 3, "state": "waiting"}),
            (200, {"state": "cancelled", "cancel_requested": True}),
        ]
        with unittest.mock.patch(
            "surface_probes._post_json", side_effect=lambda *a, **k: responses.pop(0)
        ):
            result = probe_cancel_request("http://127.0.0.1:9", "m")
        self.assertTrue(result.passed)

    def test_surface_report_hard_pass_with_soft_skip(self) -> None:
        report = SurfaceReport(base_url="u", model="m")
        from surface_probes import SurfaceProbeResult

        report.results = [
            SurfaceProbeResult("a", True, hard=True),
            SurfaceProbeResult("b", True, hard=False, skipped=True),
        ]
        self.assertTrue(report.hard_passed)
        payload = report.as_dict()
        self.assertEqual(payload["kind"], "surface_probes")
        self.assertTrue(payload["hard_passed"])
        json.dumps(payload)  # serializable


if __name__ == "__main__":
    unittest.main()
