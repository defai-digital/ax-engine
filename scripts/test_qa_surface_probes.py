#!/usr/bin/env python3
"""Offline unit tests for product-surface probe helpers (no live server)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

from surface_probes import (  # noqa: E402
    SurfaceReport,
    chat_completion_payload,
    extract_chat_content,
    probe_cancel_request,
    probe_concurrent_chat,
    probe_multimodal_image,
    probe_tools_schema,
    tiny_png_data_url,
)


class SurfaceProbeHelperTests(unittest.TestCase):
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

        with mock.patch("surface_probes._post_json", side_effect=fake_post):
            result = probe_concurrent_chat("http://127.0.0.1:9", "m", workers=2)
        self.assertTrue(result.passed)
        self.assertEqual(result.name, "concurrent_chat")

    def test_tools_schema_soft_skip_on_422(self) -> None:
        with mock.patch(
            "surface_probes._post_json", return_value=(422, {"error": "no tools"})
        ):
            result = probe_tools_schema("http://127.0.0.1:9", "m")
        self.assertTrue(result.passed)
        self.assertTrue(result.skipped)
        self.assertFalse(result.hard)

    def test_tools_schema_fails_on_500(self) -> None:
        with mock.patch(
            "surface_probes._post_json", return_value=(500, {"error": "panic"})
        ):
            result = probe_tools_schema("http://127.0.0.1:9", "m")
        self.assertFalse(result.passed)
        self.assertTrue(result.hard)

    def test_multimodal_soft_skip_on_400(self) -> None:
        with mock.patch(
            "surface_probes._post_json", return_value=(400, {"error": "no vision"})
        ):
            result = probe_multimodal_image("http://127.0.0.1:9", "m")
        self.assertTrue(result.skipped)

    def test_cancel_skipped_on_404(self) -> None:
        with mock.patch(
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

        with mock.patch("surface_probes._post_json", side_effect=fake_post):
            result = probe_cancel_request("http://127.0.0.1:9", "m")
        self.assertTrue(result.passed)
        self.assertFalse(result.skipped)

    def test_cancel_accepts_http_201_submit(self) -> None:
        """Server returns 201 Created for /v1/requests submit."""
        responses = [
            (201, {"request_id": 3, "state": "waiting"}),
            (200, {"state": "cancelled", "cancel_requested": True}),
        ]
        with mock.patch(
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
