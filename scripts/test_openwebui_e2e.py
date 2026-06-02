#!/usr/bin/env python3
"""Unit tests for the OpenWebUI integration smoke probe."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).with_name("openwebui_e2e.py")
MODULE_SPEC = importlib.util.spec_from_file_location("openwebui_e2e", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
openwebui_e2e = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = openwebui_e2e
MODULE_SPEC.loader.exec_module(openwebui_e2e)


MODEL_ID = "ax-model"
ASSISTANT_TEXT = "AGI means artificial general intelligence."


class OpenWebUIE2ETests(unittest.TestCase):
    def test_docker_openai_base_url_rewrites_loopback_host(self) -> None:
        self.assertEqual(
            openwebui_e2e.docker_openai_base_url("http://127.0.0.1:9000/v1"),
            "http://host.docker.internal:9000/v1",
        )
        self.assertEqual(
            openwebui_e2e.docker_openai_base_url("http://localhost:9000/v1"),
            "http://host.docker.internal:9000/v1",
        )
        self.assertEqual(
            openwebui_e2e.docker_openai_base_url("http://10.0.0.2:9000/v1"),
            "http://10.0.0.2:9000/v1",
        )

    def test_detect_corruption_catches_report_failure_shape(self) -> None:
        reasons = openwebui_e2e.detect_corruption(
            "AGI stands for Artificial General Intelligence.\n\n!\n!\n!\n!\n!",
            "what is agi ?",
        )
        self.assertIn("repeated punctuation-only lines", reasons)
        self.assertIn("repeated punctuation token pattern", reasons)

    def test_detect_corruption_catches_backend_disconnect_text(self) -> None:
        reasons = openwebui_e2e.detect_corruption(
            "upstream unavailable: ax-engine child is not reachable",
            "what is agi ?",
        )
        self.assertTrue(any("upstream unavailable" in reason for reason in reasons))
        self.assertTrue(any("child is not reachable" in reason for reason in reasons))

    def test_run_probe_accepts_clean_mock_openwebui_proxy(self) -> None:
        def fake_request_json(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            *,
            timeout: float,
        ) -> dict[str, Any]:
            del method, url, timeout
            assert payload is not None
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ASSISTANT_TEXT},
                        "finish_reason": "stop",
                    }
                ],
            }

        with (
            patch.object(openwebui_e2e, "wait_for_openwebui", return_value=None),
            patch.object(openwebui_e2e, "list_openwebui_models", return_value=[MODEL_ID]),
            patch.object(openwebui_e2e, "request_json", side_effect=fake_request_json),
        ):
            result = openwebui_e2e.run_probe(
                openwebui_base_url="http://127.0.0.1:8080",
                model_id=MODEL_ID,
                prompt="what is agi ?",
                max_tokens=16,
                timeout_secs=5,
            )

        self.assertTrue(result.ok)
        self.assertTrue(result.model_visible)
        self.assertEqual(result.corruption_reasons, [])
        self.assertIn("artificial general intelligence", result.assistant_text.lower())

    def test_run_probe_fails_when_model_is_missing(self) -> None:
        with (
            patch.object(openwebui_e2e, "wait_for_openwebui", return_value=None),
            patch.object(openwebui_e2e, "list_openwebui_models", return_value=[MODEL_ID]),
        ):
            result = openwebui_e2e.run_probe(
                openwebui_base_url="http://127.0.0.1:8080",
                model_id="missing",
                prompt="what is agi ?",
                max_tokens=16,
                timeout_secs=5,
            )

        self.assertFalse(result.ok)
        self.assertFalse(result.model_visible)
        self.assertIn("model not visible", result.corruption_reasons[0])


if __name__ == "__main__":
    unittest.main()
