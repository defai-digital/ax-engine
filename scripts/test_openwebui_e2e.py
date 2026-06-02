#!/usr/bin/env python3
"""Unit tests for the OpenWebUI integration smoke probe."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
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

    def test_print_docker_openai_base_url_does_not_require_model_id(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = openwebui_e2e.main(
                ["--print-docker-openai-base-url", "http://127.0.0.1:9000/v1"]
            )

        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue().strip(), "http://host.docker.internal:9000/v1")

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
            bearer_token: str | None = None,
        ) -> dict[str, Any]:
            del method, url, timeout, bearer_token
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
            patch.object(openwebui_e2e, "signin_openwebui", return_value=None),
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
        self.assertEqual(result.observed_models, [MODEL_ID])
        self.assertEqual(result.corruption_reasons, [])
        self.assertIn("artificial general intelligence", result.assistant_text.lower())

    def test_run_probe_reports_observed_models_when_model_is_missing(self) -> None:
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
        self.assertEqual(result.observed_models, [MODEL_ID])
        self.assertIn("model not visible", result.corruption_reasons[0])

    def test_run_probe_ax_direct_accepts_clean_response(self) -> None:
        def fake_request_json(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            *,
            timeout: float,
            bearer_token: str | None = None,
        ) -> dict[str, Any]:
            del method, timeout, bearer_token
            if url.endswith("/v1/models"):
                return {"data": [{"id": MODEL_ID}]}
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
            patch.object(openwebui_e2e, "wait_for_ax_direct", return_value=None),
            patch.object(openwebui_e2e, "request_json", side_effect=fake_request_json),
        ):
            result = openwebui_e2e.run_probe(
                openwebui_base_url="http://127.0.0.1:8765",
                model_id=MODEL_ID,
                prompt="what is agi ?",
                max_tokens=16,
                timeout_secs=5,
                ax_direct=True,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.chat_path, openwebui_e2e.AX_DIRECT_CHAT_PATH)
        self.assertIn("artificial general intelligence", result.assistant_text.lower())

    def test_run_probe_ax_direct_catches_ngram_corruption(self) -> None:
        corrupted = "AGI stands for Artificial General Intelligence.\n\n!\n!\n!\n!\n!"

        def fake_request_json(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            *,
            timeout: float,
            bearer_token: str | None = None,
        ) -> dict[str, Any]:
            del method, timeout, bearer_token
            if url.endswith("/v1/models"):
                return {"data": [{"id": MODEL_ID}]}
            assert payload is not None
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": corrupted},
                        "finish_reason": "stop",
                    }
                ],
            }

        with (
            patch.object(openwebui_e2e, "wait_for_ax_direct", return_value=None),
            patch.object(openwebui_e2e, "request_json", side_effect=fake_request_json),
        ):
            result = openwebui_e2e.run_probe(
                openwebui_base_url="http://127.0.0.1:8765",
                model_id=MODEL_ID,
                prompt="what is agi ?",
                max_tokens=96,
                timeout_secs=5,
                ax_direct=True,
            )

        self.assertFalse(result.ok)
        self.assertTrue(any("punctuation" in r for r in result.corruption_reasons))

    def test_write_report_includes_observed_models(self) -> None:
        result = openwebui_e2e.ProbeResult(
            ok=False,
            model_visible=False,
            model_id="missing",
            observed_models=[MODEL_ID],
            assistant_text="",
            corruption_reasons=["model not visible through OpenWebUI proxy: missing"],
            openwebui_base_url="http://127.0.0.1:8080",
            chat_path=openwebui_e2e.OPENWEBUI_PROXY_CHAT_PATH,
        )
        report_path = Path(self._testMethodName).with_suffix(".json")
        try:
            openwebui_e2e.write_report(report_path, result)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        finally:
            report_path.unlink(missing_ok=True)

        self.assertEqual(payload["observed_models"], [MODEL_ID])


if __name__ == "__main__":
    unittest.main()
