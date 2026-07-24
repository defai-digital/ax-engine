#!/usr/bin/env python3
"""Unit tests for the delegated vLLM soak runner."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("soak_ax_vllm.py")
MODULE_SPEC = importlib.util.spec_from_file_location("soak_ax_vllm", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
runner = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runner
MODULE_SPEC.loader.exec_module(runner)


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int = 200):
        super().__init__(payload)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()


class DelegatedVllmSoakTests(unittest.TestCase):
    def test_plaintext_remote_endpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "restricted to loopback"):
            runner.normalize_base_url(
                "http://gpu.example.test:31418",
                allow_insecure_http=False,
            )

        self.assertEqual(
            runner.normalize_base_url(
                "http://127.0.0.1:31418",
                allow_insecure_http=False,
            ),
            "http://127.0.0.1:31418/v1",
        )

    def test_payload_uses_ordered_text_then_image_parts(self):
        image = runner.ImageInput(
            data_uri="data:image/png;base64,AA==",
            mime_type="image/png",
            byte_count=1,
            sha256="0" * 64,
        )

        payload = runner.build_payload(
            model="ax-ocr",
            prompt="<image>document parsing.",
            image=image,
            max_tokens=64,
            stream=True,
        )

        content = payload["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "image_url"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertNotIn("vllm_xargs", payload)
        self.assertNotIn("skip_special_tokens", payload)

    def test_stream_parser_requires_done_and_collects_usage(self):
        response = FakeResponse(
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"ok"},'
            b'"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
            b"data: [DONE]\n\n"
        )

        result = runner.read_stream_response(
            response,
            request_started=runner.time.perf_counter(),
        )

        self.assertTrue(result["done"])
        self.assertEqual(result["text"], "ok")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["usage"]["completion_tokens"], 1)

    def test_generation_transport_is_called_once_on_failure(self):
        calls = []

        def fail_once(_request, *, timeout):
            calls.append(timeout)
            raise OSError("ambiguous transport failure")

        with self.assertRaisesRegex(OSError, "ambiguous"):
            runner.execute_generation(
                "http://127.0.0.1:31418/v1",
                payload={"model": "ax-ocr"},
                api_key=None,
                timeout=3.0,
                stream=False,
                urlopen=fail_once,
            )

        self.assertEqual(calls, [3.0])

    def test_model_identity_must_match_exactly(self):
        payload = json.dumps({"data": [{"id": "ax-ocr-other", "object": "model"}]}).encode()

        with self.assertRaisesRegex(ValueError, "exact model identity"):
            runner.verify_model_identity(
                "http://127.0.0.1:31418/v1",
                model="ax-ocr",
                api_key=None,
                timeout=1.0,
                urlopen=lambda *_args, **_kwargs: FakeResponse(payload),
            )

    def test_summary_enforces_failure_and_growth_gates(self):
        samples = [
            {
                "ok": True,
                "elapsed_seconds": 1.0,
                "response_sha256": "a",
                "ttft_seconds": None,
                "telemetry": {
                    "process_rss_mib": {"ax-engine": 100.0},
                    "gpu": {"memory_used_mib": 1000.0},
                },
            },
            {
                "ok": False,
                "telemetry": {
                    "process_rss_mib": {"ax-engine": 900.0},
                    "gpu": {"memory_used_mib": 2000.0},
                },
            },
        ]

        summary = runner.summarize(
            samples,
            duration_completed=True,
            warmup_samples=0,
            max_failures=0,
            max_consecutive_failures=0,
            require_stable_response=True,
            require_telemetry=True,
            max_process_rss_growth_mib=512.0,
            max_gpu_memory_growth_mib=512.0,
        )

        self.assertEqual(summary["status"], "fail")
        self.assertTrue(any("failures" in reason for reason in summary["reasons"]))
        self.assertTrue(any("RSS growth" in reason for reason in summary["reasons"]))
        self.assertTrue(any("GPU memory growth" in reason for reason in summary["reasons"]))

    def test_alternate_mode_is_deterministic(self):
        self.assertEqual(runner.mode_for_iteration("alternate", 0), "non-stream")
        self.assertEqual(runner.mode_for_iteration("alternate", 1), "stream")
        self.assertEqual(runner.mode_for_iteration("stream", 2), "stream")

    def test_nvidia_optional_metrics_accept_not_available(self):
        self.assertIsNone(runner.optional_nvidia_float("[N/A]"))
        self.assertIsNone(runner.optional_nvidia_float("Not Supported"))
        self.assertEqual(runner.optional_nvidia_float("42.5"), 42.5)


if __name__ == "__main__":
    unittest.main()
