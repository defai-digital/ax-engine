#!/usr/bin/env python3
"""Unit tests for bench_gemma4_multimodal.py.

No companion test file previously existed; scope limited to the specific
clock-origin bug fixed here rather than full coverage of the script.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).with_name("bench_gemma4_multimodal.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_gemma4_multimodal", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


def _prepared_case(**overrides: object) -> "mod.PreparedCase":
    defaults = dict(
        case_id="c1",
        description="",
        modalities=["image"],
        fixture_ids=["img1"],
        input_tokens=[1, 2, 3],
        multimodal_inputs={},
        original_prompt_tokens=3,
        expanded_prompt_tokens=3,
        image_soft_tokens=[0],
        audio_soft_tokens=[],
        video_soft_tokens=[],
        video_frame_counts=[],
        span_order=[],
        video_timestamp_seconds=[],
        chat_content=[{"type": "text", "text": "hi"}],
        chat_enabled=True,
    )
    defaults.update(overrides)
    return mod.PreparedCase(**defaults)


class _FakeStreamResponse:
    status = 200

    def __iter__(self):
        return iter(
            [
                b"event:response\n",
                b'data: {"response": {"output_tokens": [1, 2, 3], '
                b'"route": {"execution_plan": "prefill"}}}\n',
                b"\n",
            ]
        )

    def read(self, _n: int = -1) -> bytes:
        return b""


_FAKE_CHAT_RESPONSE_BODY = json.dumps(
    {
        "choices": [{"message": {"content": "hello", "reasoning_content": ""}}],
        "usage": {"completion_tokens": 2, "prompt_tokens": 3},
    }
).encode("utf-8")


class _FakeJsonResponse:
    status = 200

    def read(self) -> bytes:
        # Precomputed above (before any json.dumps patching) so this fake
        # response's own body construction can't contaminate a test that
        # patches json.dumps to measure payload-serialization timing.
        return _FAKE_CHAT_RESPONSE_BODY


class _FakeConnection:
    response_factory = _FakeStreamResponse

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def request(self, *_args: object, **_kwargs: object) -> None:
        pass

    def getresponse(self):
        return self.response_factory()

    def close(self) -> None:
        pass


def _slow_dumps(real_dumps):
    def _inner(*args: object, **kwargs: object) -> str:
        time.sleep(0.05)
        return real_dumps(*args, **kwargs)

    return _inner


class ClockOriginTests(unittest.TestCase):
    def test_run_native_one_ttft_includes_payload_serialization_time(self) -> None:
        # Regression test: `started` used to be captured after json.dumps(...)
        # built the (potentially large multimodal) request payload,
        # silently excluding serialization time from client-wall metrics.
        real_dumps = json.dumps
        with mock.patch("http.client.HTTPConnection", _FakeConnection), mock.patch(
            "json.dumps", side_effect=_slow_dumps(real_dumps)
        ):
            result = mod.run_native_one(
                "http://127.0.0.1:1234",
                "test-model",
                _prepared_case(),
                max_output_tokens=8,
                timeout_s=5,
            )

        self.assertIsNotNone(result["client_wall_ttft_ms"])
        self.assertGreaterEqual(result["client_wall_ttft_ms"], 40.0)
        self.assertGreaterEqual(result["client_wall_total_ms"], 40.0)

    def test_run_chat_one_wall_ms_includes_payload_serialization_time(self) -> None:
        real_dumps = json.dumps

        class _JsonConnection(_FakeConnection):
            response_factory = _FakeJsonResponse

        with mock.patch("http.client.HTTPConnection", _JsonConnection), mock.patch(
            "json.dumps", side_effect=_slow_dumps(real_dumps)
        ):
            result = mod.run_chat_one(
                "http://127.0.0.1:1234",
                "test-model",
                _prepared_case(),
                max_output_tokens=8,
                timeout_s=5,
            )

        self.assertGreaterEqual(result["client_wall_ms"], 40.0)
        self.assertGreaterEqual(result["non_streaming_total_ms"], 40.0)


if __name__ == "__main__":
    unittest.main()
