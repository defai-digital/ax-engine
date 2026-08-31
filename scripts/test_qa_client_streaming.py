#!/usr/bin/env python3
"""Regression tests: qa/client.py SSE streaming TTFT clock origin.

Exercises the shipped helpers in qa/client.py (not a reimplementation).
"""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
QA_DIR = REPO_ROOT / "qa"
if str(QA_DIR) not in sys.path:
    sys.path.insert(0, str(QA_DIR))

import client  # noqa: E402


class _FakeResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines
        self._index = 0

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, _size: int) -> bytes:
        if self._index >= len(self._lines):
            return b""
        chunk = self._lines[self._index]
        self._index += 1
        return chunk


def _slow_dumps(real_dumps):
    def _inner(*args: object, **kwargs: object) -> str:
        time.sleep(0.05)
        return real_dumps(*args, **kwargs)

    return _inner


class QaClientStreamingClockOriginTests(unittest.TestCase):
    def test_stream_sse_ttft_includes_payload_serialization_time(self) -> None:
        # Regression test: `start` used to be captured after json.dumps(...)
        # built the request payload, silently excluding serialization time
        # from the reported ttft_ms.
        real_dumps = json.dumps
        lines = [
            b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        with mock.patch("urllib.request.urlopen", return_value=_FakeResponse(lines)), \
            mock.patch("json.dumps", side_effect=_slow_dumps(real_dumps)):
            response = client._stream_sse("http://127.0.0.1:1/v1/chat/completions", {"a": 1})

        self.assertGreaterEqual(response.ttft_ms, 40.0)

    def test_stream_generate_sse_ttft_includes_payload_serialization_time(self) -> None:
        real_dumps = json.dumps
        lines = [
            b'event: response\ndata: {"response": {"output_tokens": [1], "output_text": "hi"}}\n\n',
            b"data: [DONE]\n\n",
        ]
        with mock.patch("urllib.request.urlopen", return_value=_FakeResponse(lines)), \
            mock.patch("json.dumps", side_effect=_slow_dumps(real_dumps)):
            response = client._stream_generate_sse(
                "http://127.0.0.1:1/v1/generate/stream", {"a": 1}, tokenizer=None
            )

        self.assertGreaterEqual(response.ttft_ms, 40.0)


if __name__ == "__main__":
    unittest.main()
