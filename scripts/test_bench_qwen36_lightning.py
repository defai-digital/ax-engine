#!/usr/bin/env python3
"""Unit tests for bench_qwen36_lightning.py.

No companion test file previously existed; scope limited to the specific
bug fixed here rather than full coverage of the script.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).with_name("bench_qwen36_lightning.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_qwen36_lightning", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


class _FakeResponse:
    def read(self) -> bytes:
        return json.dumps({"usage": {"completion_tokens": 4}}).encode()


class _FakeConnection:
    last_body: bytes | None = None

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def request(self, _method: str, _path: str, body: bytes, headers: dict) -> None:
        _FakeConnection.last_body = body

    def getresponse(self) -> _FakeResponse:
        return _FakeResponse()

    def close(self) -> None:
        pass


class ChatRequestTests(unittest.TestCase):
    def test_chat_request_sends_configured_top_k(self) -> None:
        # Regression test: the request body only sent temperature/top_p,
        # silently dropping SAMPLING["top_k"]. The server treats an
        # omitted top_k as 0 (disabled), so every measured run actually
        # used unbounded sampling instead of the documented top_k=20
        # methodology.
        with mock.patch("http.client.HTTPConnection", _FakeConnection):
            mod._chat_request(port=1234, prompt="hi", max_tokens=8, thinking=True)

        sent = json.loads(_FakeConnection.last_body)
        self.assertIn("top_k", sent)
        self.assertEqual(sent["top_k"], mod.SAMPLING["top_k"])


if __name__ == "__main__":
    unittest.main()
