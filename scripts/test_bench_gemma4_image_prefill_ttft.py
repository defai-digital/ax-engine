#!/usr/bin/env python3
"""Unit tests for bench_gemma4_image_prefill_ttft.py.

No companion test file previously existed; scope limited to the specific
clock-origin bug fixed here rather than full coverage of the script. PIL and
tokenizers (unconditional top-level imports in the script) are not installed
in this environment and are unused by the code path under test, so they are
stubbed in sys.modules before loading the script.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

if "PIL" not in sys.modules:
    pil_module = types.ModuleType("PIL")
    pil_image_module = types.ModuleType("PIL.Image")
    pil_module.Image = pil_image_module  # type: ignore[attr-defined]
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = pil_image_module

if "tokenizers" not in sys.modules:
    tokenizers_module = types.ModuleType("tokenizers")

    class _StubTokenizer:
        @staticmethod
        def from_file(*args: object, **kwargs: object) -> "_StubTokenizer":
            raise NotImplementedError

    tokenizers_module.Tokenizer = _StubTokenizer  # type: ignore[attr-defined]
    sys.modules["tokenizers"] = tokenizers_module

SCRIPT_PATH = Path(__file__).with_name("bench_gemma4_image_prefill_ttft.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "bench_gemma4_image_prefill_ttft", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


class _FakeResponse:
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

    def read(self, _n: int) -> bytes:
        return b""


class _FakeConnection:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def request(self, *_args: object, **_kwargs: object) -> None:
        pass

    def getresponse(self) -> _FakeResponse:
        return _FakeResponse()

    def close(self) -> None:
        pass


class BenchGemma4ImagePrefillTtftTests(unittest.TestCase):
    def test_client_wall_ttft_includes_payload_serialization_time(self) -> None:
        # Regression test: `started` used to be captured with
        # time.perf_counter() *after* json.dumps(...) built the request
        # payload (which includes potentially large input_tokens/
        # multimodal_inputs), silently excluding serialization time from
        # every reported client-wall latency metric.
        real_dumps = json.dumps

        def slow_dumps(*args: object, **kwargs: object) -> str:
            time.sleep(0.05)
            return real_dumps(*args, **kwargs)

        request = mod.PreparedImageRequest(
            input_tokens=[1, 2, 3],
            multimodal_inputs={},
            original_prompt_tokens=3,
            expanded_prompt_tokens=3,
            image_soft_tokens=0,
        )

        with mock.patch("http.client.HTTPConnection", _FakeConnection), mock.patch(
            "json.dumps", side_effect=slow_dumps
        ):
            result = mod.run_one(
                "http://127.0.0.1:1234", "test-model", request, max_output_tokens=8
            )

        self.assertIsNotNone(result["client_wall_ttft_ms"])
        self.assertGreaterEqual(result["client_wall_ttft_ms"], 40.0)
        self.assertGreaterEqual(result["client_wall_total_ms"], 40.0)


if __name__ == "__main__":
    unittest.main()
