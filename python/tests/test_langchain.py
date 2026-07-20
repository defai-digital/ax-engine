"""
Unit tests for ax_engine.langchain.

Uses a real embedded HTTP server (no mocks) to validate the full request/
response cycle without needing a running ax-engine-server.

langchain-core is an optional dependency. Tests are skipped if it is not
installed, so the suite remains runnable in minimal environments.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# The Python source tree (parent of this test file).
_SOURCE_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Guard: skip entire module if langchain-core is absent
# ---------------------------------------------------------------------------
try:
    from langchain_core.messages import (
        AIMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

_SKIP = not _LANGCHAIN_AVAILABLE
_SKIP_REASON = "langchain-core not installed"


# ---------------------------------------------------------------------------
# Bootstrap: make ax_engine importable without the compiled Rust extension
# ---------------------------------------------------------------------------

def _install_ax_engine_stub() -> None:
    """Insert a minimal stub for ax_engine._ax_engine so that ax_engine and
    ax_engine.langchain can be imported without the compiled PyO3 extension."""
    if "ax_engine._ax_engine" in sys.modules:
        return
    # Ensure the source tree is on the path.
    src = str(_SOURCE_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)
    # Clear any stale ax_engine modules.
    for name in list(sys.modules):
        if name == "ax_engine" or name.startswith("ax_engine."):
            del sys.modules[name]
    # Stub out the native extension.
    native = types.ModuleType("ax_engine._ax_engine")
    native.Session = object  # type: ignore[attr-defined]
    native.EngineError = RuntimeError  # type: ignore[attr-defined]
    native.EngineBackendError = RuntimeError  # type: ignore[attr-defined]
    native.EngineInferenceError = RuntimeError  # type: ignore[attr-defined]
    native.EngineStateError = RuntimeError  # type: ignore[attr-defined]
    sys.modules["ax_engine._ax_engine"] = native


def _remove_ax_engine_stub() -> None:
    """Remove the ax_engine stub from sys.modules and sys.path so that
    subsequent tests that need the real extension can import it cleanly."""
    for name in list(sys.modules):
        if name == "ax_engine" or name.startswith("ax_engine."):
            del sys.modules[name]
    src = str(_SOURCE_ROOT)
    if src in sys.path:
        sys.path.remove(src)


# ---------------------------------------------------------------------------
# Minimal embedded HTTP server
# ---------------------------------------------------------------------------

class _HeldOpenSse:
    def __init__(self, first_frame):
        self.first_frame = first_frame
        self.release = threading.Event()


class _Handler(BaseHTTPRequestHandler):
    """Serves canned responses configured by the test."""

    def log_message(self, *_):
        pass  # suppress noise

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server._last_request_path = self.path
        self.server._last_request_body = json.loads(body) if body else {}

        response = self.server._response
        if response is None:
            self.send_response(500)
            self.end_headers()
            return

        if isinstance(response, (str, _HeldOpenSse)):
            # SSE stream
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if isinstance(response, _HeldOpenSse):
                self.wfile.write(response.first_frame.encode())
                self.wfile.flush()
                response.release.wait(timeout=2)
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.wfile.write(response.encode())
            self.wfile.flush()
        else:
            # JSON response
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


class _TestServer:
    def __init__(self):
        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self._server._response = None
        self._server._last_request_path = None
        self._server._last_request_body = None
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self):
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}"

    def set_response(self, response):
        self._server._response = response

    @property
    def last_path(self):
        return self._server._last_request_path

    @property
    def last_body(self):
        return self._server._last_request_body

    def close(self):
        self._server.shutdown()
        self._server.server_close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat_response(content="Hello!", finish_reason="stop"):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen3_dense",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _completion_response(text="world"):
    return {
        "id": "cmpl-1",
        "object": "text_completion",
        "created": 1234567890,
        "model": "qwen3_dense",
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def _chat_sse(*deltas, finish_reason="stop"):
    """Build an SSE stream string for streaming chat completions."""
    lines = []
    for i, delta in enumerate(deltas):
        fr = finish_reason if i == len(deltas) - 1 else None
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": fr}],
        }
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


def _chat_tool_sse():
    deltas = [
        {
            "tool_calls": [{
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "weather", "arguments": '{"city":'},
            }]
        },
        {
            "tool_calls": [{
                "index": 0,
                "function": {"arguments": '"Toronto"}'},
            }]
        },
    ]
    lines = []
    for delta in deltas:
        chunk = {
            "id": "chatcmpl-tool",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    terminal = {
        "id": "chatcmpl-tool",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    lines.append(f"data: {json.dumps(terminal)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


def _completion_sse(*texts):
    lines = []
    for text in texts:
        chunk = {
            "id": "cmpl-1",
            "choices": [{"index": 0, "text": text, "finish_reason": None}],
        }
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestAXEngineChatModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_ax_engine_stub()
        cls.srv = _TestServer()

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()
        _remove_ax_engine_stub()

    def _make_chat(self, **kwargs):
        from ax_engine.langchain import AXEngineChatModel
        return AXEngineChatModel(base_url=self.srv.base_url, timeout=5, **kwargs)

    def test_invoke_returns_ai_message(self):
        self.srv.set_response(_chat_response("Hi there!"))
        chat = self._make_chat(max_tokens=32)
        result = chat.invoke([HumanMessage(content="Hello!")])
        self.assertIsInstance(result, AIMessage)
        self.assertEqual(result.content, "Hi there!")

    def test_null_message_content_becomes_empty_string(self):
        self.srv.set_response({
            "id": "chatcmpl-null",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None},
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        chat = self._make_chat()
        result = chat.invoke([HumanMessage(content="x")])
        self.assertEqual(result.content, "")

    def test_response_preserves_tool_calls(self):
        self.srv.set_response({
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"city":"Toronto"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })
        chat = self._make_chat()
        result = chat.invoke([HumanMessage(content="What is the weather?")])
        self.assertEqual(result.content, "")
        self.assertEqual(
            result.tool_calls,
            [{
                "name": "weather",
                "args": {"city": "Toronto"},
                "id": "call-1",
                "type": "tool_call",
            }],
        )
        self.assertEqual(result.additional_kwargs["tool_calls"][0]["id"], "call-1")

    def test_request_path(self):
        self.srv.set_response(_chat_response())
        chat = self._make_chat()
        chat.invoke([HumanMessage(content="x")])
        self.assertEqual(self.srv.last_path, "/v1/chat/completions")

    def test_request_body_messages(self):
        self.srv.set_response(_chat_response())
        chat = self._make_chat()
        chat.invoke([
            SystemMessage(content="You are AX."),
            HumanMessage(content="Hello"),
        ])
        body = self.srv.last_body
        self.assertEqual(len(body["messages"]), 2)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][1]["role"], "user")

    def test_request_body_preserves_tool_history(self):
        self.srv.set_response(_chat_response())
        chat = self._make_chat()
        chat.invoke([
            HumanMessage(content="What is the weather?"),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "weather",
                    "args": {"city": "Toronto"},
                    "id": "call-1",
                    "type": "tool_call",
                }],
            ),
            ToolMessage(content="sunny", tool_call_id="call-1", name="weather"),
            FunctionMessage(content="legacy result", name="legacy_weather"),
        ])

        messages = self.srv.last_body["messages"]
        assistant_call = messages[1]["tool_calls"][0]
        self.assertEqual(assistant_call["id"], "call-1")
        self.assertEqual(assistant_call["function"]["name"], "weather")
        self.assertEqual(
            json.loads(assistant_call["function"]["arguments"]),
            {"city": "Toronto"},
        )
        self.assertEqual(
            messages[2],
            {
                "role": "tool",
                "name": "weather",
                "tool_call_id": "call-1",
                "content": "sunny",
            },
        )
        self.assertEqual(
            messages[3],
            {"role": "function", "name": "legacy_weather", "content": "legacy result"},
        )

    def test_bind_tools_forwards_openai_contract_and_specific_choice(self):
        self.srv.set_response(_chat_response())
        chat = self._make_chat()
        bound = chat.bind_tools(
            [{
                "name": "weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }],
            tool_choice="weather",
        )
        bound.invoke([HumanMessage(content="What is the weather?")])

        self.assertEqual(self.srv.last_body["tools"][0]["type"], "function")
        self.assertEqual(
            self.srv.last_body["tools"][0]["function"]["name"],
            "weather",
        )
        self.assertEqual(
            self.srv.last_body["tool_choice"],
            {"type": "function", "function": {"name": "weather"}},
        )

    def test_sampling_params_forwarded(self):
        self.srv.set_response(_chat_response())
        chat = self._make_chat(
            max_tokens=64,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.1,
            seed=42,
        )
        chat.invoke([HumanMessage(content="x")])
        body = self.srv.last_body
        self.assertEqual(body["max_tokens"], 64)
        self.assertAlmostEqual(body["temperature"], 0.5)
        self.assertAlmostEqual(body["top_p"], 0.9)
        self.assertEqual(body["top_k"], 40)
        self.assertAlmostEqual(body["min_p"], 0.05)
        self.assertAlmostEqual(body["repetition_penalty"], 1.1)
        self.assertEqual(body["seed"], 42)

    def test_stop_forwarded(self):
        self.srv.set_response(_chat_response())
        chat = self._make_chat(stop=["<|end|>"])
        chat.invoke([HumanMessage(content="x")])
        self.assertEqual(self.srv.last_body["stop"], ["<|end|>"])

    def test_empty_per_call_stop_overrides_constructor_default(self):
        self.srv.set_response(_chat_response())
        chat = self._make_chat(stop=["<|end|>"])
        chat.invoke([HumanMessage(content="x")], stop=[])
        self.assertEqual(self.srv.last_body["stop"], [])

    def test_stream_yields_chunks(self):
        self.srv.set_response(_chat_sse("Hello", " world"))
        chat = self._make_chat()
        chunks = list(chat.stream([HumanMessage(content="x")]))
        text = "".join(c.content for c in chunks)
        self.assertEqual(text, "Hello world")

    def test_stream_yields_before_connection_closes(self):
        first = {
            "choices": [{"index": 0, "delta": {"content": "now"}, "finish_reason": None}]
        }
        response = _HeldOpenSse(f"data: {json.dumps(first)}\n\n")
        self.srv.set_response(response)
        chat = self._make_chat()
        stream = chat.stream([HumanMessage(content="x")])

        started = time.monotonic()
        first_chunk = next(stream)
        elapsed = time.monotonic() - started
        response.release.set()
        list(stream)

        self.assertEqual(first_chunk.content, "now")
        self.assertLess(elapsed, 0.5, "first SSE event was buffered until the response closed")

    def test_stream_preserves_fragmented_tool_calls(self):
        self.srv.set_response(_chat_tool_sse())
        chat = self._make_chat()
        chunks = list(chat.stream([HumanMessage(content="What is the weather?")]))
        combined = sum(chunks[1:], chunks[0])
        self.assertEqual(
            combined.tool_calls,
            [{
                "name": "weather",
                "args": {"city": "Toronto"},
                "id": "call-1",
                "type": "tool_call",
            }],
        )

    def test_stream_flushes_trailing_event_without_blank_line(self):
        # Server closes without a final \n\n after the last data frame.
        trailing = (
            'data: {"id":"c1","object":"chat.completion.chunk","choices":'
            '[{"index":0,"delta":{"content":"tail"},"finish_reason":null}]}'
        )
        self.srv.set_response(trailing)
        chat = self._make_chat()
        chunks = list(chat.stream([HumanMessage(content="x")]))
        text = "".join(c.content for c in chunks)
        self.assertEqual(text, "tail")

    def test_stream_keeps_mixed_line_ending_events_separate(self):
        first = {
            "choices": [{"index": 0, "delta": {"content": "one"}, "finish_reason": None}]
        }
        second = {
            "choices": [{"index": 0, "delta": {"content": "two"}, "finish_reason": "stop"}]
        }
        self.srv.set_response(
            f"data: {json.dumps(first)}\n\n"
            f"data: {json.dumps(second)}\r\n\r\n"
            "data: [DONE]\n\n"
        )
        chat = self._make_chat()
        chunks = list(chat.stream([HumanMessage(content="x")]))
        self.assertEqual("".join(chunk.content for chunk in chunks), "onetwo")

    def test_stream_request_has_stream_true(self):
        self.srv.set_response(_chat_sse("ok"))
        chat = self._make_chat()
        list(chat.stream([HumanMessage(content="x")]))
        self.assertTrue(self.srv.last_body.get("stream"))

    def test_model_forwarded_when_set(self):
        self.srv.set_response(_chat_response())
        chat = self._make_chat(model="qwen3_dense")
        chat.invoke([HumanMessage(content="x")])
        self.assertEqual(self.srv.last_body.get("model"), "qwen3_dense")

    def test_model_absent_when_not_set(self):
        self.srv.set_response(_chat_response())
        chat = self._make_chat()
        chat.invoke([HumanMessage(content="x")])
        self.assertNotIn("model", self.srv.last_body)

    def test_llm_type(self):
        chat = self._make_chat()
        self.assertEqual(chat._llm_type, "ax-engine")

    def test_http_error_raises_runtime_error(self):
        # server returns empty/invalid JSON → will raise
        self.srv.set_response(None)
        chat = self._make_chat()
        with self.assertRaises(RuntimeError):
            chat.invoke([HumanMessage(content="x")])


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestAXEngineLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_ax_engine_stub()
        cls.srv = _TestServer()

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()
        _remove_ax_engine_stub()

    def _make_llm(self, **kwargs):
        from ax_engine.langchain import AXEngineLLM
        return AXEngineLLM(base_url=self.srv.base_url, timeout=5, **kwargs)

    def test_invoke_returns_string(self):
        self.srv.set_response(_completion_response("world"))
        llm = self._make_llm(max_tokens=16)
        result = llm.invoke("Hello")
        self.assertEqual(result, "world")

    def test_null_completion_text_becomes_empty_string(self):
        self.srv.set_response({
            "id": "cmpl-null",
            "object": "text_completion",
            "choices": [{"index": 0, "text": None, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        })
        llm = self._make_llm()
        self.assertEqual(llm.invoke("x"), "")

    def test_request_path(self):
        self.srv.set_response(_completion_response())
        llm = self._make_llm()
        llm.invoke("test")
        self.assertEqual(self.srv.last_path, "/v1/completions")

    def test_prompt_forwarded(self):
        self.srv.set_response(_completion_response())
        llm = self._make_llm()
        llm.invoke("Once upon a time")
        self.assertEqual(self.srv.last_body["prompt"], "Once upon a time")

    def test_sampling_params_forwarded(self):
        self.srv.set_response(_completion_response())
        llm = self._make_llm(
            max_tokens=32,
            temperature=0.8,
            top_p=0.95,
            seed=7,
        )
        llm.invoke("x")
        body = self.srv.last_body
        self.assertEqual(body["max_tokens"], 32)
        self.assertAlmostEqual(body["temperature"], 0.8)
        self.assertEqual(body["seed"], 7)

    def test_empty_per_call_stop_overrides_constructor_default(self):
        self.srv.set_response(_completion_response())
        llm = self._make_llm(stop=["<|end|>"])
        llm.invoke("x", stop=[])
        self.assertEqual(self.srv.last_body["stop"], [])

    def test_stream_yields_text(self):
        self.srv.set_response(_completion_sse("Hello", " there"))
        llm = self._make_llm()
        chunks = list(llm.stream("Hi"))
        # langchain-core LLM.stream() yields the text of each GenerationChunk
        # directly as strings in newer versions, or GenerationChunk objects.
        if chunks and isinstance(chunks[0], str):
            text = "".join(chunks)
        else:
            text = "".join(c.text for c in chunks)
        self.assertEqual(text, "Hello there")

    def test_stream_request_has_stream_true(self):
        self.srv.set_response(_completion_sse("x"))
        llm = self._make_llm()
        list(llm.stream("x"))
        self.assertTrue(self.srv.last_body.get("stream"))

    def test_llm_type(self):
        llm = self._make_llm()
        self.assertEqual(llm._llm_type, "ax-engine")


if __name__ == "__main__":
    unittest.main()
