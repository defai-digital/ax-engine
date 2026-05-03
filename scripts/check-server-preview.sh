#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-127.0.0.1}"
LOG_FILE="$(mktemp "${TMPDIR:-/tmp}/ax-engine-server-check.XXXXXX.log")"
UPSTREAM_LOG_FILE="$(mktemp "${TMPDIR:-/tmp}/ax-engine-upstream-check.XXXXXX.log")"

allocate_port() {
    "$PYTHON_BIN" - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

PORT="$(allocate_port)"
COMPAT_PORT="$(allocate_port)"
UPSTREAM_PORT="$(allocate_port)"

SERVER_PID=""
UPSTREAM_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ -n "$UPSTREAM_PID" ]] && kill -0 "$UPSTREAM_PID" 2>/dev/null; then
        kill "$UPSTREAM_PID" 2>/dev/null || true
        wait "$UPSTREAM_PID" 2>/dev/null || true
    fi
    rm -f "$LOG_FILE"
    rm -f "$UPSTREAM_LOG_FILE"
}

trap cleanup EXIT

cd "$ROOT_DIR"

: "${AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR:?AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR is required for MLX server preview smoke}"
cargo run -p ax-engine-server -- --host "$HOST" --port "$PORT" --model-id qwen3_5_9b_q4 --mlx --mlx-model-artifacts-dir "$AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR" >"$LOG_FILE" 2>&1 &
SERVER_PID="$!"

AX_ENGINE_SERVER_URL="http://${HOST}:${PORT}" "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


BASE_URL = os.environ["AX_ENGINE_SERVER_URL"]


def request_json(method: str, path: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def request_text(method: str, path: str, payload: dict | None = None) -> str:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode("utf-8")


def parse_openai_sse_payloads(body: str) -> list[dict]:
    payloads: list[dict] = []
    current_data: list[str] = []
    for line in body.splitlines():
        if line.startswith("data: "):
            value = line[len("data: ") :]
            if value == "[DONE]":
                break
            current_data.append(value)
            continue
        if not line and current_data:
            payloads.append(json.loads("\n".join(current_data)))
            current_data = []
    if current_data:
        payloads.append(json.loads("\n".join(current_data)))
    return payloads


for _ in range(100):
    try:
        health = request_json("GET", "/health")
        break
    except (urllib.error.URLError, ConnectionError):
        time.sleep(0.1)
else:
    raise RuntimeError("ax-engine-server preview did not become ready in time")

assert health["status"] == "ok"
assert health["runtime"]["selected_backend"] == "mlx"
assert health["runtime"]["support_tier"] == "mlx_preview"
assert health["runtime"]["resolution_policy"] == "mlx_only"
assert health["runtime"]["mlx_runtime"]["runner"] in {
    "deterministic",
    "metal_bringup",
}

runtime = request_json("GET", "/v1/runtime")
assert runtime["service"] == "ax-engine-server"
assert runtime["model_id"] == "qwen3_5_9b_q4"
assert runtime["runtime"]["selected_backend"] == "mlx"
assert runtime["runtime"]["mlx_runtime"]["runner"] in {
    "deterministic",
    "metal_bringup",
}

generate = request_json(
    "POST",
    "/v1/generate",
    {
        "model": "qwen3_5_9b_q4",
        "input_tokens": [1, 2, 3],
        "max_output_tokens": 2,
    },
)
assert generate["status"] == "finished"
assert generate["output_tokens"] == [4, 5]
assert generate["runtime"]["support_tier"] == "mlx_preview"

submit = request_json(
    "POST",
    "/v1/requests",
    {
        "model": "qwen3_5_9b_q4",
        "input_tokens": [7, 8, 9],
        "max_output_tokens": 2,
    },
)
request_id = submit["request_id"]
assert request_id >= 1
assert submit["state"] == "waiting"

cancel = request_json("POST", f"/v1/requests/{request_id}/cancel")
assert cancel["state"] == "cancelled"
assert cancel["cancel_requested"] is True

sse_body = request_text(
    "POST",
    "/v1/generate/stream",
    {
        "model": "qwen3_5_9b_q4",
        "input_tokens": [1, 2, 3],
        "max_output_tokens": 2,
    },
)

events: list[tuple[str, dict]] = []
current_name: str | None = None
current_data: list[str] = []
for line in sse_body.splitlines():
    if not line:
        if current_name is not None:
            events.append((current_name, json.loads("\n".join(current_data))))
            current_name = None
            current_data = []
        continue
    if line.startswith("event: "):
        current_name = line[len("event: ") :]
    elif line.startswith("data: "):
        current_data.append(line[len("data: ") :])

if current_name is not None:
    events.append((current_name, json.loads("\n".join(current_data))))

assert [name for name, _ in events] == ["request", "step", "step", "step", "response"]
assert events[0][1]["request"]["state"] == "waiting"
assert events[-1][1]["response"]["output_tokens"] == [4, 5]
PY

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

AX_ENGINE_LLAMA_CPP_UPSTREAM_PORT="$UPSTREAM_PORT" "$PYTHON_BIN" - <<'PY' >"$UPSTREAM_LOG_FILE" 2>&1 &
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PORT = int(os.environ["AX_ENGINE_LLAMA_CPP_UPSTREAM_PORT"])


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/completion":
            self.send_error(404)
            return

        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))

        if payload.get("stream"):
            body = (
                'data: {"content":"llama","tokens":[41],"stop":false}\n\n'
                'data: {"content":" stream","tokens":[42],"stop":true,"stop_type":"limit"}\n\n'
                'data: [DONE]\n\n'
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        prompt = payload.get("prompt")
        if isinstance(prompt, list):
            content = "llama tokens"
            tokens = [31, 32]
        else:
            content = f"llama::{prompt}"
            tokens = [21, 22]

        body = json.dumps(
            {
                "content": content,
                "tokens": tokens,
                "stop": True,
                "stop_type": "limit",
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
server.serve_forever()
PY
UPSTREAM_PID="$!"

cargo run -p ax-engine-server -- \
  --host "$HOST" \
  --port "$COMPAT_PORT" \
  --support-tier llama_cpp \
  --llama-server-url "http://${HOST}:${UPSTREAM_PORT}" >"$LOG_FILE" 2>&1 &
SERVER_PID="$!"

AX_ENGINE_SERVER_URL="http://${HOST}:${COMPAT_PORT}" "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


BASE_URL = os.environ["AX_ENGINE_SERVER_URL"]


def request_json(method: str, path: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def request_text(method: str, path: str, payload: dict | None = None) -> str:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode("utf-8")


def parse_openai_sse_payloads(body: str) -> list[dict]:
    payloads: list[dict] = []
    current_data: list[str] = []
    for line in body.splitlines():
        if line.startswith("data: "):
            value = line[len("data: ") :]
            if value == "[DONE]":
                break
            current_data.append(value)
            continue
        if not line and current_data:
            payloads.append(json.loads("\n".join(current_data)))
            current_data = []
    if current_data:
        payloads.append(json.loads("\n".join(current_data)))
    return payloads


for _ in range(100):
    try:
        health = request_json("GET", "/health")
        break
    except (urllib.error.URLError, ConnectionError):
        time.sleep(0.1)
else:
    raise RuntimeError("llama.cpp ax-engine-server preview did not become ready in time")

assert health["status"] == "ok"
assert health["runtime"]["selected_backend"] == "llama_cpp"
assert health["runtime"]["support_tier"] == "llama_cpp"
assert health["runtime"]["resolution_policy"] == "allow_llama_cpp"
assert "mlx_runtime" not in health["runtime"]

runtime = request_json("GET", "/v1/runtime")
assert runtime["runtime"]["selected_backend"] == "llama_cpp"
assert runtime["runtime"]["support_tier"] == "llama_cpp"
assert "mlx_runtime" not in runtime["runtime"]

text_generate = request_json(
    "POST",
    "/v1/generate",
    {
        "model": "qwen3_dense",
        "input_text": "hello llama.cpp",
        "max_output_tokens": 2,
    },
)
assert text_generate["prompt_text"] == "hello llama.cpp"
assert text_generate["output_text"] == "llama::hello llama.cpp"
assert text_generate["runtime"]["selected_backend"] == "llama_cpp"

token_generate = request_json(
    "POST",
    "/v1/generate",
    {
        "model": "qwen3_dense",
        "input_tokens": [1, 2, 3],
        "max_output_tokens": 2,
    },
)
assert token_generate["output_tokens"] == [31, 32]
assert token_generate["output_text"] == "llama tokens"

openai_completion = request_json(
    "POST",
    "/v1/completions",
    {
        "model": "qwen3_dense",
        "prompt": "hello openai completion",
        "max_tokens": 2,
    },
)
assert openai_completion["object"] == "text_completion"
assert openai_completion["choices"][0]["text"] == "llama::hello openai completion"
assert openai_completion["choices"][0]["finish_reason"] == "length"

openai_chat = request_json(
    "POST",
    "/v1/chat/completions",
    {
        "model": "qwen3_dense",
        "messages": [
            {
                "role": "user",
                "content": "hello openai chat"
            }
        ],
        "max_tokens": 2,
    },
)
assert openai_chat["object"] == "chat.completion"
assert openai_chat["choices"][0]["message"]["role"] == "assistant"
assert openai_chat["choices"][0]["message"]["content"] == "llama::user: hello openai chat\nassistant:"

submit = request_json(
    "POST",
    "/v1/requests",
    {
        "model": "qwen3_dense",
        "input_tokens": [1, 2, 3],
        "max_output_tokens": 2,
    },
)
request_id = submit["request_id"]
assert submit["state"] == "waiting"
assert submit["route"]["execution_plan"] == "llama_cpp.server_completion_stream"

first_step = request_json("POST", "/v1/step")
assert first_step["scheduled_requests"] == 1
assert first_step["scheduled_tokens"] == 1
assert first_step["ttft_events"] == 1

running = request_json("GET", f"/v1/requests/{request_id}")
assert running["state"] == "running"
assert running["output_tokens"] == [41]

second_step = request_json("POST", "/v1/step")
assert second_step["scheduled_requests"] == 1
assert second_step["scheduled_tokens"] == 1

finished = request_json("GET", f"/v1/requests/{request_id}")
assert finished["state"] == "finished"
assert finished["output_tokens"] == [41, 42]

cancel_submit = request_json(
    "POST",
    "/v1/requests",
    {
        "model": "qwen3_dense",
        "input_tokens": [7, 8, 9],
        "max_output_tokens": 2,
    },
)
cancel_request_id = cancel_submit["request_id"]
cancel = request_json("POST", f"/v1/requests/{cancel_request_id}/cancel")
assert cancel["state"] == "cancelled"
assert cancel["cancel_requested"] is True

sse_body = request_text(
    "POST",
    "/v1/generate/stream",
    {
        "model": "qwen3_dense",
        "input_tokens": [1, 2, 3],
        "max_output_tokens": 2,
    },
)

events: list[tuple[str, dict]] = []
current_name: str | None = None
current_data: list[str] = []
for line in sse_body.splitlines():
    if not line:
        if current_name is not None:
            events.append((current_name, json.loads("\n".join(current_data))))
            current_name = None
            current_data = []
        continue
    if line.startswith("event: "):
        current_name = line[len("event: ") :]
    elif line.startswith("data: "):
        current_data.append(line[len("data: ") :])

if current_name is not None:
    events.append((current_name, json.loads("\n".join(current_data))))

assert [name for name, _ in events] == ["request", "step", "step", "response"]
assert events[0][1]["runtime"]["support_tier"] == "llama_cpp"
assert events[1][1]["step"]["ttft_events"] == 1
assert events[-1][1]["response"]["output_tokens"] == [41, 42]
assert events[-1][1]["response"]["output_text"] == "llama stream"

openai_completion_stream = request_text(
    "POST",
    "/v1/completions",
    {
        "model": "qwen3_dense",
        "prompt": "hello openai stream",
        "max_tokens": 2,
        "stream": True,
    },
)
openai_completion_payloads = parse_openai_sse_payloads(openai_completion_stream)
assert openai_completion_payloads[0]["choices"][0]["text"] == "llama"
assert openai_completion_payloads[1]["choices"][0]["text"] == " stream"
assert openai_completion_payloads[-1]["choices"][0]["finish_reason"] == "length"
assert "data: [DONE]" in openai_completion_stream

openai_chat_stream = request_text(
    "POST",
    "/v1/chat/completions",
    {
        "model": "qwen3_dense",
        "messages": [
            {
                "role": "user",
                "content": "hello openai chat stream"
            }
        ],
        "max_tokens": 2,
        "stream": True,
    },
)
openai_chat_payloads = parse_openai_sse_payloads(openai_chat_stream)
assert openai_chat_payloads[0]["choices"][0]["delta"]["role"] == "assistant"
assert openai_chat_payloads[0]["choices"][0]["delta"]["content"] == "llama"
assert openai_chat_payloads[1]["choices"][0]["delta"]["content"] == " stream"
assert openai_chat_payloads[-1]["choices"][0]["finish_reason"] == "length"
assert "data: [DONE]" in openai_chat_stream
PY

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
kill "$UPSTREAM_PID" 2>/dev/null || true
wait "$UPSTREAM_PID" 2>/dev/null || true
UPSTREAM_PID=""

OPENAI_UPSTREAM_PORT="$(allocate_port)"

AX_ENGINE_LLAMA_CPP_UPSTREAM_PORT="$OPENAI_UPSTREAM_PORT" "$PYTHON_BIN" - <<'PY' >"$UPSTREAM_LOG_FILE" 2>&1 &
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PORT = int(os.environ["AX_ENGINE_LLAMA_CPP_UPSTREAM_PORT"])


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/completions":
            self.send_error(404)
            return

        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        prompt = payload.get("prompt", "")

        if payload.get("stream"):
            body = (
                'data: {"choices":[{"text":"llama","finish_reason":null}]}\n\n'
                'data: {"choices":[{"text":" stream","finish_reason":"length"}]}\n\n'
                'data: [DONE]\n\n'
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = json.dumps(
            {
                "choices": [
                    {
                        "text": f"llama::{prompt}",
                        "finish_reason": "length",
                    }
                ]
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
server.serve_forever()
PY
UPSTREAM_PID="$!"

printf "Retired llama.cpp backends are no longer launched by the preview smoke check; non-MLX inference routes through llama.cpp only.\n"
