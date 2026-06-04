"""OpenAI-compatible API client for AX Engine QA testing."""

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QaResponse:
    text: str
    finish_reason: Optional[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: float = 0.0
    elapsed_ms: float = 0.0
    stream: bool = False
    error: Optional[str] = None


def _build_messages(system: Optional[str], user: str) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def _build_payload(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    stream: bool,
    repetition_penalty: Optional[float] = None,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    return payload


def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _stream_sse(url: str, payload: dict, timeout: int = 120) -> QaResponse:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    ttft_ms = 0.0
    finish_reason = None
    prompt_tokens = 0
    completion_tokens = 0
    start = time.monotonic()

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            first_chunk = True
            buffer = ""
            done = False
            while not done:
                raw = resp.read(4096)
                if not raw:
                    break
                buffer += raw.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:") :].strip()
                    if payload_str == "[DONE]":
                        done = True
                        break
                    try:
                        event = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    if first_chunk:
                        ttft_ms = (time.monotonic() - start) * 1000
                        first_chunk = False
                    choices = event.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            chunks.append(content)
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                    usage = event.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return QaResponse(
            text="",
            finish_reason=None,
            elapsed_ms=elapsed,
            stream=True,
            error=str(e),
        )

    elapsed = (time.monotonic() - start) * 1000
    text = "".join(chunks)
    if completion_tokens == 0 and text:
        completion_tokens = max(1, len(text.split()))
    return QaResponse(
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        ttft_ms=ttft_ms,
        elapsed_ms=elapsed,
        stream=True,
    )


def send_request(
    base_url: str,
    model: str,
    system: Optional[str],
    user: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    stream: bool = False,
    repetition_penalty: Optional[float] = None,
    timeout: int = 120,
) -> QaResponse:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    messages = _build_messages(system, user)
    payload = _build_payload(
        model, messages, max_tokens, temperature, stream, repetition_penalty
    )

    if stream:
        return _stream_sse(url, payload, timeout=timeout)

    start = time.monotonic()
    try:
        result = _post_json(url, payload, timeout=timeout)
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return QaResponse(
            text="",
            finish_reason=None,
            elapsed_ms=elapsed,
            stream=False,
            error=str(e),
        )

    elapsed = (time.monotonic() - start) * 1000
    choices = result.get("choices", [])
    text = ""
    finish_reason = None
    if choices:
        text = choices[0].get("message", {}).get("content", "")
        finish_reason = choices[0].get("finish_reason")
    usage = result.get("usage", {})
    return QaResponse(
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        ttft_ms=elapsed,
        elapsed_ms=elapsed,
        stream=False,
    )
