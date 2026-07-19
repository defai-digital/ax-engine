"""
LangChain integration for AX Engine.

Provides two classes:

- ``AXEngineChatModel``: a LangChain ``BaseChatModel`` backed by the
  ax-engine-server ``/v1/chat/completions`` endpoint.
- ``AXEngineLLM``: a LangChain ``LLM`` backed by ``/v1/completions``.

Both classes require ``langchain-core`` (``pip install langchain-core``) and
a running ``ax-engine-server`` instance.

Example::

    from ax_engine.langchain import AXEngineChatModel
    from langchain_core.messages import HumanMessage

    chat = AXEngineChatModel(base_url="http://127.0.0.1:31418", max_tokens=256)
    response = chat.invoke([HumanMessage(content="Hello!")])
    print(response.content)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

try:
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.language_models.llms import LLM
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
    from langchain_core.outputs import (
        ChatGeneration,
        ChatGenerationChunk,
        ChatResult,
        GenerationChunk,
    )
except ImportError as _e:
    raise ImportError(
        "ax_engine.langchain requires langchain-core. Install it with: pip install langchain-core"
    ) from _e

_DEFAULT_BASE_URL = "http://127.0.0.1:31418"


def _message_to_openai(message: BaseMessage) -> dict:
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = getattr(message, "role", "user")
    return {"role": role, "content": message.content}


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            detail = json.loads(body).get("error", {}).get("message", "")
        except Exception:
            detail = body.decode(errors="replace")
        raise RuntimeError(f"ax-engine HTTP {exc.code}: {detail or exc.reason}") from exc


def _parse_sse_block(block: bytes) -> tuple[str | None, str] | None:
    """Parse one SSE event block into (event_type, raw_data), or None if empty."""
    # Per the SSE spec, an event may carry multiple `data:` lines that are
    # concatenated with a newline; keeping only the last one would drop content.
    data_lines: list[str] = []
    event_type: str | None = None
    for line in block.splitlines():
        line = line.decode()
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
        elif line.startswith("event:"):
            event_type = line[len("event:") :].lstrip()
    if not data_lines:
        return None
    return event_type, "\n".join(data_lines)


def _sse_error_message(raw_data: str) -> str:
    try:
        error_data = json.loads(raw_data)
        if isinstance(error_data, dict):
            err = error_data.get("error")
            if isinstance(err, dict):
                return err.get("message", "") or "unknown error"
            if err:
                return str(err)
            return "unknown error"
        return str(error_data)
    except (json.JSONDecodeError, AttributeError):
        return raw_data or "unknown error"


def _stream_sse(url: str, payload: dict, timeout: int) -> Iterator[dict]:
    """Yield decoded SSE data objects from a streaming HTTP response."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buffer = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buffer += chunk
            while b"\n\n" in buffer or b"\r\n\r\n" in buffer:
                for sep in (b"\r\n\r\n", b"\n\n"):
                    idx = buffer.find(sep)
                    if idx != -1:
                        block, buffer = buffer[:idx], buffer[idx + len(sep) :]
                        break
                else:
                    break
                parsed = _parse_sse_block(block)
                if parsed is None:
                    continue
                event_type, raw_data = parsed
                if raw_data == "[DONE]":
                    return
                # Surface mid-stream error events to the caller, matching the
                # JavaScript SDK's AxEngineStreamError behavior.
                if event_type == "error":
                    raise RuntimeError(
                        f"ax-engine stream error: {_sse_error_message(raw_data)}"
                    )
                try:
                    yield json.loads(raw_data)
                except json.JSONDecodeError:
                    continue

        # Flush a trailing event if the server closed without a final blank line.
        if buffer.strip():
            parsed = _parse_sse_block(buffer)
            if parsed is not None:
                event_type, raw_data = parsed
                if raw_data == "[DONE]":
                    return
                if event_type == "error":
                    raise RuntimeError(
                        f"ax-engine stream error: {_sse_error_message(raw_data)}"
                    )
                try:
                    yield json.loads(raw_data)
                except json.JSONDecodeError:
                    pass


class AXEngineChatModel(BaseChatModel):
    """
    LangChain ``BaseChatModel`` that calls ax-engine-server ``/v1/chat/completions``.

    Parameters
    ----------
    base_url : str
        Base URL of the ax-engine-server instance (default ``http://127.0.0.1:31418``).
    model : str, optional
        Model identifier forwarded in the request body.
    max_tokens : int, optional
        Maximum number of output tokens.
    temperature : float, optional
    top_p : float, optional
    top_k : int, optional
    min_p : float, optional
    repetition_penalty : float, optional
    stop : str or list[str], optional
    seed : int, optional
    timeout : int
        HTTP request timeout in seconds (default 300).
    """

    base_url: str = _DEFAULT_BASE_URL
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    timeout: int = 300

    @property
    def _llm_type(self) -> str:
        return "ax-engine"

    def _build_request(self, messages: list[BaseMessage], stop: list[str] | None = None) -> dict:
        req: dict = {"messages": [_message_to_openai(m) for m in messages]}
        if self.model is not None:
            req["model"] = self.model
        if self.max_tokens is not None:
            req["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            req["temperature"] = self.temperature
        if self.top_p is not None:
            req["top_p"] = self.top_p
        if self.top_k is not None:
            req["top_k"] = self.top_k
        if self.min_p is not None:
            req["min_p"] = self.min_p
        if self.repetition_penalty is not None:
            req["repetition_penalty"] = self.repetition_penalty
        effective_stop = stop or self.stop
        if effective_stop is not None:
            req["stop"] = effective_stop
        if self.seed is not None:
            req["seed"] = self.seed
        return req

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        resp = _post_json(url, self._build_request(messages, stop), self.timeout)
        choices = resp.get("choices", [])
        if not choices:
            raise RuntimeError("Server returned empty choices array")
        choice = choices[0]
        # JSON null content (e.g. tool-call turns) must become "", not None.
        content = (choice.get("message") or {}).get("content")
        text = "" if content is None else content
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=text),
                    generation_info={"finish_reason": choice.get("finish_reason")},
                )
            ],
            llm_output={"token_usage": resp.get("usage")},
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        req = {**self._build_request(messages, stop), "stream": True}
        for chunk_data in _stream_sse(url, req, self.timeout):
            choice = (chunk_data.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            text = delta.get("content") or ""
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=text),
                generation_info={"finish_reason": choice.get("finish_reason")},
            )
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk


class AXEngineLLM(LLM):
    """
    LangChain ``LLM`` that calls ax-engine-server ``/v1/completions``.

    Parameters
    ----------
    base_url : str
        Base URL of the ax-engine-server instance (default ``http://127.0.0.1:31418``).
    model : str, optional
    max_tokens : int, optional
    temperature : float, optional
    top_p : float, optional
    top_k : int, optional
    min_p : float, optional
    repetition_penalty : float, optional
    stop : str or list[str], optional
    seed : int, optional
    timeout : int
        HTTP request timeout in seconds (default 300).
    """

    base_url: str = _DEFAULT_BASE_URL
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    timeout: int = 300

    @property
    def _llm_type(self) -> str:
        return "ax-engine"

    def _build_request(self, prompt: str, stop: list[str] | None = None) -> dict:
        req: dict = {"prompt": prompt}
        if self.model is not None:
            req["model"] = self.model
        if self.max_tokens is not None:
            req["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            req["temperature"] = self.temperature
        if self.top_p is not None:
            req["top_p"] = self.top_p
        if self.top_k is not None:
            req["top_k"] = self.top_k
        if self.min_p is not None:
            req["min_p"] = self.min_p
        if self.repetition_penalty is not None:
            req["repetition_penalty"] = self.repetition_penalty
        effective_stop = stop or self.stop
        if effective_stop is not None:
            req["stop"] = effective_stop
        if self.seed is not None:
            req["seed"] = self.seed
        return req

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        url = self.base_url.rstrip("/") + "/v1/completions"
        resp = _post_json(url, self._build_request(prompt, stop), self.timeout)
        choices = resp.get("choices", [])
        if not choices:
            raise RuntimeError("Server returned empty choices array")
        text = choices[0].get("text")
        return "" if text is None else text

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        url = self.base_url.rstrip("/") + "/v1/completions"
        req = {**self._build_request(prompt, stop), "stream": True}
        for chunk_data in _stream_sse(url, req, self.timeout):
            text = (chunk_data.get("choices") or [{}])[0].get("text") or ""
            chunk = GenerationChunk(text=text)
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk
