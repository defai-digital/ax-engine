"""
LangChain integration for AX Engine.

Provides two classes:

- ``AXEngineChatModel``: a LangChain ``BaseChatModel`` backed by the
  ax-engine-server ``/v1/chat/completions`` endpoint.
- ``AXEngineLLM``: a LangChain ``LLM`` backed by ``/v1/completions``.

Both classes require ``langchain-core>=0.3.11``
(``pip install "langchain-core>=0.3.11"``) and
a running ``ax-engine-server`` instance.

Example::

    from ax_engine.langchain import AXEngineChatModel
    from langchain_core.messages import HumanMessage

    chat = AXEngineChatModel(base_url="http://127.0.0.1:31418", max_tokens=256)
    response = chat.invoke([HumanMessage(content="Hello!")])
    print(response.content)
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterator, Sequence
from typing import Any

try:
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.language_models.llms import LLM
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        convert_to_openai_messages,
    )
    from langchain_core.outputs import (
        ChatGeneration,
        ChatGenerationChunk,
        ChatResult,
        GenerationChunk,
    )
    from langchain_core.utils.function_calling import convert_to_openai_tool
except ImportError as _e:
    raise ImportError(
        "ax_engine.langchain requires langchain-core>=0.3.11. Install it with: "
        'pip install "langchain-core>=0.3.11"'
    ) from _e

_DEFAULT_BASE_URL = "http://127.0.0.1:31418"

# Sampling parameters that bind()/invoke() kwargs may override per call.
_SAMPLING_PARAM_KEYS = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "seed",
)
# Arguments LangChain routes itself (named parameters such as stop and
# run_manager, or bind_tools() injections); they are not sampling overrides
# and must not be mistaken for unsupported extras.
_ROUTED_KWARGS = frozenset(
    {
        "tools",
        "tool_choice",
        "stop",
        "run_manager",
        "callbacks",
        "tags",
        "run_name",
        "run_id",
    }
)
# Standard OpenAI chat request keys that the ax-engine server accepts.
# bind()/invoke() may set them per call -- for example bind(response_format=)
# or bind_tools(..., parallel_tool_calls=False) -- and they are forwarded
# verbatim into the request body instead of being rejected as unknown.
_FORWARDED_REQUEST_KEYS = frozenset(
    {
        "response_format",
        "parallel_tool_calls",
        "stream_options",
        "logprobs",
        "top_logprobs",
        "user",
        "metadata",
        "n",
    }
)
_SUPPORTED_KWARG_KEYS = sorted(set(_SAMPLING_PARAM_KEYS) | _FORWARDED_REQUEST_KEYS)


def _merge_sampling_kwargs(req: dict, kwargs: dict) -> None:
    """Merge per-call kwargs over the instance defaults in ``req``.

    Values arriving through ``bind()`` or per-call kwargs win over the
    instance fields; an explicit ``None`` clears the key, mirroring how the
    server treats JSON null sampling parameters as absent. Recognised
    sampling parameters override the instance defaults, standard OpenAI
    chat request keys (``response_format``, ``parallel_tool_calls``, ...)
    are forwarded into the request body, and the contents of an
    ``extra_body`` dict are merged into it. LangChain-routed arguments
    (``tools``, ``tool_choice``, ``stop``, callbacks/run_manager style)
    keep their existing handling. Anything else is ignored with a warning:
    raising would break callers that bind standard kwargs the shim does
    not model explicitly.
    """
    for key, value in kwargs.items():
        if key in _ROUTED_KWARGS:
            continue
        if key == "extra_body":
            if value is None:
                continue
            if isinstance(value, dict):
                req.update(value)
            else:
                warnings.warn(
                    f"Ignoring extra_body={value!r}: expected a dict",
                    stacklevel=2,
                )
            continue
        if key in _SAMPLING_PARAM_KEYS or key in _FORWARDED_REQUEST_KEYS:
            if value is None:
                req.pop(key, None)
            else:
                req[key] = value
        else:
            warnings.warn(
                f"Ignoring unsupported argument {key!r}; supported arguments: "
                + ", ".join(_SUPPORTED_KWARG_KEYS),
                stacklevel=2,
            )


def _normalize_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, str):
        return tool_choice
    if tool_choice == "any":
        return "required"
    if tool_choice in {"auto", "none", "required"}:
        return tool_choice
    return {"type": "function", "function": {"name": tool_choice}}


def _message_additional_kwargs(message: dict[str, Any]) -> dict[str, Any]:
    additional_kwargs: dict[str, Any] = {}
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        additional_kwargs["tool_calls"] = tool_calls
    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        additional_kwargs["function_call"] = function_call
    return additional_kwargs


def _tool_call_chunks(message: dict[str, Any]) -> list[Any]:
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []
    chunks: list[Any] = []
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue
        function = raw_tool_call.get("function")
        if not isinstance(function, dict):
            function = {}
        chunks.append(
            {
                "name": function.get("name"),
                "args": function.get("arguments"),
                "id": raw_tool_call.get("id"),
                "index": raw_tool_call.get("index"),
                "type": "tool_call_chunk",
            }
        )
    return chunks


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
        raise _http_error_to_runtime(exc) from exc


def _http_error_to_runtime(exc: urllib.error.HTTPError) -> RuntimeError:
    """Map an HTTP failure to the SDK's RuntimeError with the server message."""
    body = exc.read()
    try:
        detail = json.loads(body).get("error", {}).get("message", "")
    except Exception:
        detail = body.decode(errors="replace")
    return RuntimeError(f"ax-engine HTTP {exc.code}: {detail or exc.reason}")


def _parse_sse_block(block: bytes) -> tuple[str | None, str] | None:
    """Parse one SSE event block into (event_type, raw_data), or None if empty."""
    # Per the SSE spec, an event may carry multiple `data:` lines that are
    # concatenated with a newline; keeping only the last one would drop content.
    data_lines: list[str] = []
    event_type: str | None = None
    for raw_line in block.splitlines():
        line = raw_line.decode()
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
    # Same HTTP failure contract as the blocking path: a 4xx/5xx on the
    # streaming route must surface as RuntimeError, not a raw HTTPError.
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise _http_error_to_runtime(exc) from exc
    # A clean EOF without the [DONE] sentinel means the server died
    # mid-stream; that must surface as an error, never a short success.
    seen_done = False
    with resp:
        buffer = b""
        while True:
            # HTTPResponse.read(n) waits for n bytes or EOF, which can buffer a
            # live SSE stream indefinitely. read1() performs at most one socket
            # read and returns as soon as an event frame is available.
            chunk = resp.read1(4096)
            if not chunk:
                break
            buffer += chunk
            while True:
                separators = [
                    (idx, separator)
                    for separator in (b"\r\n\r\n", b"\n\n")
                    if (idx := buffer.find(separator)) != -1
                ]
                if not separators:
                    break
                idx, separator = min(separators, key=lambda match: match[0])
                block, buffer = buffer[:idx], buffer[idx + len(separator) :]
                parsed = _parse_sse_block(block)
                if parsed is None:
                    continue
                event_type, raw_data = parsed
                if raw_data == "[DONE]":
                    seen_done = True
                    break
                # Surface mid-stream error events to the caller, matching the
                # JavaScript SDK's AxEngineStreamError behavior.
                if event_type == "error":
                    raise RuntimeError(f"ax-engine stream error: {_sse_error_message(raw_data)}")
                try:
                    yield json.loads(raw_data)
                except json.JSONDecodeError as exc:
                    # A non-JSON data frame is a broken stream, not a keepalive
                    # (those are `:` comment lines); do not report success.
                    raise RuntimeError(
                        f"ax-engine stream returned a non-JSON event: {raw_data[:120]!r}"
                    ) from exc
            if seen_done:
                break

        # Flush a trailing event if the server closed without a final blank line.
        if not seen_done and buffer.strip():
            parsed = _parse_sse_block(buffer)
            if parsed is not None:
                event_type, raw_data = parsed
                if raw_data == "[DONE]":
                    seen_done = True
                elif event_type == "error":
                    raise RuntimeError(f"ax-engine stream error: {_sse_error_message(raw_data)}")
                else:
                    with contextlib.suppress(json.JSONDecodeError):
                        yield json.loads(raw_data)
    if not seen_done:
        raise RuntimeError("ax-engine stream ended prematurely")


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

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Any:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = _normalize_tool_choice(tool_choice)
        return self.bind(tools=formatted_tools, **kwargs)

    def _build_request(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        req: dict = {"messages": convert_to_openai_messages(messages)}
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
        effective_stop = stop if stop is not None else self.stop
        if effective_stop is not None:
            req["stop"] = effective_stop
        if self.seed is not None:
            req["seed"] = self.seed
        if kwargs.get("tools") is not None:
            req["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice") is not None:
            req["tool_choice"] = kwargs["tool_choice"]
        # bind()/invoke() sampling kwargs override the instance defaults.
        _merge_sampling_kwargs(req, kwargs)
        return req

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        resp = _post_json(url, self._build_request(messages, stop, **kwargs), self.timeout)
        choices = resp.get("choices", [])
        if not choices:
            raise RuntimeError("Server returned empty choices array")
        choice = choices[0]
        # JSON null content (e.g. tool-call turns) must become "", not None.
        response_message = choice.get("message") or {}
        content = response_message.get("content")
        text = "" if content is None else content
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=text,
                        additional_kwargs=_message_additional_kwargs(response_message),
                    ),
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
        req = {**self._build_request(messages, stop, **kwargs), "stream": True}
        for chunk_data in _stream_sse(url, req, self.timeout):
            choice = (chunk_data.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            text = delta.get("content") or ""
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=text,
                    additional_kwargs=_message_additional_kwargs(delta),
                    tool_call_chunks=_tool_call_chunks(delta),
                ),
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

    def _build_request(
        self,
        prompt: str,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
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
        effective_stop = stop if stop is not None else self.stop
        if effective_stop is not None:
            req["stop"] = effective_stop
        if self.seed is not None:
            req["seed"] = self.seed
        # invoke()/bind() sampling kwargs override the instance defaults.
        _merge_sampling_kwargs(req, kwargs)
        return req

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        url = self.base_url.rstrip("/") + "/v1/completions"
        resp = _post_json(url, self._build_request(prompt, stop, **kwargs), self.timeout)
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
        req = {**self._build_request(prompt, stop, **kwargs), "stream": True}
        for chunk_data in _stream_sse(url, req, self.timeout):
            text = (chunk_data.get("choices") or [{}])[0].get("text") or ""
            chunk = GenerationChunk(text=text)
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk
