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

    chat = AXEngineChatModel(base_url="http://127.0.0.1:8080", max_tokens=256)
    response = chat.invoke([HumanMessage(content="Hello!")])
    print(response.content)
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.language_models.llms import LLM
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        SystemMessage,
        HumanMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult, GenerationChunk
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
except ImportError as _e:
    raise ImportError(
        "ax_engine.langchain requires langchain-core. "
        "Install it with: pip install langchain-core"
    ) from _e

_DEFAULT_BASE_URL = "http://127.0.0.1:8080"


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
        raise RuntimeError(
            f"ax-engine HTTP {exc.code}: {detail or exc.reason}"
        ) from exc


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
                        block, buffer = buffer[:idx], buffer[idx + len(sep):]
                        break
                else:
                    break
                raw_data = None
                for line in block.splitlines():
                    line = line.decode()
                    if line.startswith("data:"):
                        raw_data = line[len("data:"):].lstrip()
                if raw_data is None:
                    continue
                if raw_data == "[DONE]":
                    return
                try:
                    yield json.loads(raw_data)
                except json.JSONDecodeError:
                    continue


class AXEngineChatModel(BaseChatModel):
    """
    LangChain ``BaseChatModel`` that calls ax-engine-server ``/v1/chat/completions``.

    Parameters
    ----------
    base_url : str
        Base URL of the ax-engine-server instance (default ``http://127.0.0.1:8080``).
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
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    timeout: int = 300

    @property
    def _llm_type(self) -> str:
        return "ax-engine"

    def _build_request(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> dict:
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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        resp = _post_json(url, self._build_request(messages, stop), self.timeout)
        choice = resp["choices"][0]
        text = choice["message"]["content"]
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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
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
        Base URL of the ax-engine-server instance (default ``http://127.0.0.1:8080``).
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
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    timeout: int = 300

    @property
    def _llm_type(self) -> str:
        return "ax-engine"

    def _build_request(self, prompt: str, stop: Optional[List[str]] = None) -> dict:
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
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        url = self.base_url.rstrip("/") + "/v1/completions"
        resp = _post_json(url, self._build_request(prompt, stop), self.timeout)
        return resp["choices"][0]["text"]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
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
