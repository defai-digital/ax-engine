"""OpenAI-compatible API client for AX Engine QA testing."""

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


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
        message = choices[0].get("message") or {}
        # Some backends return JSON null for empty content; normalize to "".
        content = message.get("content")
        text = "" if content is None else str(content)
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


@lru_cache(maxsize=16)
def _load_chat_template(tokenizer_path: Optional[str]) -> Optional[str]:
    if not tokenizer_path:
        return None
    template_path = Path(tokenizer_path).with_name("chat_template.jinja")
    if not template_path.exists():
        return None
    return template_path.read_text()


@lru_cache(maxsize=16)
def _load_tokenizer_config(tokenizer_path: Optional[str]) -> dict[str, Any]:
    if not tokenizer_path:
        return {}
    config_path = Path(tokenizer_path).with_name("tokenizer_config.json")
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except Exception:
        return {}


def _render_with_chat_template(
    tokenizer_path: Optional[str], system: Optional[str], user: str
) -> Optional[str]:
    template = _load_chat_template(tokenizer_path)
    if not template:
        return None

    try:
        from jinja2 import Environment
    except Exception:
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    config = _load_tokenizer_config(tokenizer_path)
    env = Environment(trim_blocks=False, lstrip_blocks=False)

    def tojson(value, ensure_ascii=True):
        return json.dumps(value, ensure_ascii=ensure_ascii)

    def raise_exception(message):
        raise ValueError(message)

    env.filters["tojson"] = tojson
    rendered = env.from_string(template).render(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tools=None,
        bos_token=config.get("bos_token") or "",
        eos_token=config.get("eos_token") or "",
        raise_exception=raise_exception,
    )
    return rendered


# Keep in lockstep with crates/ax-engine-server/src/chat.rs GPT_OSS_DEFAULT_SYSTEM.
_GPT_OSS_DEFAULT_SYSTEM = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "\n"
    "Reasoning: low\n"
    "\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)


def _is_qwen_non_thinking_only(model_id: str) -> bool:
    """Mirror chat.rs is_qwen_non_thinking_only_model / is_qwen_coder_model."""
    normalized = "".join(
        ch if ch.isalnum() else "-" for ch in model_id.lower()
    )
    if normalized == "qwen3":
        return True
    return "qwen3-coder-next" in normalized or "qwen3-coder" in normalized


def _qwen_assistant_prefill(model_id: str) -> str:
    """Mirror chat.rs Qwen generation prompt with enable_thinking=false."""
    if _is_qwen_non_thinking_only(model_id):
        return "<|im_start|>assistant\n"
    # Skip CoT so short QA answers do not burn budget on thinking tokens.
    return "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def strip_gpt_oss_harmony_output(text: str) -> str:
    """Extract user-facing text from a GPT-OSS Harmony generation.

    Mirrors crates/ax-engine-server/src/chat.rs::strip_gpt_oss_harmony_output.
    """
    final_open = "<|channel|>final<|message|>"
    analysis_open = "<|channel|>analysis<|message|>"
    commentary_open = "<|channel|>commentary<|message|>"

    idx = text.rfind(final_open)
    if idx >= 0:
        body = text[idx + len(final_open) :]
        for marker in ("<|return|>", "<|end|>", "<|start|>"):
            cut = body.find(marker)
            if cut >= 0:
                body = body[:cut]
        return body.strip()

    cleaned = text
    for open_tag in (analysis_open, commentary_open):
        while True:
            start = cleaned.find(open_tag)
            if start < 0:
                break
            after = start + len(open_tag)
            rest = cleaned[after:]
            end_rel = len(rest)
            for marker in ("<|end|>", "<|return|>", "<|start|>"):
                pos = rest.find(marker)
                if pos >= 0:
                    end_rel = min(end_rel, pos)
            end = after + end_rel
            if cleaned[end:].startswith("<|end|>"):
                end += len("<|end|>")
            elif cleaned[end:].startswith("<|return|>"):
                end += len("<|return|>")
            cleaned = cleaned[:start] + cleaned[end:]

    for token in (
        "<|start|>assistant",
        "<|start|>",
        "<|end|>",
        "<|return|>",
        "<|message|>",
        "<|channel|>",
        "<|call|>",
        "<|endoftext|>",
    ):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def strip_gemma4_channel_output(text: str) -> str:
    """Drop Gemma 4 thinking-channel framing from decoded chat text.

    Mirrors the server chat path that strips ``<|channel>…<channel|>`` spans
    (and a leading bare channel-name line such as ``thought``).
    """
    open_tag = "<|channel>"
    close_tag = "<channel|>"
    cleaned = text
    while True:
        start = cleaned.find(open_tag)
        if start < 0:
            break
        after = start + len(open_tag)
        close = cleaned.find(close_tag, after)
        if close < 0:
            cleaned = cleaned[:start]
            break
        cleaned = cleaned[:start] + cleaned[close + len(close_tag) :]

    # Stray close: everything before the first close is channel continuation.
    close = cleaned.find(close_tag)
    if close >= 0 and open_tag not in cleaned[:close]:
        cleaned = cleaned[close + len(close_tag) :]

    # Drop a bare channel-name first line (e.g. "thought\n…").
    if "\n" in cleaned:
        first, rest = cleaned.split("\n", 1)
        if first.strip().lower() in {"thought", "analysis", "commentary", "final"}:
            cleaned = rest
    return cleaned.strip()


def strip_chat_output_text(model_id: str, text: str) -> str:
    """Apply server-equivalent chat decode cleanup for the generate path."""
    if not text:
        return text
    normalized = model_id.lower()
    if "gpt-oss" in normalized or "gpt_oss" in normalized or "gptoss" in normalized:
        return strip_gpt_oss_harmony_output(text)
    if (
        "gemma-4" in normalized
        or "gemma4" in normalized
        or "diffusiongemma" in normalized
        or "diffusion-gemma" in normalized
        or "diffusion_gemma" in normalized
    ):
        return strip_gemma4_channel_output(text)
    return text


def _render_chat_prompt(
    model_id: str, system: Optional[str], user: str, tokenizer_path: Optional[str] = None
) -> str:
    """Render a chat prompt using the model's native template.

    Prefer ``chat_template.jinja`` next to the tokenizer when present. Fallbacks
    must match crates/ax-engine-server/src/chat.rs::render_prompt_internal so
    ``/v1/generate`` QA (client-side tokenization) frames families correctly.
    """
    rendered = _render_with_chat_template(tokenizer_path, system, user)
    if rendered is not None:
        return rendered

    normalized = model_id.lower()
    parts: list[str] = []

    if "qwen" in normalized:
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{user}<|im_end|>\n")
        parts.append(_qwen_assistant_prefill(model_id))
    elif "glm" in normalized:
        parts.append("[gMASK]<sop>")
        if system:
            parts.append(f"<|system|>{system}")
        parts.append(f"<|user|>{user}")
        # Server pre-fills empty think block before the answer.
        parts.append("<|assistant|></think>")
    elif (
        "gemma-4" in normalized
        or "gemma4" in normalized
        or "diffusiongemma" in normalized
        or "diffusion-gemma" in normalized
        or "diffusion_gemma" in normalized
    ):
        # Gemma 4 markers are `<|turn>` / `<turn|>` (not `<|turn|>`).
        # Server emits a real system turn when present (chat.rs).
        parts.append("<bos>")
        if system:
            parts.append(f"<|turn>system\n{system}<turn|>\n")
        parts.append(f"<|turn>user\n{user}<turn|>\n")
        parts.append("<|turn>model\n<|channel>thought\n<channel|>")
    elif (
        "llama-4" in normalized
        or "llama4" in normalized
        or "llama_4" in normalized
        or "llama-3" in normalized
        or "llama3" in normalized
        or "llama_3" in normalized
        or "llama" in normalized
    ):
        parts.append("<|begin_of_text|>")
        if system:
            parts.append(
                f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            )
        parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    elif (
        "mistral" in normalized
        or "ministral" in normalized
        or "devstral" in normalized
        or "codestral" in normalized
    ) and "mixtral" not in normalized:
        parts.append("<s>")
        if system:
            parts.append(f"[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]")
        parts.append(f"[INST]{user}[/INST]")
    elif "gpt-oss" in normalized or "gpt_oss" in normalized or "gptoss" in normalized:
        parts.append("<|start|>system<|message|>")
        parts.append(_GPT_OSS_DEFAULT_SYSTEM)
        parts.append("<|end|>")
        if system:
            parts.append("<|start|>developer<|message|># Instructions\n\n")
            parts.append(system)
            parts.append("<|end|>")
        parts.append(f"<|start|>user<|message|>{user}<|end|>")
        parts.append("<|start|>assistant<|channel|>final<|message|>")
    else:
        if system:
            parts.append(f"system: {system}\n")
        parts.append(f"user: {user}\n")
        parts.append("assistant:")

    return "".join(parts)


def _stream_generate_sse(
    url: str,
    payload: dict,
    tokenizer: Any,
    timeout: int = 120,
    model_id: str = "",
) -> QaResponse:
    """Stream from /v1/generate endpoint with token-level SSE."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    all_tokens: list[int] = []
    response_tokens: list[int] = []
    response_text: Optional[str] = None
    ttft_ms = 0.0
    finish_reason = None
    start = time.monotonic()
    first_chunk = True

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buffer = ""
            current_event: Optional[str] = None
            data_lines: list[str] = []
            done = False

            def flush_frame() -> Optional[tuple[Optional[str], str]]:
                nonlocal current_event, data_lines
                if not data_lines:
                    current_event = None
                    return None
                frame = (current_event, "\n".join(data_lines))
                current_event = None
                data_lines = []
                return frame

            while not done:
                raw = resp.read(4096)
                if not raw:
                    break
                buffer += raw.decode("utf-8", errors="replace")
                frames: list[tuple[Optional[str], str]] = []
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.rstrip("\r")
                    if not line:
                        frame = flush_frame()
                        if frame is not None:
                            frames.append(frame)
                        continue
                    if line.startswith("event:"):
                        current_event = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:") :].strip())

                for event_name, payload_str in frames:
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

                    if event_name == "step":
                        delta = event.get("delta_tokens", [])
                        if delta:
                            all_tokens.extend(delta)
                    elif event_name == "response":
                        response = event.get("response", {})
                        response_tokens = response.get("output_tokens", [])
                        response_text = response.get("output_text")
                        finish_reason = response.get("finish_reason")
                    else:
                        delta = event.get("delta_tokens", [])
                        if delta:
                            all_tokens.extend(delta)
                        fr = event.get("finish_reason")
                        if fr:
                            finish_reason = fr

            frame = flush_frame()
            if frame is not None and frame[1] != "[DONE]":
                event_name, payload_str = frame
                try:
                    event = json.loads(payload_str)
                except json.JSONDecodeError:
                    event = {}
                if event_name == "response":
                    response = event.get("response", {})
                    response_tokens = response.get("output_tokens", [])
                    response_text = response.get("output_text")
                    finish_reason = response.get("finish_reason")
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return QaResponse(
            text="", finish_reason=None, elapsed_ms=elapsed, stream=True, error=str(e)
        )

    elapsed = (time.monotonic() - start) * 1000
    output_tokens = response_tokens or all_tokens
    text = response_text if response_text is not None else tokenizer.decode(output_tokens)
    text = strip_chat_output_text(model_id, text)
    return QaResponse(
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=payload.get("input_tokens", []).__len__(),
        completion_tokens=len(output_tokens),
        total_tokens=len(payload.get("input_tokens", [])) + len(output_tokens),
        ttft_ms=ttft_ms,
        elapsed_ms=elapsed,
        stream=True,
    )


def send_generate_request(
    base_url: str,
    model: str,
    system: Optional[str],
    user: str,
    tokenizer: Any,
    tokenizer_path: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    stream: bool = False,
    repetition_penalty: Optional[float] = None,
    timeout: int = 120,
) -> QaResponse:
    """Send request via /v1/generate endpoint (MLX preview path with client-side tokenization)."""
    prompt = _render_chat_prompt(model, system, user, tokenizer_path=tokenizer_path)
    input_tokens = tokenizer.encode(prompt).ids

    endpoint = "generate/stream" if stream else "generate"
    url = f"{base_url.rstrip('/')}/v1/{endpoint}"
    sampling: dict[str, Any] = {"temperature": temperature}
    if repetition_penalty is None and temperature <= 0.0:
        repetition_penalty = 1.1
    if repetition_penalty is not None:
        sampling["repetition_penalty"] = repetition_penalty

    payload: dict[str, Any] = {
        "model_id": model,
        "input_tokens": input_tokens,
        "max_output_tokens": max_tokens,
        "sampling": sampling,
    }

    if stream:
        return _stream_generate_sse(
            url, payload, tokenizer, timeout=timeout, model_id=model
        )

    start = time.monotonic()
    try:
        result = _post_json(url, payload, timeout=timeout)
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return QaResponse(
            text="", finish_reason=None, elapsed_ms=elapsed, stream=False, error=str(e)
        )

    elapsed = (time.monotonic() - start) * 1000
    output_tokens = result.get("output_tokens", [])
    text = tokenizer.decode(output_tokens) if output_tokens else ""
    text = strip_chat_output_text(model, text)
    return QaResponse(
        text=text,
        finish_reason=result.get("finish_reason"),
        prompt_tokens=len(input_tokens),
        completion_tokens=len(output_tokens),
        total_tokens=len(input_tokens) + len(output_tokens),
        ttft_ms=elapsed,
        elapsed_ms=elapsed,
        stream=False,
    )
