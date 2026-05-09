from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


ALLOWED_CHAT_ROLES = {"system", "user", "assistant", "tool", "function"}


class OpenAiShimError(ValueError):
    pass


def render_chat_prompt(messages: list[dict[str, Any]], model_id: str) -> str:
    if not messages:
        raise OpenAiShimError("chat.completions requires at least one message")

    template = chat_prompt_template(model_id)
    prompt_parts: list[str] = []
    if template == "llama3":
        prompt_parts.append("<|begin_of_text|>")

    for message in messages:
        role = str(message.get("role", "")).strip()
        if role not in ALLOWED_CHAT_ROLES:
            raise OpenAiShimError(
                "unsupported chat role; expected one of system, user, assistant, tool, function"
            )
        content = render_chat_content(message.get("content"))
        if template == "qwen_chatml":
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        elif template == "llama3":
            prompt_parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )
        else:
            safe_content = content.replace("\\", "\\\\").replace("\n", "\\n")
            prompt_parts.append(f"{role}: {safe_content}\n")

    if template == "qwen_chatml":
        prompt_parts.append("<|im_start|>assistant\n")
    elif template == "llama3":
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    else:
        prompt_parts.append("assistant:")
    return "".join(prompt_parts)


def chat_prompt_template(model_id: str) -> str:
    normalized = model_id.lower()
    if "qwen" in normalized:
        return "qwen_chatml"
    if "llama-3" in normalized or "llama3" in normalized or "llama_3" in normalized:
        return "llama3"
    return "plain_role_prefix"


def render_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered: list[str] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                raise OpenAiShimError(
                    "AX OpenAI MLX shim currently accepts text-only chat messages"
                )
            text = part.get("text")
            if not isinstance(text, str):
                raise OpenAiShimError("text chat content parts require a text field")
            rendered.append(text)
        return "".join(rendered)
    raise OpenAiShimError("chat message content must be a string or text parts")


def prompt_to_tokens(prompt: Any, tokenizer: Any) -> tuple[list[int], str | None]:
    if isinstance(prompt, str):
        return list(tokenizer.encode(prompt).ids), prompt
    if isinstance(prompt, list) and all(isinstance(token, int) for token in prompt):
        return [int(token) for token in prompt], None
    raise OpenAiShimError("completions prompt must be a string or token id array")


def finish_reason(reason: str | None) -> str | None:
    if reason == "max_output_tokens":
        return "length"
    if reason == "stop":
        return "stop"
    return None


def usage(prompt_tokens: list[int], output_tokens: list[int]) -> dict[str, int]:
    prompt_count = len(prompt_tokens)
    completion_count = len(output_tokens)
    return {
        "prompt_tokens": prompt_count,
        "completion_tokens": completion_count,
        "total_tokens": prompt_count + completion_count,
    }


def create_app(
    *,
    model_id: str = "qwen3_dense",
    mlx_model_artifacts_dir: str | None = None,
    tokenizer_path: str | Path,
    session_factory: Callable[..., Any] | None = None,
    session_kwargs: dict[str, Any] | None = None,
) -> Any:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "AX OpenAI MLX shim requires fastapi, uvicorn, and tokenizers. "
            "Install with: pip install 'ax-engine[openai]'"
        ) from error

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    session = build_session(
        model_id=model_id,
        mlx_model_artifacts_dir=mlx_model_artifacts_dir,
        session_factory=session_factory,
        session_kwargs=session_kwargs,
    )
    lock = threading.Lock()
    app = FastAPI(title="AX Engine MLX OpenAI Shim")

    @app.on_event("shutdown")
    def close_session() -> None:
        close = getattr(session, "close", None)
        if callable(close):
            close()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "ax-engine-openai-mlx-shim"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "ax-engine-v4"}],
        }

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        payload = await request.json()
        error = validate_model(payload, model_id) or require_max_tokens(payload)
        if error is not None:
            return openai_error(*error)

        try:
            input_tokens, _prompt_text = prompt_to_tokens(payload.get("prompt"), tokenizer)
        except OpenAiShimError as error:
            return openai_error(400, str(error))

        if bool(payload.get("stream", False)):
            events = stream_completion_chunks(
                session,
                lock,
                tokenizer,
                model_id,
                input_tokens,
                int(payload["max_tokens"]),
                payload,
                "completion",
            )
            return StreamingResponse(events, media_type="text/event-stream")

        with lock:
            result = session.generate(
                input_tokens,
                max_output_tokens=int(payload["max_tokens"]),
                temperature=float(payload.get("temperature", 0.0)),
                top_p=float(payload.get("top_p", 1.0)),
                top_k=int(payload.get("top_k", 0)),
                repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
                seed=int(payload.get("seed", 0)),
                metadata=payload.get("metadata"),
            )
        text = tokenizer.decode(list(result.output_tokens))
        return {
            "id": f"cmpl-{result.request_id}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {"index": 0, "text": text, "finish_reason": finish_reason(result.finish_reason)}
            ],
            "usage": usage(input_tokens, list(result.output_tokens)),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        payload = await request.json()
        error = validate_model(payload, model_id) or require_max_tokens(payload)
        if error is not None:
            return openai_error(*error)

        try:
            prompt = render_chat_prompt(payload.get("messages") or [], model_id)
            input_tokens, _prompt_text = prompt_to_tokens(prompt, tokenizer)
        except OpenAiShimError as error:
            return openai_error(400, str(error))

        if bool(payload.get("stream", False)):
            events = stream_completion_chunks(
                session,
                lock,
                tokenizer,
                model_id,
                input_tokens,
                int(payload["max_tokens"]),
                payload,
                "chat",
            )
            return StreamingResponse(events, media_type="text/event-stream")

        with lock:
            result = session.generate(
                input_tokens,
                max_output_tokens=int(payload["max_tokens"]),
                temperature=float(payload.get("temperature", 0.0)),
                top_p=float(payload.get("top_p", 1.0)),
                top_k=int(payload.get("top_k", 0)),
                repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
                seed=int(payload.get("seed", 0)),
                metadata=payload.get("metadata"),
            )
        text = tokenizer.decode(list(result.output_tokens))
        return {
            "id": f"chatcmpl-{result.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason(result.finish_reason),
                }
            ],
            "usage": usage(input_tokens, list(result.output_tokens)),
        }

    def openai_error(status: int, message: str) -> Any:
        return JSONResponse(
            status_code=status,
            content={"error": {"code": "invalid_request", "message": message}},
        )

    return app


def build_session(
    *,
    model_id: str,
    mlx_model_artifacts_dir: str | None,
    session_factory: Callable[..., Any] | None,
    session_kwargs: dict[str, Any] | None,
) -> Any:
    if session_factory is None:
        from . import Session

        session_factory = Session
    kwargs = {"model_id": model_id, "mlx": True}
    if mlx_model_artifacts_dir is not None:
        kwargs["mlx_model_artifacts_dir"] = mlx_model_artifacts_dir
    kwargs.update(session_kwargs or {})
    return session_factory(**kwargs)


def validate_model(payload: dict[str, Any], model_id: str) -> tuple[int, str] | None:
    requested = payload.get("model")
    if requested is not None and requested != model_id:
        return 400, f"requested model {requested} does not match configured model {model_id}"
    return None


def require_max_tokens(payload: dict[str, Any]) -> tuple[int, str] | None:
    max_tokens = payload.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        return 400, "OpenAI-compatible MLX shim requires max_tokens > 0"
    return None


def stream_completion_chunks(
    session: Any,
    lock: threading.Lock,
    tokenizer: Any,
    model_id: str,
    input_tokens: list[int],
    max_tokens: int,
    payload: dict[str, Any],
    kind: str,
) -> Iterator[str]:
    stream_id = f"{'chatcmpl' if kind == 'chat' else 'cmpl'}-{int(time.time() * 1000)}"
    created = int(time.time())
    with lock:
        for event in session.stream_generate(
            input_tokens,
            max_output_tokens=max_tokens,
            temperature=float(payload.get("temperature", 0.0)),
            top_p=float(payload.get("top_p", 1.0)),
            top_k=int(payload.get("top_k", 0)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
            seed=int(payload.get("seed", 0)),
            metadata=payload.get("metadata"),
        ):
            if event.event == "step" and event.delta_tokens:
                text = tokenizer.decode(list(event.delta_tokens))
                yield sse_chunk(stream_id, created, model_id, text, None, kind)
            elif event.event == "response" and event.response is not None:
                yield sse_chunk(
                    stream_id,
                    created,
                    model_id,
                    "",
                    finish_reason(event.response.finish_reason),
                    kind,
                )
    yield "data: [DONE]\n\n"


def sse_chunk(
    stream_id: str,
    created: int,
    model_id: str,
    text: str,
    reason: str | None,
    kind: str,
) -> str:
    if kind == "chat":
        choice: dict[str, Any] = {
            "index": 0,
            "delta": {"content": text},
            "finish_reason": reason,
        }
        if text:
            choice["delta"]["role"] = "assistant"
        object_name = "chat.completion.chunk"
    else:
        choice = {"index": 0, "text": text, "finish_reason": reason}
        object_name = "text_completion.chunk"
    payload = {
        "id": stream_id,
        "object": object_name,
        "created": created,
        "model": model_id,
        "choices": [choice],
    }
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the AX Engine MLX OpenAI shim")
    parser.add_argument("--model-id", default="qwen3_dense")
    parser.add_argument("--mlx-model-artifacts-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    app = create_app(
        model_id=args.model_id,
        mlx_model_artifacts_dir=args.mlx_model_artifacts_dir,
        tokenizer_path=args.tokenizer,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
