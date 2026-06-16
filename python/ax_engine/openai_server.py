import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


ALLOWED_CHAT_ROLES = {"system", "user", "assistant", "tool", "function"}
QWEN_CHATML_ASSISTANT_GENERATION_PROMPT = (
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_NO_THINK = "<|im_start|>assistant\n"
MODEL_OWNER = "ax-engine"


class OpenAiShimError(ValueError):
    pass


def render_chat_prompt(
    messages: list[dict[str, Any]],
    model_id: str,
    tools: Any = None,
    tool_choice: Any = None,
) -> str:
    if not isinstance(messages, list):
        raise OpenAiShimError("chat.completions messages must be a list")
    if not messages:
        raise OpenAiShimError("chat.completions requires at least one message")

    template = chat_prompt_template(model_id)
    prompt_parts: list[str] = []
    if template == "llama3":
        prompt_parts.append("<|begin_of_text|>")

    qwen_tool_style = qwen_tool_contract_style(model_id)
    rendered_messages: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise OpenAiShimError("chat.completions message entries must be objects")
        role = str(message.get("role", "")).strip()
        if role not in ALLOWED_CHAT_ROLES:
            raise OpenAiShimError(
                "unsupported chat role; expected one of system, user, assistant, tool, function"
            )
        content = render_chat_content(message.get("content"))
        if role == "assistant":
            rendered_tool_calls = render_assistant_tool_calls(
                message.get("tool_calls"), qwen_tool_style
            )
            if rendered_tool_calls:
                if content.strip():
                    content += (
                        "\n\n"
                        if qwen_tool_style in {"function_xml", "coder_xml"}
                        else "\n"
                    )
                content += rendered_tool_calls
        rendered_messages.append((role, content))

    if template == "qwen_chatml":
        tool_contract = render_tool_contract_system_message(
            tools, tool_choice, qwen_tool_style
        )
        if tool_contract is not None:
            if rendered_messages and rendered_messages[0][0] == "system":
                role, content = rendered_messages[0]
                rendered_messages[0] = (role, f"{content}\n\n{tool_contract}")
            else:
                if qwen_tool_style == "coder_xml":
                    tool_contract = (
                        "You are Qwen, a helpful AI assistant that can interact "
                        "with a computer to solve tasks.\n\n"
                        f"{tool_contract}"
                    )
                rendered_messages.insert(
                    0,
                    ("system", tool_contract),
                )

    qwen_tool_response_open = False
    for role, content in rendered_messages:
        if template == "qwen_chatml":
            if role in {"tool", "function"}:
                if not qwen_tool_response_open:
                    prompt_parts.append("<|im_start|>user\n")
                    qwen_tool_response_open = True
                prompt_parts.append(f"<tool_response>\n{content}\n</tool_response>\n")
            else:
                if qwen_tool_response_open:
                    prompt_parts.append("<|im_end|>\n")
                    qwen_tool_response_open = False
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        elif template == "llama3":
            prompt_parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )
        else:
            safe_content = content.replace("\\", "\\\\").replace("\n", "\\n")
            prompt_parts.append(f"{role}: {safe_content}\n")

    if qwen_tool_response_open:
        prompt_parts.append("<|im_end|>\n")

    if template == "qwen_chatml":
        prompt_parts.append(qwen_assistant_generation_prompt(model_id))
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
    if content is None:
        return ""
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


def qwen_assistant_generation_prompt(model_id: str) -> str:
    if is_qwen_non_thinking_only_model(model_id):
        return QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_NO_THINK
    return QWEN_CHATML_ASSISTANT_GENERATION_PROMPT


def is_qwen_non_thinking_only_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return normalized == "qwen3" or is_qwen_coder_model(model_id)


def is_qwen_coder_model(model_id: str) -> bool:
    normalized = normalize_model_id_token(model_id)
    return "qwen3-coder-next" in normalized or "qwen3-coder" in normalized


def uses_qwen_coder_xml_tool_contract(model_id: str) -> bool:
    if is_qwen_non_thinking_only_model(model_id):
        return True
    normalized = model_id.lower()
    return (
        "qwen3.6" in normalized
        or "qwen3_6" in normalized
        or "qwen3-6" in normalized
        or "qwen36" in normalized
    )


def qwen_tool_contract_style(model_id: str) -> str:
    normalized = normalize_model_id_token(model_id)
    if uses_qwen_coder_xml_tool_contract(model_id):
        return "coder_xml"
    if any(marker in normalized for marker in ("qwen3-next", "qwen3-5", "qwen35")):
        return "function_xml"
    return "json_tools"


def normalize_model_id_token(model_id: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in model_id.lower())


def render_tool_contract_system_message(
    tools: Any, tool_choice: Any, style: str = "json_tools"
) -> str | None:
    if not openai_value_is_present(tools):
        return None
    if style == "function_xml":
        return render_qwen_function_tool_contract_system_message(tools, tool_choice)
    if style == "coder_xml":
        return render_qwen_coder_tool_contract_system_message(tools, tool_choice)
    return render_json_tool_contract_system_message(tools, tool_choice)


def render_json_tool_contract_system_message(tools: Any, tool_choice: Any) -> str:
    lines = [
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]
    if isinstance(tools, list):
        lines.extend(json.dumps(tool, separators=(",", ":")) for tool in tools)
    else:
        lines.append(json.dumps(tools, separators=(",", ":")))
    lines.extend(
        [
            "</tools>",
            "",
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
            "<tool_call>",
            '{"name": <function-name>, "arguments": <args-json-object>}',
            "</tool_call>",
        ]
    )
    if tool_choice_forces_tool_call(tool_choice):
        lines.append(
            "The current tool_choice requires using a tool when a matching function is available."
        )
    return "\n".join(lines)


def render_qwen_function_tool_contract_system_message(
    tools: Any, tool_choice: Any
) -> str:
    lines = [
        "# Tools",
        "",
        "You have access to the following functions:",
        "",
        "<tools>",
    ]
    if isinstance(tools, list):
        lines.extend(json.dumps(tool, separators=(",", ":")) for tool in tools)
    else:
        lines.append(json.dumps(tools, separators=(",", ":")))
    lines.extend(
        [
            "</tools>",
            "",
            "If you choose to call a function ONLY reply in the following format with NO suffix:",
            "",
            "<tool_call>",
            "<function=example_function_name>",
            "<parameter=example_parameter_1>",
            "value_1",
            "</parameter>",
            "<parameter=example_parameter_2>",
            "This is the value for the second parameter",
            "that can span",
            "multiple lines",
            "</parameter>",
            "</function>",
            "</tool_call>",
            "",
            "<IMPORTANT>",
            "Reminder:",
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags",
            "- Required parameters MUST be specified",
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after",
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls",
            "</IMPORTANT>",
        ]
    )
    if tool_choice_forces_tool_call(tool_choice):
        lines.append(
            "The current tool_choice requires using a tool when a matching function is available."
        )
    return "\n".join(lines)


def render_qwen_coder_tool_contract_system_message(tools: Any, tool_choice: Any) -> str:
    lines = [
        "# Tools",
        "",
        "You have access to the following tools:",
        "",
        "<tools>",
    ]
    lines.extend(render_xml_tool_blocks(tools))
    lines.extend(
        [
            "</tools>",
            "",
            "If you choose to call a tool ONLY reply in the following format with NO suffix:",
            "",
            "<tool_call>",
            "<function=example_function_name>",
            "<parameter=example_parameter_1>",
            "value_1",
            "</parameter>",
            "<parameter=example_parameter_2>",
            "value_2",
            "</parameter>",
            "</function>",
            "</tool_call>",
            "",
            "<IMPORTANT>",
            "Reminder:",
            "- Function calls MUST follow the specified format: the tool calling block MUST begin with an opening <tool_call> tag and end with a closing </tool_call> tag.",
            "- Required parameters MUST be specified",
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after",
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls",
            "</IMPORTANT>",
        ]
    )
    if tool_choice_forces_tool_call(tool_choice):
        lines.append(
            "The current tool_choice requires using a tool when a matching function is available."
        )
    return "\n".join(lines)


def render_xml_tool_blocks(tools: Any) -> list[str]:
    if isinstance(tools, list):
        return [block for tool in tools if (block := render_xml_tool_block(tool))]
    block = render_xml_tool_block(tools)
    return [block] if block else []


def render_xml_tool_block(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = function.get("name")
    if not isinstance(name, str):
        return None

    lines = ["<function>", f"<name>{escape_xml_text(name)}</name>"]
    description = function.get("description")
    if isinstance(description, str):
        lines.append(
            f"<description>{escape_xml_text(description.strip())}</description>"
        )
    lines.append("<parameters>")
    parameters = function.get("parameters")
    if isinstance(parameters, dict):
        properties = parameters.get("properties")
        if isinstance(properties, dict):
            for param_name, param_fields in properties.items():
                if not isinstance(param_fields, dict):
                    param_fields = {}
                lines.append("<parameter>")
                lines.append(f"<name>{escape_xml_text(str(param_name))}</name>")
                if "type" in param_fields:
                    lines.append(
                        f"<type>{escape_xml_text(stringify_json_scalar(param_fields['type']))}</type>"
                    )
                param_description = param_fields.get("description")
                if isinstance(param_description, str):
                    lines.append(
                        f"<description>{escape_xml_text(param_description.strip())}</description>"
                    )
                for key, value in param_fields.items():
                    if key in {"name", "type", "description"}:
                        continue
                    lines.append(render_extra_xml_field(key, value))
                lines.append("</parameter>")
        for key, value in parameters.items():
            if key in {"type", "properties"}:
                continue
            lines.append(render_extra_xml_field(key, value))
    lines.append("</parameters>")
    for key, value in function.items():
        if key in {"type", "name", "description", "parameters"}:
            continue
        lines.append(render_extra_xml_field(key, value))
    lines.append("</function>")
    return "\n".join(lines)


def render_assistant_tool_calls(
    tool_calls: Any, style: str = "json_tools"
) -> str | None:
    if isinstance(tool_calls, dict):
        calls = [tool_calls]
    elif isinstance(tool_calls, list):
        calls = tool_calls
    else:
        return None
    rendered = [
        rendered
        for call in calls
        if (rendered := render_assistant_tool_call(call, style)) is not None
    ]
    return "\n".join(rendered) if rendered else None


def render_assistant_tool_call(tool_call: Any, style: str = "json_tools") -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        return None
    arguments = normalize_tool_arguments(function.get("arguments"))
    if style in {"function_xml", "coder_xml"}:
        return render_qwen_xml_tool_call(function["name"], arguments)
    name = json.dumps(function["name"])
    arguments_json = json.dumps(arguments, separators=(",", ":"))
    return (
        f'<tool_call>\n{{"name": {name}, "arguments": {arguments_json}}}\n</tool_call>'
    )


def render_qwen_xml_tool_call(name: str, arguments: Any) -> str:
    lines = ["<tool_call>", f"<function={escape_xml_text(name)}>"]
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            lines.append(f"<parameter={escape_xml_text(str(key))}>")
            lines.append(escape_xml_text(stringify_json_scalar(value)))
            lines.append("</parameter>")
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def render_extra_xml_field(key: str, value: Any) -> str:
    escaped_key = escape_xml_text(str(key))
    return f"<{escaped_key}>{escape_xml_text(stringify_json_scalar(value))}</{escaped_key}>"


def stringify_json_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def escape_xml_text(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def unescape_xml_text(value: str) -> str:
    """Reverse of escape_xml_text: restore &lt; &gt; &amp; to < > &."""
    return value.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def normalize_tool_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    if arguments is None:
        return {}
    return arguments


def openai_tools_are_enabled(tools: Any, tool_choice: Any) -> bool:
    return openai_value_is_present(tools) or tool_choice_forces_tool_call(tool_choice)


def tool_choice_forces_tool_call(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "auto", "none", "false", "off"}
    if isinstance(value, (list, dict)):
        return bool(value)
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


def openai_value_is_present(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    if isinstance(value, (int, float)):
        return value != 0
    return True


def extract_tool_calls(content: str) -> tuple[str, list[dict[str, Any]] | None]:
    remaining = content
    calls: list[dict[str, Any]] = []
    while True:
        start = remaining.find("<tool_call>")
        if start < 0:
            break
        body_start = start + len("<tool_call>")
        end = remaining.find("</tool_call>", body_start)
        suffix_start = len(remaining) if end < 0 else end + len("</tool_call>")
        body_end = len(remaining) if end < 0 else end
        function = parse_tool_call_body(remaining[body_start:body_end].strip())
        if function is None:
            break
        calls.append(
            {
                "id": f"call_{len(calls)}",
                "type": "function",
                "function": function,
            }
        )
        remaining = remaining[:start] + remaining[suffix_start:]
    return remaining.strip(), calls or None


def parse_tool_call_body(body: str) -> dict[str, str] | None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return parse_qwen_function_tool_call(body)
    return parse_tool_call_function(payload)


def parse_tool_call_function(payload: Any) -> dict[str, str] | None:
    if not isinstance(payload, dict):
        return None
    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = payload.get("name")
        arguments = payload.get("arguments")
    if not isinstance(name, str):
        return None
    if isinstance(arguments, str):
        arguments_text = arguments
    elif arguments is None:
        arguments_text = "{}"
    else:
        arguments_text = json.dumps(arguments, separators=(",", ":"))
    return {"name": name, "arguments": arguments_text}


def parse_qwen_function_tool_call(body: str) -> dict[str, str] | None:
    function_marker = "<function="
    function_start = body.find(function_marker)
    if function_start < 0:
        return None
    name_start = function_start + len(function_marker)
    name_end = body.find(">", name_start)
    if name_end < 0:
        return None
    name = unescape_xml_text(body[name_start:name_end].strip())
    if not name:
        return None

    body_start = name_end + 1
    close_start = body.find("</function>", body_start)
    inner = body[body_start:] if close_start < 0 else body[body_start:close_start]
    parameters = parse_qwen_tool_parameters(inner)
    if not parameters and inner.strip():
        try:
            arguments = json.dumps(json.loads(inner), separators=(",", ":"))
        except json.JSONDecodeError:
            return None
    else:
        arguments = json.dumps(parameters, separators=(",", ":"))
    return {"name": name, "arguments": arguments}


def parse_qwen_tool_parameters(body: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    parameter_marker = "<parameter="
    offset = 0
    while True:
        parameter_start = body.find(parameter_marker, offset)
        if parameter_start < 0:
            break
        name_start = parameter_start + len(parameter_marker)
        name_end = body.find(">", name_start)
        if name_end < 0:
            break
        name = unescape_xml_text(body[name_start:name_end].strip())
        if not name:
            offset = name_end + 1
            continue
        value_start = name_end + 1
        value_end = _qwen_parameter_value_end(body, value_start)
        raw_value = unescape_xml_text(body[value_start:value_end].strip())
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        parameters[name] = value
        offset = value_end
    return parameters


def _qwen_parameter_value_end(body: str, value_start: int) -> int:
    """End index (exclusive) of a `<parameter=>` value, matching the reference
    `qwen3_coder_xml` parser's alternation: an explicit `</parameter>` close is
    preferred, but a missing close is treated as an implicit terminator at the
    next `<parameter=`, `</function>`, or end of body. Qwen3-Coder models
    frequently truncate or omit the closing tag; the earlier `break`-on-missing
    dropped the whole tool call onto the plain-text path.
    """
    candidates = [
        body.find(marker, value_start) for marker in ("<parameter=", "</function>")
    ]
    next_implicit = min((idx for idx in candidates if idx >= 0), default=len(body))
    explicit = body.find("</parameter>", value_start)
    # Prefer the explicit close only when it belongs to *this* parameter, i.e.
    # it precedes the next parameter / function close. An explicit close that
    # lands past the next delimiter belongs to a later parameter, so this one
    # was truncated and must end at the implicit delimiter instead of greedily
    # absorbing the following parameter.
    if 0 <= explicit <= next_implicit:
        return explicit
    return next_implicit


def prompt_to_tokens(prompt: Any, tokenizer: Any) -> tuple[list[int], str | None]:
    if isinstance(prompt, str):
        return list(tokenizer.encode(prompt).ids), prompt
    if isinstance(prompt, list) and all(
        isinstance(token, int) and not isinstance(token, bool) for token in prompt
    ):
        return [int(token) for token in prompt], None
    raise OpenAiShimError("completions prompt must be a string or token id array")


def finish_reason(reason: str | None) -> str | None:
    if reason == "max_output_tokens":
        return "length"
    if reason == "stop":
        return "stop"
    if reason == "cancelled":
        return "cancel"
    if reason == "content_filter":
        return "content_filter"
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
            "data": [{"id": model_id, "object": "model", "owned_by": MODEL_OWNER}],
        }

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        payload = await request.json()
        error = (
            validate_payload_object(payload)
            or validate_model(payload, model_id)
            or require_max_tokens(payload)
            or validate_sampling_params(payload)
        )
        if error is not None:
            return openai_error(*error)

        try:
            input_tokens, _prompt_text = prompt_to_tokens(
                payload.get("prompt"), tokenizer
            )
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

        temperature = float(payload.get("temperature", 0.0))
        default_rp = 1.1 if temperature <= 0.0 else 1.0
        with lock:
            result = session.generate(
                input_tokens,
                max_output_tokens=int(payload["max_tokens"]),
                temperature=temperature,
                top_p=float(payload.get("top_p", 1.0)),
                top_k=int(payload.get("top_k", 0)),
                repetition_penalty=float(payload.get("repetition_penalty", default_rp)),
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
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": finish_reason(result.finish_reason),
                }
            ],
            "usage": usage(input_tokens, list(result.output_tokens)),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        payload = await request.json()
        error = (
            validate_payload_object(payload)
            or validate_model(payload, model_id)
            or require_max_tokens(payload)
            or validate_sampling_params(payload)
        )
        if error is not None:
            return openai_error(*error)

        try:
            prompt = render_chat_prompt(
                payload.get("messages") or [],
                model_id,
                payload.get("tools"),
                payload.get("tool_choice"),
            )
            input_tokens, _prompt_text = prompt_to_tokens(prompt, tokenizer)
        except OpenAiShimError as error:
            return openai_error(400, str(error))

        parse_tool_calls = openai_tools_are_enabled(
            payload.get("tools"), payload.get("tool_choice")
        )
        if bool(payload.get("stream", False)):
            if parse_tool_calls:
                temperature = float(payload.get("temperature", 0.0))
                default_rp = 1.1 if temperature <= 0.0 else 1.0
                with lock:
                    result = session.generate(
                        input_tokens,
                        max_output_tokens=int(payload["max_tokens"]),
                        temperature=temperature,
                        top_p=float(payload.get("top_p", 1.0)),
                        top_k=int(payload.get("top_k", 0)),
                        repetition_penalty=float(
                            payload.get("repetition_penalty", default_rp)
                        ),
                        seed=int(payload.get("seed", 0)),
                        metadata=payload.get("metadata"),
                    )
                text = tokenizer.decode(list(result.output_tokens))
                events = stream_buffered_tool_chat_chunks(
                    model_id,
                    result.request_id,
                    text,
                    finish_reason(result.finish_reason),
                )
                return StreamingResponse(events, media_type="text/event-stream")
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

        temperature = float(payload.get("temperature", 0.0))
        default_rp = 1.1 if temperature <= 0.0 else 1.0
        with lock:
            result = session.generate(
                input_tokens,
                max_output_tokens=int(payload["max_tokens"]),
                temperature=temperature,
                top_p=float(payload.get("top_p", 1.0)),
                top_k=int(payload.get("top_k", 0)),
                repetition_penalty=float(payload.get("repetition_penalty", default_rp)),
                seed=int(payload.get("seed", 0)),
                metadata=payload.get("metadata"),
            )
        text = tokenizer.decode(list(result.output_tokens))
        message: dict[str, Any] = {"role": "assistant", "content": text}
        response_finish_reason = finish_reason(result.finish_reason)
        if parse_tool_calls:
            content, tool_calls = extract_tool_calls(text)
            if tool_calls:
                message["content"] = content
                message["tool_calls"] = tool_calls
                response_finish_reason = "tool_calls"
        return {
            "id": f"chatcmpl-{result.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": response_finish_reason,
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
        return (
            400,
            f"requested model {requested} does not match configured model {model_id}",
        )
    return None


def validate_payload_object(payload: Any) -> tuple[int, str] | None:
    if not isinstance(payload, dict):
        return 400, "OpenAI-compatible MLX shim request body must be a JSON object"
    return None


def require_max_tokens(payload: dict[str, Any]) -> tuple[int, str] | None:
    max_tokens = payload.get("max_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        return 400, "OpenAI-compatible MLX shim requires max_tokens > 0"
    return None


def validate_sampling_params(payload: dict[str, Any]) -> tuple[int, str] | None:
    for key in ("temperature", "top_p", "repetition_penalty"):
        value = payload.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            return 400, f"OpenAI-compatible MLX shim requires {key} to be numeric"
    for key in ("top_k", "seed"):
        value = payload.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            return 400, f"OpenAI-compatible MLX shim requires {key} to be an integer"
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
    temperature = float(payload.get("temperature", 0.0))
    default_rp = 1.1 if temperature <= 0.0 else 1.0
    accumulated_tokens: list[int] = []
    prev_text_len = 0
    role_emitted = False
    with lock:
        generator = session.stream_generate(
            input_tokens,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=float(payload.get("top_p", 1.0)),
            top_k=int(payload.get("top_k", 0)),
            repetition_penalty=float(payload.get("repetition_penalty", default_rp)),
            seed=int(payload.get("seed", 0)),
            metadata=payload.get("metadata"),
        )
    for event in generator:
        if event.event == "step" and event.delta_tokens:
            accumulated_tokens.extend(event.delta_tokens)
            full_text = tokenizer.decode(accumulated_tokens)
            new_text = full_text[prev_text_len:]
            prev_text_len = len(full_text)
            if new_text:
                emit_role = kind == "chat" and not role_emitted
                role_emitted = role_emitted or emit_role
                yield sse_chunk(
                    stream_id,
                    created,
                    model_id,
                    new_text,
                    None,
                    kind,
                    emit_role=emit_role,
                )
        elif event.event == "response" and event.response is not None:
            # Flush any remaining text from incomplete UTF-8 sequences
            if accumulated_tokens:
                final_text = tokenizer.decode(accumulated_tokens)
                remaining = final_text[prev_text_len:]
                if remaining:
                    emit_role = kind == "chat" and not role_emitted
                    role_emitted = role_emitted or emit_role
                    yield sse_chunk(
                        stream_id,
                        created,
                        model_id,
                        remaining,
                        None,
                        kind,
                        emit_role=emit_role,
                    )
            # OpenAI spec: the role must appear in at least one chunk. If no
            # content chunks were emitted (0-token completion), emit a role-only
            # chunk before the finish_reason chunk so clients can read the role.
            if kind == "chat" and not role_emitted:
                yield sse_chunk(
                    stream_id, created, model_id, "", None, kind, emit_role=True
                )
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
    *,
    emit_role: bool = False,
) -> str:
    if kind == "chat":
        delta: dict[str, Any] = {"content": text}
        if emit_role:
            delta["role"] = "assistant"
        choice: dict[str, Any] = {
            "index": 0,
            "delta": delta,
            "finish_reason": reason,
        }
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


def stream_buffered_tool_chat_chunks(
    model_id: str,
    request_id: int,
    text: str,
    terminal_reason: str | None,
) -> Iterator[str]:
    stream_id = f"chatcmpl-{request_id}"
    created = int(time.time())
    content, tool_calls = extract_tool_calls(text)
    role_emitted = False
    if content:
        yield chat_sse_payload(
            stream_id,
            created,
            model_id,
            {"role": "assistant", "content": content},
            None,
        )
        role_emitted = True
    if tool_calls:
        delta_tool_calls = [
            {
                "index": index,
                "id": call["id"],
                "type": call["type"],
                "function": {
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                },
            }
            for index, call in enumerate(tool_calls)
        ]
        delta: dict[str, Any] = {"tool_calls": delta_tool_calls}
        if not role_emitted:
            delta["role"] = "assistant"
        yield chat_sse_payload(stream_id, created, model_id, delta, None)
        terminal_reason = "tool_calls"
        role_emitted = True
    if not role_emitted:
        yield chat_sse_payload(
            stream_id,
            created,
            model_id,
            {"role": "assistant", "content": ""},
            None,
        )
    yield chat_sse_payload(stream_id, created, model_id, {}, terminal_reason)
    yield "data: [DONE]\n\n"


def chat_sse_payload(
    stream_id: str,
    created: int,
    model_id: str,
    delta: dict[str, Any],
    reason: str | None,
) -> str:
    payload = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
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
