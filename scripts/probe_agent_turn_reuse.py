#!/usr/bin/env python3
"""Measure per-turn prefix reuse for an agent tool-calling chat session.

Replays a deterministic agent loop against `ax-engine-server` over
`/v1/chat/completions` (streaming, greedy): a long system prompt with tool
schemas, a task, then for each turn the model's own reply (content, reasoning
and tool calls, exactly as returned) followed by synthetic tool results, or a
"Continue." user turn when the model called no tool. This is the shape where
the chat template re-renders the previous turn and only part of the last
prompt is repeated.

Per turn it records TTFT (first streamed delta), prompt tokens and the
physically reused prompt tokens (`usage.prompt_tokens_details.cached_tokens`),
so `recomputed = prompt - cached` shows how much the follow-up re-prefills.

`--semantic-cuts off,on` starts one server per setting with
`AX_MLX_PREFIX_SEMANTIC_CUTS` unset/`1` and runs the same session on each.
Semantic cuts change the prefill partition, so replies may differ between
settings; the report records reply hashes but makes no identity claim.

Requires a local pack and Apple Silicon; a measurement tool, not a CI gate.

    python scripts/probe_agent_turn_reuse.py \\
        --server target/release/ax-engine-server --model-dir /path/to/pack \\
        --model-id qwen3.8-27b --turns 8 --semantic-cuts off,on --output out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ax.agent_turn_reuse_probe.v1"
SEMANTIC_CUTS_ENV = "AX_MLX_PREFIX_SEMANTIC_CUTS"
SETTINGS = ("off", "on")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the workspace and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

TASK = (
    "The test `parser::tests::nested_arrays` fails in this Rust workspace. "
    "Inspect the relevant files with the tools, find the cause and propose a fix."
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_settings(value: str) -> list[str]:
    settings = [item.strip() for item in value.split(",") if item.strip()]
    if not settings or any(item not in SETTINGS for item in settings):
        raise argparse.ArgumentTypeError(f"settings must be a comma list of {SETTINGS}")
    return settings


def system_prompt(lines: int) -> str:
    """A long, stable agent system prompt (the reusable prefix)."""
    rules = [
        f"Rule {index}: keep edits minimal, explain intent, and verify with the "
        f"workspace tests before claiming rule {index} is satisfied."
        for index in range(lines)
    ]
    return "You are a careful coding agent working in a Rust repository.\n" + "\n".join(rules)


def tool_result(name: str, arguments: str, lines: int) -> str:
    """Deterministic synthetic output for a tool call."""
    digest = hashlib.sha256(f"{name}\0{arguments}".encode()).hexdigest()[:12]
    body = [
        f"{digest} {index:04d}: fn item_{index}() -> usize {{ {index} }}" for index in range(lines)
    ]
    return "\n".join(body)


def iter_sse_data(lines: Iterable[bytes]) -> Iterable[tuple[str, str]]:
    """Yield `(event, data)` pairs from raw SSE lines."""
    event = "message"
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            event = "message"
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            yield event, line[len("data:") :].strip()


def accumulate_stream(events: Iterable[tuple[str, str]], clock=time.perf_counter) -> dict[str, Any]:
    """Fold chat stream chunks into one assistant turn plus timing and usage."""
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    first_delta_at = None
    usage: dict[str, Any] = {}
    finish_reason = None
    error = None
    for event, data in events:
        if data == "[DONE]":
            break
        payload = json.loads(data)
        if event == "error" or "error" in payload:
            error = payload.get("error", payload)
            continue
        if payload.get("usage"):
            usage = payload["usage"]
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            if first_delta_at is None and (
                delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")
            ):
                first_delta_at = clock()
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(
                    int(call.get("index", 0)),
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if call.get("id"):
                    slot["id"] = call["id"]
                function = call.get("function") or {}
                slot["function"]["name"] += function.get("name") or ""
                slot["function"]["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
    if any(reasoning):
        message["reasoning_content"] = "".join(reasoning)
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return {
        "message": message,
        "first_delta_at": first_delta_at,
        "usage": usage,
        "finish_reason": finish_reason,
        "error": error,
    }


def next_messages(
    messages: list[dict[str, Any]], reply: dict[str, Any], tool_result_lines: int
) -> list[dict[str, Any]]:
    """Append the assistant reply and the next agent input."""
    out = [*messages, reply]
    calls = reply.get("tool_calls") or []
    for index, call in enumerate(calls):
        function = call["function"]
        out.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id") or f"call_{index}",
                "content": tool_result(function["name"], function["arguments"], tool_result_lines),
            }
        )
    if not calls:
        out.append({"role": "user", "content": "Continue."})
    return out


def turn_row(turn: int, started: float, result: dict[str, Any]) -> dict[str, Any]:
    usage = result["usage"]
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    first = result["first_delta_at"]
    message = result["message"]
    return {
        "turn": turn,
        "ttft_ms": None if first is None else round((first - started) * 1000.0, 1),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "recomputed_tokens": max(prompt - cached, 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "finish_reason": result["finish_reason"],
        "tool_calls": len(message.get("tool_calls") or []),
        "reply_sha256": hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest(),
        "error": result["error"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Follow-up turns only: turn 0 is a cold prompt in every setting."""
    follow = [row for row in rows if row["turn"] > 0 and row["error"] is None]
    ttfts = [row["ttft_ms"] for row in follow if row["ttft_ms"] is not None]
    return {
        "followup_turns": len(follow),
        "recomputed_tokens": sum(row["recomputed_tokens"] for row in follow),
        "cached_tokens": sum(row["cached_tokens"] for row in follow),
        "ttft_ms_total": round(sum(ttfts), 1),
        "ttft_ms_mean": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
    }


def stream_chat(url: str, payload: dict[str, Any], timeout: float) -> tuple[float, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return started, accumulate_stream(iter_sse_data(response))


def wait_ready(port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except OSError:
            time.sleep(1.0)
    raise SystemExit("server did not become ready")


def stop_server(server: subprocess.Popen[bytes]) -> None:
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


def run_session(args: argparse.Namespace) -> list[dict[str, Any]]:
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(args.system_lines)},
        {"role": "user", "content": TASK},
    ]
    rows = []
    for turn in range(args.turns):
        payload: dict[str, Any] = {
            "model": args.model_id,
            "messages": messages,
            "tools": TOOLS,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "seed": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if args.no_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        started, result = stream_chat(url, payload, args.request_timeout)
        row = turn_row(turn, started, result)
        rows.append(row)
        print(
            f"turn={turn:2d} prompt={row['prompt_tokens']:6d} cached={row['cached_tokens']:6d} "
            f"recomputed={row['recomputed_tokens']:6d} ttft_ms={row['ttft_ms']} "
            f"calls={row['tool_calls']} finish={row['finish_reason']}",
            file=sys.stderr,
        )
        if result["error"] is not None:
            break
        reply = result["message"]
        if args.drop_reasoning:
            reply = {key: value for key, value in reply.items() if key != "reasoning_content"}
        messages = next_messages(messages, reply, args.tool_result_lines)
    return rows


def run_setting(args: argparse.Namespace, setting: str) -> dict[str, Any]:
    env = {**os.environ, "AX_MLX_NATIVE_CONFIRM": "1"}
    env.pop(SEMANTIC_CUTS_ENV, None)
    if setting == "on":
        env[SEMANTIC_CUTS_ENV] = "1"
    cmd = [
        str(args.server),
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(args.model_dir),
        "--model-id",
        args.model_id,
        "--port",
        str(args.port),
        *args.server_arg,
    ]
    print(f"== semantic cuts {setting}", file=sys.stderr)
    server = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(args.port, args.ready_timeout)
        rows = run_session(args)
    finally:
        stop_server(server)
    return {"semantic_cuts": setting, "rows": rows, "summary": summarize(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--port", type=positive_int, default=31498)
    parser.add_argument("--turns", type=positive_int, default=8)
    parser.add_argument("--system-lines", type=positive_int, default=200)
    parser.add_argument("--tool-result-lines", type=positive_int, default=120)
    parser.add_argument("--max-tokens", type=positive_int, default=512)
    parser.add_argument("--semantic-cuts", type=parse_settings, default=list(SETTINGS))
    parser.add_argument(
        "--drop-reasoning",
        action="store_true",
        help="Replay assistant turns without reasoning_content, as some clients do.",
    )
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra ax-engine-server argument (repeatable), e.g. "
        "--server-arg=--mlx-mtp-policy --server-arg=disabled.",
    )
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "turns": args.turns,
        "system_lines": args.system_lines,
        "tool_result_lines": args.tool_result_lines,
        "max_tokens": args.max_tokens,
        "drop_reasoning": args.drop_reasoning,
        "no_thinking": args.no_thinking,
        "server_args": args.server_arg,
        "settings": [run_setting(args, setting) for setting in args.semantic_cuts],
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
