#!/usr/bin/env python3
"""OpenWebUI integration smoke for AX OpenAI-compatible serving.

This probes OpenWebUI as an OpenAI proxy, not as a browser UI.  The goal is to
catch integration failures where OpenWebUI is correctly configured but the AX
backend disconnects or emits obviously corrupted text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = "what is agi ?"
DEFAULT_MAX_TOKENS = 96
DEFAULT_TIMEOUT_SECS = 120.0
OPENWEBUI_PROXY_MODELS_PATH = "/openai/v1/models"
OPENWEBUI_PROXY_CHAT_PATH = "/openai/v1/chat/completions"
BACKEND_ERROR_PATTERNS = (
    "upstream unavailable",
    "child is not reachable",
    "server disconnected",
    "connection refused",
    "503 service unavailable",
)
EXPECTED_HINTS_BY_PROMPT = {
    "agi": ("artificial", "general", "intelligence"),
    "two sum": ("two", "sum"),
}


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    model_visible: bool
    model_id: str
    assistant_text: str
    corruption_reasons: list[str]
    openwebui_base_url: str
    chat_path: str


def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def openwebui_url(base_url: str, path: str) -> str:
    return normalize_base_url(base_url) + path


def docker_openai_base_url(base_url: str) -> str:
    """Rewrite loopback AX URLs so the OpenWebUI container can reach the host."""

    parsed = urllib.parse.urlparse(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return base_url

    host = "host.docker.internal"
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    return urllib.parse.urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float,
) -> dict[str, Any]:
    headers = {"accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    if not raw:
        return {}
    return json.loads(raw)


def wait_for_openwebui(base_url: str, timeout_secs: float) -> None:
    deadline = time.monotonic() + timeout_secs
    last_error: str | None = None
    while time.monotonic() < deadline:
        for path in ("/health", "/"):
            try:
                url = openwebui_url(base_url, path)
                request = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(request, timeout=5) as response:
                    if 200 <= response.status < 500:
                        return
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = str(error)
        time.sleep(0.5)
    raise RuntimeError(f"OpenWebUI did not become ready: {last_error or 'timeout'}")


def list_openwebui_models(base_url: str, timeout: float) -> list[str]:
    payload = request_json(
        "GET",
        openwebui_url(base_url, OPENWEBUI_PROXY_MODELS_PATH),
        timeout=timeout,
    )
    models = payload.get("data")
    if not isinstance(models, list):
        return []
    result: list[str] = []
    for item in models:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            result.append(item["id"])
    return result


def chat_completion_payload(model_id: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }


def extract_assistant_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(first.get("text"), str):
        return first["text"]
    return ""


def single_punctuation_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.fullmatch(r"\s*[!`*_#=\-.]{1,8}\s*", line))


def detect_corruption(text: str, prompt: str) -> list[str]:
    reasons: list[str] = []
    lowered = text.lower()
    if not text.strip():
        reasons.append("empty assistant content")
    for pattern in BACKEND_ERROR_PATTERNS:
        if pattern in lowered:
            reasons.append(f"backend error text: {pattern}")
    if single_punctuation_line_count(text) >= 4:
        reasons.append("repeated punctuation-only lines")
    if re.search(r"([!`*_#=\-.])\s*(?:\n\s*\1\s*){3,}", text):
        reasons.append("repeated punctuation token pattern")
    if len(text) >= 80:
        punctuation = sum(1 for char in text if char in "!`*_#=-")
        if punctuation / max(1, len(text)) > 0.20:
            reasons.append("high punctuation ratio")

    prompt_lower = prompt.lower()
    for key, hints in EXPECTED_HINTS_BY_PROMPT.items():
        if key in prompt_lower and not any(hint in lowered for hint in hints):
            reasons.append(f"missing expected semantic hint for {key}")
    return reasons


def run_probe(
    *,
    openwebui_base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    timeout_secs: float,
) -> ProbeResult:
    wait_for_openwebui(openwebui_base_url, timeout_secs)
    models = list_openwebui_models(openwebui_base_url, timeout_secs)
    model_visible = model_id in models
    if not model_visible:
        return ProbeResult(
            ok=False,
            model_visible=False,
            model_id=model_id,
            assistant_text="",
            corruption_reasons=[f"model not visible through OpenWebUI proxy: {model_id}"],
            openwebui_base_url=openwebui_base_url,
            chat_path=OPENWEBUI_PROXY_CHAT_PATH,
        )

    response = request_json(
        "POST",
        openwebui_url(openwebui_base_url, OPENWEBUI_PROXY_CHAT_PATH),
        chat_completion_payload(model_id, prompt, max_tokens),
        timeout=timeout_secs,
    )
    assistant_text = extract_assistant_text(response)
    corruption_reasons = detect_corruption(assistant_text, prompt)
    return ProbeResult(
        ok=not corruption_reasons,
        model_visible=model_visible,
        model_id=model_id,
        assistant_text=assistant_text,
        corruption_reasons=corruption_reasons,
        openwebui_base_url=openwebui_base_url,
        chat_path=OPENWEBUI_PROXY_CHAT_PATH,
    )


def write_report(path: Path, result: ProbeResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "ax.openwebui_e2e.v1",
                "ok": result.ok,
                "model_visible": result.model_visible,
                "model_id": result.model_id,
                "openwebui_base_url": result.openwebui_base_url,
                "chat_path": result.chat_path,
                "corruption_reasons": result.corruption_reasons,
                "assistant_text": result.assistant_text,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openwebui-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout-secs", type=float, default=DEFAULT_TIMEOUT_SECS)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--print-docker-openai-base-url",
        metavar="AX_BASE_URL",
        help="print the AX base URL rewritten for an OpenWebUI Docker container",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_docker_openai_base_url:
        print(docker_openai_base_url(args.print_docker_openai_base_url))
        return 0

    result = run_probe(
        openwebui_base_url=args.openwebui_base_url,
        model_id=args.model_id,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        timeout_secs=args.timeout_secs,
    )
    if args.report is not None:
        write_report(args.report, result)
    if result.ok:
        print(f"[openwebui-e2e] ok model={result.model_id}")
        return 0
    print(
        "[openwebui-e2e] failed: " + "; ".join(result.corruption_reasons),
        file=sys.stderr,
    )
    if result.assistant_text:
        print(result.assistant_text, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
