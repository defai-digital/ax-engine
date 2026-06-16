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
OPENWEBUI_PROXY_MODELS_PATH = "/openai/models"
OPENWEBUI_PROXY_CHAT_PATH = "/openai/chat/completions"
OPENWEBUI_SIGNIN_PATH = "/api/v1/auths/signin"
AX_DIRECT_MODELS_PATH = "/v1/models"
AX_DIRECT_CHAT_PATH = "/v1/chat/completions"
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
    observed_models: list[str]
    assistant_text: str
    corruption_reasons: list[str]
    openwebui_base_url: str
    chat_path: str


def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def openwebui_url(base_url: str, path: str) -> str:
    return normalize_base_url(base_url) + path


def docker_openai_base_url(base_url: str) -> str:
    """Rewrite loopback AX URLs so the OpenWebUI container can reach the host.

    OpenWebUI expects OPENAI_API_BASE_URL to include the /v1 prefix (it
    appends /chat/completions directly).  If the caller supplies a bare
    host:port base URL, append /v1 so OpenWebUI can reach the endpoints.
    """

    parsed = urllib.parse.urlparse(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        # Non-loopback: still ensure /v1 suffix for OpenWebUI.
        path = parsed.path.rstrip("/")
        if not path.endswith("/v1"):
            parsed = parsed._replace(path=path + "/v1")
        return urllib.parse.urlunparse(parsed)

    host = "host.docker.internal"
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = path + "/v1"
    return urllib.parse.urlunparse(
        (
            parsed.scheme,
            netloc,
            path,
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
    bearer_token: str | None = None,
) -> dict[str, Any]:
    headers = {"accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    if bearer_token:
        headers["authorization"] = f"Bearer {bearer_token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    if not raw:
        return {}
    return json.loads(raw)


def signin_openwebui(base_url: str, timeout: float) -> str | None:
    """Obtain a JWT bearer token from OpenWebUI by signing in as the default admin.

    Returns the token string on success, or None if sign-in fails (e.g. auth disabled).
    """
    try:
        payload = request_json(
            "POST",
            openwebui_url(base_url, OPENWEBUI_SIGNIN_PATH),
            {"email": "user@example.com", "password": "password"},
            timeout=timeout,
        )
        token = payload.get("token")
        if isinstance(token, str) and token:
            return token
    except Exception:
        pass
    return None


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


def list_openwebui_models(base_url: str, timeout: float, bearer_token: str | None = None) -> list[str]:
    payload = request_json(
        "GET",
        openwebui_url(base_url, OPENWEBUI_PROXY_MODELS_PATH),
        timeout=timeout,
        bearer_token=bearer_token,
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
        punctuation = sum(1 for char in text if char in "!`*_#=-.")
        if punctuation / max(1, len(text)) > 0.20:
            reasons.append("high punctuation ratio")

    prompt_lower = prompt.lower()
    for key, hints in EXPECTED_HINTS_BY_PROMPT.items():
        if key in prompt_lower and not any(hint in lowered for hint in hints):
            reasons.append(f"missing expected semantic hint for {key}")
    return reasons


def wait_for_ax_direct(base_url: str, timeout_secs: float) -> None:
    deadline = time.monotonic() + timeout_secs
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            url = openwebui_url(base_url, AX_DIRECT_MODELS_PATH)
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise RuntimeError(f"AX Engine did not become ready: {last_error or 'timeout'}")


def list_ax_direct_models(base_url: str, timeout: float) -> list[str]:
    payload = request_json(
        "GET",
        openwebui_url(base_url, AX_DIRECT_MODELS_PATH),
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


def run_probe(
    *,
    openwebui_base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    timeout_secs: float,
    ax_direct: bool = False,
) -> ProbeResult:
    chat_path = AX_DIRECT_CHAT_PATH if ax_direct else OPENWEBUI_PROXY_CHAT_PATH

    bearer_token: str | None = None
    if ax_direct:
        wait_for_ax_direct(openwebui_base_url, timeout_secs)
        models = list_ax_direct_models(openwebui_base_url, timeout_secs)
    else:
        wait_for_openwebui(openwebui_base_url, timeout_secs)
        bearer_token = signin_openwebui(openwebui_base_url, timeout_secs)
        models = list_openwebui_models(openwebui_base_url, timeout_secs, bearer_token=bearer_token)

    model_visible = model_id in models
    if not model_visible:
        label = "AX Engine" if ax_direct else "OpenWebUI proxy"
        return ProbeResult(
            ok=False,
            model_visible=False,
            model_id=model_id,
            observed_models=models,
            assistant_text="",
            corruption_reasons=[f"model not visible through {label}: {model_id}"],
            openwebui_base_url=openwebui_base_url,
            chat_path=chat_path,
        )

    response = request_json(
        "POST",
        openwebui_url(openwebui_base_url, chat_path),
        chat_completion_payload(model_id, prompt, max_tokens),
        timeout=timeout_secs,
        bearer_token=bearer_token,
    )
    assistant_text = extract_assistant_text(response)
    corruption_reasons = detect_corruption(assistant_text, prompt)
    return ProbeResult(
        ok=not corruption_reasons,
        model_visible=model_visible,
        model_id=model_id,
        observed_models=models,
        assistant_text=assistant_text,
        corruption_reasons=corruption_reasons,
        openwebui_base_url=openwebui_base_url,
        chat_path=chat_path,
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
                "observed_models": result.observed_models,
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
    parser.add_argument("--model-id")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout-secs", type=float, default=DEFAULT_TIMEOUT_SECS)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--ax-direct",
        action="store_true",
        help=(
            "probe AX Engine directly at --openwebui-base-url/v1 instead of routing "
            "through an OpenWebUI Docker proxy; no Docker required"
        ),
    )
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
    if not args.model_id:
        parser.error("--model-id is required unless --print-docker-openai-base-url is used")

    result = run_probe(
        openwebui_base_url=args.openwebui_base_url,
        model_id=args.model_id,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        timeout_secs=args.timeout_secs,
        ax_direct=args.ax_direct,
    )
    if args.report is not None:
        write_report(args.report, result)
    mode_label = "ax-direct" if args.ax_direct else "openwebui-e2e"
    if result.ok:
        print(f"[{mode_label}] ok model={result.model_id}")
        return 0
    print(
        f"[{mode_label}] failed: " + "; ".join(result.corruption_reasons),
        file=sys.stderr,
    )
    if not result.model_visible:
        print(
            f"[{mode_label}] observed models: "
            + (", ".join(result.observed_models) if result.observed_models else "<none>"),
            file=sys.stderr,
        )
    if result.assistant_text:
        print(result.assistant_text, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
