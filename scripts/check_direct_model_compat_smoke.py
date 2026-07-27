#!/usr/bin/env python3
"""Smoke native MLX Qwen/Gemma compatibility through OpenAI and Ollama surfaces.

The check is artifact-gated. With no configured local MLX model artifacts it
prints a JSON skip result and exits zero, so CI and local review can include the
gate without requiring model downloads on every host.

The Qwen request shapes cover the fields OpenClaw currently sends through its
OpenAI-completions and native Ollama transports.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN_ARTIFACTS_ENV = "AX_ENGINE_QWEN_CODER_NEXT_ARTIFACTS_DIR"
QWEN3_VL_ARTIFACTS_ENV = "AX_ENGINE_QWEN3_VL_ARTIFACTS_DIR"
QWEN35_ARTIFACTS_ENV = "AX_ENGINE_QWEN35_9B_ARTIFACTS_DIR"
QWEN36_27B_ARTIFACTS_ENV = "AX_ENGINE_QWEN36_27B_ARTIFACTS_DIR"
QWEN36_ARTIFACTS_ENV = "AX_ENGINE_QWEN36_35B_ARTIFACTS_DIR"
GEMMA4_ARTIFACTS_ENV = "AX_ENGINE_GEMMA4_ARTIFACTS_DIR"
LEGACY_MLX_ARTIFACTS_ENV = "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR"
RAW_TOOL_MARKERS = ("<tool_call>", "<|tool_call>", "<tool_code>", "<|python_tag|>")


@dataclass(frozen=True)
class SmokeTarget:
    kind: str
    model_id: str
    artifacts_dir: Path


def parser() -> argparse.ArgumentParser:
    parsed = argparse.ArgumentParser(
        description=(
            "Start ax-engine-server against native MLX Qwen3-Coder-Next, "
            "Qwen3-VL, Qwen3.5 9B, Qwen3.6 27B/35B, or Gemma 4 "
            "artifacts and verify OpenAI/Ollama tool-call compatibility surfaces."
        )
    )
    parsed.add_argument("--qwen-artifacts", type=Path)
    parsed.add_argument("--qwen3-vl-artifacts", type=Path)
    parsed.add_argument("--qwen35-artifacts", type=Path)
    parsed.add_argument("--qwen36-27b-artifacts", type=Path)
    parsed.add_argument("--qwen36-artifacts", type=Path)
    parsed.add_argument("--gemma4-artifacts", type=Path)
    parsed.add_argument(
        "--qwen-model-id",
        default="ax-engine/qwen3-coder-next",
        help="model id to expose while testing Qwen3-Coder-Next artifacts",
    )
    parsed.add_argument(
        "--qwen3-vl-model-id",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="model id to expose while testing Qwen3-VL artifacts",
    )
    parsed.add_argument(
        "--qwen35-model-id",
        default="Qwen3.5-9B-4bit",
        help="model id to expose while testing Qwen3.5 9B artifacts",
    )
    parsed.add_argument(
        "--qwen36-27b-model-id",
        default="Qwen3.6-27B-4bit",
        help="model id to expose while testing Qwen3.6 27B artifacts",
    )
    parsed.add_argument(
        "--qwen36-model-id",
        default="Qwen3.6-35B-A3B-4bit",
        help="model id to expose while testing Qwen3.6 35B-A3B artifacts",
    )
    parsed.add_argument(
        "--gemma4-model-id",
        default="gemma4-e2b-it",
        help="model id to expose while testing Gemma 4 artifacts",
    )
    parsed.add_argument(
        "--server-bin",
        type=Path,
        help="prebuilt ax-engine-server binary; otherwise target/debug is used or built",
    )
    parsed.add_argument(
        "--release",
        action="store_true",
        help="use/build target/release/ax-engine-server instead of target/debug",
    )
    parsed.add_argument(
        "--no-build",
        action="store_true",
        help="fail if the requested server binary does not already exist",
    )
    parsed.add_argument("--ready-timeout-sec", type=float, default=180.0)
    parsed.add_argument("--request-timeout-sec", type=float, default=180.0)
    parsed.add_argument(
        "--expect-tool-call",
        action="store_true",
        default=os.environ.get("AX_DIRECT_MODEL_COMPAT_EXPECT_TOOL_CALLS") == "1",
        help="require at least one parsed tool call instead of only validating envelopes",
    )
    return parsed


def resolve_smoke_targets(
    args: argparse.Namespace,
    env: Mapping[str, str] | None = None,
) -> list[SmokeTarget]:
    if env is None:
        env = os.environ
    qwen = args.qwen_artifacts or _env_path(env, QWEN_ARTIFACTS_ENV)
    if qwen is None:
        qwen = _env_path(env, LEGACY_MLX_ARTIFACTS_ENV)
    qwen3_vl = args.qwen3_vl_artifacts or _env_path(env, QWEN3_VL_ARTIFACTS_ENV)
    qwen35 = args.qwen35_artifacts or _env_path(env, QWEN35_ARTIFACTS_ENV)
    qwen36_27b = args.qwen36_27b_artifacts or _env_path(
        env, QWEN36_27B_ARTIFACTS_ENV
    )
    qwen36 = args.qwen36_artifacts or _env_path(env, QWEN36_ARTIFACTS_ENV)
    gemma4 = args.gemma4_artifacts or _env_path(env, GEMMA4_ARTIFACTS_ENV)

    targets = []
    if qwen is not None:
        targets.append(
            SmokeTarget(
                kind="qwen3-coder-next",
                model_id=args.qwen_model_id,
                artifacts_dir=Path(qwen),
            )
        )
    if qwen3_vl is not None:
        targets.append(
            SmokeTarget(
                kind="qwen3-vl",
                model_id=args.qwen3_vl_model_id,
                artifacts_dir=Path(qwen3_vl),
            )
        )
    if qwen35 is not None:
        targets.append(
            SmokeTarget(
                kind="qwen3.5-9b",
                model_id=args.qwen35_model_id,
                artifacts_dir=Path(qwen35),
            )
        )
    if qwen36_27b is not None:
        targets.append(
            SmokeTarget(
                kind="qwen3.6-27b",
                model_id=args.qwen36_27b_model_id,
                artifacts_dir=Path(qwen36_27b),
            )
        )
    if qwen36 is not None:
        targets.append(
            SmokeTarget(
                kind="qwen3.6-35b-a3b",
                model_id=args.qwen36_model_id,
                artifacts_dir=Path(qwen36),
            )
        )
    if gemma4 is not None:
        targets.append(
            SmokeTarget(
                kind="gemma4",
                model_id=args.gemma4_model_id,
                artifacts_dir=Path(gemma4),
            )
        )
    return targets


def build_openai_tool_request(model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Read README.md and report the first heading. Use the "
                    "read_file tool if you need repository contents."
                ),
            }
        ],
        "tools": [_read_file_tool_spec()],
        "tool_choice": "auto",
        "temperature": 0,
        "max_completion_tokens": 96,
        "chat_template_kwargs": {"enable_thinking": False},
        "store": False,
        "stream": False,
    }


def build_ollama_chat_request(model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Read README.md and report the first heading. Use the "
                    "read_file tool if you need repository contents."
                ),
            }
        ],
        "tools": [_read_file_tool_spec()],
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": 16_384,
            "num_predict": 96,
        },
        "stream": False,
    }


def main() -> int:
    args = parser().parse_args()
    targets = resolve_smoke_targets(args)
    if not targets:
        print(
            json.dumps(
                {
                    "schema": "ax.direct_model_compat_smoke.v1",
                    "status": "skipped",
                    "reason": (
                        f"set {QWEN_ARTIFACTS_ENV}, {QWEN3_VL_ARTIFACTS_ENV}, "
                        f"{QWEN35_ARTIFACTS_ENV}, {QWEN36_27B_ARTIFACTS_ENV}, "
                        f"{QWEN36_ARTIFACTS_ENV}, and/or {GEMMA4_ARTIFACTS_ENV} "
                        "to run native MLX "
                        "compatibility smoke checks"
                    ),
                    "results": [],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    try:
        server_bin = ensure_server_binary(args)
        results = [run_target(args, server_bin, target) for target in targets]
    except (OSError, SmokeFailure) as error:
        print(
            json.dumps(
                {
                    "schema": "ax.direct_model_compat_smoke.v1",
                    "status": "failed",
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "schema": "ax.direct_model_compat_smoke.v1",
                "status": "passed",
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_target(
    args: argparse.Namespace, server_bin: Path, target: SmokeTarget
) -> dict[str, Any]:
    if not target.artifacts_dir.is_dir():
        raise SmokeFailure(
            f"{target.kind} artifacts dir does not exist: {target.artifacts_dir}"
        )
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = Path(tempfile.gettempdir()) / (
        f"ax-engine-direct-compat-{target.kind}-{port}.log"
    )
    command = [
        str(server_bin),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-id",
        target.model_id,
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(target.artifacts_dir),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            health = wait_for_health(
                process,
                f"{base_url}/health",
                args.ready_timeout_sec,
                log_path,
            )
            models = http_json(
                "GET", f"{base_url}/v1/models", None, args.request_timeout_sec
            )
            model_card = assert_model_metadata(models, target.model_id)
            ollama_show = http_json(
                "POST",
                f"{base_url}/api/show",
                {"name": target.model_id},
                args.request_timeout_sec,
            )
            ollama_capabilities = assert_ollama_show_response(
                ollama_show, model_card
            )
            openai = http_json(
                "POST",
                f"{base_url}/v1/chat/completions",
                build_openai_tool_request(target.model_id),
                args.request_timeout_sec,
            )
            openai_tool_calls = assert_openai_chat_response(openai, args.expect_tool_call)
            ollama = http_json(
                "POST",
                f"{base_url}/api/chat",
                build_ollama_chat_request(target.model_id),
                args.request_timeout_sec,
            )
            ollama_tool_calls = assert_ollama_chat_response(ollama, args.expect_tool_call)
            return {
                "kind": target.kind,
                "model_id": target.model_id,
                "artifacts_dir": str(target.artifacts_dir),
                "backend": health.get("runtime", {}).get("selected_backend"),
                "openai_tool_calling_supported": model_card["ax_engine"][
                    "openai_tool_calling_supported"
                ],
                "capabilities_toolcall": model_card["capabilities"]["toolcall"],
                "ollama_capabilities": ollama_capabilities,
                "openai_tool_call_count": openai_tool_calls,
                "ollama_tool_call_count": ollama_tool_calls,
                "log_path": str(log_path),
            }
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def ensure_server_binary(args: argparse.Namespace) -> Path:
    if args.server_bin:
        server_bin = args.server_bin
        if not server_bin.is_file():
            raise SmokeFailure(f"server binary does not exist: {server_bin}")
        return server_bin
    profile = "release" if args.release else "debug"
    server_bin = REPO_ROOT / "target" / profile / "ax-engine-server"
    if args.no_build:
        if server_bin.is_file():
            return server_bin
        raise SmokeFailure(f"server binary does not exist: {server_bin}")
    command = ["cargo", "build", "-p", "ax-engine-server"]
    if args.release:
        command.append("--release")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not server_bin.is_file():
        raise SmokeFailure(f"cargo build did not produce {server_bin}")
    return server_bin


def wait_for_health(
    process: subprocess.Popen[str],
    url: str,
    timeout_sec: float,
    log_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SmokeFailure(
                f"server exited before readiness with status {process.returncode}; "
                f"see {log_path}"
            )
        try:
            health = http_json("GET", url, None, timeout_sec=5.0)
            if health.get("status") == "ok":
                backend = str(health.get("runtime", {}).get("selected_backend", "")).lower()
                if backend != "mlx":
                    raise SmokeFailure(
                        f"expected native MLX backend from /health, got {backend!r}"
                    )
                return health
            last_error = f"unexpected health response: {health}"
        except SmokeFailure as error:
            last_error = str(error)
        time.sleep(1)
    raise SmokeFailure(f"timed out waiting for /health: {last_error}; see {log_path}")


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout_sec: float,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise SmokeFailure(f"{method} {url} failed with HTTP {error.code}: {body}")
    except urllib.error.URLError as error:
        raise SmokeFailure(f"{method} {url} failed: {error.reason}")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise SmokeFailure(f"{method} {url} returned invalid JSON: {error}")
    if not isinstance(decoded, dict):
        raise SmokeFailure(f"{method} {url} returned non-object JSON: {decoded!r}")
    return decoded


def assert_model_metadata(models: dict[str, Any], model_id: str) -> dict[str, Any]:
    data = models.get("data")
    if not isinstance(data, list) or not data:
        raise SmokeFailure("/v1/models did not return a non-empty data list")
    model_card = next(
        (
            candidate
            for candidate in data
            if isinstance(candidate, dict) and candidate.get("id") == model_id
        ),
        None,
    )
    if model_card is None:
        returned_ids = [
            candidate.get("id")
            for candidate in data
            if isinstance(candidate, dict) and candidate.get("id") is not None
        ]
        raise SmokeFailure(
            f"/v1/models did not include expected id {model_id!r}; got {returned_ids!r}"
        )
    if not isinstance(model_card, dict):
        raise SmokeFailure("/v1/models matching entry is not an object")
    capabilities = model_card.get("capabilities")
    ax_engine = model_card.get("ax_engine")
    if not isinstance(capabilities, dict) or not capabilities.get("toolcall"):
        raise SmokeFailure("/v1/models did not advertise capabilities.toolcall=true")
    if not isinstance(ax_engine, dict) or not ax_engine.get(
        "openai_tool_calling_supported"
    ):
        raise SmokeFailure(
            "/v1/models did not advertise ax_engine.openai_tool_calling_supported=true"
        )
    return model_card


def assert_openai_chat_response(response: dict[str, Any], expect_tool_call: bool) -> int:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SmokeFailure("OpenAI chat response did not include choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise SmokeFailure("OpenAI chat choices[0] is not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise SmokeFailure("OpenAI chat choices[0].message is not an object")
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise SmokeFailure("OpenAI chat message.tool_calls is not a list")
    content = message.get("content") or ""
    if isinstance(content, str):
        assert_no_raw_tool_markers(content, "OpenAI chat content")
    if expect_tool_call and not tool_calls:
        raise SmokeFailure("OpenAI chat response did not include parsed tool_calls")
    return len(tool_calls)


def assert_ollama_show_response(
    response: dict[str, Any], model_card: dict[str, Any]
) -> list[str]:
    capabilities = response.get("capabilities")
    if not isinstance(capabilities, list) or "completion" not in capabilities:
        raise SmokeFailure("/api/show did not advertise completion capability")
    if "tools" not in capabilities:
        raise SmokeFailure("/api/show did not advertise tools capability")
    model_info = response.get("model_info")
    if not isinstance(model_info, dict):
        raise SmokeFailure("/api/show did not return model_info")
    context_length = model_info.get("ax_engine.context_length")
    if context_length != model_card.get("context_length"):
        raise SmokeFailure(
            "/api/show context length does not match /v1/models: "
            f"{context_length!r} != {model_card.get('context_length')!r}"
        )
    advertised = model_card.get("capabilities")
    if isinstance(advertised, dict):
        modalities = advertised.get("input")
        if (
            isinstance(modalities, dict)
            and modalities.get("image") is True
            and "vision" not in capabilities
        ):
            raise SmokeFailure("/api/show omitted vision for an image-capable model")
        if advertised.get("reasoning") is True and "thinking" not in capabilities:
            raise SmokeFailure("/api/show omitted thinking for a reasoning-capable model")
    return [str(capability) for capability in capabilities]


def assert_ollama_chat_response(response: dict[str, Any], expect_tool_call: bool) -> int:
    message = response.get("message")
    if not isinstance(message, dict):
        raise SmokeFailure("Ollama chat response did not include message")
    if response.get("done") is not True:
        raise SmokeFailure("Ollama chat response did not finish with done=true")
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise SmokeFailure("Ollama chat message.tool_calls is not a list")
    content = message.get("content") or ""
    if isinstance(content, str):
        assert_no_raw_tool_markers(content, "Ollama chat content")
    if expect_tool_call and not tool_calls:
        raise SmokeFailure("Ollama chat response did not include parsed tool_calls")
    return len(tool_calls)


def assert_no_raw_tool_markers(content: str, context: str) -> None:
    marker = next((marker for marker in RAW_TOOL_MARKERS if marker in content), None)
    if marker is not None:
        raise SmokeFailure(f"{context} leaked unparsed tool marker {marker!r}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _env_path(env: Mapping[str, str], name: str) -> Path | None:
    value = env.get(name, "").strip()
    if not value:
        return None
    return Path(value)


def _read_file_tool_spec() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the current repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative file path.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


class SmokeFailure(RuntimeError):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
