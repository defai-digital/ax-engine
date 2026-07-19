"""Product-surface probes beyond the chat question bank.

These exercise serving paths that pure Q&A sampling does not cover:

* concurrent chat completions
* stream vs non-stream smoke (+ content parity)
* request cancel via ``/v1/requests``
* OpenAI tools schema acceptance (no panic / structured HTTP response)
* multimodal image chat (**capability-aware**: soft-skip only when the model
  does not advertise image input)
* fail-closed media policy (remote URL reject, public video reject)

Best practice: soft-skips are allowed only when the model card does **not**
claim the capability. Advertised vision that returns 4xx/empty is a hard fail.

All probes target a **running** server. Pure helpers and payload builders are
unit-tested offline in ``scripts/test_qa_surface_probes.py``.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


@dataclass
class SurfaceProbeResult:
    name: str
    passed: bool
    detail: str = ""
    hard: bool = True
    elapsed_ms: float = 0.0
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SurfaceReport:
    base_url: str
    model: str
    results: list[SurfaceProbeResult] = field(default_factory=list)

    @property
    def hard_passed(self) -> bool:
        return all(r.passed or r.skipped for r in self.results if r.hard)

    @property
    def summary_line(self) -> str:
        hard = [r for r in self.results if r.hard and not r.skipped]
        ok = sum(1 for r in hard if r.passed)
        skip = sum(1 for r in self.results if r.skipped)
        return f"surface hard {ok}/{len(hard)} passed ({skip} skipped)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "surface_probes",
            "base_url": self.base_url,
            "model": self.model,
            "hard_passed": self.hard_passed,
            "summary": self.summary_line,
            "results": [r.as_dict() for r in self.results],
        }


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float = 60.0,
) -> tuple[int, dict[str, Any] | str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", 200) or 200
            try:
                return int(status), json.loads(body)
            except json.JSONDecodeError:
                return int(status), body
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        try:
            return int(exc.code), json.loads(raw)
        except json.JSONDecodeError:
            return int(exc.code), raw
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, f"connection_error: {exc}"


def _get_json(url: str, timeout: float = 30.0) -> tuple[int, dict[str, Any] | str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", 200) or 200
            try:
                return int(status), json.loads(body)
            except json.JSONDecodeError:
                return int(status), body
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        try:
            return int(exc.code), json.loads(raw)
        except json.JSONDecodeError:
            return int(exc.code), raw
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, f"connection_error: {exc}"


def fetch_model_card(
    base_url: str,
    model: str,
    *,
    timeout: float = 30.0,
) -> Optional[dict[str, Any]]:
    """Return the ``/v1/models`` card matching ``model``, or None."""
    status, body = _get_json(f"{base_url.rstrip('/')}/v1/models", timeout=timeout)
    if status != 200 or not isinstance(body, dict):
        return None
    data = body.get("data") or []
    if not isinstance(data, list):
        return None
    for item in data:
        if isinstance(item, dict) and str(item.get("id") or "") == model:
            return item
    # Single-model servers often expose one card under a different id alias.
    if len(data) == 1 and isinstance(data[0], dict):
        return data[0]
    return None


def model_advertises_image(card: Optional[dict[str, Any]]) -> bool:
    """True when the model card claims image / native multimodal input."""
    if not card:
        return False
    caps = card.get("capabilities") or {}
    if isinstance(caps, dict):
        inp = caps.get("input") or {}
        if isinstance(inp, dict) and inp.get("image") is True:
            return True
        if caps.get("attachment") is True and isinstance(inp, dict):
            # attachment alone is weak; require explicit image when present.
            if "image" in inp:
                return bool(inp.get("image"))
    ax = card.get("ax_engine") or {}
    if isinstance(ax, dict):
        if ax.get("native_multimodal_input_supported") is True:
            return True
        if ax.get("gemma4_unified_multimodal_input_supported") is True:
            return True
        if ax.get("openai_tokenized_multimodal_input_supported") is True:
            return True
    return False


def normalize_answer_text(text: str) -> str:
    """Collapse whitespace for stream/non-stream parity comparison."""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def extract_sse_chat_text(raw: str) -> str:
    """Best-effort content extraction from OpenAI-style chat SSE."""
    pieces: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        choices = obj.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta") or {}
        if isinstance(delta, dict) and delta.get("content"):
            pieces.append(str(delta["content"]))
            continue
        message = choices[0].get("message") or {}
        if isinstance(message, dict) and message.get("content"):
            pieces.append(str(message["content"]))
    return "".join(pieces)


def chat_completion_payload(
    model: str,
    content: str | list[dict[str, Any]],
    *,
    max_tokens: int = 32,
    temperature: float = 0.0,
    stream: bool = False,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal OpenAI chat.completions body (test helper)."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


def tiny_png_data_url() -> str:
    """1x1 blue PNG as data URL (stdlib only, no Pillow)."""
    # Minimal valid 1x1 PNG (blue-ish) — fixed fixture bytes.
    # Generated offline; stable for unit tests.
    png = bytes(
        [
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,
            0x08,
            0x02,
            0x00,
            0x00,
            0x00,
            0x90,
            0x77,
            0x53,
            0xDE,
            0x00,
            0x00,
            0x00,
            0x0C,
            0x49,
            0x44,
            0x41,
            0x54,
            0x08,
            0xD7,
            0x63,
            0xF8,
            0xCF,
            0xC0,
            0x00,
            0x00,
            0x03,
            0x01,
            0x01,
            0x00,
            0x18,
            0xDD,
            0x8D,
            0xB4,
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,
            0x44,
            0xAE,
            0x42,
            0x60,
            0x82,
        ]
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def extract_chat_content(body: dict[str, Any] | str) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        return None
    return str(content)


def probe_concurrent_chat(
    base_url: str,
    model: str,
    *,
    workers: int = 3,
    timeout: float = 90.0,
) -> SurfaceProbeResult:
    name = "concurrent_chat"
    start = time.monotonic()
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = chat_completion_payload(model, "Reply with the single word ok.", max_tokens=16)

    def one(_i: int) -> tuple[int, Optional[str]]:
        status, body = _post_json(url, payload, timeout=timeout)
        content = extract_chat_content(body) if status == 200 else None
        return status, content

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(one, i) for i in range(workers)]
            outcomes = [f.result() for f in futs]
    except Exception as exc:
        return SurfaceProbeResult(
            name,
            False,
            f"exception: {exc}",
            elapsed_ms=(time.monotonic() - start) * 1000,
        )

    elapsed = (time.monotonic() - start) * 1000
    bad = [o for o in outcomes if o[0] != 200 or o[1] is None]
    if bad:
        return SurfaceProbeResult(
            name,
            False,
            f"{len(bad)}/{workers} failed: {bad[:2]!r}",
            elapsed_ms=elapsed,
        )
    return SurfaceProbeResult(
        name, True, f"{workers} parallel chat completions ok", elapsed_ms=elapsed
    )


def probe_stream_and_nonstream(
    base_url: str,
    model: str,
    *,
    timeout: float = 90.0,
    require_parity: bool = True,
) -> SurfaceProbeResult:
    """Stream + non-stream smoke; optionally require normalized content parity."""
    name = "stream_nonstream"
    start = time.monotonic()
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    # Fixed short token answer — parity is meaningful at temperature 0.
    prompt = "Reply with the single digit 7 and nothing else."

    status_ns, body_ns = _post_json(
        url,
        chat_completion_payload(
            model, prompt, max_tokens=16, temperature=0.0, stream=False
        ),
        timeout=timeout,
    )
    content_ns = extract_chat_content(body_ns) if status_ns == 200 else None

    stream_payload = chat_completion_payload(
        model, prompt, max_tokens=16, temperature=0.0, stream=True
    )
    data = json.dumps(stream_payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    stream_ok = False
    stream_detail = ""
    stream_text = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        stream_ok = "data:" in raw and raw.strip() != ""
        stream_text = extract_sse_chat_text(raw)
        stream_detail = f"sse_bytes={len(raw)} stream_chars={len(stream_text)}"
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        stream_detail = f"stream_error={exc}"
        stream_ok = False
    except Exception as exc:
        stream_detail = f"stream_error={exc}"
        stream_ok = False

    elapsed = (time.monotonic() - start) * 1000
    if status_ns != 200 or content_ns is None:
        return SurfaceProbeResult(
            name,
            False,
            f"non-stream failed status={status_ns} content={content_ns!r}; {stream_detail}",
            elapsed_ms=elapsed,
        )
    if not stream_ok:
        return SurfaceProbeResult(
            name, False, f"stream failed; {stream_detail}", elapsed_ms=elapsed
        )
    if require_parity:
        ns_n = normalize_answer_text(content_ns)
        st_n = normalize_answer_text(stream_text)
        if not st_n:
            return SurfaceProbeResult(
                name,
                False,
                f"stream produced no extractable content; {stream_detail}",
                elapsed_ms=elapsed,
            )
        # Accept exact match or shared digit token (some templates add prose).
        if ns_n != st_n and not (
            "7" in ns_n and "7" in st_n and abs(len(ns_n) - len(st_n)) <= 24
        ):
            return SurfaceProbeResult(
                name,
                False,
                f"stream/non-stream mismatch ns={ns_n[:60]!r} st={st_n[:60]!r}",
                elapsed_ms=elapsed,
            )
        stream_detail += " parity=ok"
    return SurfaceProbeResult(
        name,
        True,
        f"non-stream chars={len(content_ns)}; {stream_detail}",
        elapsed_ms=elapsed,
    )


def probe_cancel_request(
    base_url: str,
    model: str,
    *,
    timeout: float = 60.0,
) -> SurfaceProbeResult:
    """Submit a generate request then cancel via /v1/requests/{id}/cancel."""
    name = "cancel_request"
    start = time.monotonic()
    base = base_url.rstrip("/")
    submit_url = f"{base}/v1/requests"
    # Short token prompt; cancel before or after scheduling.
    status, body = _post_json(
        submit_url,
        {
            "model": model,
            "input_tokens": list(range(1, 17)),
            "max_output_tokens": 64,
        },
        timeout=timeout,
    )
    if status in (404, 405):
        return SurfaceProbeResult(
            name,
            True,
            f"cancel API unavailable (HTTP {status}); skipped",
            hard=False,
            skipped=True,
            elapsed_ms=(time.monotonic() - start) * 1000,
        )
    # Submit commonly returns 201 Created (also accept 200).
    if status not in (200, 201) or not isinstance(body, dict):
        return SurfaceProbeResult(
            name,
            False,
            f"submit failed status={status} body={str(body)[:200]}",
            elapsed_ms=(time.monotonic() - start) * 1000,
        )
    request_id = body.get("request_id")
    if request_id is None:
        # Some paths finish instantly; treat finished-without-id as soft skip.
        state = body.get("state")
        if state in ("finished", "completed"):
            return SurfaceProbeResult(
                name,
                True,
                "request finished before cancel handle available",
                hard=False,
                skipped=True,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        return SurfaceProbeResult(
            name,
            False,
            f"no request_id in {body!r}",
            elapsed_ms=(time.monotonic() - start) * 1000,
        )

    cancel_url = f"{base}/v1/requests/{request_id}/cancel"
    c_status, c_body = _post_json(cancel_url, {}, timeout=timeout)
    elapsed = (time.monotonic() - start) * 1000
    if c_status != 200 or not isinstance(c_body, dict):
        return SurfaceProbeResult(
            name,
            False,
            f"cancel HTTP {c_status}: {str(c_body)[:200]}",
            elapsed_ms=elapsed,
        )
    state = str(c_body.get("state", ""))
    # Accept cancelled or already-finished races.
    ok = state in ("cancelled", "finished", "completed") or bool(
        c_body.get("cancel_requested")
    )
    return SurfaceProbeResult(
        name,
        ok,
        f"state={state} cancel_requested={c_body.get('cancel_requested')}",
        elapsed_ms=elapsed,
    )


def probe_tools_schema(
    base_url: str,
    model: str,
    *,
    timeout: float = 90.0,
) -> SurfaceProbeResult:
    """Send a tools-bearing chat request; require non-5xx and non-null body path."""
    name = "tools_schema"
    start = time.monotonic()
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    status, body = _post_json(
        url,
        chat_completion_payload(
            model,
            "What is the weather in Paris? Use a tool if available.",
            max_tokens=64,
            tools=tools,
        ),
        timeout=timeout,
    )
    elapsed = (time.monotonic() - start) * 1000
    if status >= 500:
        return SurfaceProbeResult(
            name, False, f"server error HTTP {status}: {str(body)[:200]}", elapsed_ms=elapsed
        )
    if status in (400, 422):
        # Unsupported model/tool combo is a soft skip (delegated / no tool DSL).
        return SurfaceProbeResult(
            name,
            True,
            f"tools rejected with HTTP {status} (soft skip)",
            hard=False,
            skipped=True,
            elapsed_ms=elapsed,
        )
    if status != 200:
        return SurfaceProbeResult(
            name, False, f"unexpected HTTP {status}: {str(body)[:200]}", elapsed_ms=elapsed
        )
    if not isinstance(body, dict) or not body.get("choices"):
        return SurfaceProbeResult(
            name, False, f"missing choices: {str(body)[:200]}", elapsed_ms=elapsed
        )
    # Content or tool_calls both OK — engine must not return empty panic shape.
    msg = (body.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content")
    tool_calls = msg.get("tool_calls")
    if content is None and not tool_calls:
        return SurfaceProbeResult(
            name, False, "message has neither content nor tool_calls", elapsed_ms=elapsed
        )
    return SurfaceProbeResult(
        name,
        True,
        f"content_null={content is None} tool_calls={bool(tool_calls)}",
        elapsed_ms=elapsed,
    )


def probe_multimodal_image(
    base_url: str,
    model: str,
    *,
    timeout: float = 120.0,
    require_image: Optional[bool] = None,
) -> SurfaceProbeResult:
    """Send a tiny inline image.

    Soft-skip on 4xx only when the model does **not** advertise image input
    (or ``require_image`` is explicitly False). Advertised vision that rejects
    or returns empty content is a hard failure.
    """
    name = "multimodal_image"
    start = time.monotonic()
    if require_image is None:
        card = fetch_model_card(base_url, model, timeout=min(timeout, 30.0))
        require_image = model_advertises_image(card)
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    content = [
        {"type": "text", "text": "Describe this image in three words."},
        {"type": "image_url", "image_url": {"url": tiny_png_data_url()}},
    ]
    status, body = _post_json(
        url,
        chat_completion_payload(model, content, max_tokens=32, temperature=0.0),
        timeout=timeout,
    )
    elapsed = (time.monotonic() - start) * 1000
    if status in (400, 415, 422):
        if require_image:
            return SurfaceProbeResult(
                name,
                False,
                (
                    f"model advertises image but multimodal rejected "
                    f"HTTP {status}: {str(body)[:160]}"
                ),
                elapsed_ms=elapsed,
            )
        return SurfaceProbeResult(
            name,
            True,
            f"multimodal rejected HTTP {status} (soft skip; no image capability)",
            hard=False,
            skipped=True,
            elapsed_ms=elapsed,
        )
    if status >= 500 or status == 0:
        return SurfaceProbeResult(
            name,
            False,
            f"server/connection error HTTP {status}: {str(body)[:200]}",
            elapsed_ms=elapsed,
        )
    if status != 200:
        return SurfaceProbeResult(
            name, False, f"unexpected HTTP {status}: {str(body)[:200]}", elapsed_ms=elapsed
        )
    text = extract_chat_content(body)
    if text is None or not str(text).strip():
        return SurfaceProbeResult(
            name, False, "empty multimodal content", elapsed_ms=elapsed
        )
    stripped = str(text).strip()
    if stripped == "thought" or stripped.startswith("thought\n"):
        return SurfaceProbeResult(
            name,
            False,
            f"thinking-channel leak in multimodal content: {stripped[:80]!r}",
            elapsed_ms=elapsed,
        )
    return SurfaceProbeResult(
        name,
        True,
        f"image chat returned {len(text)} chars (require_image={require_image})",
        elapsed_ms=elapsed,
    )


def probe_remote_media_rejected(
    base_url: str,
    model: str,
    *,
    timeout: float = 60.0,
) -> SurfaceProbeResult:
    """Product policy: remote media URLs must fail closed (4xx, not 5xx/200)."""
    name = "remote_media_rejected"
    start = time.monotonic()
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    content = [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/remote-image-must-reject.png"},
        },
    ]
    status, body = _post_json(
        url,
        chat_completion_payload(model, content, max_tokens=16, temperature=0.0),
        timeout=timeout,
    )
    elapsed = (time.monotonic() - start) * 1000
    if status >= 500 or status == 0:
        return SurfaceProbeResult(
            name,
            False,
            f"server/connection error HTTP {status}: {str(body)[:160]}",
            elapsed_ms=elapsed,
        )
    if status == 200:
        return SurfaceProbeResult(
            name,
            False,
            "remote image URL was accepted (expected fail-closed 4xx)",
            elapsed_ms=elapsed,
        )
    if 400 <= status < 500:
        return SurfaceProbeResult(
            name,
            True,
            f"remote media rejected HTTP {status}",
            elapsed_ms=elapsed,
        )
    return SurfaceProbeResult(
        name,
        False,
        f"unexpected HTTP {status}: {str(body)[:160]}",
        elapsed_ms=elapsed,
    )


def probe_video_rejected(
    base_url: str,
    model: str,
    *,
    timeout: float = 60.0,
) -> SurfaceProbeResult:
    """Product policy: public routes reject video (unsupported_modality)."""
    name = "video_rejected"
    start = time.monotonic()
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    # Minimal 1x1 GIF as data URL — still a video_url modality at the API.
    gif = (
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff"
        b"\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00"
        b"\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    )
    content = [
        {"type": "text", "text": "How many frames?"},
        {
            "type": "video_url",
            "video_url": {
                "url": "data:image/gif;base64," + base64.b64encode(gif).decode("ascii")
            },
        },
    ]
    status, body = _post_json(
        url,
        chat_completion_payload(model, content, max_tokens=16, temperature=0.0),
        timeout=timeout,
    )
    elapsed = (time.monotonic() - start) * 1000
    if status >= 500 or status == 0:
        return SurfaceProbeResult(
            name,
            False,
            f"server/connection error HTTP {status}: {str(body)[:160]}",
            elapsed_ms=elapsed,
        )
    if status == 200:
        return SurfaceProbeResult(
            name,
            False,
            "video_url accepted on public chat route (expected reject)",
            elapsed_ms=elapsed,
        )
    if 400 <= status < 500:
        return SurfaceProbeResult(
            name,
            True,
            f"video rejected HTTP {status}",
            elapsed_ms=elapsed,
        )
    return SurfaceProbeResult(
        name,
        False,
        f"unexpected HTTP {status}: {str(body)[:160]}",
        elapsed_ms=elapsed,
    )


DEFAULT_PROBES: list[Callable[..., SurfaceProbeResult]] = [
    probe_concurrent_chat,
    probe_stream_and_nonstream,
    probe_cancel_request,
    probe_tools_schema,
    probe_multimodal_image,
    probe_remote_media_rejected,
    probe_video_rejected,
]


def run_surface_probes(
    base_url: str,
    model: str,
    *,
    timeout: float = 90.0,
    include_multimodal: bool = True,
    include_tools: bool = True,
    include_cancel: bool = True,
    include_media_policy: bool = True,
    concurrency_workers: int = 3,
    require_stream_parity: bool = True,
) -> SurfaceReport:
    report = SurfaceReport(base_url=base_url, model=model)
    report.results.append(
        probe_concurrent_chat(
            base_url, model, workers=concurrency_workers, timeout=timeout
        )
    )
    report.results.append(
        probe_stream_and_nonstream(
            base_url,
            model,
            timeout=timeout,
            require_parity=require_stream_parity,
        )
    )
    if include_cancel:
        report.results.append(probe_cancel_request(base_url, model, timeout=timeout))
    if include_tools:
        report.results.append(probe_tools_schema(base_url, model, timeout=timeout))
    if include_media_policy:
        report.results.append(
            probe_remote_media_rejected(base_url, model, timeout=timeout)
        )
        report.results.append(probe_video_rejected(base_url, model, timeout=timeout))
    if include_multimodal:
        report.results.append(probe_multimodal_image(base_url, model, timeout=timeout))
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="AX Engine product-surface probes")
    parser.add_argument("--base-url", default="http://127.0.0.1:31418")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--skip-multimodal", action="store_true")
    parser.add_argument("--skip-tools", action="store_true")
    parser.add_argument("--skip-cancel", action="store_true")
    parser.add_argument("--skip-media-policy", action="store_true")
    parser.add_argument(
        "--no-stream-parity",
        action="store_true",
        help="Only require SSE non-empty (skip content parity)",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)

    report = run_surface_probes(
        args.base_url,
        args.model,
        timeout=args.timeout,
        include_multimodal=not args.skip_multimodal,
        include_tools=not args.skip_tools,
        include_cancel=not args.skip_cancel,
        include_media_policy=not args.skip_media_policy,
        concurrency_workers=args.workers,
        require_stream_parity=not args.no_stream_parity,
    )
    for r in report.results:
        flag = "SKIP" if r.skipped else ("PASS" if r.passed else "FAIL")
        print(f"  [{flag}] {r.name}: {r.detail} ({r.elapsed_ms:.0f}ms)")
    print(report.summary_line)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.as_dict(), indent=2))
        print(f"JSON: {path}")
    return 0 if report.hard_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
