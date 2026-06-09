#!/usr/bin/env python3
"""Measure Gemma 4 unified image-prefill TTFT through a running AX server."""
from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import statistics
import sys
import time
import urllib.parse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "python" / "ax_engine" / "gemma4_unified.py"
DEFAULT_PROMPT = (
    "<bos><|turn>user\n"
    "What color is this image?<|image|><turn|>\n"
    "<|turn>model\n<|channel>thought\n<channel|>"
)


@dataclass(frozen=True)
class PreparedImageRequest:
    input_tokens: list[int]
    multimodal_inputs: dict[str, Any]
    original_prompt_tokens: int
    expanded_prompt_tokens: int
    image_soft_tokens: int


def load_gemma4_helper():
    spec = importlib.util.spec_from_file_location("ax_engine_gemma4_unified_bench", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load Gemma4 helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def prepare_image_request(model_dir: Path, prompt: str, image_bytes: bytes) -> PreparedImageRequest:
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompt_tokens = tokenizer.encode(prompt).ids
    helper = load_gemma4_helper()
    request = helper.prepare_gemma4_unified_image_request(
        model_dir,
        prompt_tokens,
        [image_bytes],
    )
    image_soft_tokens = int(request.soft_token_counts[0])
    return PreparedImageRequest(
        input_tokens=[int(token) for token in request.input_tokens],
        multimodal_inputs=request.multimodal_inputs,
        original_prompt_tokens=len(prompt_tokens),
        expanded_prompt_tokens=len(request.input_tokens),
        image_soft_tokens=image_soft_tokens,
    )


def iter_sse_json_events(lines: Iterable[str]) -> Iterator[tuple[str, Any]]:
    event_name = ""
    data_parts: list[str] = []

    def flush() -> tuple[str, Any] | None:
        nonlocal event_name, data_parts
        name = event_name
        data = "\n".join(data_parts)
        event_name = ""
        data_parts = []
        if not data:
            return None
        return name, json.loads(data)

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if line == "":
            frame = flush()
            if frame is not None:
                yield frame
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            value = line[len("data:") :]
            data_parts.append(value[1:] if value.startswith(" ") else value)

    frame = flush()
    if frame is not None:
        yield frame


def is_prefill_step(step: dict[str, Any], *, seen_prefill: bool) -> bool:
    route = step.get("route") or {}
    labels = [
        str(route.get("execution_plan") or "").lower(),
        str(route.get("attention_route") or "").lower(),
    ]
    if any("prefill" in label for label in labels):
        return True
    if any("decode" in label for label in labels):
        return False
    return not seen_prefill or int(step.get("scheduled_tokens") or 0) > 1


def run_one(url: str, model: str, request: PreparedImageRequest, max_output_tokens: int) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url.rstrip("/"))
    conn = http.client.HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port or 80, timeout=300)
    payload = json.dumps(
        {
            "model": model,
            "input_tokens": request.input_tokens,
            "multimodal_inputs": request.multimodal_inputs,
            "max_output_tokens": max_output_tokens,
            "sampling": {"temperature": 0.0, "ignore_eos": True},
        }
    ).encode("utf-8")
    started = time.perf_counter()
    first_output_wall_s: float | None = None
    try:
        conn.request(
            "POST",
            "/v1/generate/stream",
            body=payload,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"HTTP {response.status}: {response.read(500).decode(errors='replace')}"
            )

        prefill_us = 0
        seen_prefill = False
        output_tokens = 0
        execution_plan = None
        decoded_lines = (line.decode("utf-8", errors="replace") for line in response)
        for event_name, obj in iter_sse_json_events(decoded_lines):
            if event_name == "step":
                step = obj.get("step") or {}
                current_output_len = obj.get("request", {}).get("output_len")
                if current_output_len:
                    output_tokens = int(current_output_len)
                    if first_output_wall_s is None:
                        first_output_wall_s = time.perf_counter() - started
                if is_prefill_step(step, seen_prefill=seen_prefill):
                    prefill_us += int(step.get("runner_time_us") or 0)
                    seen_prefill = True
                    route = step.get("route") or {}
                    execution_plan = route.get("execution_plan") or execution_plan
            elif event_name == "response":
                response_obj = obj.get("response") or {}
                tokens = response_obj.get("output_tokens")
                if isinstance(tokens, list):
                    output_tokens = len(tokens)
                    if tokens and first_output_wall_s is None:
                        first_output_wall_s = time.perf_counter() - started
                route = response_obj.get("route") or {}
                execution_plan = execution_plan or route.get("execution_plan")
    finally:
        conn.close()

    total_wall_ms = (time.perf_counter() - started) * 1000.0
    runner_prefill_ms = prefill_us / 1000.0
    return {
        "runner_prefill_ttft_ms": runner_prefill_ms,
        "client_wall_ttft_ms": (
            first_output_wall_s * 1000.0 if first_output_wall_s is not None else None
        ),
        "client_wall_total_ms": total_wall_ms,
        "prefill_tok_s": (
            request.expanded_prompt_tokens / (runner_prefill_ms / 1000.0)
            if runner_prefill_ms > 0
            else None
        ),
        "output_tokens": output_tokens,
        "execution_plan": execution_plan,
    }


def summarize(values: list[float | None]) -> dict[str, float | None]:
    present = [value for value in values if value is not None]
    if not present:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": sum(present) / len(present),
        "median": statistics.median(present),
        "min": min(present),
        "max": max(present),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="gemma-4-12B-it")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / ".internal/models/gemma-4-12B-it-4bit",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image = solid_png(64, 64, (220, 20, 20))
    request = prepare_image_request(args.model_dir, args.prompt, image)

    for _ in range(args.warmup):
        run_one(args.url, args.model, request, args.max_output_tokens)

    runs = [
        run_one(args.url, args.model, request, args.max_output_tokens)
        for _ in range(args.repetitions)
    ]
    artifact = {
        "schema": "ax.gemma4_image_prefill_ttft.v1",
        "model": args.model,
        "server_url": args.url,
        "model_dir": str(args.model_dir),
        "prompt": {
            "original_tokens": request.original_prompt_tokens,
            "expanded_tokens": request.expanded_prompt_tokens,
            "image_soft_tokens": request.image_soft_tokens,
        },
        "max_output_tokens": args.max_output_tokens,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "runs": runs,
        "summary": {
            "runner_prefill_ttft_ms": summarize(
                [run["runner_prefill_ttft_ms"] for run in runs]
            ),
            "client_wall_ttft_ms": summarize([run["client_wall_ttft_ms"] for run in runs]),
            "prefill_tok_s": summarize([run["prefill_tok_s"] for run in runs]),
        },
    }

    text = json.dumps(artifact, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
