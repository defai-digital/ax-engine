#!/usr/bin/env python3
"""Benchmark AX Engine runtime modes through the public /v1/generate API.

This is intentionally an end-to-end AX API latency harness. It does not claim
raw model/kernel apples-to-apples throughput. Use it to reproduce user-facing
mode comparisons such as mlx_preview default, mlx_preview with n-gram disabled,
and mlx_lm_delegated.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Mode:
    name: str
    base_url: str
    input_kind: str


def parse_mode(value: str) -> Mode:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--mode must be name,base_url,input_kind where input_kind is text or tokens"
        )
    name, base_url, input_kind = (part.strip() for part in parts)
    if input_kind not in {"text", "tokens"}:
        raise argparse.ArgumentTypeError("mode input_kind must be text or tokens")
    return Mode(name=name, base_url=base_url.rstrip("/"), input_kind=input_kind)


def parse_tokens(value: str | None, path: Path | None) -> list[int]:
    if value and path:
        raise SystemExit("use either --tokens or --tokens-file, not both")
    if path:
        raw = path.read_text().strip()
    else:
        raw = value or ""
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        tokens = json.loads(raw)
    else:
        tokens = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not all(isinstance(token, int) and token >= 0 for token in tokens):
        raise SystemExit("tokens must be non-negative integers")
    return tokens


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def request_payload(
    mode: Mode,
    model_id: str,
    prompt: str,
    tokens: list[int],
    max_output_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "max_output_tokens": max_output_tokens,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "seed": 0,
        },
    }
    if mode.input_kind == "text":
        payload["input_text"] = prompt
    else:
        if not tokens:
            raise SystemExit(f"mode {mode.name} uses token input but no --tokens were provided")
        payload["input_tokens"] = tokens
    return payload


def post_json(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any] | str]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = response.read().decode("utf-8")
            return response.status, json.loads(data)
    except urllib.error.HTTPError as error:
        data = error.read().decode("utf-8", errors="replace")
        try:
            return error.code, json.loads(data)
        except json.JSONDecodeError:
            return error.code, data


def run_mode(
    mode: Mode,
    model_id: str,
    prompt: str,
    tokens: list[int],
    max_output_tokens: int,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    payload = request_payload(mode, model_id, prompt, tokens, max_output_tokens)
    url = f"{mode.base_url}/v1/generate"
    observations: list[dict[str, Any]] = []

    for index in range(warmup + runs):
        started = time.perf_counter()
        status, response = post_json(url, payload)
        elapsed = time.perf_counter() - started
        observation = {
            "phase": "warmup" if index < warmup else "measured",
            "status": status,
            "wall_time_s": elapsed,
            "finish_reason": response.get("finish_reason") if isinstance(response, dict) else None,
            "output_token_count": (
                response.get("output_token_count")
                if isinstance(response, dict) and response.get("output_token_count") is not None
                else len(response.get("output_tokens", [])) if isinstance(response, dict) else None
            ),
            "error": response.get("error") if isinstance(response, dict) else response,
        }
        observations.append(observation)

    measured = [item for item in observations if item["phase"] == "measured"]
    successful_times = sorted(
        item["wall_time_s"] for item in measured if 200 <= item["status"] < 300
    )
    return {
        "name": mode.name,
        "base_url": mode.base_url,
        "input_kind": mode.input_kind,
        "summary": {
            "ok_runs": len(successful_times),
            "error_runs": len(measured) - len(successful_times),
            "median_wall_time_s": statistics.median(successful_times)
            if successful_times
            else None,
            "p25_wall_time_s": percentile(successful_times, 0.25)
            if successful_times
            else None,
            "p75_wall_time_s": percentile(successful_times, 0.75)
            if successful_times
            else None,
        },
        "observations": observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        action="append",
        type=parse_mode,
        required=True,
        help="name,base_url,input_kind. Example: preview_default,http://127.0.0.1:19091,tokens",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens")
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokens = parse_tokens(args.tokens, args.tokens_file)
    result = {
        "methodology": {
            "scope": "end_to_end_ax_generate_api_latency",
            "warmup": args.warmup,
            "runs": args.runs,
            "max_output_tokens": args.max_output_tokens,
            "notes": [
                "Start each AX server mode separately before running this harness.",
                "Use token input for repo-owned MLX preview and text input for mlx_lm_delegated.",
                "This is product-path latency evidence, not raw model throughput evidence.",
            ],
        },
        "modes": [
            run_mode(
                mode,
                args.model_id,
                args.prompt,
                tokens,
                args.max_output_tokens,
                args.warmup,
                args.runs,
            )
            for mode in args.mode
        ],
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
