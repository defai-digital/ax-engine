#!/usr/bin/env python3
"""Probe prefix-cache reuse on a hybrid (linear-attention) model server.

Starts `ax-engine-server` on the repo-owned MLX path and sends deterministic
token-id requests (`/v1/generate`, greedy, one output token by default) shaped like
the traffic that matters for agents:

* `cold`     shared prefix P plus suffix S1 (nothing cached yet)
* `sibling`  P plus a different suffix S2 (new conversation, same system prompt)
* `followup` P + S1 + a generated-looking middle + a new tail (next chat turn)
* `repeat`   exactly the `sibling` request again

For each request it records wall time (TTFT proxy at one output token) and the
prefix-cache route counters, so a run answers: how much of the shared prefix is
re-prefilled, and how much does the cache capacity default matter?

Deterministic: fixed token ids, greedy sampling, no randomness. Requires a
local pack and Apple Silicon; it is a measurement tool, not a CI gate.

    python scripts/probe_hybrid_prefix_reuse.py \\
        --server target/release/ax-engine-server --model-dir /path/to/pack \\
        --model-id qwen3.8-27b --prefix-tokens 6144 --output out.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ax.hybrid_prefix_reuse_probe.v1"
COUNTER_KEYS = (
    "ax_mlx_prefix_cache_hits",
    "ax_mlx_prefix_cache_misses",
    "ax_mlx_prefix_cache_blocked",
    "ax_mlx_prefix_cache_blocked_policy_disabled",
    "ax_mlx_prefix_cache_blocked_unsupported_layout",
    "ax_mlx_prefix_cache_blocked_trim_failure",
    "ax_mlx_prefix_cache_warmup_tokens",
    "ax_mlx_prefix_cache_reused_tokens",
    "ax_mlx_prefix_cache_stores",
    "ax_mlx_prefix_cache_evictions",
    "ax_mlx_prefix_cache_entries",
    "ax_mlx_prefix_cache_bytes_kib",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def token_run(length: int, *, seed: int, start: int, span: int) -> list[int]:
    """Deterministic pseudo-text token ids in [start, start + span)."""
    state = seed * 2654435761 % (1 << 32)
    out: list[int] = []
    for _ in range(length):
        state = (state * 1664525 + 1013904223) % (1 << 32)
        out.append(start + (state >> 8) % span)
    return out


def build_requests(
    prefix_tokens: int, suffix_tokens: int, middle_tokens: int
) -> list[tuple[str, list[int]]]:
    prefix = token_run(prefix_tokens, seed=1, start=1000, span=20000)
    s1 = token_run(suffix_tokens, seed=2, start=30000, span=20000)
    s2 = token_run(suffix_tokens, seed=3, start=30000, span=20000)
    middle = token_run(middle_tokens, seed=4, start=1000, span=20000)
    tail = token_run(suffix_tokens, seed=5, start=30000, span=20000)
    return [
        ("cold", prefix + s1),
        ("sibling", prefix + s2),
        ("followup", prefix + s1 + middle + tail),
        ("repeat", prefix + s2),
    ]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_ready(port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except OSError:
            time.sleep(1.0)
    raise SystemExit("server did not become ready")


def find_route(response: dict[str, Any]) -> dict[str, int]:
    """Locate crossover decisions wherever the response nests the route report."""
    stack: list[Any] = [response]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            decisions = node.get("crossover_decisions")
            if isinstance(decisions, dict):
                return {k: int(v) for k, v in decisions.items()}
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return {}


def find_output_tokens(response: dict[str, Any]) -> list[int]:
    """Generated token ids, compared across runs as a token-identity check."""
    tokens = response.get("output_tokens")
    return [int(t) for t in tokens] if isinstance(tokens, list) else []


def stop_server(server: subprocess.Popen[bytes]) -> None:
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    env = {**os.environ, "AX_MLX_NATIVE_CONFIRM": "1"}
    env.pop("AX_MLX_PREFIX_CACHE_MAX_BYTES", None)
    env.pop("AX_MLX_PREFIX_CACHE_MAX_ENTRIES", None)
    if args.cache_max_bytes is not None:
        env["AX_MLX_PREFIX_CACHE_MAX_BYTES"] = str(args.cache_max_bytes)
    cmd = [
        str(args.server),
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(args.model_dir),
        "--model-id",
        args.model_id,
        "--port",
        str(args.port),
        "--mlx-mtp-policy",
        "disabled",
        *args.server_arg,
    ]

    def start_server() -> subprocess.Popen[bytes]:
        server = subprocess.Popen(
            cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            wait_ready(args.port, args.ready_timeout)
        except BaseException:
            stop_server(server)
            raise
        return server

    rows: list[dict[str, Any]] = []
    url = f"http://127.0.0.1:{args.port}/v1/generate"
    server = None if args.fresh_server_per_request else start_server()
    try:
        for round_index in range(args.rounds):
            for name, tokens in build_requests(
                args.prefix_tokens, args.suffix_tokens, args.middle_tokens
            ):
                if round_index > 0 and name == "cold":
                    # Fresh prefix per round so rounds do not contaminate each other.
                    tokens = [t + 7 * round_index for t in tokens]
                if args.fresh_server_per_request:
                    # True cold reference: no prefix state of any kind survives.
                    server = start_server()
                payload = {
                    "model": args.model_id,
                    "input_tokens": tokens,
                    "max_output_tokens": args.max_output_tokens,
                    "sampling": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 0,
                        "seed": 1,
                        "ignore_eos": args.ignore_eos,
                    },
                }
                started = time.perf_counter()
                response = post_json(url, payload, args.request_timeout)
                wall_ms = (time.perf_counter() - started) * 1000.0
                if args.fresh_server_per_request and server is not None:
                    stop_server(server)
                    server = None
                route = find_route(response)
                rows.append(
                    {
                        "round": round_index,
                        "request": name,
                        "prompt_tokens": len(tokens),
                        "wall_ms": round(wall_ms, 1),
                        "output_tokens": find_output_tokens(response),
                        "route": {k: route[k] for k in COUNTER_KEYS if k in route},
                    }
                )
                print(
                    f"round={round_index} {name:9s} tokens={len(tokens):6d} "
                    f"wall_ms={wall_ms:9.1f} "
                    f"reused={route.get('ax_mlx_prefix_cache_reused_tokens', 0)} "
                    f"warmup={route.get('ax_mlx_prefix_cache_warmup_tokens', 0)} "
                    f"hits={route.get('ax_mlx_prefix_cache_hits', 0)}",
                    file=sys.stderr,
                )
    finally:
        if server is not None:
            stop_server(server)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "middle_tokens": args.middle_tokens,
        "max_output_tokens": args.max_output_tokens,
        "cache_max_bytes": args.cache_max_bytes,
        "fresh_server_per_request": args.fresh_server_per_request,
        "ignore_eos": args.ignore_eos,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--port", type=positive_int, default=31499)
    parser.add_argument("--prefix-tokens", type=positive_int, default=6144)
    parser.add_argument("--suffix-tokens", type=positive_int, default=64)
    parser.add_argument("--middle-tokens", type=positive_int, default=96)
    parser.add_argument("--rounds", type=positive_int, default=1)
    parser.add_argument(
        "--max-output-tokens",
        type=positive_int,
        default=1,
        help="Greedy tokens per request; >1 makes `output_tokens` a useful identity check "
        "between a run with prefix reuse and one without.",
    )
    parser.add_argument(
        "--cache-max-bytes",
        type=int,
        default=None,
        help="AX_MLX_PREFIX_CACHE_MAX_BYTES for the server; omit for the product default.",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra ax-engine-server argument (repeatable), e.g. --server-arg=--total-blocks "
        "--server-arg=4096.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Generate exactly --max-output-tokens even past EOS, so a model that "
        "ends synthetic prompts at once still yields a useful identity check.",
    )
    parser.add_argument(
        "--fresh-server-per-request",
        action="store_true",
        help="Start a new server for every request: the true cold reference for "
        "`output_tokens`, free of any prefix restore or recompute fallback.",
    )
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_probe(args)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
