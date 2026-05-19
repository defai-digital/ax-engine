#!/usr/bin/env python3
"""Drive a long-running p=2048 prefill load against ax-engine-server so an
external CPU profiler (Apple sample(1) or xctrace) can attach to the running
server and capture stable hot-path samples.

Usage:
  python3 scripts/cpu_trace_prefill.py \
    --token-ids-path benchmarks/.../prompt-2048-gen-16-a7760438e4a5.json \
    --port 8731 \
    --requests 20

The harness starts the server, waits for /health, then issues `--requests`
synchronous SSE prefill+gen=1 requests in series. Server PID is printed to
stderr so the operator can launch `sample <pid> <duration>` against it.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


def start_server(binary: str, model_dir: str, port: int) -> subprocess.Popen:
    cmd = [
        binary,
        "--mlx",
        "--mlx-model-artifacts-dir",
        model_dir,
        "--port",
        str(port),
        "--prefill-chunk",
        "2048",
        "--max-batch-tokens",
        "2048",
        "--disable-ngram-acceleration",
    ]
    env = {**os.environ, "AX_MLX_NATIVE_CONFIRM": "1"}
    # Inherit stderr to the parent so diagnostic eprintln from the server is visible.
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=None)


def wait_for_health(port: int, proc: subprocess.Popen, timeout: float = 60.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server died: rc={proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.1)
    raise TimeoutError("server /health did not respond")


def post_one(port: int, token_ids: list[int], max_tokens: int) -> float:
    """Issue a single SSE generate request and return elapsed seconds."""
    body = json.dumps(
        {
            "input_tokens": token_ids,
            "max_output_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/generate/stream",
        data=body,
        headers={"content-type": "application/json", "accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120.0) as r:
        for _ in r:
            pass
    return time.monotonic() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", default="target/release/ax-engine-server")
    p.add_argument(
        "--model-dir",
        default=(
            "/Users/akiralam/.cache/huggingface/hub/"
            "models--mlx-community--gemma-4-e2b-it-4bit/snapshots/"
            "99d9a53ff828d365a8ecae538e45f80a08d612cd"
        ),
    )
    p.add_argument("--port", type=int, default=8731)
    p.add_argument("--token-ids-path", required=True)
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--max-output-tokens", type=int, default=1)
    p.add_argument(
        "--ready-marker", default="/tmp/cpu_trace_ready",
        help="touch this file after server is up; allows external sample driver to start",
    )
    args = p.parse_args()

    prompt_doc = json.load(open(args.token_ids_path))
    token_ids = prompt_doc["token_ids"]
    print(f"prompt tokens: {len(token_ids)}", file=sys.stderr)

    proc = start_server(args.binary, args.model_dir, args.port)
    print(f"SERVER_PID={proc.pid}", file=sys.stderr)
    try:
        wait_for_health(args.port, proc)
        # Warmup: 2 requests
        for i in range(2):
            dt = post_one(args.port, token_ids, args.max_output_tokens)
            print(f"warmup {i}: {dt*1000:.1f}ms", file=sys.stderr)

        # Touch ready marker so external profiler can start now.
        try:
            os.makedirs(os.path.dirname(args.ready_marker) or ".", exist_ok=True)
            open(args.ready_marker, "w").close()
        except Exception as e:
            print(f"warn: ready marker write failed: {e}", file=sys.stderr)

        # Measure loop.
        t0 = time.monotonic()
        for i in range(args.requests):
            dt = post_one(args.port, token_ids, args.max_output_tokens)
            print(f"req {i}: {dt*1000:.1f}ms", file=sys.stderr)
        total = time.monotonic() - t0
        print(f"DONE: {args.requests} reqs in {total:.3f}s, mean {total/args.requests*1000:.1f}ms/req", file=sys.stderr)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
