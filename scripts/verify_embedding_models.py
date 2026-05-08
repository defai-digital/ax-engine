#!/usr/bin/env python3
"""Verify ax-engine embedding output matches mlx-lm reference.

Exit codes mirror ax-engine-bench convention:
  0 = all tests passed
  2 = contract failure (server didn't respond / bad format)
  3 = correctness failure (cosine similarity below threshold)

Usage:
    python scripts/verify_embedding_models.py \
        --model-dir .internal/models/qwen3-embedding-0.6b-8bit \
        --hf-model Qwen/Qwen3-Embedding-0.6B \
        [--port 8082]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "AX Engine achieves 318 tokens per second on Apple M4",
    "Apple Silicon on-chip memory bandwidth enables low-latency inference",
    "What is the capital of France?",
    "Hello world",
]

COSINE_THRESHOLD = 0.9990  # bf16 rounding budget


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def http_post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()}") from e


# ---------------------------------------------------------------------------
# Reference embeddings via mlx-lm
# ---------------------------------------------------------------------------

def compute_reference_embeddings(
    model_dir: Path,
    sentences: list[str],
) -> list[list[float]]:
    """Compute reference embeddings by loading the mlx-community model via mlx-lm.

    Uses model.model() (the transformer body) directly to get hidden states,
    then applies last-token pooling and L2 normalization.  This is the same
    computation path that Qwen3-Embedding documentation specifies.
    """
    print(f"  Loading reference model from {model_dir}")
    from mlx_lm import load  # type: ignore[import]
    import mlx.core as mx

    model, tokenizer = load(str(model_dir))

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    embeddings: list[list[float]] = []
    for sentence in sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=False) + [eos_id]
        x = mx.array([tokens])  # [1, seq]
        hidden = model.model(x)  # [1, seq, hidden] via transformer body (no lm_head)
        last = hidden[0, -1, :].astype(mx.float32)
        norm = mx.sqrt(mx.sum(last * last))
        normalized = last / (norm + 1e-12)
        mx.eval(normalized)
        embeddings.append(normalized.tolist())

    return embeddings


# ---------------------------------------------------------------------------
# ax-engine embeddings via HTTP
# ---------------------------------------------------------------------------

def compute_ax_embeddings(
    server_url: str,
    tokenizer_path: str,
    sentences: list[str],
) -> list[list[float]]:
    """Tokenize sentences and call /v1/embeddings on the running server."""
    from transformers import AutoTokenizer

    print(f"  Loading tokenizer: {tokenizer_path}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    eos_id = tok.eos_token_id

    embeddings: list[list[float]] = []
    for sentence in sentences:
        tokens = tok.encode(sentence, add_special_tokens=False)
        tokens = tokens + [eos_id]  # append EOS for Qwen3-Embedding
        resp = http_post(
            f"{server_url}/v1/embeddings",
            {"input": tokens, "pooling": "last", "normalize": True},
        )
        vec = resp["data"][0]["embedding"]
        embeddings.append(vec)

    return embeddings


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def start_server(
    model_dir: Path,
    port: int,
) -> subprocess.Popen:
    server_bin = REPO_ROOT / "target" / "release" / "ax-engine-server"
    if not server_bin.exists():
        server_bin = REPO_ROOT / "target" / "debug" / "ax-engine-server"
    if not server_bin.exists():
        raise FileNotFoundError(
            "ax-engine-server binary not found. Run: cargo build -p ax-engine-server"
        )

    cmd = [
        str(server_bin),
        "--model-id", "qwen3_dense",
        "--mlx",
        "--mlx-model-artifacts-dir", str(model_dir),
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    print(f"  Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Verify embedding model correctness")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to model directory with model-manifest.json",
    )
    parser.add_argument(
        "--hf-model",
        help="Deprecated — ignored. Reference now uses --model-dir directly.",
    )
    parser.add_argument("--port", type=int, default=8082, help="Server port (default 8082)")
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Assume server is already running on --port",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not (model_dir / "model-manifest.json").exists():
        print(f"ERROR: model-manifest.json not found in {model_dir}", file=sys.stderr)
        return 1

    server_url = f"http://127.0.0.1:{args.port}"
    server_proc: subprocess.Popen | None = None

    try:
        # 1. Start ax-engine-server
        if not args.skip_server:
            print("\n[1/3] Starting ax-engine-server")
            try:
                server_proc = start_server(model_dir, args.port)
            except FileNotFoundError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 2

            print(f"  Waiting for server on port {args.port}…")
            if not wait_for_port("127.0.0.1", args.port, timeout=120.0):
                print("ERROR: Server did not start within 120s", file=sys.stderr)
                if server_proc:
                    out, _ = server_proc.communicate(timeout=5)
                    print(out, file=sys.stderr)
                return 2
            print("  Server ready.")

        # 2. Compute reference embeddings
        print("\n[2/3] Computing reference embeddings via mlx-lm")
        try:
            ref_embeddings = compute_reference_embeddings(model_dir, SENTENCES)
        except Exception as e:
            print(f"ERROR computing reference: {e}", file=sys.stderr)
            return 2

        # 3. Compute ax-engine embeddings
        print("\n[3/3] Computing ax-engine embeddings")
        try:
            ax_embeddings = compute_ax_embeddings(server_url, str(model_dir), SENTENCES)
        except Exception as e:
            print(f"ERROR computing ax-engine embeddings: {e}", file=sys.stderr)
            return 2

        # 4. Compare
        print("\nResults:")
        print(f"{'Sentence':<50} {'cosine_sim':>12} {'pass':>6}")
        print("-" * 72)
        all_pass = True
        for sentence, ref_vec, ax_vec in zip(SENTENCES, ref_embeddings, ax_embeddings):
            sim = cosine_sim(ref_vec, ax_vec)
            passed = sim >= COSINE_THRESHOLD
            if not passed:
                all_pass = False
            label = "✓" if passed else "✗ FAIL"
            short = sentence[:48]
            print(f"{short:<50} {sim:>12.6f} {label:>6}")

        print()
        if all_pass:
            print(f"PASS — all cosine similarities ≥ {COSINE_THRESHOLD}")
            return 0
        else:
            print(f"FAIL — one or more cosine similarities < {COSINE_THRESHOLD}")
            return 3

    finally:
        if server_proc is not None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    sys.exit(main())
