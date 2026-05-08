#!/usr/bin/env python3
"""Benchmark embedding throughput: ax-engine vs mlx-lm vs mlx-swift-lm.

Measures ms/sentence and tok/s for each backend on the same set of sentences
using the same pre-tokenized input (EOS-appended for Qwen3-Embedding).

Exit codes:
  0  all backends ran
  1  usage error
  2  a required backend failed to run (non-optional backends only)

Usage:
    # Build ax-engine-server first:
    cargo build -p ax-engine-server --release

    # Build the Swift embed bench (optional):
    cd scripts/mlx-swift-embed-bench
    swift build -c release
    cd ../..

    python scripts/bench_embedding_models.py \\
        --model-dir .internal/models/qwen3-embedding-0.6b-8bit \\
        --model-label qwen3-embedding-0.6b-8bit \\
        [--trials 5] \\
        [--cooldown 2] \\
        [--port 8083] \\
        [--swift-bench-binary .build/release/mlx-swift-embed-bench] \\
        [--output-dir benchmarks/results/embedding]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "AX Engine achieves 318 tokens per second on Apple M4",
    "Apple Silicon on-chip memory bandwidth enables low-latency inference",
    "What is the capital of France?",
    "Hello world",
    "Machine learning models require significant computational resources",
    "Natural language processing enables computers to understand human text",
    "The transformer architecture revolutionized deep learning",
    "Embeddings capture semantic relationships between words and phrases",
    "On-device inference preserves user privacy and reduces latency",
]

AXENGINE_PORT_DEFAULT = 8083


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def wait_for_port(host: str, port: int, timeout: float = 120.0) -> bool:
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
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def stddev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_sentences(model_dir: Path, sentences: list[str]) -> list[list[int]]:
    """Tokenize sentences using HF AutoTokenizer, appending EOS (Qwen3-Embedding style)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    eos_id = tok.eos_token_id
    result = []
    for s in sentences:
        ids = tok.encode(s, add_special_tokens=False)
        if eos_id is not None:
            ids = ids + [eos_id]
        result.append(ids)
    return result


# ---------------------------------------------------------------------------
# Backend: mlx-lm (Python)
# ---------------------------------------------------------------------------

def bench_mlx_lm(
    model_dir: Path,
    token_ids_list: list[list[int]],
    trials: int,
    cooldown: float,
) -> dict[str, Any]:
    print("\n[mlx-lm] Loading model…", file=sys.stderr)
    from mlx_lm import load
    import mlx.core as mx

    model, tokenizer = load(str(model_dir))

    total_tokens = sum(len(ids) for ids in token_ids_list)
    n_sentences = len(token_ids_list)

    # Warmup
    print("  [mlx-lm] warmup…", file=sys.stderr)
    for ids in token_ids_list:
        x = mx.array([ids])
        hidden = model.model(x)
        last = hidden[0, -1, :].astype(mx.float32)
        norm = mx.sqrt(mx.sum(last * last))
        normalized = last / (norm + 1e-12)
        mx.eval(normalized)

    trial_results = []
    for i in range(1, trials + 1):
        if cooldown > 0:
            time.sleep(cooldown)
        t0 = time.perf_counter()
        for ids in token_ids_list:
            x = mx.array([ids])
            hidden = model.model(x)
            last = hidden[0, -1, :].astype(mx.float32)
            norm = mx.sqrt(mx.sum(last * last))
            normalized = last / (norm + 1e-12)
            mx.eval(normalized)
        elapsed = time.perf_counter() - t0
        ms_per_sentence = elapsed * 1000.0 / n_sentences
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
        trial_results.append({
            "ms_per_sentence": ms_per_sentence,
            "tokens_per_sec": tokens_per_sec,
        })
        print(
            f"    trial {i}: {ms_per_sentence:.1f}ms/sentence "
            f"{tokens_per_sec:.1f} tok/s",
            file=sys.stderr,
        )

    ms_vals = [t["ms_per_sentence"] for t in trial_results]
    tps_vals = [t["tokens_per_sec"] for t in trial_results]
    return {
        "engine": "mlx_lm",
        "mean_ms_per_sentence": mean(ms_vals),
        "median_ms_per_sentence": median(ms_vals),
        "stddev_ms_per_sentence": stddev(ms_vals),
        "mean_tokens_per_sec": mean(tps_vals),
        "median_tokens_per_sec": median(tps_vals),
        "trials": trial_results,
    }


# ---------------------------------------------------------------------------
# Backend: ax-engine (HTTP)
# ---------------------------------------------------------------------------

def start_axengine(model_dir: Path, port: int) -> subprocess.Popen:
    server_bin = REPO_ROOT / "target" / "release" / "ax-engine-server"
    if not server_bin.exists():
        server_bin = REPO_ROOT / "target" / "debug" / "ax-engine-server"
    if not server_bin.exists():
        raise FileNotFoundError(
            "ax-engine-server not found. Run: cargo build -p ax-engine-server --release"
        )
    cmd = [
        str(server_bin),
        "--model-id", "qwen3_dense",
        "--support-tier", "mlx-preview",
        "--mlx",
        "--mlx-model-artifacts-dir", str(model_dir),
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    print(f"  [ax-engine] starting: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def bench_ax_engine(
    model_dir: Path,
    token_ids_list: list[list[int]],
    trials: int,
    cooldown: float,
    port: int,
    skip_server: bool = False,
) -> dict[str, Any]:
    server_proc: subprocess.Popen | None = None
    server_url = f"http://127.0.0.1:{port}"
    total_tokens = sum(len(ids) for ids in token_ids_list)
    n_sentences = len(token_ids_list)

    try:
        if not skip_server:
            print("\n[ax-engine] Starting server…", file=sys.stderr)
            server_proc = start_axengine(model_dir, port)
            print(f"  [ax-engine] waiting for port {port}…", file=sys.stderr)
            if not wait_for_port("127.0.0.1", port, timeout=120.0):
                raise RuntimeError("ax-engine-server did not start within 120s")
            print("  [ax-engine] server ready.", file=sys.stderr)
        else:
            print(f"\n[ax-engine] using running server on port {port}", file=sys.stderr)

        # Warmup
        print("  [ax-engine] warmup…", file=sys.stderr)
        for ids in token_ids_list:
            http_post(f"{server_url}/v1/embeddings",
                      {"input": ids, "pooling": "last", "normalize": True})

        trial_results = []
        for i in range(1, trials + 1):
            if cooldown > 0:
                time.sleep(cooldown)
            t0 = time.perf_counter()
            for ids in token_ids_list:
                http_post(f"{server_url}/v1/embeddings",
                          {"input": ids, "pooling": "last", "normalize": True})
            elapsed = time.perf_counter() - t0
            ms_per_sentence = elapsed * 1000.0 / n_sentences
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
            trial_results.append({
                "ms_per_sentence": ms_per_sentence,
                "tokens_per_sec": tokens_per_sec,
            })
            print(
                f"    trial {i}: {ms_per_sentence:.1f}ms/sentence "
                f"{tokens_per_sec:.1f} tok/s",
                file=sys.stderr,
            )

        ms_vals = [t["ms_per_sentence"] for t in trial_results]
        tps_vals = [t["tokens_per_sec"] for t in trial_results]
        return {
            "engine": "ax_engine",
            "mean_ms_per_sentence": mean(ms_vals),
            "median_ms_per_sentence": median(ms_vals),
            "stddev_ms_per_sentence": stddev(ms_vals),
            "mean_tokens_per_sec": mean(tps_vals),
            "median_tokens_per_sec": median(tps_vals),
            "trials": trial_results,
        }

    finally:
        if server_proc is not None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


# ---------------------------------------------------------------------------
# Backend: mlx-swift-lm (subprocess)
# ---------------------------------------------------------------------------

def bench_mlx_swift(
    binary: Path,
    model_dir: Path,
    token_ids_list: list[list[int]],
    sentences: list[str],
    trials: int,
    delay: float,
) -> dict[str, Any]:
    if not binary.exists():
        raise FileNotFoundError(
            f"mlx-swift-embed-bench binary not found at {binary}. "
            "Build with: cd scripts/mlx-swift-embed-bench && swift build -c release"
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        payload = [
            {"label": s, "token_ids": ids}
            for s, ids in zip(sentences, token_ids_list)
        ]
        json.dump(payload, f)
        token_ids_path = f.name

    try:
        cmd = [
            str(binary),
            "--model", str(model_dir),
            "--token-ids-path", token_ids_path,
            "--trials", str(trials),
            "--delay", str(delay),
        ]
        print(f"\n[mlx-swift] {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(cmd, capture_output=False, text=False,
                                stdout=subprocess.PIPE, stderr=None)
        if result.returncode != 0:
            raise RuntimeError(
                f"mlx-swift-embed-bench exited {result.returncode}"
            )
        output = json.loads(result.stdout)
    finally:
        os.unlink(token_ids_path)

    trial_results = output.get("trials", [])
    ms_vals = [t.get("ms_per_sentence", 0) for t in trial_results]
    tps_vals = [t.get("tokens_per_sec", 0) for t in trial_results]
    return {
        "engine": "mlx_swift",
        "mean_ms_per_sentence": mean(ms_vals),
        "median_ms_per_sentence": median(ms_vals),
        "stddev_ms_per_sentence": stddev(ms_vals),
        "mean_tokens_per_sec": mean(tps_vals),
        "median_tokens_per_sec": median(tps_vals),
        "trials": trial_results,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict[str, dict]) -> None:
    engines = ["mlx_lm", "ax_engine", "mlx_swift"]
    print(f"\n{'Engine':<20} {'mean ms/sentence':>18} {'median ms/sentence':>20} {'mean tok/s':>12}")
    print("-" * 74)
    for engine in engines:
        r = results.get(engine)
        if r is None:
            print(f"{engine:<20} {'(skipped)':>18}")
            continue
        print(
            f"{engine:<20} "
            f"{r['mean_ms_per_sentence']:>18.1f} "
            f"{r['median_ms_per_sentence']:>20.1f} "
            f"{r['mean_tokens_per_sec']:>12.1f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark embedding models")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Path to model directory with model-manifest.json")
    parser.add_argument("--model-label", default=None,
                        help="Short label for output filenames (default: model-dir basename)")
    parser.add_argument("--trials", type=int, default=5, help="Timed trials per backend")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Seconds between trials")
    parser.add_argument("--port", type=int, default=AXENGINE_PORT_DEFAULT,
                        help=f"ax-engine server port (default {AXENGINE_PORT_DEFAULT})")
    parser.add_argument("--skip-ax-server", action="store_true",
                        help="Assume ax-engine-server already running on --port")
    parser.add_argument("--skip-swift", action="store_true",
                        help="Skip mlx-swift benchmark")
    parser.add_argument("--swift-bench-binary", type=Path,
                        default=REPO_ROOT / "scripts/mlx-swift-bench/.build/arm64-apple-macosx/release/mlx-swift-embed-bench",
                        help="Path to compiled mlx-swift-embed-bench binary")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "benchmarks" / "results" / "embedding",
                        help="Directory for result JSON files")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not (model_dir / "model-manifest.json").exists():
        print(f"ERROR: model-manifest.json not found in {model_dir}", file=sys.stderr)
        return 1

    label = args.model_label or model_dir.name
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    run_dir = args.output_dir / f"{timestamp}-{label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tokenizing {len(SENTENCES)} sentences…", file=sys.stderr)
    token_ids_list = tokenize_sentences(model_dir, SENTENCES)
    token_counts = [len(ids) for ids in token_ids_list]
    print(f"  token counts: {token_counts}", file=sys.stderr)

    all_results: dict[str, dict] = {}
    errors: list[str] = []

    # --- mlx-lm ---
    try:
        all_results["mlx_lm"] = bench_mlx_lm(model_dir, token_ids_list, args.trials, args.cooldown)
    except Exception as e:
        print(f"\n[mlx-lm] FAILED: {e}", file=sys.stderr)
        errors.append(f"mlx_lm: {e}")

    # --- ax-engine ---
    try:
        all_results["ax_engine"] = bench_ax_engine(
            model_dir, token_ids_list, args.trials, args.cooldown,
            args.port, skip_server=args.skip_ax_server,
        )
    except Exception as e:
        print(f"\n[ax-engine] FAILED: {e}", file=sys.stderr)
        errors.append(f"ax_engine: {e}")

    # --- mlx-swift ---
    if not args.skip_swift:
        swift_bin = args.swift_bench_binary
        if not swift_bin.is_absolute():
            swift_bin = REPO_ROOT / swift_bin
        try:
            all_results["mlx_swift"] = bench_mlx_swift(
                swift_bin, model_dir, token_ids_list, SENTENCES, args.trials, args.cooldown,
            )
        except FileNotFoundError as e:
            print(f"\n[mlx-swift] SKIPPED (binary not found): {e}", file=sys.stderr)
        except Exception as e:
            print(f"\n[mlx-swift] FAILED: {e}", file=sys.stderr)
            errors.append(f"mlx_swift: {e}")

    # --- summary ---
    print_summary(all_results)

    # --- write output ---
    output = {
        "schema_version": "ax.embedding_bench.v1",
        "timestamp": datetime.now().isoformat(),
        "model_dir": str(model_dir),
        "model_label": label,
        "sentences": SENTENCES,
        "token_counts": token_counts,
        "trials": args.trials,
        "cooldown_s": args.cooldown,
        "results": all_results,
    }
    out_path = run_dir / "embedding_bench.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote results to {out_path}", file=sys.stderr)

    if errors and "mlx_lm" in str(errors) or "ax_engine" in str(errors):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
