#!/usr/bin/env python3
"""Benchmark embedding throughput: ax-engine vs mlx-lm vs mlx-swift-lm.

Measures ms/sentence and tok/s for each backend on the same set of sentences
using the same pre-tokenized input (EOS-appended for Qwen3-Embedding).

Backends
--------
mlx_lm        Python: mlx_lm.load() → model.model(x) → last-token pool
ax_engine_py  Python: ax_engine.Session.embed() — direct call, no HTTP
ax_engine     HTTP:   ax-engine-server /v1/embeddings (shows HTTP overhead)
mlx_swift     Swift:  MLXEmbedders.EmbedderModelFactory via subprocess

ax_engine_py is the apple-to-apple comparison vs mlx_lm and mlx_swift:
same Python call depth, no network, just Python → Rust → MLX → Metal.

Exit codes:
  0  all backends ran
  1  usage error
  2  a required backend failed (mlx_lm or ax_engine_py)

Usage:
    # Build Python extension first (release for fair benchmark):
    maturin develop --release

    # Build ax-engine-server (optional, for HTTP comparison):
    cargo build -p ax-engine-server --release

    python scripts/bench_embedding_models.py \\
        --model-dir .internal/models/qwen3-embedding-0.6b-8bit \\
        --model-label qwen3-embedding-0.6b-8bit \\
        [--trials 5] \\
        [--cooldown 2] \\
        [--port 8083] \\
        [--skip-ax-http]     # skip HTTP backend \\
        [--skip-swift]       # skip mlx-swift backend \\
        [--output-dir benchmarks/results/embedding]
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
import tempfile
import time
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
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
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


def _trial_stats(ms_vals: list[float], tps_vals: list[float], engine: str) -> dict[str, Any]:
    return {
        "engine": engine,
        "mean_ms_per_sentence": mean(ms_vals),
        "median_ms_per_sentence": median(ms_vals),
        "stddev_ms_per_sentence": stddev(ms_vals),
        "mean_tokens_per_sec": mean(tps_vals),
        "median_tokens_per_sec": median(tps_vals),
    }


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
# Backend: mlx-lm (Python direct)
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

    model, _tokenizer = load(str(model_dir))
    total_tokens = sum(len(ids) for ids in token_ids_list)
    n = len(token_ids_list)

    print("  [mlx-lm] warmup…", file=sys.stderr)
    for ids in token_ids_list:
        x = mx.array([ids])
        h = model.model(x)
        last = h[0, -1, :].astype(mx.float32)
        norm = mx.sqrt(mx.sum(last * last))
        mx.eval(last / (norm + 1e-12))

    trial_rows = []
    for i in range(1, trials + 1):
        if cooldown > 0:
            time.sleep(cooldown)
        t0 = time.perf_counter()
        for ids in token_ids_list:
            x = mx.array([ids])
            h = model.model(x)
            last = h[0, -1, :].astype(mx.float32)
            norm = mx.sqrt(mx.sum(last * last))
            mx.eval(last / (norm + 1e-12))
        elapsed = time.perf_counter() - t0
        ms = elapsed * 1000.0 / n
        tps = total_tokens / elapsed if elapsed > 0 else 0.0
        trial_rows.append({"ms_per_sentence": ms, "tokens_per_sec": tps})
        print(f"    trial {i}: {ms:.1f}ms/sentence  {tps:.1f} tok/s", file=sys.stderr)

    ms_vals = [t["ms_per_sentence"] for t in trial_rows]
    tps_vals = [t["tokens_per_sec"] for t in trial_rows]
    return {**_trial_stats(ms_vals, tps_vals, "mlx_lm"), "trials": trial_rows}


# ---------------------------------------------------------------------------
# Backend: ax-engine-py (Python direct — apple-to-apple with mlx-lm)
# ---------------------------------------------------------------------------

def bench_ax_engine_py(
    model_dir: Path,
    token_ids_list: list[list[int]],
    trials: int,
    cooldown: float,
) -> dict[str, Any]:
    print("\n[ax-engine-py] Loading session…", file=sys.stderr)
    try:
        sys.path.insert(0, str(REPO_ROOT / "python"))
        import ax_engine
    except ImportError:
        raise ImportError(
            "ax_engine not installed. Run: maturin develop --release"
        )

    session = ax_engine.Session(
        model_id="qwen3_dense",
        mlx=True,
        support_tier="mlx_preview",
        mlx_model_artifacts_dir=str(model_dir),
    )

    total_tokens = sum(len(ids) for ids in token_ids_list)
    n = len(token_ids_list)

    # Use embed_bytes for a fair comparison vs mlx-lm: mlx-lm's bench only
    # mx.eval()s the result (no read-back into Python), so ax-engine should
    # also avoid the per-element PyFloat allocation that list[float] forces.
    # embed_bytes returns the raw f32 buffer as Python bytes (O(1) materialise).
    embed_fast = getattr(session, "embed_bytes", None)
    if embed_fast is None:
        embed_fast = lambda ids: session.embed(ids, pooling="last", normalize=True)
    else:
        _bytes_fn = embed_fast
        embed_fast = lambda ids: _bytes_fn(ids, pooling="last", normalize=True)

    print("  [ax-engine-py] warmup…", file=sys.stderr)
    for ids in token_ids_list:
        embed_fast(ids)

    trial_rows = []
    for i in range(1, trials + 1):
        if cooldown > 0:
            time.sleep(cooldown)
        t0 = time.perf_counter()
        for ids in token_ids_list:
            embed_fast(ids)
        elapsed = time.perf_counter() - t0
        ms = elapsed * 1000.0 / n
        tps = total_tokens / elapsed if elapsed > 0 else 0.0
        trial_rows.append({"ms_per_sentence": ms, "tokens_per_sec": tps})
        print(f"    trial {i}: {ms:.1f}ms/sentence  {tps:.1f} tok/s", file=sys.stderr)

    session.close()
    ms_vals = [t["ms_per_sentence"] for t in trial_rows]
    tps_vals = [t["tokens_per_sec"] for t in trial_rows]
    return {**_trial_stats(ms_vals, tps_vals, "ax_engine_py"), "trials": trial_rows}


# ---------------------------------------------------------------------------
# Backend: ax-engine HTTP (shows serialization overhead)
# ---------------------------------------------------------------------------

def _start_axengine_server(model_dir: Path, port: int) -> subprocess.Popen:
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
    print(f"  [ax-engine-http] starting: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def bench_ax_engine_http(
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
    n = len(token_ids_list)

    try:
        if not skip_server:
            print("\n[ax-engine-http] Starting server…", file=sys.stderr)
            server_proc = _start_axengine_server(model_dir, port)
            print(f"  [ax-engine-http] waiting for port {port}…", file=sys.stderr)
            if not wait_for_port("127.0.0.1", port, timeout=120.0):
                raise RuntimeError("ax-engine-server did not start within 120s")
            print("  [ax-engine-http] server ready.", file=sys.stderr)
        else:
            print(f"\n[ax-engine-http] using running server on port {port}", file=sys.stderr)

        print("  [ax-engine-http] warmup…", file=sys.stderr)
        for ids in token_ids_list:
            http_post(f"{server_url}/v1/embeddings",
                      {"input": ids, "pooling": "last", "normalize": True})

        trial_rows = []
        for i in range(1, trials + 1):
            if cooldown > 0:
                time.sleep(cooldown)
            t0 = time.perf_counter()
            for ids in token_ids_list:
                http_post(f"{server_url}/v1/embeddings",
                          {"input": ids, "pooling": "last", "normalize": True})
            elapsed = time.perf_counter() - t0
            ms = elapsed * 1000.0 / n
            tps = total_tokens / elapsed if elapsed > 0 else 0.0
            trial_rows.append({"ms_per_sentence": ms, "tokens_per_sec": tps})
            print(f"    trial {i}: {ms:.1f}ms/sentence  {tps:.1f} tok/s", file=sys.stderr)

        ms_vals = [t["ms_per_sentence"] for t in trial_rows]
        tps_vals = [t["tokens_per_sec"] for t in trial_rows]
        return {**_trial_stats(ms_vals, tps_vals, "ax_engine_http"), "trials": trial_rows}

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
            f"mlx-swift-embed-bench binary not found at {binary}.\n"
            "Build with: cd scripts/mlx-swift-bench && swift build -c release --product mlx-swift-embed-bench"
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            [{"label": s, "token_ids": ids} for s, ids in zip(sentences, token_ids_list)],
            f,
        )
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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=False)
        if result.returncode != 0:
            raise RuntimeError(f"mlx-swift-embed-bench exited {result.returncode}")
        output = json.loads(result.stdout)
    finally:
        os.unlink(token_ids_path)

    trial_rows = output.get("trials", [])
    ms_vals = [t.get("ms_per_sentence", 0) for t in trial_rows]
    tps_vals = [t.get("tokens_per_sec", 0) for t in trial_rows]
    return {**_trial_stats(ms_vals, tps_vals, "mlx_swift"), "trials": trial_rows}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict[str, dict]) -> None:
    engines = ["mlx_lm", "ax_engine_py", "mlx_swift", "ax_engine_http"]
    labels = {
        "mlx_lm": "mlx-lm (Python)",
        "ax_engine_py": "ax-engine-py (direct)",
        "mlx_swift": "mlx-swift-lm",
        "ax_engine_http": "ax-engine-http",
    }
    print(f"\n{'Engine':<26} {'mean ms/sent':>14} {'median ms/sent':>16} {'mean tok/s':>12}")
    print("-" * 72)
    for engine in engines:
        r = results.get(engine)
        label = labels.get(engine, engine)
        if r is None:
            print(f"{label:<26} {'(skipped)':>14}")
            continue
        print(
            f"{label:<26} "
            f"{r['mean_ms_per_sentence']:>14.1f} "
            f"{r['median_ms_per_sentence']:>16.1f} "
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
                        help=f"ax-engine HTTP server port (default {AXENGINE_PORT_DEFAULT})")
    parser.add_argument("--skip-ax-http", action="store_true",
                        help="Skip the HTTP backend (keep only direct Python/Swift comparisons)")
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

    # --- mlx-lm (reference) ---
    try:
        all_results["mlx_lm"] = bench_mlx_lm(model_dir, token_ids_list, args.trials, args.cooldown)
    except Exception as e:
        print(f"\n[mlx-lm] FAILED: {e}", file=sys.stderr)
        errors.append(f"mlx_lm: {e}")

    # --- ax-engine-py (apple-to-apple direct) ---
    try:
        all_results["ax_engine_py"] = bench_ax_engine_py(
            model_dir, token_ids_list, args.trials, args.cooldown,
        )
    except Exception as e:
        print(f"\n[ax-engine-py] FAILED: {e}", file=sys.stderr)
        errors.append(f"ax_engine_py: {e}")

    # --- mlx-swift-lm ---
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

    # --- ax-engine HTTP (shows serialization overhead) ---
    if not args.skip_ax_http:
        try:
            all_results["ax_engine_http"] = bench_ax_engine_http(
                model_dir, token_ids_list, args.trials, args.cooldown,
                args.port, skip_server=args.skip_ax_server,
            )
        except Exception as e:
            print(f"\n[ax-engine-http] FAILED: {e}", file=sys.stderr)
            errors.append(f"ax_engine_http: {e}")

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

    required_failed = any(k in str(errors) for k in ("mlx_lm", "ax_engine_py"))
    return 2 if required_failed else 0


if __name__ == "__main__":
    sys.exit(main())
