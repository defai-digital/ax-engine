#!/usr/bin/env python3
"""Apple-to-apple prefill + decode benchmark: SwiftLM, mlx_lm, and ax-engine MLX mode.

All three paths share the same underlying Apple MLX C++ stack.
  SwiftLM      — Swift/Hummingbird server wrapping mlx-swift + MLXLLM.
                 Non-streaming only (streaming path has a crash in the fork).
                 Two-step measurement:
                   Step 1: unique prompt, max_tokens=1 → elapsed ≈ prefill time
                   Step 2: same prompt (KV cached), max_tokens=N → elapsed ≈ decode time
  mlx_lm       — Python CLI. Verbose output gives "Prompt: N tok, X tok/s" and
                 "Generation: N tok, X tok/s" directly from the MLX engine.
  ax-engine    — Rust server wrapping mlx-sys FFI.  /v1/generate/stream SSE
                 events carry runner_time_us per step (server-side compute time,
                 no HTTP overhead). Step 0 = prefill; steps 1..N = decode.

Metrics:
  prefill_tok_s = prompt_tokens / prefill_s
  decode_tok_s  = (output_tokens - 1) / decode_s

Usage
─────
  # Build SwiftLM (one-time, ~5 min):
  #   cd .internal/reference/SwiftLM
  #   git submodule update --init
  #   bash build.sh    (builds Metal kernels + SwiftLM binary)
  #
  # Build ax-engine-server:
  #   cargo build -p ax-engine-server --release
  #
  python3 scripts/bench_swiftlm_vs_axengine_mlx.py

Optional flags:
  --swiftlm-binary   path to SwiftLM binary
  --model-id         HuggingFace model ID (default: mlx-community/Qwen3-4B-4bit)
  --model-dir        local model snapshot directory
  --prompt-tokens    comma-separated prompt lengths (default: 64,256,512)
  --decode-tokens    output tokens per request (default: 128)
  --repetitions      runs per cell excluding warmup (default: 3)
  --cooldown         seconds between runs (default: 5)
  --output           write JSON results to this file
  --skip-swiftlm     skip SwiftLM path
  --skip-mlxlm       skip mlx_lm path
  --skip-axengine    skip ax-engine path
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SWIFTLM_DIR = REPO_ROOT / ".internal/reference/SwiftLM"
SWIFTLM_BINARY_DEFAULT = SWIFTLM_DIR / ".build/arm64-apple-macosx/release/SwiftLM"
AX_ENGINE_SERVER = REPO_ROOT / "target/release/ax-engine-server"

MODEL_HF_ID = "mlx-community/Qwen3-4B-4bit"
MODEL_DIR_DEFAULT = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit"
    / "snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"
)

SWIFTLM_PORT = 8090
AXENGINE_PORT = 8091

# mlx_lm verbose output patterns.
_MLX_PREFILL_RE = re.compile(r"Prompt:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec")
_MLX_GEN_RE     = re.compile(r"Generation:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec")

_cached_tokens: dict[str, list[int]] = {}

# Counter used to generate unique prefixes per measurement, preventing KV cache hits.
_run_counter = 0


def next_run_tag() -> str:
    global _run_counter
    _run_counter += 1
    return f"[uid={_run_counter}] "


# ── Tokenizer helper ──────────────────────────────────────────────────────────

def tokenize(prompt: str, model_dir: Path) -> list[int]:
    key = f"{model_dir}:{prompt}"
    if key not in _cached_tokens:
        print("  [tokenize] encoding prompt...", file=sys.stderr)
        import mlx_lm
        _, tok = mlx_lm.load(str(model_dir))
        ids = tok.encode(prompt)
        _cached_tokens[key] = ids
        del tok
        print(f"  [tokenize] {len(ids)} tokens", file=sys.stderr)
    return _cached_tokens[key]


def make_prompt(target_tokens: int) -> str:
    phrase = "The quick brown fox jumps over the lazy dog. "
    chars_needed = int(target_tokens * 4.4)
    return (phrase * ((chars_needed // len(phrase)) + 2))[:chars_needed]


# ── Server helpers ────────────────────────────────────────────────────────────

def wait_for_server(url: str, timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def kill_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


_METAL_CRYPTEX = (
    "/private/var/run/com.apple.security.cryptexd/mnt"
    "/com.apple.MobileAsset.MetalToolchain-v17.5.188.0.f4nJJ1"
    "/Metal.xctoolchain/usr/metal/current/bin"
)


def _env_with_metal() -> dict:
    env = dict(os.environ)
    existing = env.get("PATH", "")
    if _METAL_CRYPTEX not in existing:
        env["PATH"] = f"{_METAL_CRYPTEX}:{existing}"
    return env


def _ensure_mlx_metallib(binary_dir: Path) -> None:
    """mlx-c looks for mlx.metallib co-located with the binary."""
    import shutil
    default_lib = binary_dir / "default.metallib"
    mlx_lib = binary_dir / "mlx.metallib"
    if default_lib.exists() and not mlx_lib.exists():
        shutil.copy2(default_lib, mlx_lib)


def start_swiftlm(binary: Path, model_id: str, port: int) -> subprocess.Popen:
    _ensure_mlx_metallib(binary.parent)
    cmd = [str(binary), "--model", model_id, "--port", str(port), "--temp", "0"]
    print(f"  [SwiftLM] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=_env_with_metal())


def start_axengine(
    binary: Path, model_dir: Path, port: int, *, no_speculative: bool = False
) -> subprocess.Popen:
    cmd = [str(binary), "--mlx-native",
           "--native-model-artifacts-dir", str(model_dir), "--port", str(port)]
    if no_speculative:
        cmd.append("--no-speculative-decode")
    env = {**os.environ, "AX_MLX_NATIVE_CONFIRM": "1"}
    print(f"  [ax-engine] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)


# ── SwiftLM measurement (non-streaming, two-step) ────────────────────────────
#
# SwiftLM's streaming path has a crash in the SharpAI mlx-swift fork.
# Non-streaming is stable.  timings.predicted_ms = genStart to last token,
# so it includes prefill.  To isolate the two phases we use:
#
#   Step 1: unique prompt P, max_tokens=1
#     → client elapsed ≈ prefill_time + ~1 decode step
#     → Since 1 decode << prefill, elapsed ≈ prefill_time
#     → prefill_tok_s = len(P) / elapsed
#
#   Step 2: SAME prompt P, max_tokens=N  (MLXLLM re-uses KV cache for P)
#     → KV cache hit for the prompt prefix → elapsed ≈ decode_time
#     → decode_tok_s = (N - 1) / elapsed
#
# A unique run-tag is prepended to each prompt to prevent cross-rep cache hits.

def _swiftlm_post(port: int, prompt: str, max_tokens: int) -> tuple[float, dict]:
    """POST /v1/completions (non-streaming). Returns (elapsed_s, response_obj)."""
    payload = json.dumps({
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    t0 = time.monotonic()
    conn.request("POST", "/v1/completions", body=payload,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    elapsed = time.monotonic() - t0
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"SwiftLM HTTP {resp.status}: {data[:200].decode(errors='replace')}")
    return elapsed, json.loads(data)


def swiftlm_one_run(port: int, prompt: str, max_tokens: int) -> dict:
    """Two-step measurement. Returns prefill_tok_s and decode_tok_s."""
    tag = next_run_tag()
    unique_prompt = tag + prompt

    # Step 1: prefill measurement (max_tokens=1, fresh prompt)
    elapsed1, obj1 = _swiftlm_post(port, unique_prompt, 1)
    pt = obj1.get("usage", {}).get("prompt_tokens", 0)
    prefill_s = elapsed1  # ~prefill + tiny 1-token decode; the latter is <5ms
    prefill_tps = pt / prefill_s if prefill_s > 0 else 0.0

    # Step 2: decode measurement (same prompt, KV cache hit → elapsed ≈ decode only)
    elapsed2, obj2 = _swiftlm_post(port, unique_prompt, max_tokens)
    ct = obj2.get("usage", {}).get("completion_tokens", max_tokens)
    decode_s = elapsed2
    decode_tps = max(ct - 1, 0) / decode_s if decode_s > 0 else 0.0

    return {
        "prompt_tokens": pt,
        "output_tokens": ct,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tok_s": prefill_tps,
        "decode_tok_s": decode_tps,
    }


def bench_swiftlm(
    port: int,
    prompt: str,
    prompt_tokens_hint: int,
    max_tokens: int,
    reps: int,
    cooldown: float,
) -> dict:
    print(f"  [SwiftLM] prompt~{prompt_tokens_hint}tok → decode {max_tokens}tok", file=sys.stderr)

    def one(tag: str) -> dict:
        r = swiftlm_one_run(port, prompt, max_tokens)
        print(
            f"    {tag}: prefill={r['prefill_tok_s']:.1f} tok/s  decode={r['decode_tok_s']:.1f} tok/s"
            f"  (prefill_s={r['prefill_s']:.3f} decode_s={r['decode_s']:.3f} pt={r['prompt_tokens']})",
            file=sys.stderr,
        )
        return r

    one("warmup")
    if cooldown > 0:
        time.sleep(cooldown)

    runs = []
    for i in range(reps):
        runs.append(one(f"rep {i+1}"))
        if cooldown > 0 and i < reps - 1:
            time.sleep(cooldown)

    def med(k: str) -> float:
        return statistics.median(r[k] for r in runs)

    # Use the caller's token count, not the server-reported count.
    # The unique prefix tag added to each prompt for cache-busting inflates the
    # server-reported prompt_tokens; the timing is measured for the full tagged
    # prompt but we normalise rates back to the nominal token count so the table
    # rows align with the other paths.

    return {
        "prompt_tokens": prompt_tokens_hint,
        "decode_tokens": max_tokens,
        "prefill_tok_s": {
            "median": med("prefill_tok_s"),
            "min":    min(r["prefill_tok_s"] for r in runs),
            "max":    max(r["prefill_tok_s"] for r in runs),
        },
        "decode_tok_s": {
            "median": med("decode_tok_s"),
            "min":    min(r["decode_tok_s"] for r in runs),
            "max":    max(r["decode_tok_s"] for r in runs),
        },
        "prefill_s": {"median": med("prefill_s")},
        "decode_s":  {"median": med("decode_s")},
    }


# ── mlx_lm measurement ────────────────────────────────────────────────────────

def mlxlm_one_run(model_id: str, prompt: str, max_tokens: int) -> dict:
    cmd = [
        "python3", "-m", "mlx_lm", "generate",
        "--model", model_id,
        "--prompt", prompt,
        "--ignore-chat-template",
        "--max-tokens", str(max_tokens),
        "--temp", "0", "--top-p", "1", "--top-k", "0", "--seed", "1234",
        "--verbose", "true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [mlx_lm ERROR] exit={result.returncode}: {result.stderr[:200]}", file=sys.stderr)
        return {}
    combined = result.stdout + result.stderr
    m_pre = _MLX_PREFILL_RE.search(combined)
    m_gen = _MLX_GEN_RE.search(combined)
    out: dict = {}
    if m_pre:
        out["prompt_tokens"]  = int(m_pre.group(1))
        out["prefill_tok_s"]  = float(m_pre.group(2))
    if m_gen:
        out["output_tokens"]  = int(m_gen.group(1))
        out["decode_tok_s"]   = float(m_gen.group(2))
    return out


def bench_mlxlm(
    model_id: str,
    prompt: str,
    prompt_tokens_hint: int,
    max_tokens: int,
    reps: int,
    cooldown: float,
) -> dict:
    print(f"  [mlx_lm] prompt~{prompt_tokens_hint}tok → decode {max_tokens}tok", file=sys.stderr)

    def one(tag: str) -> dict:
        r = mlxlm_one_run(model_id, prompt, max_tokens)
        print(
            f"    {tag}: prefill={r.get('prefill_tok_s',0):.1f} tok/s  "
            f"decode={r.get('decode_tok_s',0):.1f} tok/s  "
            f"(pt={r.get('prompt_tokens','?')} ot={r.get('output_tokens','?')})",
            file=sys.stderr,
        )
        return r

    one("warmup")
    if cooldown > 0:
        time.sleep(cooldown)

    runs = []
    for i in range(reps):
        r = one(f"rep {i+1}")
        if r:
            runs.append(r)
        if cooldown > 0 and i < reps - 1:
            time.sleep(cooldown)

    if not runs:
        return {"prompt_tokens": prompt_tokens_hint, "decode_tokens": max_tokens,
                "prefill_tok_s": {"median": 0, "min": 0, "max": 0},
                "decode_tok_s":  {"median": 0, "min": 0, "max": 0}}

    def med(k: str) -> float:
        vals = [r[k] for r in runs if k in r]
        return statistics.median(vals) if vals else 0.0

    actual_pt = int(statistics.median(r["prompt_tokens"] for r in runs if "prompt_tokens" in r)) or prompt_tokens_hint

    return {
        "prompt_tokens": actual_pt,
        "decode_tokens": max_tokens,
        "prefill_tok_s": {
            "median": med("prefill_tok_s"),
            "min":    min(r.get("prefill_tok_s", 0) for r in runs),
            "max":    max(r.get("prefill_tok_s", 0) for r in runs),
        },
        "decode_tok_s": {
            "median": med("decode_tok_s"),
            "min":    min(r.get("decode_tok_s", 0) for r in runs),
            "max":    max(r.get("decode_tok_s", 0) for r in runs),
        },
    }


# ── ax-engine measurement (SSE, server-side runner_time_us) ───────────────────

def axengine_one_run(port: int, tokens: list[int], max_tokens: int) -> dict:
    payload = json.dumps({"input_tokens": tokens, "max_output_tokens": max_tokens}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    conn.request("POST", "/v1/generate/stream", body=payload,
                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"ax-engine HTTP {resp.status}: {resp.read(200).decode(errors='replace')}")

    prefill_us: Optional[int] = None
    decode_us = 0
    n_out = 0
    current_event = ""

    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
            continue
        if not line.startswith("data:"):
            continue
        try:
            obj = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if current_event == "step":
            step = obj.get("step", {})
            runner_us = step.get("runner_time_us", 0)
            n_out = obj.get("request", {}).get("output_len", 0)
            if prefill_us is None:
                prefill_us = runner_us
            else:
                decode_us += runner_us
        elif current_event == "response":
            n_out = len(obj.get("response", {}).get("output_tokens", [])) or n_out
    conn.close()

    prompt_len = len(tokens)
    prefill_s = (prefill_us or 0) / 1_000_000
    decode_s = decode_us / 1_000_000
    return {
        "prompt_tokens":  prompt_len,
        "output_tokens":  n_out,
        "prefill_s":      prefill_s,
        "decode_s":       decode_s,
        "prefill_tok_s":  prompt_len / prefill_s if prefill_s > 0 else 0.0,
        "decode_tok_s":   max(n_out - 1, 0) / decode_s if decode_s > 0 else 0.0,
    }


def bench_axengine(
    port: int,
    tokens: list[int],
    max_tokens: int,
    reps: int,
    cooldown: float,
) -> dict:
    prompt_len = len(tokens)
    print(f"  [ax_engine] prompt={prompt_len}tok → decode {max_tokens}tok", file=sys.stderr)

    def one(tag: str) -> dict:
        r = axengine_one_run(port, tokens, max_tokens)
        print(
            f"    {tag}: prefill={r['prefill_tok_s']:.1f} tok/s  decode={r['decode_tok_s']:.1f} tok/s"
            f"  (prefill_s={r['prefill_s']:.3f} decode_s={r['decode_s']:.3f} out={r['output_tokens']})",
            file=sys.stderr,
        )
        return r

    one("warmup")
    if cooldown > 0:
        time.sleep(cooldown)

    runs = []
    for i in range(reps):
        runs.append(one(f"rep {i+1}"))
        if cooldown > 0 and i < reps - 1:
            time.sleep(cooldown)

    def med(k: str) -> float:
        return statistics.median(r[k] for r in runs)

    return {
        "prompt_tokens": prompt_len,
        "decode_tokens": max_tokens,
        "prefill_tok_s": {
            "median": med("prefill_tok_s"),
            "min":    min(r["prefill_tok_s"] for r in runs),
            "max":    max(r["prefill_tok_s"] for r in runs),
        },
        "decode_tok_s": {
            "median": med("decode_tok_s"),
            "min":    min(r["decode_tok_s"] for r in runs),
            "max":    max(r["decode_tok_s"] for r in runs),
        },
        "prefill_s": {"median": med("prefill_s")},
        "decode_s":  {"median": med("decode_s")},
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(doc: dict) -> None:
    model = doc.get("model", "?")
    path_cells: dict[str, list[dict]] = doc.get("results", {})
    paths = list(path_cells.keys())

    all_prompt_lens = sorted({
        cell["prompt_tokens"]
        for p in paths
        for cell in path_cells[p]
    })

    # Timing method note per path.
    _method = {
        "swiftlm":      "two-step non-stream",
        "mlx_lm":       "mlx_lm --verbose",
        "ax_engine_mlx": "runner_time_us SSE",
    }

    print("\n" + "=" * 92)
    print(f"  {model}  — prefill + decode benchmark  (Apple MLX stack comparison)")
    print("=" * 92)
    print(f"{'Engine':<20} {'Prompt tok':>10} {'Prefill tok/s':>14} {'Decode tok/s':>13}  Method")
    print("-" * 92)

    for ptok in all_prompt_lens:
        rows: list[tuple[str, float, float]] = []
        for path in paths:
            cell = next((c for c in path_cells[path] if c["prompt_tokens"] == ptok), None)
            if cell is None:
                continue
            pre = cell["prefill_tok_s"]["median"]
            dec = cell["decode_tok_s"]["median"]
            method = _method.get(path, path)
            rows.append((path, pre, dec))
            print(f"{'  ' + path:<20} {ptok:>10}  {pre:>13.1f}  {dec:>12.1f}  [{method}]")

        # Print ratio rows vs mlx_lm reference.
        ref = next(((pre, dec) for p, pre, dec in rows if p == "mlx_lm"), None)
        if ref:
            ref_pre, ref_dec = ref
            for path, pre, dec in rows:
                if path == "mlx_lm":
                    continue
                pre_x = pre / max(ref_pre, 0.001)
                dec_x = dec / max(ref_dec, 0.001)
                label = f"vs mlx_lm → {path}"
                print(
                    f"{'    ' + label:<28} {'':>4}  {pre_x:>12.2f}x  {dec_x:>12.2f}x"
                )
        print()
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefill+decode benchmark: SwiftLM, mlx_lm, and ax-engine MLX mode"
    )
    parser.add_argument("--swiftlm-binary", type=Path, default=SWIFTLM_BINARY_DEFAULT)
    parser.add_argument("--model-id", default=MODEL_HF_ID)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR_DEFAULT)
    parser.add_argument("--prompt-tokens", default="64,256,512")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-swiftlm",  action="store_true")
    parser.add_argument("--skip-mlxlm",    action="store_true")
    parser.add_argument("--skip-axengine", action="store_true")
    parser.add_argument("--swiftlm-port", type=int, default=SWIFTLM_PORT)
    parser.add_argument("--axengine-port", type=int, default=AXENGINE_PORT)
    parser.add_argument(
        "--axengine-no-speculative", action="store_true",
        help="Start ax-engine-server with --no-speculative-decode for a clean "
             "single-token greedy baseline (disables n-gram speculation).",
    )
    args = parser.parse_args()

    prompt_lengths = [int(x.strip()) for x in args.prompt_tokens.split(",")]

    print(f"\n=== MLX stack benchmark: SwiftLM / mlx_lm / ax-engine ===", file=sys.stderr)
    print(f"  model:         {args.model_id}", file=sys.stderr)
    print(f"  prompt_tokens: {prompt_lengths}", file=sys.stderr)
    print(f"  decode_tokens: {args.decode_tokens}", file=sys.stderr)
    print(f"  repetitions:   {args.repetitions} + 1 warmup", file=sys.stderr)
    if not args.skip_axengine:
        spec_label = "OFF (--no-speculative-decode)" if args.axengine_no_speculative else "ON (EMA adaptive)"
        print(f"  ax-engine speculation: {spec_label}", file=sys.stderr)

    # Pre-tokenize prompts for ax-engine (needs raw token IDs).
    prompts: list[tuple[str, list[int]]] = []
    for ptok in prompt_lengths:
        text = make_prompt(ptok)
        tokens = tokenize(text, args.model_dir)
        prompts.append((text, tokens))
        print(f"  prompt {ptok}: {len(tokens)} actual tokens", file=sys.stderr)

    procs: list[subprocess.Popen] = []
    results_data: dict = {}

    try:
        # ── mlx_lm ───────────────────────────────────────────────────────────
        if not args.skip_mlxlm:
            print("\n[mlx_lm] Benchmarking...", file=sys.stderr)
            cells = []
            for text, tokens in prompts:
                cell = bench_mlxlm(
                    args.model_id, text, len(tokens),
                    args.decode_tokens, args.repetitions, args.cooldown,
                )
                cells.append(cell)
            results_data["mlx_lm"] = cells

        # ── SwiftLM ───────────────────────────────────────────────────────────
        if not args.skip_swiftlm:
            if not args.swiftlm_binary.exists():
                print(
                    f"\nERROR: SwiftLM binary not found at {args.swiftlm_binary}\n"
                    f"Build it:\n"
                    f"  cd {SWIFTLM_DIR}\n"
                    f"  git submodule update --init\n"
                    f"  bash build.sh",
                    file=sys.stderr,
                )
                sys.exit(1)

            print("\n[SwiftLM] Starting server...", file=sys.stderr)
            sw_proc = start_swiftlm(args.swiftlm_binary, args.model_id, args.swiftlm_port)
            procs.append(sw_proc)

            if not wait_for_server(f"http://127.0.0.1:{args.swiftlm_port}/health"):
                print("ERROR: SwiftLM did not become ready in 180s", file=sys.stderr)
                sys.exit(1)
            print("[SwiftLM] Ready.", file=sys.stderr)

            sw_cells = []
            for text, tokens in prompts:
                cell = bench_swiftlm(
                    args.swiftlm_port, text, len(tokens),
                    args.decode_tokens, args.repetitions, args.cooldown,
                )
                sw_cells.append(cell)
            results_data["swiftlm"] = sw_cells

            print("\n[SwiftLM] Stopping...", file=sys.stderr)
            kill_proc(sw_proc)
            procs.remove(sw_proc)
            time.sleep(3)

        # ── ax-engine MLX mode ───────────────────────────────────────────────
        if not args.skip_axengine:
            if not AX_ENGINE_SERVER.exists():
                print(
                    f"\nERROR: ax-engine-server not found at {AX_ENGINE_SERVER}\n"
                    "Build it: cargo build -p ax-engine-server --release",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not args.model_dir.exists():
                print(f"\nERROR: model dir not found: {args.model_dir}", file=sys.stderr)
                sys.exit(1)

            print("\n[ax-engine] Starting server...", file=sys.stderr)
            ax_proc = start_axengine(
                AX_ENGINE_SERVER, args.model_dir, args.axengine_port,
                no_speculative=args.axengine_no_speculative,
            )
            procs.append(ax_proc)

            if not wait_for_server(f"http://127.0.0.1:{args.axengine_port}/health"):
                print("ERROR: ax-engine-server did not become ready in 120s", file=sys.stderr)
                if ax_proc.stderr:
                    out = ax_proc.stderr.read(2000)
                    print(f"  stderr: {out.decode(errors='replace')}", file=sys.stderr)
                sys.exit(1)
            print("[ax-engine] Ready.", file=sys.stderr)

            ax_cells = []
            for _, tokens in prompts:
                cell = bench_axengine(
                    args.axengine_port, tokens,
                    args.decode_tokens, args.repetitions, args.cooldown,
                )
                ax_cells.append(cell)
            results_data["ax_engine_mlx"] = ax_cells

    finally:
        for p in procs:
            kill_proc(p)

    doc = {
        "model": args.model_id,
        "decode_tokens": args.decode_tokens,
        "repetitions": args.repetitions,
        "results": results_data,
    }

    print_summary(doc)
    print(json.dumps(doc, indent=2))

    if args.output:
        args.output.write_text(json.dumps(doc, indent=2))
        print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
