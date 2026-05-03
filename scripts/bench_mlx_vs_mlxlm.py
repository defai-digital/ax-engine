#!/usr/bin/env python3
"""Benchmark ax-engine MLX mode vs direct mlx_lm.

Compares two paths for Qwen3-4B-4bit:
  1. mlx_lm direct      — python3 -m mlx_lm generate (baseline)
  2. ax-engine MLX mode — MlxRunner via ax-bench --mlx

Metrics captured:
  - Wall-clock latency  (end-to-end subprocess time)
  - Tokens/sec from mlx_lm verbose output (for path 1 only)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AX_BENCH = REPO_ROOT / "target/release/ax-bench"

MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"
MODEL_HF_ID = "mlx-community/Qwen3-4B-4bit"

DEFAULT_PROMPT = (
    "Write a concise Python function that checks whether a number is prime, "
    "explain the time complexity, and include one short example."
)
DEFAULT_MAX_TOKENS = 64
DEFAULT_REPETITIONS = 3
DEFAULT_COOLDOWN = 8.0

MLX_PROMPT_RE = re.compile(r"Prompt:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec")
MLX_GEN_RE = re.compile(r"Generation:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec")

_last_end: float | None = None


def cooldown(label: str, secs: float) -> None:
    global _last_end
    if _last_end is not None and secs > 0:
        elapsed = time.monotonic() - _last_end
        wait = secs - elapsed
        if wait > 0:
            print(f"  [cooldown] {label}: waiting {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)


def run_cmd(cmd: list[str], label: str, cooldown_secs: float) -> dict:
    cooldown(label, cooldown_secs)
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    global _last_end
    _last_end = time.monotonic()
    stdout = result.stdout + result.stderr
    metrics: dict = {}
    m = MLX_PROMPT_RE.search(stdout)
    if m:
        metrics["prompt_tps"] = float(m.group(2))
    m = MLX_GEN_RE.search(stdout)
    if m:
        metrics["gen_tps"] = float(m.group(2))
    return {"elapsed_sec": elapsed, "metrics": metrics, "returncode": result.returncode}


def bench(cmd: list[str], label: str, reps: int, cooldown_secs: float) -> dict:
    print(f"  Benchmarking [{label}] ({reps} reps)...", file=sys.stderr)
    # Warmup
    run_cmd(cmd, f"{label} warmup", cooldown_secs)
    runs = []
    for i in range(reps):
        r = run_cmd(cmd, f"{label} rep {i+1}", cooldown_secs)
        runs.append(r)
        print(f"    rep {i+1}: {r['elapsed_sec']:.3f}s  metrics={r['metrics']}", file=sys.stderr)
    elapsed_vals = [r["elapsed_sec"] for r in runs]
    all_metrics_keys = {k for r in runs for k in r["metrics"]}
    metrics_agg = {}
    for k in all_metrics_keys:
        vals = [r["metrics"][k] for r in runs if k in r["metrics"]]
        if vals:
            metrics_agg[k] = {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}
    return {
        "label": label,
        "command": cmd,
        "reps": reps,
        "elapsed_sec": {
            "mean": statistics.mean(elapsed_vals),
            "min": min(elapsed_vals),
            "max": max(elapsed_vals),
        },
        "metrics": metrics_agg,
    }


def mlx_lm_cmd(prompt: str, max_tokens: int, verbose: bool) -> list[str]:
    return [
        "python3", "-m", "mlx_lm", "generate",
        "--model", MODEL_HF_ID,
        "--prompt", prompt,
        "--ignore-chat-template",
        "--max-tokens", str(max_tokens),
        "--temp", "0", "--top-p", "1", "--top-k", "0", "--seed", "1234",
        "--verbose", "true" if verbose else "false",
    ]


def tokenize_prompt(prompt: str) -> list[int]:
    """Tokenize using the model's tokenizer via mlx_lm."""
    import mlx_lm
    model, tokenizer = mlx_lm.load(str(MODEL_DIR))
    token_ids = tokenizer.encode(prompt)
    del model
    return token_ids


_cached_tokens: dict[str, list[int]] = {}


def get_token_ids(prompt: str) -> str:
    """Return comma-separated token IDs for the prompt (cached)."""
    if prompt not in _cached_tokens:
        print("  [tokenize] encoding prompt...", file=sys.stderr)
        _cached_tokens[prompt] = tokenize_prompt(prompt)
        print(f"  [tokenize] {len(_cached_tokens[prompt])} tokens", file=sys.stderr)
    return ",".join(str(t) for t in _cached_tokens[prompt])


def ax_mlx_cmd(prompt: str, max_tokens: int) -> list[str]:
    return [
        str(AX_BENCH), "generate",
        "--tokens", get_token_ids(prompt),
        "--max-output-tokens", str(max_tokens),
        "--mlx",
        "--mlx-model-artifacts-dir", str(MODEL_DIR),
        "--json",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ax-engine MLX mode vs mlx_lm")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--cooldown-seconds", type=float, default=DEFAULT_COOLDOWN)
    parser.add_argument("--output", help="Save JSON results here")
    parser.add_argument("--skip-ax-engine", action="store_true", help="Skip ax-engine MLX path")
    args = parser.parse_args()

    if not AX_BENCH.exists():
        print(f"ERROR: ax-bench not found at {AX_BENCH}. Run: cargo build -p ax-bench --release", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Benchmark: Qwen3-4B-4bit ===", file=sys.stderr)
    print(f"  prompt tokens: ~{len(args.prompt.split())} words", file=sys.stderr)
    print(f"  max_tokens: {args.max_tokens}", file=sys.stderr)
    print(f"  repetitions: {args.repetitions}", file=sys.stderr)

    results = {}

    # 1. mlx_lm direct (with verbose for tok/s)
    results["mlx_lm_direct"] = bench(
        mlx_lm_cmd(args.prompt, args.max_tokens, verbose=True),
        "mlx_lm_direct",
        args.repetitions,
        args.cooldown_seconds,
    )

    # 2. ax-engine MLX mode
    if not args.skip_ax_engine:
        results["ax_mlx"] = bench(
            ax_mlx_cmd(args.prompt, args.max_tokens),
            "ax_mlx",
            args.repetitions,
            args.cooldown_seconds,
        )

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Path':<22} {'Mean (s)':>10} {'Min (s)':>10} {'Max (s)':>10}  Gen tok/s")
    print("-" * 60)
    for key, r in results.items():
        e = r["elapsed_sec"]
        gen_tps = r["metrics"].get("gen_tps", {}).get("mean", "-")
        gen_str = f"{gen_tps:.1f}" if isinstance(gen_tps, float) else str(gen_tps)
        print(f"{key:<22} {e['mean']:>10.3f} {e['min']:>10.3f} {e['max']:>10.3f}  {gen_str}")

    base = results["mlx_lm_direct"]["elapsed_sec"]["mean"]
    print("\nOverhead vs mlx_lm_direct:")
    for key, r in results.items():
        if key == "mlx_lm_direct":
            continue
        delta = r["elapsed_sec"]["mean"] - base
        pct = (delta / base) * 100
        print(f"  {key}: {delta:+.3f}s ({pct:+.1f}%)")

    output = {
        "model": MODEL_HF_ID,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "results": results,
    }
    print("\n" + json.dumps(output, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
