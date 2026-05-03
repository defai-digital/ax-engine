#!/usr/bin/env python3
"""Benchmark aligned GGUF and MLX model pairs for AX Engine research.

This script compares local GGUF models through `llama-bench` against MLX
models through `mlx_lm`. It is intended for repo-grounded performance checks
when evaluating llama.cpp backend tradeoffs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx_lm


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPT = (
    "Write a concise Python function that checks whether a number is prime, "
    "explain the time complexity, and include one short example."
)
DEFAULT_MAX_TOKENS = 64
DEFAULT_WARMUP_TOKENS = 8

PROMPT_TPS_RE = re.compile(r"Prompt:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec")
GEN_TPS_RE = re.compile(
    r"Generation:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec"
)
PEAK_MEMORY_RE = re.compile(r"Peak memory:\s+([0-9.]+)\s+GB")


@dataclass(frozen=True)
class ModelPair:
    key: str
    label: str
    gguf_path: Path
    mlx_model_id: str


MODEL_PAIRS = {
    "qwen3_5_9b": ModelPair(
        key="qwen3_5_9b",
        label="Qwen3.5-9B 4-bit",
        gguf_path=REPO_ROOT / ".internal/models/Qwen3.5-9B-Q4_K_M.gguf",
        mlx_model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
    ),
    "gemma4_26b_a4b": ModelPair(
        key="gemma4_26b_a4b",
        label="Gemma 4 26B A4B 4-bit",
        gguf_path=REPO_ROOT / ".internal/models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf",
        mlx_model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
    ),
}


def build_formatted_prompt(tokenizer: Any, prompt: str) -> tuple[str, int]:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if isinstance(encoded, dict):
        token_ids = list(encoded["input_ids"])
    else:
        token_ids = list(encoded)
    return formatted, len(token_ids)


def run_llama_bench(model_path: Path, prompt_tokens: int, max_tokens: int) -> dict[str, Any]:
    command = [
        "llama-bench",
        "-m",
        str(model_path),
        "-p",
        str(prompt_tokens),
        "-n",
        str(max_tokens),
        "-r",
        "1",
        "-o",
        "json",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"llama-bench failed for {model_path}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    payload = json.loads(completed.stdout)
    prompt_entry = next(item for item in payload if item["n_prompt"] > 0)
    gen_entry = next(item for item in payload if item["n_gen"] > 0)
    return {
        "backend": "gguf",
        "tool": "llama-bench",
        "model_path": str(model_path),
        "model_type": gen_entry["model_type"],
        "prompt_tokens": prompt_entry["n_prompt"],
        "max_output_tokens": gen_entry["n_gen"],
        "prompt_tokens_per_sec": prompt_entry["avg_ts"],
        "generation_tokens_per_sec": gen_entry["avg_ts"],
        "raw": {
            "prompt": prompt_entry,
            "generation": gen_entry,
        },
    }


def parse_mlx_verbose_metrics(verbose_output: str) -> dict[str, Any]:
    prompt_match = PROMPT_TPS_RE.search(verbose_output)
    gen_match = GEN_TPS_RE.search(verbose_output)
    memory_match = PEAK_MEMORY_RE.search(verbose_output)
    if not prompt_match or not gen_match:
        raise RuntimeError(f"Could not parse mlx_lm verbose output:\n{verbose_output}")
    return {
        "prompt_tokens_reported": int(prompt_match.group(1)),
        "prompt_tokens_per_sec": float(prompt_match.group(2)),
        "generation_tokens_reported": int(gen_match.group(1)),
        "generation_tokens_per_sec": float(gen_match.group(2)),
        "peak_memory_gb": float(memory_match.group(1)) if memory_match else None,
    }


def run_mlx_bench(
    model_id: str,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    warmup_tokens: int,
) -> dict[str, Any]:
    formatted_prompt, prompt_tokens = build_formatted_prompt(tokenizer, prompt)

    _ = mlx_lm.generate(
        model,
        tokenizer,
        prompt="Hi",
        max_tokens=warmup_tokens,
        verbose=False,
    )

    capture = io.StringIO()
    started_at = time.perf_counter()
    with contextlib.redirect_stdout(capture):
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=True,
        )
    elapsed = time.perf_counter() - started_at
    verbose_output = capture.getvalue()
    metrics = parse_mlx_verbose_metrics(verbose_output)
    output_tokens = len(tokenizer.encode(response))

    return {
        "backend": "mlx",
        "tool": "mlx_lm.generate",
        "model_id": model_id,
        "prompt_tokens": prompt_tokens,
        "max_output_tokens": max_tokens,
        "output_tokens": output_tokens,
        "wall_time_sec": elapsed,
        "generation_tokens_per_sec_wall": output_tokens / elapsed if elapsed > 0 else 0.0,
        **metrics,
    }


def benchmark_pair(pair: ModelPair, prompt: str, max_tokens: int, warmup_tokens: int) -> dict[str, Any]:
    if not pair.gguf_path.is_file():
        raise FileNotFoundError(f"GGUF model not found: {pair.gguf_path}")

    model, tokenizer = mlx_lm.load(pair.mlx_model_id)
    _, prompt_tokens = build_formatted_prompt(tokenizer, prompt)

    print(f"\n=== {pair.label} ===", file=sys.stderr)
    print(f"GGUF: {pair.gguf_path}", file=sys.stderr)
    gguf = run_llama_bench(pair.gguf_path, prompt_tokens, max_tokens)

    print(f"MLX:  {pair.mlx_model_id}", file=sys.stderr)
    mlx = run_mlx_bench(
        pair.mlx_model_id,
        model,
        tokenizer,
        prompt,
        max_tokens,
        warmup_tokens,
    )

    return {
        "key": pair.key,
        "label": pair.label,
        "prompt": prompt,
        "prompt_tokens_target": prompt_tokens,
        "gguf": gguf,
        "mlx": mlx,
        "speedup_gguf_over_mlx": (
            gguf["generation_tokens_per_sec"] / mlx["generation_tokens_per_sec"]
            if mlx["generation_tokens_per_sec"] > 0
            else None
        ),
        "speedup_mlx_over_gguf": (
            mlx["generation_tokens_per_sec"] / gguf["generation_tokens_per_sec"]
            if gguf["generation_tokens_per_sec"] > 0
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark aligned GGUF and MLX model pairs")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=list(MODEL_PAIRS.keys()),
        choices=sorted(MODEL_PAIRS.keys()),
        help="Model pairs to benchmark",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--warmup-tokens", type=int, default=DEFAULT_WARMUP_TOKENS)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file path for saving benchmark results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.time()
    results = {
        "schema_version": "ax.llama.bench.v1",
        "started_at_unix": started_at,
        "pairs": [],
    }

    for key in args.pairs:
        pair = MODEL_PAIRS[key]
        results["pairs"].append(
            benchmark_pair(pair, args.prompt, args.max_tokens, args.warmup_tokens)
        )

    output = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
        print(f"saved benchmark results to {args.output}", file=sys.stderr)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
