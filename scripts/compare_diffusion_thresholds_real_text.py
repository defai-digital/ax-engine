#!/usr/bin/env python3
"""Compare DiffusionGemma convergence thresholds with real-text multi-block prompts.

Runs the direct bench with pre-tokenized real text prompts under several threshold
configurations and reports per-prompt median denoise steps per block, convergence
signals, block decode throughput, total wall time, and output token count.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--diffusiongemma-26B-A4B-it-4bit"
    / "snapshots"
    / "252183330817f96e9cba0b20cc400b2947a575cf"
)

PROMPTS: dict[str, list[int]] = {
    "relativity": [155122, 506, 6073, 529, 74413, 528, 3606, 3755, 236761],
    "autumn": [6974, 496, 2822, 27355, 1003, 16273, 6895, 236761],
    "photosynthesis": [82858, 506, 1657, 529, 93036, 236761],
    "climate": [3689, 659, 506, 1689, 9454, 529, 10022, 2352, 236881],
}

# Threshold configurations under test.  The "default" config matches the
# documented production defaults.  Telemetry runs with real-text multi-block
# prompts (max_output_tokens >= 512) show that all tested configs converge in
# the minimum one denoise step per 256-token block via the acceptance signal,
# so the stricter defaults do not sacrifice speed while remaining safer against
# false convergence on harder prompts.
CONFIGS: dict[str, dict[str, str]] = {
    "default": {
        "AX_DIFFUSION_ENTROPY_THRESHOLD": "0.005",
        "AX_DIFFUSION_ACCEPTANCE_RATE_THRESHOLD": "0.075",
        "AX_DIFFUSION_ENTROPY_PLATEAU_DELTA": "0.001",
    },
    "entropy_0.05": {
        "AX_DIFFUSION_ENTROPY_THRESHOLD": "0.05",
        "AX_DIFFUSION_ACCEPTANCE_RATE_THRESHOLD": "0.05",
        "AX_DIFFUSION_ENTROPY_PLATEAU_DELTA": "0.001",
    },
    "entropy_0.1": {
        "AX_DIFFUSION_ENTROPY_THRESHOLD": "0.1",
        "AX_DIFFUSION_ACCEPTANCE_RATE_THRESHOLD": "0.05",
        "AX_DIFFUSION_ENTROPY_PLATEAU_DELTA": "0.001",
    },
    "best_sweep": {
        "AX_DIFFUSION_ENTROPY_THRESHOLD": "0.1",
        "AX_DIFFUSION_ACCEPTANCE_RATE_THRESHOLD": "0.1",
        "AX_DIFFUSION_ENTROPY_PLATEAU_DELTA": "0.001",
    },
}

WARMUP = 1
MEASURE = 3
DEFAULT_MAX_OUTPUT_TOKENS = 512


@dataclass(frozen=True)
class Trial:
    prompt: str
    config: str
    max_output_tokens: int
    denoise_steps: int
    converged_blocks: int
    converged_strict: int
    converged_acceptance: int
    converged_plateau: int
    min_entropy: float
    min_acceptance_rate: float
    block_decode_tok_s: float
    block_wall_us: int
    total_wall_us: int
    output_tokens: int
    finish_reason: str


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def run_one(
    *,
    bench_bin: Path,
    model_dir: Path,
    prompt_name: str,
    tokens: list[int],
    config_name: str,
    env_overrides: dict[str, str],
    max_output_tokens: int,
) -> Trial:
    command = [
        str(bench_bin),
        "generate",
        "--mlx",
        "--model-id",
        "diffusiongemma",
        "--tokens",
        ",".join(str(t) for t in tokens),
        "--max-output-tokens",
        str(max_output_tokens),
        "--mlx-model-artifacts-dir",
        str(model_dir),
        "--json",
    ]
    env = dict(os.environ)
    env.update(env_overrides)
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"bench failed for {prompt_name}/{config_name}: {result.stdout}")
    response = json.loads(result.stdout)
    counters = response.get("route", {}).get("crossover_decisions", {})

    def int_counter(key: str) -> int:
        value = counters.get(key)
        if not isinstance(value, int):
            raise ValueError(f"missing integer counter {key}")
        return value

    def int_counter_or(key: str, default: int) -> int:
        value = counters.get(key)
        return value if isinstance(value, int) else default

    block_wall_us = int_counter("ax_mlx_diffusion_block_wall_us")
    diffusion_blocks = int_counter_or("ax_mlx_diffusion_blocks", 0)
    # The diffusion block wall time covers all blocks generated in this request.
    # Approximate per-block throughput by assuming each block contributes canvas_size
    # tokens (the final block is also a full canvas unless max_output_tokens cuts it).
    canvas_size = 256
    block_tokens = diffusion_blocks * canvas_size if diffusion_blocks > 0 else 1
    block_decode_tok_s = (
        block_tokens / (block_wall_us / 1_000_000.0) if block_wall_us > 0 else 0.0
    )

    return Trial(
        prompt=prompt_name,
        config=config_name,
        max_output_tokens=max_output_tokens,
        denoise_steps=int_counter("ax_mlx_diffusion_denoise_steps"),
        converged_blocks=int_counter_or("ax_mlx_diffusion_converged_blocks", 0),
        converged_strict=int_counter("ax_mlx_diffusion_converged_strict"),
        converged_acceptance=int_counter("ax_mlx_diffusion_converged_acceptance"),
        converged_plateau=int_counter("ax_mlx_diffusion_converged_plateau"),
        min_entropy=int_counter("ax_mlx_diffusion_min_entropy_bp") / 10000.0,
        min_acceptance_rate=int_counter("ax_mlx_diffusion_min_acceptance_rate_bp") / 10000.0,
        block_decode_tok_s=block_decode_tok_s,
        block_wall_us=block_wall_us,
        total_wall_us=int_counter("ax_mlx_decode_wall_us"),
        output_tokens=len(response.get("output_tokens", [])),
        finish_reason=response.get("finish_reason", "unknown"),
    )


def summarize(trials: list[Trial]) -> dict[str, Any]:
    summary: dict[str, Any] = {"trials": [t.__dict__ for t in trials], "by_prompt": {}}
    print("\n" + "=" * 90)
    print("SUMMARY (median over measure runs)")
    print("=" * 90)
    for prompt_name in PROMPTS:
        summary["by_prompt"][prompt_name] = {}
        print(f"\nPrompt: {prompt_name}")
        for config_name in CONFIGS:
            subset = [t for t in trials if t.prompt == prompt_name and t.config == config_name]
            if not subset:
                continue
            blocks = sum(t.converged_blocks for t in subset)
            strict = sum(t.converged_strict for t in subset)
            accept = sum(t.converged_acceptance for t in subset)
            plateau = sum(t.converged_plateau for t in subset)
            expected_blocks = max(1, subset[0].max_output_tokens // 256)
            conv_rate = blocks / max(1, len(subset) * expected_blocks)
            row = {
                "denoise_steps": median([t.denoise_steps for t in subset]),
                "converged_strict": strict,
                "converged_acceptance": accept,
                "converged_plateau": plateau,
                "convergence_rate": conv_rate,
                "block_decode_tok_s": median([t.block_decode_tok_s for t in subset]),
                "wall_s": median([t.total_wall_us for t in subset]) / 1_000_000.0,
                "output_tokens": median([t.output_tokens for t in subset]),
                "recommended": config_name == "default" and conv_rate >= 0.9,
            }
            summary["by_prompt"][prompt_name][config_name] = row
            print(
                f"  {config_name:14s}: denoise_steps={row['denoise_steps']:.0f} "
                f"strict={strict} accept={accept} plateau={plateau} "
                f"conv_rate={conv_rate:.2f} "
                f"tok/s={row['block_decode_tok_s']:.1f} "
                f"wall_s={row['wall_s']:.2f} "
                f"out_tok={row['output_tokens']:.0f}"
                f"{'  <- recommended' if row['recommended'] else ''}"
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Target generated tokens per prompt (default: {DEFAULT_MAX_OUTPUT_TOKENS})",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to the MLX DiffusionGemma model artifacts directory",
    )
    parser.add_argument(
        "--configs",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated list of config names to run (default: all)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of the results",
    )
    args = parser.parse_args()

    bench_bin = REPO_ROOT / "target" / "release" / "ax-engine-bench"
    model_dir = args.model_dir
    if not bench_bin.exists():
        raise SystemExit(f"bench binary not found: {bench_bin}")
    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir}")

    configs = {k: v for k, v in CONFIGS.items() if args.configs is None or k in args.configs}
    if not configs:
        raise SystemExit("no configs selected")

    trials: list[Trial] = []
    for prompt_name, tokens in PROMPTS.items():
        for config_name, env_overrides in configs.items():
            print(
                f"\n[{prompt_name}] {config_name}: max_output_tokens={args.max_output_tokens} "
                f"warmup={WARMUP} measure={MEASURE}"
            )
            for _ in range(WARMUP):
                run_one(
                    bench_bin=bench_bin,
                    model_dir=model_dir,
                    prompt_name=prompt_name,
                    tokens=tokens,
                    config_name=config_name,
                    env_overrides=env_overrides,
                    max_output_tokens=args.max_output_tokens,
                )
            for i in range(MEASURE):
                trial = run_one(
                    bench_bin=bench_bin,
                    model_dir=model_dir,
                    prompt_name=prompt_name,
                    tokens=tokens,
                    config_name=config_name,
                    env_overrides=env_overrides,
                    max_output_tokens=args.max_output_tokens,
                )
                trials.append(trial)
                print(
                    f"  measure#{i}: steps={trial.denoise_steps} "
                    f"strict={trial.converged_strict} acceptance={trial.converged_acceptance} "
                    f"plateau={trial.converged_plateau} blocks={trial.converged_blocks} "
                    f"tok/s={trial.block_decode_tok_s:.1f} wall_s={trial.total_wall_us / 1_000_000.0:.2f}"
                )

    summary = summarize(trials)
    if args.output_json:
        args.output_json.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote JSON summary to {args.output_json}")


if __name__ == "__main__":
    main()
