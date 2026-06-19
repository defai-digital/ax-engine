#!/usr/bin/env python3
"""Sweep DiffusionGemma convergence thresholds and produce a summary artifact.

Runs the DiffusionGemma first-block benchmark across a grid of convergence
threshold configurations and reports per-config median denoise steps,
convergence rate, per-criterion signals, and block decode tok/s.

Usage:
  python3 scripts/sweep_diffusion_convergence.py \
      --model-dir $AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR/diffusiongemma-26B-A4B-it-4bit \
      --prompt-tokens 128 512 2048 \
      --output-dir benchmarks/results/diffusion-gemma-direct/convergence-sweep/

Grid dimensions:
  entropy_threshold:        [0.005, 0.01, 0.05, 0.1, 0.5]
  acceptance_rate_threshold: [0.01, 0.05, 0.10, 0.20]
  entropy_plateau_delta:    [0.001, 0.005, 0.01, 0.05]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "diffusion-gemma-direct"
    / "convergence-sweep"
)
SWEEP_SCHEMA = "ax.diffusion_convergence_sweep.v1"

ENTROPY_THRESHOLDS = [0.005, 0.01, 0.05, 0.1, 0.5]
ACCEPTANCE_RATE_THRESHOLDS = [0.01, 0.05, 0.10, 0.20]
ENTROPY_PLATEAU_DELTAS = [0.001, 0.005, 0.01, 0.05]

WARMUP_RUNS = 2
MEASURE_RUNS = 5


def log(message: str) -> None:
    print(f"[diffusion-sweep] {message}", flush=True)


def synthetic_tokens(prompt_tokens: int) -> list[int]:
    return [1000 + ((prompt_tokens * 9973 + i * 37) % 120000) for i in range(prompt_tokens)]


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot calculate median of an empty list")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


@dataclass(frozen=True)
class GridPoint:
    entropy_threshold: float
    acceptance_rate_threshold: float
    entropy_plateau_delta: float

    def env_overrides(self) -> dict[str, str]:
        return {
            "AX_DIFFUSION_ENTROPY_THRESHOLD": str(self.entropy_threshold),
            "AX_DIFFUSION_ACCEPTANCE_RATE_THRESHOLD": str(self.acceptance_rate_threshold),
            "AX_DIFFUSION_ENTROPY_PLATEAU_DELTA": str(self.entropy_plateau_delta),
        }

    def label(self) -> str:
        return (
            f"ent={self.entropy_threshold} "
            f"acc={self.acceptance_rate_threshold} "
            f"plateau={self.entropy_plateau_delta}"
        )


@dataclass(frozen=True)
class SweepTrial:
    denoise_steps: int
    converged: bool
    converged_strict: bool
    converged_acceptance: bool
    converged_plateau: bool
    min_entropy: float
    min_acceptance_rate: float
    block_decode_tok_s: float
    denoise_wall_us: int
    block_wall_us: int


def parse_trial(response: dict[str, Any]) -> SweepTrial:
    counters = response.get("route", {}).get("crossover_decisions", {})

    def int_counter(key: str) -> int:
        value = counters.get(key)
        if not isinstance(value, int):
            raise ValueError(f"missing integer counter {key}")
        return value

    def bp_to_f32(key: str) -> float:
        return int_counter(key) / 10000.0

    output_tokens = int_counter("output_tokens")
    block_wall_us = int_counter("diffusion_block_wall_us")
    denoise_wall_us = int_counter("diffusion_denoise_wall_us")

    block_decode_tok_s = output_tokens / (block_wall_us / 1_000_000.0) if block_wall_us > 0 else 0.0

    return SweepTrial(
        denoise_steps=int_counter("diffusion_denoise_steps"),
        converged=int_counter("diffusion_converged") > 0,
        converged_strict=int_counter("diffusion_converged_strict") > 0,
        converged_acceptance=int_counter("diffusion_converged_acceptance") > 0,
        converged_plateau=int_counter("diffusion_converged_plateau") > 0,
        min_entropy=bp_to_f32("diffusion_min_entropy_bp"),
        min_acceptance_rate=bp_to_f32("diffusion_min_acceptance_rate_bp"),
        block_decode_tok_s=block_decode_tok_s,
        denoise_wall_us=denoise_wall_us,
        block_wall_us=block_wall_us,
    )


def run_one(
    *,
    bench_bin: Path,
    model_dir: Path,
    prompt_tokens: int,
    grid: GridPoint,
    phase: str,
    index: int,
    log_dir: Path,
) -> SweepTrial:
    tokens = synthetic_tokens(prompt_tokens)
    command = [
        str(bench_bin),
        "generate",
        "--mlx",
        "--model-id",
        "diffusiongemma",
        "--tokens",
        ",".join(str(token) for token in tokens),
        "--max-output-tokens",
        "1",
        "--mlx-model-artifacts-dir",
        str(model_dir),
        "--json",
    ]
    env = dict(os.environ)
    env.update(grid.env_overrides())
    log_path = log_dir / f"p{prompt_tokens}-{phase}-{index}.log"
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"benchmark failed p={prompt_tokens} {grid.label()} {phase}#{index}; "
            f"exit={result.returncode}; log={log_path}"
        )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"benchmark output not JSON for {log_path}: {error}") from error
    return parse_trial(response)


def sweep_grid_point(
    *,
    bench_bin: Path,
    model_dir: Path,
    prompt_tokens: int,
    grid: GridPoint,
    log_dir: Path,
) -> dict[str, Any]:
    # Warmup runs (results discarded).
    for i in range(WARMUP_RUNS):
        run_one(
            bench_bin=bench_bin,
            model_dir=model_dir,
            prompt_tokens=prompt_tokens,
            grid=grid,
            phase="warmup",
            index=i,
            log_dir=log_dir,
        )

    # Measure runs.
    trials: list[SweepTrial] = []
    for i in range(MEASURE_RUNS):
        trial = run_one(
            bench_bin=bench_bin,
            model_dir=model_dir,
            prompt_tokens=prompt_tokens,
            grid=grid,
            phase="measure",
            index=i,
            log_dir=log_dir,
        )
        trials.append(trial)
        log(
            f"  p{prompt_tokens} {grid.label()} measure#{i}: "
            f"steps={trial.denoise_steps} converged={trial.converged} "
            f"tok/s={trial.block_decode_tok_s:.1f}"
        )

    convergence_rate = sum(1 for t in trials if t.converged) / len(trials)

    return {
        "entropy_threshold": grid.entropy_threshold,
        "acceptance_rate_threshold": grid.acceptance_rate_threshold,
        "entropy_plateau_delta": grid.entropy_plateau_delta,
        "denoise_steps": {
            "median": median([float(t.denoise_steps) for t in trials]),
            "min": min(t.denoise_steps for t in trials),
            "max": max(t.denoise_steps for t in trials),
        },
        "convergence_rate": convergence_rate,
        "converged_strict_count": sum(1 for t in trials if t.converged_strict),
        "converged_acceptance_count": sum(1 for t in trials if t.converged_acceptance),
        "converged_plateau_count": sum(1 for t in trials if t.converged_plateau),
        "min_entropy": {
            "median": median([t.min_entropy for t in trials]),
        },
        "min_acceptance_rate": {
            "median": median([t.min_acceptance_rate for t in trials]),
        },
        "block_decode_tok_s": {
            "median": median([t.block_decode_tok_s for t in trials]),
        },
        "block_wall_us": {
            "median": median([float(t.block_wall_us) for t in trials]),
        },
        "measure_runs": len(trials),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-bin",
        type=Path,
        default=REPO_ROOT / "target" / "debug" / "ax-engine-bench",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prompt-tokens",
        nargs="+",
        type=int,
        default=[128, 512, 2048],
    )
    args = parser.parse_args()

    if not args.bench_bin.exists():
        log(f"bench binary not found at {args.bench_bin}; run cargo build first")
        sys.exit(1)
    if not args.model_dir.exists():
        log(f"model dir not found at {args.model_dir}")
        sys.exit(1)

    grid_points = [
        GridPoint(ent, acc, plateau)
        for ent, acc, plateau in itertools.product(
            ENTROPY_THRESHOLDS,
            ACCEPTANCE_RATE_THRESHOLDS,
            ENTROPY_PLATEAU_DELTAS,
        )
    ]
    total = len(grid_points) * len(args.prompt_tokens)
    log(
        f"sweep: {len(grid_points)} grid points × {len(args.prompt_tokens)} prompt "
        f"lengths = {total} configurations ({WARMUP_RUNS}+{MEASURE_RUNS} runs each)"
    )

    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    config_index = 0
    for prompt_tokens in args.prompt_tokens:
        for grid in grid_points:
            config_index += 1
            log(
                f"[{config_index}/{total}] p{prompt_tokens} {grid.label()}"
            )
            row = sweep_grid_point(
                bench_bin=args.bench_bin,
                model_dir=args.model_dir,
                prompt_tokens=prompt_tokens,
                grid=grid,
                log_dir=log_dir,
            )
            row["prompt_tokens"] = prompt_tokens
            results.append(row)

            # Print summary line for this grid point.
            steps = row["denoise_steps"]["median"]
            conv = row["convergence_rate"]
            tok_s = row["block_decode_tok_s"]["median"]
            log(
                f"  → steps={steps:.0f} conv={conv:.0%} tok/s={tok_s:.1f}"
            )

    artifact = {
        "schema": SWEEP_SCHEMA,
        "model_dir": str(args.model_dir),
        "prompt_tokens": args.prompt_tokens,
        "grid_dimensions": {
            "entropy_threshold": ENTROPY_THRESHOLDS,
            "acceptance_rate_threshold": ACCEPTANCE_RATE_THRESHOLDS,
            "entropy_plateau_delta": ENTROPY_PLATEAU_DELTAS,
        },
        "warmup_runs": WARMUP_RUNS,
        "measure_runs": MEASURE_RUNS,
        "results": results,
    }

    output_path = args.output_dir / "sweep_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    log(f"wrote {len(results)} results to {output_path}")

    # Print best configurations per prompt length.
    for pt in args.prompt_tokens:
        pt_results = [r for r in results if r["prompt_tokens"] == pt]
        best = min(pt_results, key=lambda r: r["denoise_steps"]["median"])
        log(
            f"best for p{pt}: steps={best['denoise_steps']['median']:.0f} "
            f"conv={best['convergence_rate']:.0%} "
            f"ent={best['entropy_threshold']} acc={best['acceptance_rate_threshold']} "
            f"plateau={best['entropy_plateau_delta']}"
        )


if __name__ == "__main__":
    main()
