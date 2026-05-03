#!/usr/bin/env python3
"""Benchmark direct backend CLI runs against AX compatibility CLI fallback.

This script compares:
- direct `llama.cpp` CLI invocation vs `ax-bench generate` compatibility CLI fallback
- direct `mlx_lm.generate` CLI invocation vs `ax-bench generate` compatibility CLI fallback

It focuses on end-to-end wall-clock latency for one blocking request, because the
AX compatibility CLI path does not currently surface backend prompt/decode
throughput counters. Direct backend-reported throughput is still captured when
available and recorded as reference context.
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
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
AX_BENCH = REPO_ROOT / "target/release/ax-bench"

DEFAULT_PROMPT = (
    "Write a concise Python function that checks whether a number is prime, "
    "explain the time complexity, and include one short example."
)
DEFAULT_MAX_TOKENS = 64
DEFAULT_REPETITIONS = 5
DEFAULT_COOLDOWN_SECONDS = 20.0
OUTLIERS_REMOVED_PER_SIDE = 1

LLAMA_PERF_RE = re.compile(
    r"Prompt:\s+([0-9.]+)\s+t/s\s+\|\s+Generation:\s+([0-9.]+)\s+t/s"
)
MLX_PROMPT_RE = re.compile(r"Prompt:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec")
MLX_GENERATION_RE = re.compile(
    r"Generation:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec"
)


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

LAST_COMMAND_ENDED_AT: float | None = None


def maybe_cooldown(label: str, cooldown_seconds: float) -> None:
    global LAST_COMMAND_ENDED_AT

    if cooldown_seconds <= 0 or LAST_COMMAND_ENDED_AT is None:
        return

    elapsed_since_last = time.perf_counter() - LAST_COMMAND_ENDED_AT
    remaining = cooldown_seconds - elapsed_since_last
    if remaining <= 0:
        return

    print(
        f"[cooldown] waiting {remaining:.1f}s before {label}",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(remaining)


def run_command(
    command: list[str],
    *,
    label: str,
    cooldown_seconds: float,
) -> dict[str, Any]:
    global LAST_COMMAND_ENDED_AT

    maybe_cooldown(label, cooldown_seconds)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    LAST_COMMAND_ENDED_AT = time.perf_counter()
    combined_output = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return {
        "command": command,
        "elapsed_sec": elapsed,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "combined_output": combined_output,
    }


def ensure_success(result: dict[str, Any], label: str) -> None:
    if result["returncode"] == 0:
        return
    output = result["combined_output"].strip()
    raise RuntimeError(f"{label} failed:\n{output}")


def parse_direct_llama_metrics(output: str) -> dict[str, Any]:
    match = LLAMA_PERF_RE.search(output)
    if not match:
        raise RuntimeError(f"Could not parse llama.cpp perf summary:\n{output}")
    return {
        "prompt_tokens_per_sec": float(match.group(1)),
        "generation_tokens_per_sec": float(match.group(2)),
    }


def parse_direct_mlx_metrics(output: str) -> dict[str, Any]:
    prompt_match = MLX_PROMPT_RE.search(output)
    generation_match = MLX_GENERATION_RE.search(output)
    if not prompt_match or not generation_match:
        raise RuntimeError(f"Could not parse mlx-lm verbose summary:\n{output}")
    return {
        "prompt_tokens_reported": int(prompt_match.group(1)),
        "prompt_tokens_per_sec": float(prompt_match.group(2)),
        "generation_tokens_reported": int(generation_match.group(1)),
        "generation_tokens_per_sec": float(generation_match.group(2)),
    }


def base_direct_llama_command(pair: ModelPair, prompt: str, max_tokens: int) -> list[str]:
    return [
        "llama-cli",
        "--simple-io",
        "--no-display-prompt",
        "--single-turn",
        "--log-disable",
        "--model",
        str(pair.gguf_path),
        "--prompt",
        prompt,
        "--n-predict",
        str(max_tokens),
        "--temp",
        "0",
        "--top-p",
        "1",
        "--top-k",
        "0",
        "--repeat-penalty",
        "1",
        "--seed",
        "1234",
    ]


def direct_llama_perf_command(pair: ModelPair, prompt: str, max_tokens: int) -> list[str]:
    command = base_direct_llama_command(pair, prompt, max_tokens)
    command.insert(5, "--perf")
    return command


def ax_llama_command(pair: ModelPair, prompt: str, max_tokens: int) -> list[str]:
    return [
        str(AX_BENCH),
        "generate",
        "--prompt",
        prompt,
        "--max-output-tokens",
        str(max_tokens),
        "--support-tier",
        "compatibility",
        "--compat-cli-path",
        "llama-cli",
        "--compat-model-path",
        str(pair.gguf_path),
        "--json",
    ]


def direct_mlx_command(
    pair: ModelPair,
    prompt: str,
    max_tokens: int,
    *,
    verbose: bool,
) -> list[str]:
    return [
        "python3",
        "-m",
        "mlx_lm",
        "generate",
        "--model",
        pair.mlx_model_id,
        "--prompt",
        prompt,
        "--ignore-chat-template",
        "--max-tokens",
        str(max_tokens),
        "--temp",
        "0",
        "--top-p",
        "1",
        "--top-k",
        "0",
        "--seed",
        "1234",
        "--verbose",
        "true" if verbose else "false",
    ]


def ax_mlx_command(pair: ModelPair, prompt: str, max_tokens: int) -> list[str]:
    return [
        str(AX_BENCH),
        "generate",
        "--prompt",
        prompt,
        "--max-output-tokens",
        str(max_tokens),
        "--support-tier",
        "compatibility",
        "--compat-backend",
        "mlx",
        "--compat-cli-path",
        "python3",
        "--compat-model-path",
        pair.mlx_model_id,
        "--json",
    ]


def summarize_numeric(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def trim_runs_by_elapsed(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(runs) < 5:
        return runs, []

    indexed_runs = list(enumerate(runs))
    sorted_runs = sorted(indexed_runs, key=lambda item: (item[1]["elapsed_sec"], item[0]))
    low_outliers = sorted_runs[:OUTLIERS_REMOVED_PER_SIDE]
    high_outliers = sorted_runs[-OUTLIERS_REMOVED_PER_SIDE:]

    removed_entries = []
    removed_indexes = set()
    for index, run in low_outliers:
        removed_indexes.add(index)
        removed_entries.append(
            {
                "run_index": index + 1,
                "elapsed_sec": run["elapsed_sec"],
                "metrics": run["metrics"],
                "kind": "low",
            }
        )
    for index, run in high_outliers:
        removed_indexes.add(index)
        removed_entries.append(
            {
                "run_index": index + 1,
                "elapsed_sec": run["elapsed_sec"],
                "metrics": run["metrics"],
                "kind": "high",
            }
        )

    kept_runs = [run for index, run in enumerate(runs) if index not in removed_indexes]
    return kept_runs, removed_entries


def finalize_benchmark_result(command: list[str], repetitions: int, runs: list[dict[str, Any]]) -> dict[str, Any]:
    filtered_runs, removed_runs = trim_runs_by_elapsed(runs)
    raw_elapsed_summary = summarize_numeric([run["elapsed_sec"] for run in runs])
    elapsed_summary = summarize_numeric([run["elapsed_sec"] for run in filtered_runs])
    metric_summary: dict[str, dict[str, float]] = {}
    raw_metric_summary: dict[str, dict[str, float]] = {}
    metric_keys = sorted(
        {key for run in runs for key in run["metrics"].keys()}
    )
    for key in metric_keys:
        raw_metric_summary[key] = summarize_numeric(
            [float(run["metrics"][key]) for run in runs if key in run["metrics"]]
        )
        metric_summary[key] = summarize_numeric(
            [float(run["metrics"][key]) for run in filtered_runs if key in run["metrics"]]
        )

    return {
        "command": command,
        "repetitions": repetitions,
        "runs": runs,
        "filtered_runs": filtered_runs,
        "removed_runs": removed_runs,
        "outlier_policy": {
            "kind": "trim_min_max_elapsed",
            "removed_per_side": OUTLIERS_REMOVED_PER_SIDE,
        },
        "raw_elapsed_sec": raw_elapsed_summary,
        "elapsed_sec": elapsed_summary,
        "raw_metrics": raw_metric_summary,
        "metrics": metric_summary,
    }


def benchmark_command(
    label: str,
    command: list[str],
    parser: Any | None,
    repetitions: int,
    cooldown_seconds: float,
    *,
    warmup: bool = True,
) -> dict[str, Any]:
    runs = []
    if warmup:
        warmup_result = run_command(
            command,
            label=f"{label} warmup",
            cooldown_seconds=cooldown_seconds,
        )
        ensure_success(warmup_result, f"{label} warmup")
    for index in range(repetitions):
        result = run_command(
            command,
            label=f"{label} run {index + 1}",
            cooldown_seconds=cooldown_seconds,
        )
        ensure_success(result, f"{label} run {index + 1}")
        metrics = parser(result["combined_output"]) if parser else {}
        runs.append(
            {
                "run_index": index + 1,
                "elapsed_sec": result["elapsed_sec"],
                "metrics": metrics,
            }
        )

    return finalize_benchmark_result(command, repetitions, runs)


def benchmark_paired_commands(
    left_label: str,
    left_command: list[str],
    left_parser: Any | None,
    right_label: str,
    right_command: list[str],
    right_parser: Any | None,
    repetitions: int,
    cooldown_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    left_warmup = run_command(
        left_command,
        label=f"{left_label} warmup",
        cooldown_seconds=cooldown_seconds,
    )
    ensure_success(left_warmup, f"{left_label} warmup")
    right_warmup = run_command(
        right_command,
        label=f"{right_label} warmup",
        cooldown_seconds=cooldown_seconds,
    )
    ensure_success(right_warmup, f"{right_label} warmup")

    left_runs: list[dict[str, Any]] = []
    right_runs: list[dict[str, Any]] = []

    for round_index in range(repetitions):
        round_number = round_index + 1
        if round_index % 2 == 0:
            schedule = [
                (left_label, left_command, left_parser, left_runs),
                (right_label, right_command, right_parser, right_runs),
            ]
        else:
            schedule = [
                (right_label, right_command, right_parser, right_runs),
                (left_label, left_command, left_parser, left_runs),
            ]

        for order_in_round, (label, command, parser, target_runs) in enumerate(schedule, start=1):
            result = run_command(
                command,
                label=f"{label} run {round_number}",
                cooldown_seconds=cooldown_seconds,
            )
            ensure_success(result, f"{label} run {round_number}")
            metrics = parser(result["combined_output"]) if parser else {}
            target_runs.append(
                {
                    "run_index": round_number,
                    "order_in_round": order_in_round,
                    "elapsed_sec": result["elapsed_sec"],
                    "metrics": metrics,
                }
            )

    return (
        finalize_benchmark_result(left_command, repetitions, left_runs),
        finalize_benchmark_result(right_command, repetitions, right_runs),
    )


def benchmark_pair(
    pair: ModelPair,
    prompt: str,
    max_tokens: int,
    repetitions: int,
    cooldown_seconds: float,
) -> dict[str, Any]:
    print(f"\n=== {pair.label} ===", file=sys.stderr)

    direct_llama, ax_llama = benchmark_paired_commands(
        "direct_llama",
        base_direct_llama_command(pair, prompt, max_tokens),
        None,
        "ax_llama",
        ax_llama_command(pair, prompt, max_tokens),
        None,
        repetitions,
        cooldown_seconds,
    )
    direct_mlx, ax_mlx = benchmark_paired_commands(
        "direct_mlx",
        direct_mlx_command(pair, prompt, max_tokens, verbose=False),
        None,
        "ax_mlx",
        ax_mlx_command(pair, prompt, max_tokens),
        None,
        repetitions,
        cooldown_seconds,
    )

    return {
        "key": pair.key,
        "label": pair.label,
        "prompt": prompt,
        "max_output_tokens": max_tokens,
        "direct_llama": direct_llama,
        "ax_llama": ax_llama,
        "direct_mlx": direct_mlx,
        "ax_mlx": ax_mlx,
        "direct_reference_metrics": {
            "llama": benchmark_command(
                "direct_llama_metrics",
                direct_llama_perf_command(pair, prompt, max_tokens),
                parse_direct_llama_metrics,
                repetitions,
                cooldown_seconds,
            ),
            "mlx": benchmark_command(
                "direct_mlx_metrics",
                direct_mlx_command(pair, prompt, max_tokens, verbose=True),
                parse_direct_mlx_metrics,
                repetitions,
                cooldown_seconds,
            ),
        },
        "delta": {
            "ax_vs_direct_llama_sec": ax_llama["elapsed_sec"]["mean"]
            - direct_llama["elapsed_sec"]["mean"],
            "ax_vs_direct_llama_percent": (
                (ax_llama["elapsed_sec"]["mean"] / direct_llama["elapsed_sec"]["mean"]) - 1.0
            )
            * 100.0,
            "ax_vs_direct_mlx_sec": ax_mlx["elapsed_sec"]["mean"]
            - direct_mlx["elapsed_sec"]["mean"],
            "ax_vs_direct_mlx_percent": (
                (ax_mlx["elapsed_sec"]["mean"] / direct_mlx["elapsed_sec"]["mean"]) - 1.0
            )
            * 100.0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark direct backend CLIs against AX compatibility CLI fallback"
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=list(MODEL_PAIRS.keys()),
        choices=sorted(MODEL_PAIRS.keys()),
        help="Model pairs to benchmark",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help="Minimum idle time enforced between every subprocess invocation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file path for saving benchmark results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not AX_BENCH.is_file():
        raise FileNotFoundError(f"ax-bench binary not found: {AX_BENCH}")

    results = {
        "schema_version": "ax.direct-vs-compat-cli.v1",
        "cooldown_seconds": args.cooldown_seconds,
        "cooldown_policy": "minimum idle time between every subprocess invocation",
        "pairs": [],
    }
    for key in args.pairs:
        pair = MODEL_PAIRS[key]
        if not pair.gguf_path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {pair.gguf_path}")
        results["pairs"].append(
            benchmark_pair(
                pair=pair,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                repetitions=args.repetitions,
                cooldown_seconds=args.cooldown_seconds,
            )
        )

    rendered = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
