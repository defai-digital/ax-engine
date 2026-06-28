#!/usr/bin/env python3
"""Fair in-process embedding benchmark for mlx-lm and ax-engine.

This script measures the model/runtime path only. It intentionally excludes
HTTP, server micro-batching, cold start, and Swift adapters. Both backends
materialize the same output contract: one contiguous CPU float32 matrix shaped
`[batch_size, hidden_size]`.

Use this when publishing README-style embedding throughput claims. Keep
`bench_embedding_models.py` for legacy smoke coverage of the older API mix.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

SHORT_QUERY_SENTENCES = [
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

OUTPUT_CONTRACT = "contiguous_cpu_f32_batch_hidden"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class Workload:
    name: str
    input_kind: str
    batch_size: int
    token_ids: list[list[int]]

    @property
    def token_counts(self) -> list[int]:
        return [len(ids) for ids in self.token_ids]

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)

    @property
    def max_tokens(self) -> int:
        return max(self.token_counts) if self.token_ids else 0


def parse_csv_ints(value: str, *, name: str) -> list[int]:
    if not value.strip():
        return []
    out = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise ValueError(f"{name} values must be positive: {parsed}")
        out.append(parsed)
    if not out:
        raise ValueError(f"{name} must contain at least one positive integer")
    return out


def parse_model_spec(value: str) -> ModelSpec:
    if "=" in value:
        label, raw_path = value.split("=", 1)
    elif ":" in value:
        label, raw_path = value.split(":", 1)
    else:
        path = Path(value).expanduser()
        label = path.name
        raw_path = value
    label = label.strip()
    path = Path(raw_path).expanduser()
    if not label:
        raise ValueError(f"model label is empty in {value!r}")
    return ModelSpec(label=label, path=path)


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def model_vocab_size(model_dir: Path) -> int:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return 151_936
    config = load_json_file(config_path)
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and isinstance(text_config.get("vocab_size"), int):
        return int(text_config["vocab_size"])
    if isinstance(config.get("vocab_size"), int):
        return int(config["vocab_size"])
    return 151_936


def tokenize_short_queries(model_dir: Path) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    eos_id = tokenizer.eos_token_id
    rows = []
    for sentence in SHORT_QUERY_SENTENCES:
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        if eos_id is not None:
            token_ids = token_ids + [int(eos_id)]
        rows.append(token_ids)
    return rows


def cycle_batch(rows: list[list[int]], batch_size: int) -> list[list[int]]:
    if not rows:
        raise ValueError("cannot build a batch from an empty row set")
    return [list(rows[i % len(rows)]) for i in range(batch_size)]


def synthetic_batch(length: int, batch_size: int, vocab_size: int) -> list[list[int]]:
    upper = max(vocab_size - 1, 2)
    rows = []
    for row_idx in range(batch_size):
        start = 1 + row_idx * 997
        rows.append([1 + ((start + pos) % upper) for pos in range(length)])
    return rows


def build_workloads(
    model_dir: Path,
    batch_sizes: list[int],
    fixed_lengths: list[int],
    include_short_query: bool,
) -> list[Workload]:
    workloads: list[Workload] = []
    if include_short_query:
        short_rows = tokenize_short_queries(model_dir)
        for batch_size in batch_sizes:
            workloads.append(
                Workload(
                    name=f"short_query_b{batch_size}",
                    input_kind="short_query_text",
                    batch_size=batch_size,
                    token_ids=cycle_batch(short_rows, batch_size),
                )
            )
    vocab_size = model_vocab_size(model_dir)
    for length in fixed_lengths:
        for batch_size in batch_sizes:
            workloads.append(
                Workload(
                    name=f"fixed_{length}_b{batch_size}",
                    input_kind="synthetic_token_ids",
                    batch_size=batch_size,
                    token_ids=synthetic_batch(length, batch_size, vocab_size),
                )
            )
    return workloads


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def stddev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def trial_stats(trials: list[dict[str, float]], engine: str) -> dict[str, Any]:
    ms = [row["ms_per_batch"] for row in trials]
    tokens = [row["tokens_per_sec"] for row in trials]
    items = [row["items_per_sec"] for row in trials]
    return {
        "engine": engine,
        "median_ms_per_batch": median(ms),
        "mean_ms_per_batch": mean(ms),
        "stddev_ms_per_batch": stddev(ms),
        "median_tokens_per_sec": median(tokens),
        "mean_tokens_per_sec": mean(tokens),
        "median_items_per_sec": median(items),
        "mean_items_per_sec": mean(items),
        "trials": trials,
    }


def benchmark_step(fn, workload: Workload) -> dict[str, float]:
    started = time.perf_counter()
    output_bytes, batch_size, hidden_size = fn(workload.token_ids)
    elapsed = time.perf_counter() - started
    expected_bytes = batch_size * hidden_size * 4
    if len(output_bytes) != expected_bytes:
        raise RuntimeError(
            f"output byte length mismatch: got {len(output_bytes)}, expected {expected_bytes}"
        )
    return {
        "ms_per_batch": elapsed * 1000.0,
        "ms_per_item": elapsed * 1000.0 / workload.batch_size,
        "tokens_per_sec": workload.total_tokens / elapsed if elapsed > 0 else 0.0,
        "items_per_sec": workload.batch_size / elapsed if elapsed > 0 else 0.0,
        "hidden_size": float(hidden_size),
        "output_bytes": float(len(output_bytes)),
    }


def run_trials(
    engine: str,
    workload: Workload,
    step_fn,
    warmup: int,
    trials: int,
    cooldown: float,
) -> dict[str, Any]:
    print(f"    [{engine}] warmup x {warmup}", file=sys.stderr)
    for _ in range(warmup):
        benchmark_step(step_fn, workload)
    rows = []
    for idx in range(1, trials + 1):
        if cooldown > 0:
            time.sleep(cooldown)
        row = benchmark_step(step_fn, workload)
        rows.append(row)
        print(
            f"    [{engine}] trial {idx}: "
            f"{row['ms_per_item']:.2f} ms/item  {row['tokens_per_sec']:.1f} tok/s",
            file=sys.stderr,
        )
    return trial_stats(rows, engine)


def make_mlx_lm_step(model_dir: Path):
    print(f"  [mlx-lm] loading {model_dir}", file=sys.stderr)
    from mlx_lm import load
    import mlx.core as mx
    import numpy as np
    from transformers import AutoTokenizer

    model, _ = load(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def step(batch: list[list[int]]) -> tuple[bytes, int, int]:
        max_len = max(len(ids) for ids in batch)
        padded = [ids + [int(pad_id)] * (max_len - len(ids)) for ids in batch]
        last_positions = [len(ids) - 1 for ids in batch]
        x = mx.array(padded)
        hidden = model.model(x)
        rows = []
        for row_idx, position in enumerate(last_positions):
            last = hidden[row_idx, position, :].astype(mx.float32)
            norm = mx.sqrt(mx.sum(last * last))
            rows.append(last / (norm + 1e-12))
        matrix = mx.stack(rows, 0)
        array = np.array(matrix, dtype=np.float32, copy=True)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return array.tobytes(), int(array.shape[0]), int(array.shape[1])

    return step


def make_ax_engine_step(model_dir: Path):
    print(f"  [ax-engine-py] loading {model_dir}", file=sys.stderr)
    sys.path.insert(0, str(REPO_ROOT / "python"))
    import ax_engine

    session = ax_engine.Session(
        model_id="qwen3",
        mlx=True,
        support_tier="mlx_preview",
        mlx_model_artifacts_dir=str(model_dir),
    )

    def step(batch: list[list[int]]) -> tuple[bytes, int, int]:
        return session.embed_batch_flat_bytes(batch, pooling="last", normalize=True)

    return step, session


def compare_results(results: dict[str, Any]) -> dict[str, float]:
    mlx = results.get("mlx_lm")
    ax = results.get("ax_engine_py")
    if not mlx or not ax:
        return {}
    mlx_tps = float(mlx["median_tokens_per_sec"])
    ax_tps = float(ax["median_tokens_per_sec"])
    mlx_items = float(mlx["median_items_per_sec"])
    ax_items = float(ax["median_items_per_sec"])
    return {
        "ax_vs_mlx_lm_tokens_pct": ((ax_tps - mlx_tps) / mlx_tps * 100.0)
        if mlx_tps
        else 0.0,
        "ax_vs_mlx_lm_items_pct": ((ax_items - mlx_items) / mlx_items * 100.0)
        if mlx_items
        else 0.0,
    }


def run_model(
    spec: ModelSpec,
    batch_sizes: list[int],
    fixed_lengths: list[int],
    include_short_query: bool,
    warmup: int,
    trials: int,
    cooldown: float,
) -> dict[str, Any]:
    model_dir = spec.path.resolve()
    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"model-manifest.json not found: {manifest_path}")
    print(f"[model] {spec.label}: {model_dir}", file=sys.stderr)
    workloads = build_workloads(model_dir, batch_sizes, fixed_lengths, include_short_query)
    if not workloads:
        raise ValueError("benchmark matrix has no workloads")

    mlx_step = make_mlx_lm_step(model_dir)
    ax_step, ax_session = make_ax_engine_step(model_dir)
    rows = []
    try:
        for workload in workloads:
            print(
                f"  [workload] {workload.name} tokens={workload.token_counts}",
                file=sys.stderr,
            )
            results = {
                "mlx_lm": run_trials("mlx-lm", workload, mlx_step, warmup, trials, cooldown),
                "ax_engine_py": run_trials(
                    "ax-engine-py", workload, ax_step, warmup, trials, cooldown
                ),
            }
            rows.append(
                {
                    "workload": workload.name,
                    "input_kind": workload.input_kind,
                    "batch_size": workload.batch_size,
                    "token_counts": workload.token_counts,
                    "total_tokens": workload.total_tokens,
                    "max_tokens": workload.max_tokens,
                    "results": results,
                    "comparison": compare_results(results),
                }
            )
    finally:
        close = getattr(ax_session, "close", None)
        if close is not None:
            close()
    gc.collect()
    return {
        "model_label": spec.label,
        "model_dir": str(model_dir),
        "rows": rows,
    }


def fmt(value: float, digits: int = 1) -> str:
    if value == 0 or math.isfinite(value):
        return f"{value:,.{digits}f}"
    return "nan"


def render_summary(artifact: dict[str, Any]) -> str:
    lines = [
        "# Fair Embedding Benchmark",
        "",
        f"Output contract: `{artifact['output_contract']}`.",
        "",
        "| Model | Workload | Batch | Max tokens | mlx-lm tok/s | AX tok/s | AX vs mlx-lm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in artifact["models"]:
        for row in model["rows"]:
            mlx = row["results"]["mlx_lm"]["median_tokens_per_sec"]
            ax = row["results"]["ax_engine_py"]["median_tokens_per_sec"]
            delta = row["comparison"]["ax_vs_mlx_lm_tokens_pct"]
            lines.append(
                f"| {model['model_label']} | {row['workload']} | {row['batch_size']} | "
                f"{row['max_tokens']} | {fmt(mlx)} | {fmt(ax)} | {delta:+.1f}% |"
            )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model as label=/hf/snapshot/path, label:/hf/snapshot/path, or /path.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,8,32",
        help="Comma-separated batch sizes. Default: 1,8,32.",
    )
    parser.add_argument(
        "--lengths",
        default="16,64,256",
        help="Comma-separated synthetic token lengths. Use '' for none.",
    )
    parser.add_argument("--skip-short-query", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "embedding-fair",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        model_specs = [parse_model_spec(raw) for raw in args.model]
        batch_sizes = parse_csv_ints(args.batch_sizes, name="batch-sizes")
        fixed_lengths = parse_csv_ints(args.lengths, name="lengths") if args.lengths else []
    except ValueError as error:
        parser.error(str(error))

    run_dir = args.output_dir / datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "ax.embedding_fair.v1",
        "timestamp": datetime.now().isoformat(),
        "output_contract": OUTPUT_CONTRACT,
        "warmup": args.warmup,
        "trials": args.trials,
        "cooldown_s": args.cooldown,
        "batch_sizes": batch_sizes,
        "synthetic_lengths": fixed_lengths,
        "include_short_query": not args.skip_short_query,
        "models": [],
    }

    for spec in model_specs:
        artifact["models"].append(
            run_model(
                spec,
                batch_sizes,
                fixed_lengths,
                not args.skip_short_query,
                args.warmup,
                args.trials,
                args.cooldown,
            )
        )

    artifact_path = run_dir / "embedding_fair.json"
    summary_path = run_dir / "summary.md"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    summary_path.write_text(render_summary(artifact))
    print(f"Wrote {artifact_path}", file=sys.stderr)
    print(f"Wrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
