#!/usr/bin/env python3
"""Large-corpus embedding ingest benchmark for mlx-lm and ax-engine.

This benchmark complements `bench_embedding_fair.py`. The fair benchmark
measures one batch at a time; this script measures sustained ingestion of a
larger deterministic chunk corpus split into repeated batches. It still keeps
the same output contract for both engines: every batch materializes a
contiguous CPU float32 matrix shaped `[B,H]`.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CONTRACT = "contiguous_cpu_f32_batch_hidden"


def load_fair_bench():
    spec = importlib.util.spec_from_file_location(
        "bench_embedding_fair_runtime", SCRIPT_DIR / "bench_embedding_fair.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load bench_embedding_fair.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fair = load_fair_bench()


@dataclass(frozen=True)
class ScaleWorkload:
    name: str
    chunk_tokens: int
    batch_size: int
    total_chunks: int
    batches: list[list[list[int]]]

    @property
    def total_tokens(self) -> int:
        return self.chunk_tokens * self.total_chunks


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def build_chunk_corpus(total_chunks: int, chunk_tokens: int, vocab_size: int) -> list[list[int]]:
    if total_chunks <= 0:
        raise ValueError("total_chunks must be positive")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    return fair.synthetic_batch(chunk_tokens, total_chunks, vocab_size)


def split_batches(corpus: list[list[int]], batch_size: int) -> list[list[list[int]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [corpus[start : start + batch_size] for start in range(0, len(corpus), batch_size)]


def build_scale_workloads(
    model_dir: Path,
    chunk_lengths: list[int],
    batch_sizes: list[int],
    total_chunks: int,
) -> list[ScaleWorkload]:
    vocab_size = fair.model_vocab_size(model_dir)
    workloads = []
    for chunk_tokens in chunk_lengths:
        corpus = build_chunk_corpus(total_chunks, chunk_tokens, vocab_size)
        for batch_size in batch_sizes:
            workloads.append(
                ScaleWorkload(
                    name=f"scale_{total_chunks}x{chunk_tokens}_b{batch_size}",
                    chunk_tokens=chunk_tokens,
                    batch_size=batch_size,
                    total_chunks=total_chunks,
                    batches=split_batches(corpus, batch_size),
                )
            )
    return workloads


def validate_output(
    output: tuple[bytes, int, int],
    expected_batch_size: int,
) -> tuple[int, int]:
    output_bytes, batch_size, hidden_size = output
    if batch_size != expected_batch_size:
        raise RuntimeError(f"batch size mismatch: got {batch_size}, expected {expected_batch_size}")
    expected_bytes = batch_size * hidden_size * 4
    if len(output_bytes) != expected_bytes:
        raise RuntimeError(
            f"output byte length mismatch: got {len(output_bytes)}, expected {expected_bytes}"
        )
    return hidden_size, len(output_bytes)


def run_one_pass(
    step_fn: Callable[[list[list[int]]], tuple[bytes, int, int]],
    workload: ScaleWorkload,
) -> dict[str, Any]:
    batch_latencies_ms = []
    output_bytes = 0
    hidden_size = 0
    started = time.perf_counter()
    for batch in workload.batches:
        batch_started = time.perf_counter()
        output = step_fn(batch)
        batch_elapsed = time.perf_counter() - batch_started
        hidden_size, batch_output_bytes = validate_output(output, len(batch))
        output_bytes += batch_output_bytes
        batch_latencies_ms.append(batch_elapsed * 1000.0)
    elapsed = time.perf_counter() - started
    return {
        "wall_ms": elapsed * 1000.0,
        "tokens_per_sec": workload.total_tokens / elapsed if elapsed > 0 else 0.0,
        "chunks_per_sec": workload.total_chunks / elapsed if elapsed > 0 else 0.0,
        "output_mb_per_sec": (output_bytes / (1024.0 * 1024.0)) / elapsed
        if elapsed > 0
        else 0.0,
        "batch_p50_ms": percentile(batch_latencies_ms, 0.50),
        "batch_p95_ms": percentile(batch_latencies_ms, 0.95),
        "batch_max_ms": max(batch_latencies_ms) if batch_latencies_ms else 0.0,
        "batches": len(workload.batches),
        "hidden_size": hidden_size,
        "output_bytes": output_bytes,
    }


def trial_stats(trials: list[dict[str, Any]], engine: str) -> dict[str, Any]:
    def med(key: str) -> float:
        return statistics.median(float(row[key]) for row in trials) if trials else 0.0

    return {
        "engine": engine,
        "median_wall_ms": med("wall_ms"),
        "median_tokens_per_sec": med("tokens_per_sec"),
        "median_chunks_per_sec": med("chunks_per_sec"),
        "median_output_mb_per_sec": med("output_mb_per_sec"),
        "median_batch_p50_ms": med("batch_p50_ms"),
        "median_batch_p95_ms": med("batch_p95_ms"),
        "median_batch_max_ms": med("batch_max_ms"),
        "trials": trials,
    }


def run_trials(
    engine: str,
    workload: ScaleWorkload,
    step_fn: Callable[[list[list[int]]], tuple[bytes, int, int]],
    warmup: int,
    trials: int,
    cooldown: float,
) -> dict[str, Any]:
    print(f"    [{engine}] warmup x {warmup}", file=sys.stderr)
    for _ in range(warmup):
        run_one_pass(step_fn, workload)
    rows = []
    for idx in range(1, trials + 1):
        if cooldown > 0:
            time.sleep(cooldown)
        row = run_one_pass(step_fn, workload)
        rows.append(row)
        print(
            f"    [{engine}] trial {idx}: "
            f"{row['chunks_per_sec']:.1f} chunks/s  "
            f"{row['tokens_per_sec']:.1f} tok/s  p95={row['batch_p95_ms']:.1f} ms",
            file=sys.stderr,
        )
    return trial_stats(rows, engine)


def compare_results(results: dict[str, Any], reference_key: str) -> dict[str, float]:
    ref = results.get(reference_key)
    ax = results.get("ax_engine_py")
    if not ref or not ax:
        return {}
    ref_tps = float(ref["median_tokens_per_sec"])
    ax_tps = float(ax["median_tokens_per_sec"])
    ref_chunks = float(ref["median_chunks_per_sec"])
    ax_chunks = float(ax["median_chunks_per_sec"])
    return {
        "ax_vs_reference_tokens_pct": ((ax_tps - ref_tps) / ref_tps * 100.0)
        if ref_tps
        else 0.0,
        "ax_vs_reference_chunks_pct": ((ax_chunks - ref_chunks) / ref_chunks * 100.0)
        if ref_chunks
        else 0.0,
    }


def run_model(
    spec,
    chunk_lengths: list[int],
    batch_sizes: list[int],
    total_chunks: int,
    warmup: int,
    trials: int,
    cooldown: float,
    reference: str,
    pooling: str,
    ax_only: bool,
) -> dict[str, Any]:
    model_dir = spec.path.resolve()
    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"model-manifest.json not found: {manifest_path}")
    print(f"[model] {spec.label}: {model_dir}", file=sys.stderr)
    workloads = build_scale_workloads(model_dir, chunk_lengths, batch_sizes, total_chunks)
    if reference == "mlx_embeddings":
        ref_step = None if ax_only else fair.make_mlx_embeddings_step(model_dir)
        ref_key, ref_label, ax_model_id = "mlx_embeddings", "mlx-embeddings", "embeddinggemma"
    else:
        ref_step = None if ax_only else fair.make_mlx_lm_step(model_dir)
        ref_key, ref_label, ax_model_id = "mlx_lm", "mlx-lm", "qwen3"
    ax_step, ax_session = fair.make_ax_engine_step(model_dir, pooling=pooling, model_id=ax_model_id)
    rows = []
    try:
        for workload in workloads:
            print(
                f"  [workload] {workload.name} chunks={workload.total_chunks} "
                f"chunk_tokens={workload.chunk_tokens} batch={workload.batch_size}",
                file=sys.stderr,
            )
            results = {}
            if ref_step is not None:
                results[ref_key] = run_trials(
                    ref_label, workload, ref_step, warmup, trials, cooldown
                )
            results["ax_engine_py"] = run_trials(
                "ax-engine-py", workload, ax_step, warmup, trials, cooldown
            )
            rows.append(
                {
                    "workload": workload.name,
                    "total_chunks": workload.total_chunks,
                    "chunk_tokens": workload.chunk_tokens,
                    "batch_size": workload.batch_size,
                    "batches_per_trial": len(workload.batches),
                    "total_tokens": workload.total_tokens,
                    "results": results,
                    "comparison": compare_results(results, ref_key),
                }
            )
    finally:
        close = getattr(ax_session, "close", None)
        if close is not None:
            close()
    gc.collect()
    return {"model_label": spec.label, "model_dir": str(model_dir), "rows": rows}


def fmt(value: float, digits: int = 1) -> str:
    if value == 0 or math.isfinite(value):
        return f"{value:,.{digits}f}"
    return "nan"


def render_summary(artifact: dict[str, Any]) -> str:
    reference = artifact.get("reference", "mlx_lm")
    ref_key = "mlx_embeddings" if reference == "mlx_embeddings" else "mlx_lm"
    ref_label = "mlx-embeddings" if reference == "mlx_embeddings" else "mlx-lm"
    if artifact.get("ax_only"):
        lines = [
            "# AX-Only Embedding Ingest Scale Benchmark",
            "",
            f"Output contract: `{artifact['output_contract']}`. "
            f"Total chunks per trial: `{artifact['total_chunks']}`.",
            "",
            "| Model | Chunk tokens | Batch | Batches/trial | AX tok/s | AX chunks/s | AX p95 batch ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for model in artifact["models"]:
            for row in model["rows"]:
                ax = row["results"]["ax_engine_py"]
                lines.append(
                    f"| {model['model_label']} | {row['chunk_tokens']} | {row['batch_size']} | "
                    f"{row['batches_per_trial']} | {fmt(ax['median_tokens_per_sec'])} | "
                    f"{fmt(ax['median_chunks_per_sec'])} | {fmt(ax['median_batch_p95_ms'])} |"
                )
        lines.append("")
        return "\n".join(lines)

    lines = [
        "# Embedding Ingest Scale Benchmark",
        "",
        f"Output contract: `{artifact['output_contract']}`. "
        f"Reference: `{ref_label}`. Total chunks per trial: `{artifact['total_chunks']}`.",
        "",
        f"| Model | Chunk tokens | Batch | Batches/trial | {ref_label} tok/s | AX tok/s | AX vs {ref_label} | AX chunks/s | AX p95 batch ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in artifact["models"]:
        for row in model["rows"]:
            ref = row["results"][ref_key]["median_tokens_per_sec"]
            ax = row["results"]["ax_engine_py"]
            delta = row["comparison"]["ax_vs_reference_tokens_pct"]
            lines.append(
                f"| {model['model_label']} | {row['chunk_tokens']} | {row['batch_size']} | "
                f"{row['batches_per_trial']} | {fmt(ref)} | "
                f"{fmt(ax['median_tokens_per_sec'])} | {delta:+.1f}% | "
                f"{fmt(ax['median_chunks_per_sec'])} | {fmt(ax['median_batch_p95_ms'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--batch-sizes", default="8,32,64")
    parser.add_argument("--chunk-tokens", default="256,512")
    parser.add_argument("--total-chunks", type=int, default=512)
    parser.add_argument(
        "--reference",
        choices=["mlx_lm", "mlx_embeddings"],
        default="mlx_lm",
    )
    parser.add_argument("--pooling", choices=["last", "cls", "mean"], default="last")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--ax-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "embedding-scale",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        model_specs = [fair.parse_model_spec(raw) for raw in args.model]
        batch_sizes = fair.parse_csv_ints(args.batch_sizes, name="batch-sizes")
        chunk_lengths = fair.parse_csv_ints(args.chunk_tokens, name="chunk-tokens")
        if args.total_chunks <= 0:
            raise ValueError("total-chunks must be positive")
    except ValueError as error:
        parser.error(str(error))

    run_dir = args.output_dir / datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "ax.embedding_ingest_scale.v1",
        "timestamp": datetime.now().isoformat(),
        "output_contract": OUTPUT_CONTRACT,
        "warmup": args.warmup,
        "trials": args.trials,
        "cooldown_s": args.cooldown,
        "batch_sizes": batch_sizes,
        "chunk_tokens": chunk_lengths,
        "total_chunks": args.total_chunks,
        "reference": args.reference,
        "pooling": args.pooling,
        "ax_only": args.ax_only,
        "models": [],
    }

    for spec in model_specs:
        artifact["models"].append(
            run_model(
                spec,
                chunk_lengths,
                batch_sizes,
                args.total_chunks,
                args.warmup,
                args.trials,
                args.cooldown,
                args.reference,
                args.pooling,
                args.ax_only,
            )
        )

    artifact_path = run_dir / "embedding_ingest_scale.json"
    summary_path = run_dir / "summary.md"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    summary_path.write_text(render_summary(artifact) + "\n")
    print(f"Wrote {artifact_path}", file=sys.stderr)
    print(f"Wrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
