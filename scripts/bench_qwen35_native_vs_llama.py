#!/usr/bin/env python3
"""Benchmark Qwen3.5 GGUF in llama.cpp against AX native artifacts.

The AX native exporter currently materializes GGUF tensors into the native
artifact dtype recorded in model-manifest.json. For Q4_K_M GGUF input this means
the source model is Q4_K_M, but the AX native runtime is not executing llama.cpp's
quantized Q4 kernels unless the manifest tensor dtype is quantized.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GGUF = REPO_ROOT / ".internal/models/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_NATIVE_DIR = Path("/tmp/ax-qwen35-9b-q4-native-check")
DEFAULT_METAL_DIR = REPO_ROOT / "build/metal"
DEFAULT_LLAMA_BENCH = REPO_ROOT / ".internal/reference/llama.cpp/build-metal/bin/llama-bench"
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks/manifests/scenario/chat_qwen_short.json"
DENSE_DEQUANTIZED_EXPORT_NOTE = "source_quantization_dequantized_for_dense_native_export"


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )


def maybe_cooldown(last_finished_at: float | None, cooldown_seconds: float) -> None:
    if last_finished_at is None or cooldown_seconds <= 0:
        return
    remaining = cooldown_seconds - (time.perf_counter() - last_finished_at)
    if remaining > 0:
        time.sleep(remaining)


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode == 0:
        return
    output = "\n".join(part for part in (result.stdout, result.stderr) if part.strip())
    raise RuntimeError(f"{label} failed with exit code {result.returncode}\n{output[-4000:]}")


def median_numeric(runs: list[dict[str, Any]], key: str) -> float | None:
    values = [float(run[key]) for run in runs if run.get(key) is not None]
    return statistics.median(values) if values else None


def format_optional_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def native_artifact_summary(native_dir: Path) -> dict[str, Any]:
    manifest_path = native_dir / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dtype_counts: dict[str, int] = {}
    for tensor in manifest.get("tensors", []):
        dtype = str(tensor.get("dtype", "unknown"))
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    source_quantization = manifest.get("source_quantization") or {}
    runtime_status = manifest.get("runtime_status") or {}
    runtime_notes = runtime_status.get("notes") or []
    dense_dequantized_source = (
        bool(source_quantization.get("contains_quantized_tensors"))
        and DENSE_DEQUANTIZED_EXPORT_NOTE in runtime_notes
    )
    return {
        "manifest_path": str(manifest_path),
        "model_family": manifest.get("model_family"),
        "tensor_count": len(manifest.get("tensors", [])),
        "dtype_counts": dtype_counts,
        "source_quantization": source_quantization or None,
        "runtime_status": runtime_status or None,
        "dense_dequantized_quantized_source": dense_dequantized_source,
        "native_quantized_execution_equivalent": not dense_dequantized_source,
        "attn_output_gate": manifest.get("attn_output_gate", False),
        "partial_rotary_factor": manifest.get("partial_rotary_factor"),
        "linear_attention": manifest.get("linear_attention"),
    }


def run_llama_once(args: argparse.Namespace, run_index: int, run_dir: Path) -> dict[str, Any]:
    output_path = run_dir / f"llama-run-{run_index}.json"
    command = [
        str(args.llama_bench),
        "-m",
        str(args.gguf_path),
        "-p",
        str(args.prompt_tokens),
        "-n",
        str(args.decode_tokens),
        "-r",
        "1",
        "-ngl",
        str(args.llama_gpu_layers),
        "-fa",
        "1" if args.llama_flash_attention else "0",
        "-o",
        "json",
    ]
    started = time.perf_counter()
    result = run_command(command, cwd=REPO_ROOT, timeout_seconds=args.llama_timeout_seconds)
    elapsed = time.perf_counter() - started
    require_success(result, f"llama.cpp run {run_index}")
    output_path.write_text(result.stdout, encoding="utf-8")
    rows = json.loads(result.stdout)
    prefill = next(row for row in rows if row.get("n_prompt", 0) > 0)
    decode = next(row for row in rows if row.get("n_gen", 0) > 0)
    return {
        "run_index": run_index,
        "elapsed_sec": elapsed,
        "prefill_tok_s": float(prefill["avg_ts"]),
        "decode_tok_s": float(decode["avg_ts"]),
        "raw_path": str(output_path),
    }


def run_native_once(args: argparse.Namespace, run_index: int, run_dir: Path) -> dict[str, Any]:
    output_root = run_dir / f"native-run-{run_index}"
    output_root.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / f"native-run-{run_index}.stdout"
    env = os.environ.copy()
    env["AX_ENGINE_NATIVE_MODEL_DIR"] = str(args.native_model_dir)
    env["AX_ENGINE_METAL_BUILD_DIR"] = str(args.native_runtime_artifacts_dir)
    command = [
        str(args.ax_bench),
        "scenario",
        "--manifest",
        str(args.resolved_native_manifest),
        "--output-root",
        str(output_root),
    ]
    started = time.perf_counter()
    result = run_command(command, cwd=REPO_ROOT, env=env, timeout_seconds=args.native_timeout_seconds)
    elapsed = time.perf_counter() - started
    stdout_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    require_success(result, f"AX native run {run_index}")
    result_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    if len(result_dirs) != 1:
        raise RuntimeError(f"expected one native result dir under {output_root}, found {result_dirs}")
    metrics_path = result_dirs[0] / "metrics.json"
    trace_path = result_dirs[0] / "trace.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    observation = trace.get("observation", {})
    actual_prefill_tokens = int(observation.get("prefill_tokens") or 0)
    actual_decode_tokens = int(observation.get("decode_tokens") or 0)
    process_prefill_tok_s = actual_prefill_tokens / elapsed if elapsed > 0 else None
    process_decode_tok_s = (
        actual_decode_tokens / elapsed
        if elapsed > 0 and actual_decode_tokens > 0
        else None
    )
    return {
        "run_index": run_index,
        "elapsed_sec": elapsed,
        "prefill_tok_s": float(metrics["prefill_tok_s"]) if actual_prefill_tokens > 0 else None,
        "decode_tok_s": float(metrics["decode_tok_s"]) if actual_decode_tokens > 0 else None,
        "process_prefill_tok_s": process_prefill_tok_s,
        "process_decode_tok_s": process_decode_tok_s,
        "actual_prefill_tokens": actual_prefill_tokens,
        "actual_decode_tokens": actual_decode_tokens,
        "ttft_ms": metrics.get("ttft_ms"),
        "e2e_latency_ms": metrics.get("e2e_latency_ms"),
        "runner_time_per_token_us": metrics.get("runner_time_per_token_us"),
        "result_dir": str(result_dirs[0]),
        "stdout_path": str(stdout_path),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": runs,
        "median_prefill_tok_s": median_numeric(runs, "prefill_tok_s"),
        "median_decode_tok_s": median_numeric(runs, "decode_tok_s"),
        "median_elapsed_sec": statistics.median(run["elapsed_sec"] for run in runs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-path", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--native-model-dir", type=Path, default=DEFAULT_NATIVE_DIR)
    parser.add_argument("--native-runtime-artifacts-dir", type=Path, default=DEFAULT_METAL_DIR)
    parser.add_argument("--llama-bench", type=Path, default=DEFAULT_LLAMA_BENCH)
    parser.add_argument("--ax-bench", type=Path, default=REPO_ROOT / "target/release/ax-bench")
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/ax-qwen35-native-vs-llama"))
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=192)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cooldown-seconds", type=float, default=20.0)
    parser.add_argument("--llama-gpu-layers", type=int, default=99)
    parser.add_argument("--llama-flash-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llama-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--native-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument(
        "--allow-non-equivalent-comparison",
        action="store_true",
        help=(
            "Emit native-vs-llama speedup ratios even when AX native is executing dense "
            "artifacts exported from a quantized GGUF source."
        ),
    )
    return parser.parse_args()


def prepare_native_manifest(args: argparse.Namespace, run_dir: Path) -> Path:
    manifest = json.loads(args.scenario_manifest.read_text(encoding="utf-8"))
    manifest.setdefault("shape", {})
    manifest["shape"]["input_tokens_target"] = args.prompt_tokens
    manifest["shape"]["output_tokens_target"] = args.decode_tokens
    path = run_dir / "native-scenario.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be greater than zero")
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_root / str(int(time.time()))
    run_dir.mkdir(parents=True, exist_ok=True)
    args.resolved_native_manifest = prepare_native_manifest(args, run_dir)

    native_artifacts = native_artifact_summary(args.native_model_dir)
    native_quantized_execution_equivalent = native_artifacts[
        "native_quantized_execution_equivalent"
    ]
    comparison_contract_status = (
        "comparable"
        if native_quantized_execution_equivalent
        else (
            "non_equivalent_allowed"
            if args.allow_non_equivalent_comparison
            else "blocked_dense_dequantized_native_artifact"
        )
    )
    summary: dict[str, Any] = {
        "schema_version": "ax.qwen35.native_vs_llama.v1",
        "run_dir": str(run_dir),
        "gguf_path": str(args.gguf_path),
        "shape": {
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
        },
        "native_scenario_manifest": str(args.resolved_native_manifest),
        "repetitions": args.repetitions,
        "cooldown_seconds": args.cooldown_seconds,
        "native_artifacts": native_artifacts,
        "comparison_contract": {
            "native_quantized_execution_equivalent": native_quantized_execution_equivalent,
            "allow_non_equivalent_comparison": args.allow_non_equivalent_comparison,
            "status": comparison_contract_status,
        },
        "notes": [
            "AX native artifacts record the executed tensor dtype; Q4_K_M source GGUF does not imply native quantized Q4 execution.",
            "Each llama.cpp repetition is an independent llama-bench process with -r 1.",
            "Each AX native repetition is an independent ax-bench scenario process.",
            "AX native prefill_tok_s/decode_tok_s come from ax-bench runtime metrics and are null when that phase did not execute.",
            "AX native process_prefill_tok_s/process_decode_tok_s include process startup and model loading overhead.",
        ],
    }

    last_finished_at: float | None = None
    if not args.skip_llama:
        llama_runs = []
        for run_index in range(1, args.repetitions + 1):
            maybe_cooldown(last_finished_at, args.cooldown_seconds)
            llama_runs.append(run_llama_once(args, run_index, run_dir))
            last_finished_at = time.perf_counter()
            print(
                f"llama run {run_index}: prefill={llama_runs[-1]['prefill_tok_s']:.3f} "
                f"decode={llama_runs[-1]['decode_tok_s']:.3f}",
                file=sys.stderr,
                flush=True,
            )
        summary["llama_cpp"] = summarize_runs(llama_runs)

    if not args.skip_native:
        native_runs = []
        for run_index in range(1, args.repetitions + 1):
            maybe_cooldown(last_finished_at, args.cooldown_seconds)
            native_runs.append(run_native_once(args, run_index, run_dir))
            last_finished_at = time.perf_counter()
            print(
                f"native run {run_index}: "
                f"prefill={format_optional_rate(native_runs[-1]['prefill_tok_s'])} "
                f"decode={format_optional_rate(native_runs[-1]['decode_tok_s'])}",
                file=sys.stderr,
                flush=True,
            )
        summary["ax_native"] = summarize_runs(native_runs)

    if (
        "llama_cpp" in summary
        and "ax_native" in summary
        and (
            native_quantized_execution_equivalent
            or args.allow_non_equivalent_comparison
        )
    ):
        summary["comparison"] = {
            "contract": comparison_contract_status,
            "prefill_speedup_native_over_llama": (
                summary["ax_native"]["median_prefill_tok_s"]
                / summary["llama_cpp"]["median_prefill_tok_s"]
            )
            if summary["ax_native"]["median_prefill_tok_s"]
            and summary["llama_cpp"]["median_prefill_tok_s"]
            else None,
            "decode_speedup_native_over_llama": (
                summary["ax_native"]["median_decode_tok_s"]
                / summary["llama_cpp"]["median_decode_tok_s"]
            )
            if summary["ax_native"]["median_decode_tok_s"]
            and summary["llama_cpp"]["median_decode_tok_s"]
            else None,
        }
    elif "llama_cpp" in summary and "ax_native" in summary:
        summary["comparison_blocked"] = {
            "reason": "ax_native_dense_dequantized_artifact_is_not_native_q4_quantized_execution",
            "enable_with": "--allow-non-equivalent-comparison",
            "llama_cpp_source": "Q4_K_M GGUF quantized execution",
            "ax_native_execution": "dense native artifact exported from quantized source",
        }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
