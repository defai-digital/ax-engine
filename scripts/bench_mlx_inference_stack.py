#!/usr/bin/env python3
"""Benchmark AX Engine MLX inference against MLX reference runtimes.

The primary reference is upstream `mlx_lm.benchmark`, not the older SwiftLM
application harness. `mlx-swift-lm` is supported as an optional command adapter
because the reference package exposes libraries and benchmark helpers, but no
repo-stable inference benchmark CLI.

Examples:
  cargo build -p ax-engine-server --release

  python3 scripts/bench_mlx_inference_stack.py \
    --model-dir .internal/models/Qwen3.5-9B-MLX-4bit \
    --prompt-tokens 512,2048 \
    --generation-tokens 128 \
    --repetitions 5 \
    --cooldown 5

Optional mlx-swift-lm adapter:
  python3 scripts/bench_mlx_inference_stack.py \
    --mlx-swift-lm-command './.internal/tools/mlx-swift-bench \
      --model {model} --prompt-tokens {prompt_tokens} \
      --generation-tokens {generation_tokens} --trials {trials} --delay {delay}'

The optional adapter command must print JSON with either:
  {"prompt_tps": 123.4, "generation_tps": 56.7, "peak_memory": 12.3}
or:
  {"prefill_tok_s": 123.4, "decode_tok_s": 56.7, "peak_memory_gb": 12.3}
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
AX_ENGINE_SERVER = REPO_ROOT / "target/release/ax-engine-server"
DEFAULT_MODEL_DIR = REPO_ROOT / ".internal/models/Qwen3.5-9B-MLX-4bit"
DEFAULT_MODEL_ID = str(DEFAULT_MODEL_DIR)
DEFAULT_PROMPT_TOKENS = "512,2048"
DEFAULT_GENERATION_TOKENS = 128
DEFAULT_REPETITIONS = 5
DEFAULT_COOLDOWN = 5.0
AXENGINE_PORT = 8091


def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return "unknown"


def collect_host_metadata() -> dict[str, Any]:
    """Gather Apple Silicon host provenance for benchmark artifact labelling."""
    chip = _sysctl("machdep.cpu.brand_string") or _sysctl("hw.model")
    memory_bytes = _sysctl("hw.memsize")
    try:
        memory_gb = round(int(memory_bytes) / (1024 ** 3))
    except ValueError:
        memory_gb = "unknown"

    os_version = "unknown"
    try:
        os_version = subprocess.check_output(
            ["sw_vers", "-productVersion"], text=True
        ).strip()
    except Exception:
        pass

    return {
        "chip": chip,
        "memory_gb": memory_gb,
        "os_version": os_version,
        "platform": sys.platform,
    }


def collect_build_metadata() -> dict[str, Any]:
    """Collect git commit and build profile for artifact provenance."""
    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass

    return {
        "commit": commit,
        "build_profile": "release",
        "server_binary": str(AX_ENGINE_SERVER),
    }

MLX_AVERAGES_RE = re.compile(
    r"Averages:\s+prompt_tps=(?P<prompt>[0-9.]+),\s+"
    r"generation_tps=(?P<generation>[0-9.]+),\s+"
    r"peak_memory=(?P<memory>[0-9.]+)"
)
MLX_TRIAL_RE = re.compile(
    r"Trial\s+(?P<trial>\d+):\s+"
    r"prompt_tps=(?P<prompt>[0-9.]+),\s+"
    r"generation_tps=(?P<generation>[0-9.]+),\s+"
    r"peak_memory=(?P<memory>[0-9.]+),\s+"
    r"total_time=(?P<total>[0-9.]+)"
)

_cached_tokenizers: dict[str, Any] = {}


def wait_for_server(url: str, timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def kill_proc(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def tokenizer_for(model_dir: Path) -> Any:
    key = str(model_dir)
    if key not in _cached_tokenizers:
        print(f"  [tokenize] loading tokenizer from {model_dir}", file=sys.stderr)
        from mlx_lm import load

        loaded = load(str(model_dir))
        tokenizer = loaded[1]
        _cached_tokenizers[key] = tokenizer
    return _cached_tokenizers[key]


def make_prompt(target_tokens: int) -> str:
    phrase = (
        "AX Engine measures deterministic local inference on Apple Silicon. "
        "The benchmark prompt is intentionally plain and repeatable. "
    )
    chars_needed = max(target_tokens * 5, len(phrase))
    return (phrase * ((chars_needed // len(phrase)) + 2))[:chars_needed]


def tokenize(prompt: str, model_dir: Path) -> list[int]:
    tokenizer = tokenizer_for(model_dir)
    return list(tokenizer.encode(prompt))


def prompt_for_token_count(target_tokens: int, model_dir: Path) -> tuple[str, list[int]]:
    text = make_prompt(target_tokens)
    tokens = tokenize(text, model_dir)
    while len(tokens) < target_tokens:
        text += " " + make_prompt(target_tokens // 2)
        tokens = tokenize(text, model_dir)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
    print(f"  [tokenize] target={target_tokens} actual={len(tokens)}", file=sys.stderr)
    return text, tokens


def parse_mlx_lm_benchmark_output(output: str) -> dict[str, Any]:
    trials = []
    for match in MLX_TRIAL_RE.finditer(output):
        trials.append(
            {
                "trial": int(match.group("trial")),
                "prefill_tok_s": float(match.group("prompt")),
                "decode_tok_s": float(match.group("generation")),
                "peak_memory_gb": float(match.group("memory")),
                "total_time_s": float(match.group("total")),
            }
        )

    averages = MLX_AVERAGES_RE.search(output)
    if not averages:
        raise RuntimeError("mlx_lm.benchmark output did not contain an Averages line")

    return {
        "prefill_tok_s": {"mean": float(averages.group("prompt"))},
        "decode_tok_s": {"mean": float(averages.group("generation"))},
        "peak_memory_gb": {"mean": float(averages.group("memory"))},
        "trials": trials,
    }


def run_mlx_lm_benchmark(
    model: str,
    prompt_tokens: int,
    generation_tokens: int,
    repetitions: int,
    cooldown: float,
    prefill_step_size: int,
) -> dict[str, Any]:
    cmd = [
        "python3",
        "-m",
        "mlx_lm.benchmark",
        "--model",
        model,
        "--prompt-tokens",
        str(prompt_tokens),
        "--generation-tokens",
        str(generation_tokens),
        "--num-trials",
        str(repetitions),
        "--delay",
        str(int(cooldown)),
        "--prefill-step-size",
        str(prefill_step_size),
    ]
    print(f"  [mlx_lm] {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"mlx_lm.benchmark failed with exit={result.returncode}:\n{combined}")
    cell = parse_mlx_lm_benchmark_output(combined)
    cell.update(
        {
            "engine": "mlx_lm",
            "method": "mlx_lm.benchmark",
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
        }
    )
    return cell


def start_axengine(
    binary: Path,
    model_dir: Path,
    port: int,
    *,
    no_speculative: bool,
) -> subprocess.Popen[Any]:
    cmd = [
        str(binary),
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(model_dir),
        "--port",
        str(port),
    ]
    if no_speculative:
        cmd.append("--no-speculative-decode")
    env = {**os.environ, "AX_MLX_NATIVE_CONFIRM": "1"}
    print(f"  [ax-engine] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)


def axengine_one_run(port: int, tokens: list[int], generation_tokens: int) -> dict[str, float]:
    payload = json.dumps(
        {"input_tokens": tokens, "max_output_tokens": generation_tokens}
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    conn.request(
        "POST",
        "/v1/generate/stream",
        body=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(
            f"ax-engine HTTP {response.status}: {response.read(300).decode(errors='replace')}"
        )

    prefill_us: int | None = None
    decode_us = 0
    output_tokens = 0
    current_event = ""

    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
            continue
        if not line.startswith("data:"):
            continue
        try:
            obj = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if current_event == "step":
            runner_us = int(obj.get("step", {}).get("runner_time_us", 0))
            output_tokens = int(obj.get("request", {}).get("output_len", output_tokens))
            if prefill_us is None:
                prefill_us = runner_us
            else:
                decode_us += runner_us
        elif current_event == "response":
            response_tokens = obj.get("response", {}).get("output_tokens", [])
            output_tokens = len(response_tokens) or output_tokens

    conn.close()
    prompt_tokens = len(tokens)
    prefill_s = (prefill_us or 0) / 1_000_000
    decode_s = decode_us / 1_000_000
    measured_decode_tokens = max(output_tokens - 1, 0)
    return {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tok_s": prompt_tokens / prefill_s if prefill_s > 0 else 0.0,
        "decode_tok_s": measured_decode_tokens / decode_s if decode_s > 0 else 0.0,
        "output_tokens": float(output_tokens),
    }


def summarize_runs(runs: list[dict[str, float]], key: str) -> dict[str, float]:
    values = [run[key] for run in runs]
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def bench_axengine(
    port: int,
    tokens: list[int],
    generation_tokens: int,
    repetitions: int,
    cooldown: float,
    *,
    no_speculative: bool = False,
) -> dict[str, Any]:
    engine_key = "ax_engine_mlx_greedy" if no_speculative else "ax_engine_mlx_speculative"
    print(
        f"  [ax-engine/{engine_key}] prompt={len(tokens)} generation={generation_tokens}",
        file=sys.stderr,
    )
    axengine_one_run(port, tokens, generation_tokens)
    if cooldown > 0:
        time.sleep(cooldown)

    runs = []
    for index in range(repetitions):
        run = axengine_one_run(port, tokens, generation_tokens)
        runs.append(run)
        print(
            f"    rep {index + 1}: prefill={run['prefill_tok_s']:.1f} tok/s "
            f"decode={run['decode_tok_s']:.1f} tok/s out={run['output_tokens']:.0f}",
            file=sys.stderr,
        )
        if cooldown > 0 and index < repetitions - 1:
            time.sleep(cooldown)

    return {
        "engine": engine_key,
        "method": "server_sse_runner_time_us",
        "prompt_tokens": len(tokens),
        "generation_tokens": generation_tokens,
        "prefill_tok_s": summarize_runs(runs, "prefill_tok_s"),
        "decode_tok_s": summarize_runs(runs, "decode_tok_s"),
        "prefill_s": summarize_runs(runs, "prefill_s"),
        "decode_s": summarize_runs(runs, "decode_s"),
        "trials": runs,
    }


def parse_swift_adapter_json(stdout: str) -> dict[str, float]:
    payload = json.loads(stdout)
    prefill = payload.get("prefill_tok_s", payload.get("prompt_tps"))
    decode = payload.get("decode_tok_s", payload.get("generation_tps"))
    memory = payload.get("peak_memory_gb", payload.get("peak_memory"))
    if prefill is None or decode is None:
        raise RuntimeError("mlx-swift-lm adapter JSON must include prefill/decode throughput")
    parsed = {
        "prefill_tok_s": float(prefill),
        "decode_tok_s": float(decode),
    }
    if memory is not None:
        parsed["peak_memory_gb"] = float(memory)
    return parsed


def run_mlx_swift_lm_adapter(
    command_template: str,
    model: str,
    prompt_tokens: int,
    generation_tokens: int,
    repetitions: int,
    cooldown: float,
) -> dict[str, Any]:
    command = command_template.format(
        model=model,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        trials=repetitions,
        delay=cooldown,
    )
    argv = shlex.split(command)
    print(f"  [mlx-swift-lm] {command}", file=sys.stderr)
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"mlx-swift-lm adapter failed with exit={result.returncode}:\n"
            f"{result.stdout}{result.stderr}"
        )
    metrics = parse_swift_adapter_json(result.stdout)
    cell: dict[str, Any] = {
        "engine": "mlx_swift_lm",
        "method": "external_json_adapter",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prefill_tok_s": {"median": metrics["prefill_tok_s"]},
        "decode_tok_s": {"median": metrics["decode_tok_s"]},
    }
    if "peak_memory_gb" in metrics:
        cell["peak_memory_gb"] = {"median": metrics["peak_memory_gb"]}
    return cell


def metric_value(cell: dict[str, Any], metric: str) -> float:
    data = cell.get(metric, {})
    if "median" in data:
        return float(data["median"])
    if "mean" in data:
        return float(data["mean"])
    return 0.0


def print_summary(doc: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("AX Engine MLX inference stack benchmark")
    print("=" * 88)
    print(f"{'Engine':<18} {'Prompt tok':>10} {'Prefill tok/s':>14} {'Decode tok/s':>13}  Method")
    print("-" * 88)
    for cell in doc["results"]:
        print(
            f"{cell['engine']:<18} {cell['prompt_tokens']:>10} "
            f"{metric_value(cell, 'prefill_tok_s'):>14.1f} "
            f"{metric_value(cell, 'decode_tok_s'):>13.1f}  {cell['method']}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AX MLX against MLX reference runtimes")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--generation-tokens", type=int, default=DEFAULT_GENERATION_TOKENS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-mlx-lm", action="store_true")
    parser.add_argument("--skip-ax-engine", action="store_true")
    parser.add_argument("--axengine-port", type=int, default=AXENGINE_PORT)
    parser.add_argument("--ax-no-speculative", action="store_true",
                        help="Use greedy (no-spec) decode; emits ax_engine_mlx_greedy in JSON")
    parser.add_argument("--ax-both-modes", action="store_true",
                        help="Run ax-engine twice: once greedy, once speculative. "
                             "Emits both ax_engine_mlx_greedy and ax_engine_mlx_speculative.")
    parser.add_argument(
        "--mlx-swift-lm-command",
        help="Optional command template that returns mlx-swift-lm benchmark JSON",
    )
    args = parser.parse_args()
    if args.model == DEFAULT_MODEL_ID and args.model_dir != DEFAULT_MODEL_DIR:
        args.model = str(args.model_dir)

    if not args.skip_ax_engine and not AX_ENGINE_SERVER.exists():
        print(
            f"ERROR: ax-engine-server not found at {AX_ENGINE_SERVER}. "
            "Run: cargo build -p ax-engine-server --release",
            file=sys.stderr,
        )
        sys.exit(1)

    prompt_lengths = [int(item.strip()) for item in args.prompt_tokens.split(",") if item.strip()]
    print("\n=== AX Engine MLX inference stack ===", file=sys.stderr)
    print(f"  model: {args.model}", file=sys.stderr)
    print(f"  model_dir: {args.model_dir}", file=sys.stderr)
    print(f"  prompt_tokens: {prompt_lengths}", file=sys.stderr)
    print(f"  generation_tokens: {args.generation_tokens}", file=sys.stderr)
    print(f"  repetitions: {args.repetitions} + 1 warmup for AX", file=sys.stderr)

    prompts = []
    if not args.skip_ax_engine:
        prompts = [
            prompt_for_token_count(prompt_tokens, args.model_dir)
            for prompt_tokens in prompt_lengths
        ]

    results: list[dict[str, Any]] = []
    procs: list[subprocess.Popen[Any]] = []
    try:
        if not args.skip_mlx_lm:
            for prompt_tokens in prompt_lengths:
                results.append(
                    run_mlx_lm_benchmark(
                        args.model,
                        prompt_tokens,
                        args.generation_tokens,
                        args.repetitions,
                        args.cooldown,
                        args.prefill_step_size,
                    )
                )

        if args.mlx_swift_lm_command:
            for prompt_tokens in prompt_lengths:
                results.append(
                    run_mlx_swift_lm_adapter(
                        args.mlx_swift_lm_command,
                        args.model,
                        prompt_tokens,
                        args.generation_tokens,
                        args.repetitions,
                        args.cooldown,
                    )
                )

        if not args.skip_ax_engine:
            modes = []
            if args.ax_both_modes:
                modes = [True, False]  # greedy first, then speculative
            else:
                modes = [args.ax_no_speculative]

            for no_spec in modes:
                proc = start_axengine(
                    AX_ENGINE_SERVER,
                    args.model_dir,
                    args.axengine_port,
                    no_speculative=no_spec,
                )
                procs.append(proc)
                if not wait_for_server(f"http://127.0.0.1:{args.axengine_port}/health"):
                    stderr = proc.stderr.read(2000).decode(errors="replace") if proc.stderr else ""
                    raise RuntimeError(f"ax-engine-server did not become ready:\n{stderr}")
                for _, tokens in prompts:
                    results.append(
                        bench_axengine(
                            args.axengine_port,
                            tokens,
                            args.generation_tokens,
                            args.repetitions,
                            args.cooldown,
                            no_speculative=no_spec,
                        )
                    )
                kill_proc(proc)
                procs.remove(proc)
                if no_spec and args.ax_both_modes:
                    time.sleep(3)  # brief cooldown between modes
    finally:
        for proc in procs:
            kill_proc(proc)

    doc = {
        "schema_version": "ax.mlx_inference_stack.v1",
        "host": collect_host_metadata(),
        "build": collect_build_metadata(),
        "model": args.model,
        "model_dir": str(args.model_dir),
        "reference_contract": {
            "primary_reference": "mlx_lm.benchmark",
            "secondary_reference": "mlx-swift-lm external JSON adapter",
            "retired_reference": "SwiftLM application server",
        },
        "prompt_tokens": prompt_lengths,
        "generation_tokens": args.generation_tokens,
        "repetitions": args.repetitions,
        "cooldown": args.cooldown,
        "prefill_step_size": args.prefill_step_size,
        "concurrency": 1,
        "results": results,
    }

    print_summary(doc)
    print(json.dumps(doc, indent=2))
    if args.output:
        args.output.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
