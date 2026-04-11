#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

AX_ENV_VARS = [
    "AX_PRECOMPUTE_F16",
    "AX_METAL_F16_KV_CACHE",
    "AX_METAL_ATTN_PROFILE",
    "AX_METAL_FUSED_QKV",
    "AX_METAL_DECODE_FUSED_QKV",
    "AX_METAL_PREFILL_FA2_MODE",
    "AX_METAL_PREFILL_FA2_HD128_MODE",
    "AX_METAL_PREFILL_FA2_SIMD",
    "AX_METAL_PREFILL_BC64_MODE",
    "AX_METAL_DECODE_SPLITK_MODE",
    "AX_METAL_DECODE_SPLITK_AUTO_MIN_TOKENS",
    "AX_METAL_DECODE_SPLITK_CHUNK_SIZE",
    "AX_METAL_DECODE_SDPA",
    "AX_METAL_DECODE_HD128_N2",
    "AX_METAL_CONCURRENT_DECODE",
    "AX_METAL_BARRIERS",
]


@dataclass
class LlamaConfig:
    n_gpu_layers: int = 99
    batch_size: int = 2048
    ubatch_size: int = 512
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    flash_attn: bool = True
    threads: int = 12
    no_warmup: bool = True
    repetitions_per_outer_sample: int = 1


@dataclass
class RunConfig:
    repo_dir: Path
    ax_bench: Path
    llama_bench: Path
    model: Path
    label: str
    prompt_tokens: int
    decode_tokens: int
    decode_depth: int
    samples: int
    cooldown_seconds: int
    out_dir: Path
    timestamp: str
    llama: LlamaConfig
    ax_only: bool = False
    llama_only: bool = False
    llama_baseline: Path | None = None
    ax_baseline: Path | None = None


@dataclass
class LlamaPhaseResult:
    samples_tok_per_s: list[float]
    sample_commands: list[list[str]]


def parse_args() -> RunConfig:
    default_repo_dir = Path(os.getenv("REPO_DIR", Path(__file__).resolve().parent.parent))
    parser = argparse.ArgumentParser(
        description="Run apple-to-apple AX Engine vs llama.cpp benchmark."
    )
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--label", help="short run label")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--decode-depth", type=int)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--cooldown-seconds", type=int, default=20)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument(
        "--ax-only",
        action="store_true",
        help="Run AX Engine only (skip llama.cpp). Use --llama-baseline to compare against a previous llama.cpp result.",
    )
    parser.add_argument(
        "--llama-only",
        action="store_true",
        help="Run llama.cpp only (skip AX Engine). Use --ax-baseline to compare against a previous AX result.",
    )
    parser.add_argument(
        "--ax-baseline",
        help="Path to a previous ax.json to use as the AX baseline (implies --llama-only).",
    )
    parser.add_argument(
        "--llama-baseline",
        help="Path to a previous llama/summary.json to use as the llama.cpp baseline (implies --ax-only).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for benchmark result folders (default: <repo-dir>/benchmarks/results)",
    )
    parser.add_argument(
        "--ax-bench",
        default=None,
        help="AX bench binary path (default: <repo-dir>/target/release/ax-engine-bench)",
    )
    parser.add_argument("--llama-bench", default=None, help="llama-bench binary (default: find via PATH or /opt/homebrew/bin/llama-bench)")
    parser.add_argument(
        "--timestamp",
        help="override datetime prefix for result folder naming (YYYYMMDD-HHMMSS)",
    )
    parser.add_argument(
        "--repo-dir",
        default=str(default_repo_dir),
        help="repository directory (defaults to script location or $REPO_DIR)",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else repo_dir / "benchmarks" / "results"
    ax_bench = Path(args.ax_bench) if args.ax_bench is not None else repo_dir / "target" / "release" / "ax-engine-bench"

    if args.llama_bench is None:
        llama_path = shutil.which("llama-bench")
        args.llama_bench = llama_path or "/opt/homebrew/bin/llama-bench"

    model = Path(args.model)
    decode_depth = args.decode_depth if args.decode_depth is not None else args.prompt_tokens
    label = args.label or model.stem.replace(" ", "-")
    timestamp = args.timestamp or time.strftime("%Y%m%d-%H%M%S")
    ax_only = args.ax_only or args.llama_baseline is not None
    llama_only = args.llama_only or (args.ax_baseline is not None)
    llama_baseline = Path(args.llama_baseline) if args.llama_baseline else None
    ax_baseline = Path(args.ax_baseline) if getattr(args, "ax_baseline", None) else None

    if ax_only and llama_only:
        parser.error("cannot use both --ax-only and --llama-only")

    return RunConfig(
        repo_dir=repo_dir,
        ax_bench=ax_bench,
        llama_bench=Path(args.llama_bench),
        model=model,
        label=label,
        prompt_tokens=args.prompt_tokens,
        decode_tokens=args.decode_tokens,
        decode_depth=decode_depth,
        samples=args.samples,
        cooldown_seconds=args.cooldown_seconds,
        out_dir=out_dir,
        timestamp=timestamp,
        llama=LlamaConfig(threads=args.threads),
        ax_only=ax_only,
        llama_only=llama_only,
        llama_baseline=llama_baseline,
        ax_baseline=ax_baseline,
    )
def ensure_exists(path: Path, executable: bool = False) -> None:
    if executable:
        if not path.exists() or not path.is_file():
            raise SystemExit(f"error: executable not found: {path}")
        if not path.stat().st_mode & 0o111:
            raise SystemExit(f"error: file is not executable: {path}")
    elif not path.exists():
        raise SystemExit(f"error: path not found: {path}")


def allocate_run_dir(out_dir: Path, timestamp: str) -> Path:
    prefix = f"{timestamp}-"
    max_seq = 0
    if out_dir.exists():
        for child in out_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if len(suffix) != 3 or not suffix.isdigit():
                continue
            max_seq = max(max_seq, int(suffix))
    next_seq = max_seq + 1
    return out_dir / f"{timestamp}-{next_seq:03d}"


def run_checked(cmd: list[str], stdout_path: Path | None = None) -> str:
    if stdout_path is None:
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        return completed.stdout
    with stdout_path.open("w", encoding="utf-8") as out:
        completed = subprocess.run(cmd, check=True, text=True, stdout=out, stderr=subprocess.PIPE)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return stdout_path.read_text(encoding="utf-8")


def parse_llama_json_payload(text: str) -> list[dict]:
    start = text.find("[")
    if start == -1:
        raise RuntimeError("no JSON array found in llama-bench output")
    return json.loads(text[start:])


def extract_llama_prefill_tok_per_s(rows: list[dict], prompt_tokens: int) -> float:
    for row in rows:
        if row.get("n_prompt") == prompt_tokens and row.get("n_gen") == 0:
            return float(row["avg_ts"])
    raise RuntimeError("failed to find llama prefill row")


def extract_llama_decode_tok_per_s(rows: list[dict], decode_tokens: int, decode_depth: int) -> float:
    for row in rows:
        if (
            row.get("n_prompt") == 0
            and row.get("n_gen") == decode_tokens
            and row.get("n_depth") == decode_depth
        ):
            return float(row["avg_ts"])
    raise RuntimeError("failed to find llama decode row")


def sleep_if_needed(seconds: int) -> None:
    if seconds > 0:
        time.sleep(seconds)


def run_llama_phase(config: RunConfig, run_dir: Path, phase: str) -> LlamaPhaseResult:
    values: list[float] = []
    commands: list[list[str]] = []
    for sample_idx in range(1, config.samples + 1):
        print(
            f"--- llama.cpp benchmark phase={phase} sample={sample_idx}/{config.samples} ---",
            file=sys.stderr,
        )
        out_json = run_dir / f"{phase}-sample{sample_idx}.json"
        if phase == "prefill":
            cmd = [
                str(config.llama_bench),
                "-m",
                str(config.model),
                "-p",
                str(config.prompt_tokens),
                "-n",
                "0",
                "-d",
                "0",
                "-r",
                str(config.llama.repetitions_per_outer_sample),
            ]
        else:
            cmd = [
                str(config.llama_bench),
                "-m",
                str(config.model),
                "-p",
                "0",
                "-n",
                str(config.decode_tokens),
                "-d",
                str(config.decode_depth),
                "-r",
                str(config.llama.repetitions_per_outer_sample),
            ]
        if config.llama.no_warmup:
            cmd.append("--no-warmup")
        cmd.extend(
            [
                "-ngl",
                str(config.llama.n_gpu_layers),
                "-b",
                str(config.llama.batch_size),
                "-ub",
                str(config.llama.ubatch_size),
                "-ctk",
                config.llama.cache_type_k,
                "-ctv",
                config.llama.cache_type_v,
                "-fa",
                "1" if config.llama.flash_attn else "0",
                "-t",
                str(config.llama.threads),
                "-o",
                "json",
            ]
        )
        commands.append(cmd.copy())
        text = run_checked(cmd, stdout_path=out_json)
        rows = parse_llama_json_payload(text)
        if phase == "prefill":
            value = extract_llama_prefill_tok_per_s(rows, config.prompt_tokens)
        else:
            value = extract_llama_decode_tok_per_s(rows, config.decode_tokens, config.decode_depth)
        values.append(value)
        if sample_idx < config.samples:
            sleep_if_needed(config.cooldown_seconds)
    return LlamaPhaseResult(samples_tok_per_s=values, sample_commands=commands)


def write_llama_summary(
    config: RunConfig,
    run_dir: Path,
    prefill: LlamaPhaseResult,
    decode: LlamaPhaseResult,
) -> Path:
    summary_json = run_dir / "summary.json"
    summary_tsv = run_dir / "summary.tsv"
    summary_md = run_dir / "summary.md"
    payload = {
        "engine": {
            "name": "llama.cpp",
            "binary": str(config.llama_bench),
        },
        "model": str(config.model),
        "prompt_tokens": config.prompt_tokens,
        "decode_tokens": config.decode_tokens,
        "decode_depth": config.decode_depth,
        "samples": config.samples,
        "cooldown_s": config.cooldown_seconds,
        "prefill_median_tok_per_s": statistics.median(prefill.samples_tok_per_s),
        "prefill_mean_tok_per_s": statistics.mean(prefill.samples_tok_per_s),
        "decode_median_tok_per_s": statistics.median(decode.samples_tok_per_s),
        "decode_mean_tok_per_s": statistics.mean(decode.samples_tok_per_s),
        "prefill_samples_tok_per_s": prefill.samples_tok_per_s,
        "decode_samples_tok_per_s": decode.samples_tok_per_s,
        "prefill_sample_commands": prefill.sample_commands,
        "decode_sample_commands": decode.sample_commands,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_tsv.write_text(
        "model\tprompt_tokens\tdecode_tokens\tdecode_depth\tsamples\tcooldown_s\tprefill_median_tok_per_s\tprefill_mean_tok_per_s\tdecode_median_tok_per_s\tdecode_mean_tok_per_s\n"
        f"{config.model}\t{config.prompt_tokens}\t{config.decode_tokens}\t{config.decode_depth}\t{config.samples}\t{config.cooldown_seconds}\t"
        f"{payload['prefill_median_tok_per_s']}\t{payload['prefill_mean_tok_per_s']}\t"
        f"{payload['decode_median_tok_per_s']}\t{payload['decode_mean_tok_per_s']}\n",
        encoding="utf-8",
    )
    summary_md.write_text(
        "# llama.cpp Serial Median Benchmark\n\n"
        f"- Engine: `llama.cpp`\n"
        f"- Binary: `{config.llama_bench}`\n"
        f"- Model: `{config.model}`\n"
        f"- Prompt: `{config.prompt_tokens}`\n"
        f"- Decode: `{config.decode_tokens}` @ depth `{config.decode_depth}`\n"
        f"- Samples: `{config.samples}`\n"
        f"- Cooldown: `{config.cooldown_seconds}s`\n\n"
        "| Phase | Median tok/s | Mean tok/s |\n"
        "|---|---:|---:|\n"
        f"| Prefill | {payload['prefill_median_tok_per_s']:.1f} | {payload['prefill_mean_tok_per_s']:.1f} |\n"
        f"| Decode | {payload['decode_median_tok_per_s']:.1f} | {payload['decode_mean_tok_per_s']:.1f} |\n",
        encoding="utf-8",
    )
    return summary_json


def run_ax(config: RunConfig, run_dir: Path) -> tuple[Path, list[str]]:
    ax_json = run_dir / "ax.json"
    print(
        f"--- AX benchmark model={config.model} prompt={config.prompt_tokens} decode={config.decode_tokens} samples={config.samples} ---",
        file=sys.stderr,
    )
    cmd = [
        str(config.ax_bench),
        "bench",
        "--model",
        str(config.model),
        "--prompt-tokens",
        str(config.prompt_tokens),
        "--decode-tokens",
        str(config.decode_tokens),
        "--deterministic",
        "--samples",
        str(config.samples),
        "--cooldown-ms",
        str(config.cooldown_seconds * 1000),
        "--warmup-iters",
        "0",
        "--measure-iters",
        "1",
        "--llama-parity-preset",
        "--json-output",
        str(ax_json),
    ]
    subprocess.run(cmd, check=True, text=True)
    return ax_json, cmd


def snapshot_env(keys: list[str]) -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in keys}


def write_manifest(
    config: RunConfig,
    run_dir: Path,
    ax_json: Path,
    llama_json: Path,
    ax_command: list[str],
    llama_prefill_commands: list[list[str]],
    llama_decode_commands: list[list[str]],
) -> Path:
    manifest_json = run_dir / "manifest.json"
    payload = {
        "label": config.label,
        "model": str(config.model),
        "run_dir": str(run_dir),
        "engines": {
            "ax": {
                "name": "ax-engine",
                "binary": str(config.ax_bench),
                "command": ax_command,
                "environment": snapshot_env(AX_ENV_VARS),
                "artifact": str(ax_json),
            },
            "llama": {
                "name": "llama.cpp",
                "binary": str(config.llama_bench),
                "prefill_sample_commands": llama_prefill_commands,
                "decode_sample_commands": llama_decode_commands,
                "artifact": str(llama_json),
            },
        },
    }
    manifest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_json


def build_comparison(
    config: RunConfig,
    run_dir: Path,
    ax_json: Path,
    llama_json: Path,
    ax_command: list[str],
) -> None:
    comparison_json = run_dir / "comparison.json"
    comparison_tsv = run_dir / "comparison.tsv"
    comparison_md = run_dir / "comparison.md"

    ax = json.loads(ax_json.read_text(encoding="utf-8"))
    llama = json.loads(llama_json.read_text(encoding="utf-8"))
    ax_prefill = float(ax["prefill_tok_per_sec_median"])
    ax_decode = float(ax["decode_tok_per_sec_median"])
    llama_prefill = float(llama["prefill_median_tok_per_s"])
    llama_decode = float(llama["decode_median_tok_per_s"])

    payload = {
        "label": config.label,
        "model": str(config.model),
        "prompt_tokens": config.prompt_tokens,
        "decode_tokens": config.decode_tokens,
        "decode_depth": config.decode_depth,
        "samples": config.samples,
        "cooldown_seconds": config.cooldown_seconds,
        "methodology": {
            "aggregation": "outer_median",
            "serial": True,
            "ax": {
                "deterministic": True,
                "warmup_iters": 0,
                "measure_iters": 1,
            },
            "llama": asdict(config.llama),
        },
        "ax": {
            "engine": "ax-engine",
            "binary": str(config.ax_bench),
            "command": ax_command,
            "prefill_median_tok_per_s": ax_prefill,
            "decode_median_tok_per_s": ax_decode,
            "prefill_plan": ax.get("prefill_plan"),
            "decode_plan": ax.get("decode_plan"),
            "artifact": str(ax_json),
        },
        "llama": {
            "engine": "llama.cpp",
            "binary": str(config.llama_bench),
            "prefill_median_tok_per_s": llama_prefill,
            "decode_median_tok_per_s": llama_decode,
            "artifact": str(llama_json),
        },
        "ratios": {
            "prefill_percent": ax_prefill / llama_prefill * 100.0 if llama_prefill else None,
            "decode_percent": ax_decode / llama_decode * 100.0 if llama_decode else None,
        },
    }
    comparison_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    comparison_tsv.write_text(
        "label\tmodel\tprompt_tokens\tdecode_tokens\tdecode_depth\tsamples\tcooldown_seconds\tax_prefill_median\tllama_prefill_median\tprefill_percent\tax_decode_median\tllama_decode_median\tdecode_percent\n"
        f"{config.label}\t{config.model}\t{config.prompt_tokens}\t{config.decode_tokens}\t{config.decode_depth}\t{config.samples}\t{config.cooldown_seconds}\t"
        f"{ax_prefill}\t{llama_prefill}\t{payload['ratios']['prefill_percent']}\t"
        f"{ax_decode}\t{llama_decode}\t{payload['ratios']['decode_percent']}\n",
        encoding="utf-8",
    )
    comparison_md.write_text(
        "# Apple-to-Apple Benchmark\n\n"
        f"- Label: `{config.label}`\n"
        f"- Model: `{config.model}`\n"
        f"- Prompt: `{config.prompt_tokens}`\n"
        f"- Decode: `{config.decode_tokens}` @ depth `{config.decode_depth}`\n"
        f"- Samples: `{config.samples}`\n"
        f"- Cooldown: `{config.cooldown_seconds}s`\n"
        "- Method: serial outer-sample medians on the same machine\n\n"
        "| Engine | Prefill tok/s | Decode tok/s |\n"
        "|---|---:|---:|\n"
        f"| AX Engine | {ax_prefill:.1f} | {ax_decode:.1f} |\n"
        f"| llama.cpp | {llama_prefill:.1f} | {llama_decode:.1f} |\n\n"
        "| Ratio | Value |\n"
        "|---|---:|\n"
        f"| AX / llama prefill | {payload['ratios']['prefill_percent']:.1f}% |\n"
        f"| AX / llama decode | {payload['ratios']['decode_percent']:.1f}% |\n\n"
        f"- AX artifact: `{ax_json}`\n"
        f"- llama.cpp artifact: `{llama_json}`\n",
        encoding="utf-8",
    )


def write_ax_only_summary(config: RunConfig, run_dir: Path, ax_json: Path, ax_command: list[str]) -> None:
    """Write summary files when running AX-only (no llama.cpp comparison)."""
    ax = json.loads(ax_json.read_text(encoding="utf-8"))
    ax_prefill = float(ax["prefill_tok_per_sec_median"])
    ax_decode = float(ax["decode_tok_per_sec_median"])
    summary_md = run_dir / "summary.md"
    summary_md.write_text(
        "# AX Engine Benchmark (AX-only)\n\n"
        f"- Label: `{config.label}`\n"
        f"- Model: `{config.model}`\n"
        f"- Prompt: `{config.prompt_tokens}`\n"
        f"- Decode: `{config.decode_tokens}` @ depth `{config.decode_depth}`\n"
        f"- Samples: `{config.samples}`\n"
        f"- Cooldown: `{config.cooldown_seconds}s`\n\n"
        "| Phase | Median tok/s |\n"
        "|---|---:|\n"
        f"| Prefill | {ax_prefill:.1f} |\n"
        f"| Decode | {ax_decode:.1f} |\n\n"
        f"- Prefill plan: `{ax.get('prefill_plan', 'n/a')}`\n"
        f"- Decode plan: `{ax.get('decode_plan', 'n/a')}`\n"
        f"- Prefill CBs: `{ax.get('prefill_command_buffers', 'n/a')}`\n"
        f"- AX artifact: `{ax_json}`\n",
        encoding="utf-8",
    )


def write_llama_only_summary(config: RunConfig, run_dir: Path, llama_json: Path) -> None:
    """Write summary files when running llama-only (no AX comparison)."""
    llama = json.loads(llama_json.read_text(encoding="utf-8"))
    llama_prefill = float(llama["prefill_median_tok_per_s"])
    llama_decode = float(llama["decode_median_tok_per_s"])
    summary_md = run_dir / "summary.md"
    summary_md.write_text(
        "# llama.cpp Benchmark (llama-only)\n\n"
        f"- Label: `{config.label}`\n"
        f"- Model: `{config.model}`\n"
        f"- Prompt: `{config.prompt_tokens}`\n"
        f"- Decode: `{config.decode_tokens}` @ depth `{config.decode_depth}`\n"
        f"- Samples: `{config.samples}`\n"
        f"- Cooldown: `{config.cooldown_seconds}s`\n\n"
        "| Phase | Median tok/s | Mean tok/s |\n"
        "|---|---:|---:|\n"
        f"| Prefill | {llama_prefill:.1f} | {float(llama['prefill_mean_tok_per_s']):.1f} |\n"
        f"| Decode | {llama_decode:.1f} | {float(llama['decode_mean_tok_per_s']):.1f} |\n\n"
        f"- Prefill samples: `{llama['prefill_samples_tok_per_s']}`\n"
        f"- Decode samples: `{llama['decode_samples_tok_per_s']}`\n"
        f"- llama.cpp artifact: `{llama_json}`\n",
        encoding="utf-8",
    )


def main() -> int:
    config = parse_args()
    ensure_exists(config.model)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = allocate_run_dir(config.out_dir, config.timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── llama-only mode ──────────────────────────────────────────────────
    if config.llama_only:
        ensure_exists(config.llama_bench, executable=True)
        llama_run_dir = run_dir / "llama"
        llama_run_dir.mkdir(parents=True, exist_ok=True)
        prefill_result = run_llama_phase(config, llama_run_dir, "prefill")
        sleep_if_needed(config.cooldown_seconds)
        decode_result = run_llama_phase(config, llama_run_dir, "decode")
        llama_summary_json = write_llama_summary(
            config, llama_run_dir, prefill_result, decode_result,
        )

        if config.ax_baseline is not None:
            # Compare against a previous AX result.
            ensure_exists(config.ax_baseline)
            import shutil
            ax_json_local = run_dir / "ax.json"
            shutil.copy2(config.ax_baseline, ax_json_local)
            build_comparison(config, run_dir, ax_json_local, llama_summary_json, [])
            print(f"RUN_DIR:         {run_dir}")
            print(f"LLAMA_JSON:      {llama_summary_json}")
            print(f"AX_JSON:         {ax_json_local} (baseline from {config.ax_baseline})")
            print(f"COMPARISON_JSON: {run_dir / 'comparison.json'}")
            print(f"COMPARISON_MD:   {run_dir / 'comparison.md'}")
        else:
            write_llama_only_summary(config, run_dir, llama_summary_json)
            print(f"RUN_DIR:    {run_dir}")
            print(f"LLAMA_JSON: {llama_summary_json}")
            print(f"SUMMARY_MD: {run_dir / 'summary.md'}")
        return 0

    # ── modes that need AX ───────────────────────────────────────────────
    ensure_exists(config.ax_bench, executable=True)

    ax_json, ax_command = run_ax(config, run_dir)

    if config.ax_only and config.llama_baseline is None:
        # AX-only mode: just report AX results, no comparison.
        write_ax_only_summary(config, run_dir, ax_json, ax_command)
        print(f"RUN_DIR:    {run_dir}")
        print(f"AX_JSON:    {ax_json}")
        print(f"SUMMARY_MD: {run_dir / 'summary.md'}")
        return 0

    if config.llama_baseline is not None:
        # Use a previous llama.cpp result as the baseline — skip running llama.cpp.
        ensure_exists(config.llama_baseline)
        llama_summary_json = config.llama_baseline
        llama_run_dir = run_dir / "llama"
        llama_run_dir.mkdir(parents=True, exist_ok=True)
        # Copy baseline into the run dir for self-contained results.
        import shutil
        shutil.copy2(config.llama_baseline, llama_run_dir / "summary.json")
        llama_summary_json_local = llama_run_dir / "summary.json"
        manifest_json = write_manifest(
            config, run_dir, ax_json, llama_summary_json_local, ax_command, [], [],
        )
        build_comparison(config, run_dir, ax_json, llama_summary_json_local, ax_command)
        print(f"RUN_DIR:         {run_dir}")
        print(f"MANIFEST_JSON:   {manifest_json}")
        print(f"AX_JSON:         {ax_json}")
        print(f"LLAMA_JSON:      {llama_summary_json_local} (baseline from {config.llama_baseline})")
        print(f"COMPARISON_JSON: {run_dir / 'comparison.json'}")
        print(f"COMPARISON_MD:   {run_dir / 'comparison.md'}")
        return 0

    # Full mode: run both AX and llama.cpp.
    ensure_exists(config.llama_bench, executable=True)
    llama_run_dir = run_dir / "llama"
    llama_run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"--- llama.cpp benchmark model={config.model} prompt={config.prompt_tokens} decode={config.decode_tokens} depth={config.decode_depth} samples={config.samples} ---",
        file=sys.stderr,
    )
    prefill_result = run_llama_phase(config, llama_run_dir, "prefill")
    sleep_if_needed(config.cooldown_seconds)
    decode_result = run_llama_phase(config, llama_run_dir, "decode")
    llama_summary_json = write_llama_summary(config, llama_run_dir, prefill_result, decode_result)
    manifest_json = write_manifest(
        config,
        run_dir,
        ax_json,
        llama_summary_json,
        ax_command,
        prefill_result.sample_commands,
        decode_result.sample_commands,
    )
    build_comparison(config, run_dir, ax_json, llama_summary_json, ax_command)

    print(f"RUN_DIR:         {run_dir}")
    print(f"MANIFEST_JSON:   {manifest_json}")
    print(f"AX_JSON:         {ax_json}")
    print(f"LLAMA_JSON:      {llama_summary_json}")
    print(f"COMPARISON_JSON: {run_dir / 'comparison.json'}")
    print(f"COMPARISON_MD:   {run_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
