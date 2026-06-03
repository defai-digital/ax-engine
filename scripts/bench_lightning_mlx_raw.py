#!/usr/bin/env python3
"""Run source-derived lightning-mlx raw decode benchmarks.

This wrapper invokes the real ``vllm_mlx.cli bench`` entrypoint from the
checked-in lightning-mlx reference source.  It intentionally stays separate
from the OpenAI server prompt-suite adapter because lightning-mlx uses
different settings for raw throughput and for serving/agentic correctness.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIGHTNING_SOURCE = REPO_ROOT / ".internal" / "reference" / "lightning-mlx"
DEFAULT_PYTHON = Path("/opt/homebrew/var/mtplx/venv-0.3.7/bin/python")
DEFAULT_OUTPUT_BASE = REPO_ROOT / "benchmarks" / "results" / "mtp-fair"
HF_CACHE = Path(
    os.environ.get("HF_HUB_CACHE")
    or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
)

MODEL_ALIASES = {
    "27b": "qwen3.6-27b",
    "35b": "qwen3.6-35b",
}

MODE_NAMES = {
    "mtp": "Lightning MTP",
    "mtp-ngram": "Lightning MTP+n-gram",
}

FLOAT_LINE_RE = re.compile(r"^\s*(?P<label>[A-Za-z/ ]+):\s*(?P<value>[0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class BenchTarget:
    model_key: str
    mode: str
    alias: str

    @property
    def artifact_name(self) -> str:
        return f"{self.model_key}-{self.mode}.json"


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def load_aliases(lightning_source: Path) -> dict[str, str]:
    path = lightning_source / "vllm_mlx" / "aliases.json"
    return json.loads(path.read_text())


def hf_cache_dir(model_id: str, hf_cache: Path) -> Path:
    return hf_cache / f"models--{model_id.replace('/', '--')}"


def build_bench_command(
    *,
    python: Path,
    lightning_source: Path,
    model: str,
    mode: str,
    num_prompts: int,
    max_tokens: int,
) -> list[str]:
    cmd = [
        str(python),
        "-m",
        "vllm_mlx.cli",
        "bench",
        model,
        "--num-prompts",
        str(num_prompts),
        "--max-tokens",
        str(max_tokens),
        "--disable-prefix-cache",
        "--max-num-seqs",
        "1",
        "--prefill-batch-size",
        "1",
        "--completion-batch-size",
        "1",
        "--prefill-step-size",
        "8192",
        "--mtp-num-draft-tokens",
        "3",
        "--mtp-optimistic",
    ]
    if mode == "mtp-ngram":
        cmd += [
            "--enable-ngram",
            "--ngram-num-draft-tokens",
            "6",
            "--ngram-min-occurrences",
            "2",
            "--ngram-acceptance-mode",
            "greedy",
            "--ngram-hybrid-verify",
            "--ngram-everywhere",
            "--ngram-skip-tool-calls",
            "--ngram-self-tune",
            "--ngram-self-tune-disable-threshold",
            "0.30",
            "--ngram-auto-disable-mtp-threshold",
            "0.85",
            "--ngram-auto-disable-min-ngram",
            "0.50",
        ]
    return cmd


def parse_bench_stdout(text: str) -> dict[str, float | None]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = FLOAT_LINE_RE.match(line)
        if not match:
            continue
        key = match.group("label").strip().lower().replace("/", "_").replace(" ", "_")
        values[key] = float(match.group("value"))
    return {
        "total_time_s": values.get("total_time"),
        "prompts": values.get("prompts"),
        "prompts_per_second": values.get("prompts_second"),
        "total_prompt_tokens": values.get("total_prompt_tokens"),
        "total_completion_tokens": values.get("total_completion_tokens"),
        "total_tokens": values.get("total_tokens"),
        "completion_tokens_per_second": values.get("tokens_second"),
        "throughput_tok_s": values.get("throughput"),
    }


def run_target(
    *,
    target: BenchTarget,
    python: Path,
    lightning_source: Path,
    resolved_model: str,
    output_dir: Path,
    num_prompts: int,
    max_tokens: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / target.artifact_name
    cmd = build_bench_command(
        python=python,
        lightning_source=lightning_source,
        model=target.alias,
        mode=target.mode,
        num_prompts=num_prompts,
        max_tokens=max_tokens,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(lightning_source.resolve()), env["PYTHONPATH"]]
        if env.get("PYTHONPATH")
        else [str(lightning_source.resolve())]
    )
    started = date.today().isoformat()
    try:
        proc = subprocess.run(
            cmd,
            cwd=lightning_source,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        payload: dict[str, Any] = {
            "schema": "ax.lightning_mlx_raw_bench.v1",
            "model_key": target.model_key,
            "model_alias": target.alias,
            "resolved_model": resolved_model,
            "mode": target.mode,
            "mode_label": MODE_NAMES[target.mode],
            "started_at": started,
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "metrics": parse_bench_stdout(proc.stdout),
            "settings": {
                "source": "lightning-mlx README raw decode benchmark",
                "num_prompts": num_prompts,
                "max_tokens": max_tokens,
                "disable_prefix_cache": True,
                "max_num_seqs": 1,
                "prefill_batch_size": 1,
                "completion_batch_size": 1,
                "prefill_step_size": 8192,
                "mtp_num_draft_tokens": 3,
                "mtp_optimistic": True,
                "enable_ngram": target.mode == "mtp-ngram",
                "ngram_profile": "qwen3.6-35b preset" if target.mode == "mtp-ngram" else None,
            },
            "build": {
                "host": platform.node(),
                "platform": platform.platform(),
                "git_commit": git_value(["rev-parse", "HEAD"]),
                "git_tracked_dirty": bool(git_value(["status", "--porcelain"])),
            },
        }
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
        return artifact_path
    except Exception as exc:
        if not artifact_path.is_file():
            payload = {
                "schema": "ax.lightning_mlx_raw_bench.v1",
                "model_key": target.model_key,
                "model_alias": target.alias,
                "resolved_model": resolved_model,
                "mode": target.mode,
                "command": cmd,
                "error": str(exc),
            }
            artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_ALIASES), default=["27b", "35b"])
    parser.add_argument("--modes", nargs="+", choices=sorted(MODE_NAMES), default=["mtp", "mtp-ngram"])
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--lightning-source", type=Path, default=DEFAULT_LIGHTNING_SOURCE)
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable),
    )
    parser.add_argument("--hf-cache", type=Path, default=HF_CACHE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--require-cached",
        action="store_true",
        help="Fail before running if the resolved Lightning optimized model is not in the HF cache.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = (
        args.output_dir
        or DEFAULT_OUTPUT_BASE / f"{date.today().isoformat()}-qwen36-lightning-raw"
    )
    aliases = load_aliases(args.lightning_source)
    artifacts: list[str] = []
    for model_key in args.models:
        alias = MODEL_ALIASES[model_key]
        resolved = aliases.get(alias, alias)
        if args.require_cached and not hf_cache_dir(resolved, args.hf_cache).is_dir():
            raise FileNotFoundError(
                f"Lightning optimized model is not cached for {alias}: {resolved}"
            )
        for mode in args.modes:
            target = BenchTarget(model_key=model_key, mode=mode, alias=alias)
            path = run_target(
                target=target,
                python=args.python,
                lightning_source=args.lightning_source,
                resolved_model=resolved,
                output_dir=output_dir,
                num_prompts=args.num_prompts,
                max_tokens=args.max_tokens,
            )
            artifacts.append(str(path))
    summary = {"schema": "ax.lightning_mlx_raw_bench_summary.v1", "artifacts": artifacts}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
