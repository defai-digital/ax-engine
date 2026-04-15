#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_COMPLETION_PROMPT = """use std::collections::HashMap;

pub fn normalize_headers(raw: &[(String, String)]) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    for (key, value) in raw {
"""


DEFAULT_COMPLETION_EXTENDED_PROMPT = """use std::collections::HashMap;

pub fn normalize_headers(raw: &[(String, String)]) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    for (key, value) in raw {
        let normalized_key = key.trim().to_ascii_lowercase();
        let normalized_value = value.trim().to_string();
"""


DEFAULT_INFILL_PREFIX = """fn render_user(name: &str) -> String {
    let escaped = html_escape(name);
"""


DEFAULT_INFILL_SUFFIX = """
    format!("<p>{escaped}</p>")
}
"""


DEFAULT_INFILL_ALT_SUFFIX = """
    format!("<span class=\\"user\\">{escaped}</span>")
}
"""


@dataclass
class RunConfig:
    repo_dir: Path
    ax_bench: Path
    model: Path
    label: str
    samples: int
    cooldown_seconds: int
    max_tokens: int
    out_dir: Path
    timestamp: str
    context_length: int | None
    seed: int
    completion_prompt: str
    completion_extended_prompt: str
    infill_prefix: str
    infill_suffix: str
    infill_alt_suffix: str


@dataclass
class Scenario:
    name: str
    args: list[str]
    workload: str


def parse_args() -> RunConfig:
    default_repo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run AX-only workload benchmarks for completion and infill cache scenarios."
    )
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--label", help="short run label")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timestamp", help="override datetime prefix for result folder naming")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for result folders (default: <repo-dir>/benchmarks/results)",
    )
    parser.add_argument(
        "--ax-bench",
        default=None,
        help="AX bench binary path (default: <repo-dir>/target/release/ax-engine-bench)",
    )
    parser.add_argument("--repo-dir", default=str(default_repo_dir))
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--completion-prompt-file", default=None)
    parser.add_argument("--completion-extended-prompt-file", default=None)
    parser.add_argument("--infill-prefix-file", default=None)
    parser.add_argument("--infill-suffix-file", default=None)
    parser.add_argument("--infill-alt-suffix-file", default=None)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else repo_dir / "benchmarks" / "results"
    )
    ax_bench = (
        Path(args.ax_bench)
        if args.ax_bench is not None
        else repo_dir / "target" / "release" / "ax-engine-bench"
    )
    model = Path(args.model)
    label = args.label or model.stem.replace(" ", "-")
    timestamp = args.timestamp or time.strftime("%Y%m%d-%H%M%S")

    return RunConfig(
        repo_dir=repo_dir,
        ax_bench=ax_bench,
        model=model,
        label=label,
        samples=args.samples,
        cooldown_seconds=args.cooldown_seconds,
        max_tokens=args.max_tokens,
        out_dir=out_dir,
        timestamp=timestamp,
        context_length=args.context_length,
        seed=args.seed,
        completion_prompt=read_text(
            args.completion_prompt_file, DEFAULT_COMPLETION_PROMPT
        ),
        completion_extended_prompt=read_text(
            args.completion_extended_prompt_file, DEFAULT_COMPLETION_EXTENDED_PROMPT
        ),
        infill_prefix=read_text(args.infill_prefix_file, DEFAULT_INFILL_PREFIX),
        infill_suffix=read_text(args.infill_suffix_file, DEFAULT_INFILL_SUFFIX),
        infill_alt_suffix=read_text(
            args.infill_alt_suffix_file, DEFAULT_INFILL_ALT_SUFFIX
        ),
    )


def read_text(path: str | None, default: str) -> str:
    if path is None:
        return default
    return Path(path).read_text(encoding="utf-8")


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
    return out_dir / f"{timestamp}-{max_seq + 1:03d}"


def build_scenarios(config: RunConfig) -> list[Scenario]:
    shared = [
        "--model",
        str(config.model),
        "--deterministic",
        "--samples",
        str(config.samples),
        "--measure-iters",
        "1",
        "--warmup-iters",
        "0",
        "--cooldown-ms",
        str(config.cooldown_seconds * 1000),
        "--max-tokens",
        str(config.max_tokens),
        "--seed",
        str(config.seed),
    ]
    if config.context_length is not None:
        shared.extend(["--context-length", str(config.context_length)])

    return [
        Scenario(
            name="completion-cold",
            workload="completion",
            args=shared
            + [
                "--workload",
                "completion",
                "--label",
                "completion-cold",
                "--prompt",
                config.completion_prompt,
            ],
        ),
        Scenario(
            name="completion-warm-same-prompt",
            workload="completion",
            args=shared
            + [
                "--workload",
                "completion",
                "--label",
                "completion-warm-same-prompt",
                "--prompt",
                config.completion_prompt,
                "--prime-prompt",
                config.completion_prompt,
            ],
        ),
        Scenario(
            name="completion-warm-extended-prompt",
            workload="completion",
            args=shared
            + [
                "--workload",
                "completion",
                "--label",
                "completion-warm-extended-prompt",
                "--prompt",
                config.completion_extended_prompt,
                "--prime-prompt",
                config.completion_prompt,
            ],
        ),
        Scenario(
            name="infill-cold",
            workload="infill",
            args=shared
            + [
                "--workload",
                "infill",
                "--label",
                "infill-cold",
                "--prefix",
                config.infill_prefix,
                "--suffix",
                config.infill_suffix,
            ],
        ),
        Scenario(
            name="infill-warm-same-request",
            workload="infill",
            args=shared
            + [
                "--workload",
                "infill",
                "--label",
                "infill-warm-same-request",
                "--prefix",
                config.infill_prefix,
                "--suffix",
                config.infill_suffix,
                "--prime-prefix",
                config.infill_prefix,
                "--prime-suffix",
                config.infill_suffix,
            ],
        ),
        Scenario(
            name="infill-warm-changed-suffix",
            workload="infill",
            args=shared
            + [
                "--workload",
                "infill",
                "--label",
                "infill-warm-changed-suffix",
                "--prefix",
                config.infill_prefix,
                "--suffix",
                config.infill_alt_suffix,
                "--prime-prefix",
                config.infill_prefix,
                "--prime-suffix",
                config.infill_suffix,
            ],
        ),
    ]


def run_scenario(config: RunConfig, run_dir: Path, scenario: Scenario) -> dict | None:
    json_path = run_dir / f"{scenario.name}.json"
    cmd = [
        str(config.ax_bench),
        "workload-bench",
        *scenario.args,
        "--json-output",
        str(json_path),
    ]
    print(f"--- AX workload scenario={scenario.name} ---", file=sys.stderr)
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode == 0:
        return json.loads(json_path.read_text(encoding="utf-8"))
    if scenario.workload == "infill" and "does not support infill" in completed.stderr:
        print(f"SKIP: {scenario.name} (model does not support infill)", file=sys.stderr)
        return None
    raise SystemExit(f"error: scenario failed: {scenario.name}")


def write_manifest(config: RunConfig, run_dir: Path, scenarios: list[Scenario]) -> None:
    manifest = {
        "label": config.label,
        "model": str(config.model),
        "run_dir": str(run_dir),
        "binary": str(config.ax_bench),
        "samples": config.samples,
        "cooldown_seconds": config.cooldown_seconds,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "scenarios": [
            {
                "name": scenario.name,
                "command": [str(config.ax_bench), "workload-bench", *scenario.args],
            }
            for scenario in scenarios
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def write_summary(run_dir: Path, config: RunConfig, results: dict[str, dict]) -> None:
    summary = {
        "label": config.label,
        "model": str(config.model),
        "samples": config.samples,
        "cooldown_seconds": config.cooldown_seconds,
        "max_tokens": config.max_tokens,
        "scenarios": results,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# AX Workload Benchmark",
        "",
        f"- Label: `{config.label}`",
        f"- Model: `{config.model}`",
        f"- Samples: `{config.samples}`",
        f"- Cooldown: `{config.cooldown_seconds}s`",
        f"- Max tokens: `{config.max_tokens}`",
        "",
        "| Scenario | Request ms | First token ms | tok/s | Cache hit | Cached tok | Prompt eval tok |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['request_ms_median']:.2f} | "
            f"{result['first_token_ms_median']:.2f} | "
            f"{result['output_tok_per_sec_median']:.2f} | "
            f"{result['cache_hit_ratio_median'] * 100.0:.1f}% | "
            f"{result['cached_tokens_median']:.1f} | "
            f"{result['prompt_tokens_evaluated_median']:.1f} |"
        )
    lines.append("")
    lines.append("Artifacts:")
    for name in results:
        lines.append(f"- `{name}.json`")

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = parse_args()
    ensure_exists(config.ax_bench, executable=True)
    ensure_exists(config.model)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = allocate_run_dir(config.out_dir, config.timestamp)
    run_dir.mkdir(parents=True, exist_ok=False)

    scenarios = build_scenarios(config)
    results: dict[str, dict] = {}
    for scenario in scenarios:
        result = run_scenario(config, run_dir, scenario)
        if result is None:
            continue
        results[scenario.name] = result

    write_manifest(config, run_dir, scenarios)
    write_summary(run_dir, config, results)
    print(f"Results written to {run_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
