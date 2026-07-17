#!/usr/bin/env python3
"""Interleaved A/B for `AX_MLX_AUTO_BUFFER_CAPS` (MoE Metal buffer-cap raise).

Config A = `AX_MLX_AUTO_BUFFER_CAPS=0` (MLX default caps, prior production
behavior); config B = flag unset (auto-raise for MoE-class checkpoints).
Runs decode-trace interleaved A/B pairs per model, enforces greedy
token-checksum parity across every run of a model, and writes a JSON
artifact with per-model medians, ratios, and verdicts.

Promotion gate (ADR-003 D5): MoE models need ratio >= --min-ratio with
parity clean; dense/linear controls must stay within the neutrality band
(no regression beyond --control-tolerance).

Example:
  python3 scripts/ab_mlx_buffer_caps.py \\
    --decode-trace target/release/decode-trace \\
    --model coder-next=<dir> --model a3b=<dir> --control llama=<dir> \\
    --output-dir benchmarks/results/inference/mlx-inference/<date>-buffer-caps-ab
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

FLAG = "AX_MLX_AUTO_BUFFER_CAPS"
TOKS_RE = re.compile(r"decode tok/s = ([0-9.]+)")
CHECKSUM_RE = re.compile(r"token stream: n=(\d+) fnv1a64=([0-9a-f]{16})")


def run_once(binary: Path, model_dir: Path, steps: int, auto_on: bool) -> dict:
    env = dict(os.environ)
    env.pop("MLX_MAX_MB_PER_BUFFER", None)
    env.pop("MLX_MAX_OPS_PER_BUFFER", None)
    if auto_on:
        env.pop(FLAG, None)
    else:
        env[FLAG] = "0"
    proc = subprocess.run(
        [str(binary), str(model_dir), str(steps)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"decode-trace failed (auto_on={auto_on}): rc={proc.returncode}\n"
            f"stderr tail: {proc.stderr[-600:]}"
        )
    toks = TOKS_RE.search(proc.stdout)
    checksum = CHECKSUM_RE.search(proc.stdout)
    if not (toks and checksum):
        raise RuntimeError(f"unparseable output:\n{proc.stdout[-800:]}")
    return {
        "auto_on": auto_on,
        "tok_s": float(toks.group(1)),
        "token_fnv1a64": checksum.group(2),
        "stdout": proc.stdout,
    }


def parse_model_arg(value: str) -> tuple[str, Path]:
    name, _, path = value.partition("=")
    if not path:
        raise argparse.ArgumentTypeError("expected NAME=DIR")
    return name, Path(path).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-trace", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        type=parse_model_arg,
        help="NAME=DIR of an MoE checkpoint expected to benefit (gate: >= --min-ratio)",
    )
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        type=parse_model_arg,
        help="NAME=DIR of a dense/linear checkpoint expected to be unaffected",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--control-reps", type=int, default=3)
    parser.add_argument("--cooldown-s", type=float, default=10.0)
    parser.add_argument("--min-ratio", type=float, default=1.01)
    parser.add_argument("--control-tolerance", type=float, default=0.98)
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "flag": FLAG,
        "steps": args.steps,
        "min_ratio": args.min_ratio,
        "control_tolerance": args.control_tolerance,
        "models": {},
    }
    failures: list[str] = []

    plans = [(n, d, args.reps, True) for n, d in args.model] + [
        (n, d, args.control_reps, False) for n, d in args.control
    ]
    for name, model_dir, reps, is_target in plans:
        runs = []
        for rep in range(reps):
            for auto_on in (False, True):
                run = run_once(args.decode_trace, model_dir, args.steps, auto_on)
                log = f"{name}-rep{rep}-{'on' if auto_on else 'off'}.txt"
                (out_dir / log).write_text(run.pop("stdout"))
                run.update(rep=rep, log=log)
                runs.append(run)
                print(
                    f"{name} rep{rep} auto={int(auto_on)}: {run['tok_s']:.2f} tok/s "
                    f"sum={run['token_fnv1a64']}",
                    flush=True,
                )
                time.sleep(args.cooldown_s)
        off_med = statistics.median(r["tok_s"] for r in runs if not r["auto_on"])
        on_med = statistics.median(r["tok_s"] for r in runs if r["auto_on"])
        ratio = on_med / off_med if off_med else 0.0
        checksums = {r["token_fnv1a64"] for r in runs}
        parity = len(checksums) == 1
        if not parity:
            failures.append(f"{name}: token checksums diverge: {sorted(checksums)}")
        if is_target and ratio < args.min_ratio:
            failures.append(f"{name}: ratio {ratio:.4f} < gate {args.min_ratio}")
        if not is_target and ratio < args.control_tolerance:
            failures.append(
                f"{name} (control): ratio {ratio:.4f} < tolerance {args.control_tolerance}"
            )
        result["models"][name] = {
            "role": "target" if is_target else "control",
            "off_tok_s": [r["tok_s"] for r in runs if not r["auto_on"]],
            "on_tok_s": [r["tok_s"] for r in runs if r["auto_on"]],
            "off_median": round(off_med, 2),
            "on_median": round(on_med, 2),
            "ratio": round(ratio, 4),
            "parity": parity,
        }
        print(f"{name}: off {off_med:.2f} | on {on_med:.2f} | ratio {ratio:.4f}", flush=True)

    result["failures"] = failures
    result["verdict"] = "promoted" if not failures else "not_promoted"
    (out_dir / "ab_mlx_buffer_caps.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nverdict: {result['verdict']}")
    for f in failures:
        print(f"  gate failure: {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
