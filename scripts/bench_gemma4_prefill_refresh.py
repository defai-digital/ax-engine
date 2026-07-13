#!/usr/bin/env python3
"""Refresh AX-only Gemma 4 direct prefill/decode rows after prefill work.

Runs gemma-4-12b ffn4 and gemma-4-26b-a4b-4bit with the same contract as the
README high-water AX cells (2 warm + 5 measure, 15s cooldown, gen=128,
--ax-direct, no mlx_lm peer). Optionally enables AX_MLX_PREFILL_PROFILE=1.

Usage:
  python3 scripts/bench_gemma4_prefill_refresh.py
  AX_MLX_PREFILL_PROFILE=1 python3 scripts/bench_gemma4_prefill_refresh.py --profile
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "scripts" / "bench_mlx_inference_stack.py"
HF = Path(os.environ.get("AX_HF_CACHE", "/Volumes/Ext4T/models"))
MODELS = [
    {
        "name": "gemma-4-12b-it-ffn4",
        "repo": "ax-local/gemma-4-12B-it-4bit-ffn4",
        "dir": HF
        / "hub/models--ax-local--gemma-4-12B-it-4bit-ffn4/snapshots/v1",
    },
    {
        "name": "gemma-4-26b-a4b-it-4bit",
        "repo": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "dir": HF
        / "hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/efbeee6e582ebfd06abc9d65e90839c4b5d2116b",
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=REPO
        / "benchmarks/results/inference/mlx-inference/2026-07-12-gemma4-prefill-phase2",
    )
    ap.add_argument("--profile", action="store_true", help="set AX_MLX_PREFILL_PROFILE=1")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.profile:
        env["AX_MLX_PREFILL_PROFILE"] = "1"

    if not args.no_build:
        subprocess.check_call(
            [
                "cargo",
                "build",
                "-p",
                "ax-engine-server",
                "--release",
            ],
            cwd=REPO,
            env={**env, "RUSTFLAGS": env.get("RUSTFLAGS", "-C target-cpu=native")},
        )

    for m in MODELS:
        if not m["dir"].is_dir():
            print(f"skip missing model dir {m['dir']}", file=sys.stderr)
            continue
        out = args.output_root / f"{m['name']}.json"
        cmd = [
            sys.executable,
            str(BENCH),
            "--model",
            m["name"],
            "--model-repo-id",
            m["repo"],
            "--model-dir",
            str(m["dir"]),
            "--hf-cache-root",
            str(HF),
            "--prompt-tokens",
            "128,512,2048",
            "--generation-tokens",
            "128",
            "--seed",
            "0",
            "--repetitions",
            "5",
            "--warmup-repetitions",
            "2",
            "--cooldown",
            "15",
            "--max-load-average",
            "100",
            "--max-top-process-cpu-percent",
            "1000",
            "--ax-direct",
            "--skip-mlx-lm",
            "--no-build-ax-engine",
            "--output",
            str(out),
        ]
        print("running", m["name"], "->", out, flush=True)
        subprocess.check_call(cmd, cwd=REPO, env=env)
    print("done", args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
