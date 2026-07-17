#!/usr/bin/env python3
"""A/B harness: fused MoE router Metal kernel (`AX_MLX_MOE_ROUTER_FUSED_METAL`).

Runs `decode-trace` on an eligible MoE checkpoint (Qwen3 narrow-softmax router,
`norm_topk_prob=true`, <=1024 experts) with the fused-router flag off and on,
interleaved, and writes a JSON artifact with the promotion verdict.

Gates enforced (ADR-003 D5, `.internal/specs/TECH-SPEC-DECODE-DISPATCH-EFFICIENCY.md`):
  - parity: identical greedy token-stream checksum across every run of both configs
    (plus optional short `AX_MLX_MOE_ROUTER_TRACE` index-trace comparison);
  - route reach: fused runs report `moe router fused: attempts == hits > 0`,
    zero fallbacks; baseline runs report zero attempts;
  - throughput: median fused tok/s / median baseline tok/s >= --min-ratio
    (default 1.01).

Example:
  python3 scripts/ab_moe_router_fused.py \\
    --decode-trace target/release/decode-trace \\
    --model-dir <hf-cache>/models--mlx-community--Qwen3-Coder-Next-4bit/snapshots/<hash> \\
    --output-dir benchmarks/results/inference/mlx-inference/<date>-router-fused-ab
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

FLAG = "AX_MLX_MOE_ROUTER_FUSED_METAL"
TRACE_ENV = "AX_MLX_MOE_ROUTER_TRACE"

TOKS_RE = re.compile(r"decode tok/s = ([0-9.]+)")
CHECKSUM_RE = re.compile(r"token stream: n=(\d+) fnv1a64=([0-9a-f]{16})")
ROUTER_RE = re.compile(r"moe router fused: attempts=(\d+) hits=(\d+) fallbacks=(\d+)")


def run_decode_trace(
    binary: Path,
    model_dir: Path,
    steps: int,
    fused: bool,
    trace_path: Path | None,
) -> dict:
    env = dict(os.environ)
    env[FLAG] = "1" if fused else "0"
    env.pop(TRACE_ENV, None)
    if trace_path is not None:
        env[TRACE_ENV] = str(trace_path)
    started = time.time()
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
            f"decode-trace failed (fused={fused}): rc={proc.returncode}\n"
            f"stdout tail: {proc.stdout[-800:]}\nstderr tail: {proc.stderr[-800:]}"
        )
    out = proc.stdout
    toks = TOKS_RE.search(out)
    checksum = CHECKSUM_RE.search(out)
    router = ROUTER_RE.search(out)
    if not (toks and checksum and router):
        raise RuntimeError(f"unparseable decode-trace output:\n{out[-1200:]}")
    return {
        "fused": fused,
        "steps": steps,
        "tok_s": float(toks.group(1)),
        "token_count": int(checksum.group(1)),
        "token_fnv1a64": checksum.group(2),
        "router_attempts": int(router.group(1)),
        "router_hits": int(router.group(2)),
        "router_fallbacks": int(router.group(3)),
        "wall_s": round(time.time() - started, 1),
        "stdout": out,
    }


def normalized_router_trace(path: Path) -> list[str]:
    """Per router call, compare the top-k selection as a SET.

    `top_k_by_argpartition` returns the top-k unordered while the fused kernel
    returns them max-first (see the kernel's unit test), so raw trace lines
    differ in ordering even when the selection is identical. Sort the indices
    within each `<seq>;<i0>,<i1>,...` line before comparing.
    """
    lines = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        seq, _, idx = line.partition(";")
        indices = sorted(int(v) for v in idx.split(",") if v != "")
        lines.append(f"{seq};{','.join(str(v) for v in indices)}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-trace", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--parity-steps", type=int, default=24)
    parser.add_argument("--cooldown-s", type=float, default=10.0)
    parser.add_argument("--min-ratio", type=float, default=1.01)
    parser.add_argument(
        "--skip-parity-trace",
        action="store_true",
        help="skip the short forced-eval index-trace runs",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "flag": FLAG,
        "model_dir": str(args.model_dir),
        "steps": args.steps,
        "reps": args.reps,
        "min_ratio": args.min_ratio,
        "parity_trace": None,
        "runs": [],
    }
    failures: list[str] = []

    # ---- Phase 1: short forced-eval runs with the router index trace. ----
    if not args.skip_parity_trace:
        traces = {}
        for fused in (False, True):
            trace_file = out_dir / f"router-trace-{'fused' if fused else 'baseline'}.txt"
            trace_file.unlink(missing_ok=True)
            run = run_decode_trace(
                args.decode_trace, args.model_dir, args.parity_steps, fused, trace_file
            )
            traces[fused] = {
                "trace_file": trace_file,
                "checksum": run["token_fnv1a64"],
                "router_attempts": run["router_attempts"],
                "router_hits": run["router_hits"],
                "router_fallbacks": run["router_fallbacks"],
            }
            time.sleep(args.cooldown_s)
        base_trace = normalized_router_trace(traces[False]["trace_file"])
        fused_trace = normalized_router_trace(traces[True]["trace_file"])
        indices_match = base_trace == fused_trace and len(base_trace) > 0
        checksum_match = traces[False]["checksum"] == traces[True]["checksum"]
        result["parity_trace"] = {
            "steps": args.parity_steps,
            "indices_match": indices_match,
            "checksum_match": checksum_match,
            "baseline": {k: str(v) for k, v in traces[False].items()},
            "fused": {k: str(v) for k, v in traces[True].items()},
        }
        if not indices_match:
            failures.append("parity: router top-k index traces differ")
        if not checksum_match:
            failures.append("parity: token checksums differ in trace runs")

    # ---- Phase 2: interleaved timed runs, no trace. ----
    for rep in range(args.reps):
        for fused in (False, True):
            run = run_decode_trace(args.decode_trace, args.model_dir, args.steps, fused, None)
            log_name = f"run-rep{rep}-{'fused' if fused else 'baseline'}.txt"
            (out_dir / log_name).write_text(run.pop("stdout"))
            run["rep"] = rep
            run["log"] = log_name
            result["runs"].append(run)
            print(
                f"rep {rep} fused={int(fused)}: {run['tok_s']:.2f} tok/s "
                f"checksum={run['token_fnv1a64']} "
                f"router attempts={run['router_attempts']} hits={run['router_hits']} "
                f"fallbacks={run['router_fallbacks']}",
                flush=True,
            )
            time.sleep(args.cooldown_s)

    base_runs = [r for r in result["runs"] if not r["fused"]]
    fused_runs = [r for r in result["runs"] if r["fused"]]

    # Parity across every timed run (long-horizon greedy determinism).
    checksums = {r["token_fnv1a64"] for r in result["runs"]}
    if len(checksums) != 1:
        failures.append(f"parity: timed-run checksums diverge: {sorted(checksums)}")

    # Route reach.
    for r in base_runs:
        if r["router_attempts"] != 0:
            failures.append("route: baseline run recorded fused-router attempts")
            break
    for r in fused_runs:
        if r["router_attempts"] == 0 or r["router_attempts"] != r["router_hits"]:
            failures.append(
                f"route: fused run rep {r['rep']} attempts={r['router_attempts']} "
                f"hits={r['router_hits']}"
            )
        if r["router_fallbacks"] != 0:
            failures.append(f"route: fused run rep {r['rep']} fallbacks={r['router_fallbacks']}")

    base_median = statistics.median(r["tok_s"] for r in base_runs)
    fused_median = statistics.median(r["tok_s"] for r in fused_runs)
    ratio = fused_median / base_median if base_median > 0 else 0.0
    result["summary"] = {
        "baseline_tok_s": [r["tok_s"] for r in base_runs],
        "fused_tok_s": [r["tok_s"] for r in fused_runs],
        "baseline_median_tok_s": round(base_median, 2),
        "fused_median_tok_s": round(fused_median, 2),
        "decode_ratio": round(ratio, 4),
    }
    if ratio < args.min_ratio:
        failures.append(f"throughput: ratio {ratio:.4f} < gate {args.min_ratio}")

    result["failures"] = failures
    result["verdict"] = "promoted" if not failures else "not_promoted"
    (out_dir / "ab_moe_router_fused.json").write_text(json.dumps(result, indent=2) + "\n")

    print(
        f"\nbaseline median {base_median:.2f} tok/s | fused median {fused_median:.2f} tok/s "
        f"| ratio {ratio:.4f}"
    )
    print(f"verdict: {result['verdict']}")
    for f in failures:
        print(f"  gate failure: {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
