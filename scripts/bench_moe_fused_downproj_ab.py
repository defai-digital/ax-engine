#!/usr/bin/env python3
"""A/B harness for the fused MoE down-projection GEMV (AX_MLX_MOE_FUSED_DOWNPROJ).

Compares decode throughput and output parity between the flag OFF (standard
gather_qmm + weighted-sum) and ON (fused int4/int8 gather-GEMV) on a real MoE
checkpoint. This is the end-to-end confirmation the microbench (~2-3% decode,
synthetic per-dispatch) and the bit-exact unit tests cannot give on their own.

Two independent phases:
  * PARITY  — `ax-engine generate --deterministic` greedy decode with the flag
              OFF then ON; the output token ids MUST be identical (the kernel is
              bit-exact-gated, so any divergence is a real bug). Model-reload
              cost is irrelevant here; this checks correctness, not speed.
  * THROUGHPUT — `ax-engine scenario` (model loaded once, decode_tok_s measured
              WITHIN the process so process startup does not pollute it) run
              `--reps` times per flag state; reports median decode_tok_s and the
              speedup. The scenario manifest is deterministic single-request.

Requires a MoE checkpoint (config.json + model-manifest.json + *.safetensors).
Point --model-dir at it (or set AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR).

Examples:
  # Dry run — prints the exact commands without a model (verify the harness):
  python3 scripts/bench_moe_fused_downproj_ab.py --dry-run

  # Real A/B once a MoE checkpoint is downloaded:
  python3 scripts/bench_moe_fused_downproj_ab.py \
      --model-dir ~/.cache/ax-engine/local-artifacts/Qwen3.6-35B-A3B \
      --manifest benchmarks/manifests/scenario/moe_fused_downproj_ab.json \
      --reps 5 --release

Exit code is non-zero if output parity fails or a run errors.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

FLAG = "AX_MLX_MOE_FUSED_DOWNPROJ"
REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "benchmarks/manifests/scenario/moe_fused_downproj_ab.json"
DEFAULT_PARITY_PROMPT = "Explain what a mixture-of-experts model is in three sentences."


def bench_cmd(args: argparse.Namespace, sub_args: list[str]) -> list[str]:
    """Command prefix for the ax-engine bench binary (prebuilt or via cargo)."""
    if args.bin:
        return [args.bin, *sub_args]
    cmd = ["cargo", "run", "-q", "-p", "ax-engine-bench"]
    if args.release:
        cmd.append("--release")
    cmd += ["--", *sub_args]
    return cmd


def env_with_flag(flag_on: bool) -> dict[str, str]:
    env = dict(os.environ)
    env[FLAG] = "1" if flag_on else "0"
    return env


def extract_json(stdout: str) -> dict:
    """Parse the JSON object from bench stdout (tolerates leading log lines)."""
    start = stdout.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in output:\n{stdout[:500]}")
    return json.loads(stdout[start:])


def run_generate(args: argparse.Namespace, flag_on: bool) -> list[int]:
    """Greedy deterministic decode; returns the output token ids."""
    sub = [
        "generate",
        "--mlx",
        "--mlx-model-artifacts-dir", args.model_dir,
        "--prompt", args.parity_prompt,
        "--max-output-tokens", str(args.parity_tokens),
        "--deterministic", "true",
        "--seed", "1234",
        "--json",
    ]
    cmd = bench_cmd(args, sub)
    if args.dry_run:
        print(f"  [dry-run] {FLAG}={'1' if flag_on else '0'} {' '.join(cmd)}")
        return []
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env_with_flag(flag_on),
        cwd=REPO, check=True,
    )
    return extract_json(proc.stdout).get("output_tokens", [])


def run_scenario(args: argparse.Namespace, flag_on: bool) -> dict | None:
    """One scenario run; returns the single run's metrics dict (decode_tok_s, ...)."""
    with tempfile.TemporaryDirectory(prefix="ax-moe-ab-") as out_root:
        sub = ["scenario", "--manifest", args.manifest, "--output-root", out_root]
        cmd = bench_cmd(args, sub)
        if args.dry_run:
            print(f"  [dry-run] {FLAG}={'1' if flag_on else '0'} {' '.join(cmd)}")
            return None
        subprocess.run(
            cmd, capture_output=True, text=True, env=env_with_flag(flag_on),
            cwd=REPO, check=True,
        )
        run_dirs = [p for p in Path(out_root).iterdir() if p.is_dir()]
        if len(run_dirs) != 1:
            raise RuntimeError(f"expected one run dir under {out_root}, found {run_dirs}")
        return json.loads((run_dirs[0] / "metrics.json").read_text())


def parity_phase(args: argparse.Namespace) -> bool:
    print("== PARITY (greedy decode, flag OFF vs ON; tokens must match) ==")
    off = run_generate(args, flag_on=False)
    on = run_generate(args, flag_on=True)
    if args.dry_run:
        return True
    if off == on:
        print(f"  PASS — {len(on)} output tokens identical")
        return True
    # Report the first divergence to aid debugging.
    first = next((i for i, (a, b) in enumerate(zip(off, on)) if a != b), min(len(off), len(on)))
    print(f"  FAIL — diverge at token {first}: OFF={off[max(0, first-2):first+3]} "
          f"ON={on[max(0, first-2):first+3]} (lens {len(off)} vs {len(on)})")
    return False


def throughput_phase(args: argparse.Namespace) -> None:
    print(f"== THROUGHPUT (scenario x{args.reps} per flag; median decode_tok_s) ==")

    def measure(flag_on: bool, label: str) -> dict:
        decode, prefill, ttft = [], [], []
        for i in range(args.reps):
            m = run_scenario(args, flag_on)
            if m is None:
                continue
            mm = m["metrics"]
            decode.append(mm["decode_tok_s"])
            prefill.append(mm.get("prefill_tok_s", 0.0))
            ttft.append(mm.get("ttft_ms", 0.0))
            print(f"  {label} rep {i + 1}/{args.reps}: decode {mm['decode_tok_s']:.2f} tok/s")
        return {
            "decode": statistics.median(decode) if decode else 0.0,
            "prefill": statistics.median(prefill) if prefill else 0.0,
            "ttft": statistics.median(ttft) if ttft else 0.0,
        }

    off = measure(False, "OFF")
    on = measure(True, "ON ")
    if args.dry_run:
        return

    speedup = on["decode"] / off["decode"] if off["decode"] else float("nan")
    print()
    print("== RESULT ==")
    print(f"  decode tok/s : OFF {off['decode']:.2f}  ->  ON {on['decode']:.2f}  "
          f"({(speedup - 1) * 100:+.1f}%, {speedup:.3f}x)")
    print(f"  prefill tok/s: OFF {off['prefill']:.2f}  ->  ON {on['prefill']:.2f}")
    print(f"  ttft ms      : OFF {off['ttft']:.2f}  ->  ON {on['ttft']:.2f}")
    print()
    print(f"  Microbench predicted ~2-3% decode. Promote the flag to default-on "
          f"only if this confirms a real gain AND parity passed.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=os.environ.get("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR"),
                   help="MoE checkpoint dir (default: $AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR)")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="scenario manifest path")
    p.add_argument("--reps", type=int, default=5, help="scenario runs per flag state (median)")
    p.add_argument("--parity-prompt", default=DEFAULT_PARITY_PROMPT)
    p.add_argument("--parity-tokens", type=int, default=64, help="greedy tokens for the parity check")
    p.add_argument("--release", action="store_true", help="build/run the release profile")
    p.add_argument("--bin", help="path to a prebuilt ax-engine binary (skips cargo run)")
    p.add_argument("--skip-parity", action="store_true")
    p.add_argument("--skip-throughput", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print commands; no model needed")
    args = p.parse_args()

    if not args.dry_run and not args.model_dir:
        p.error("--model-dir (or AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR) is required unless --dry-run")
    if args.dry_run and not args.model_dir:
        args.model_dir = "<MODEL_DIR>"
    if not args.dry_run and not Path(args.model_dir).is_dir():
        p.error(f"model dir not found: {args.model_dir}")
    if not Path(args.manifest).is_file():
        p.error(f"manifest not found: {args.manifest}")
    if "REPLACE_WITH_MOE_FAMILY" in Path(args.manifest).read_text() and not args.dry_run:
        print(f"WARNING: {args.manifest} still has the placeholder model.family — set it to "
              f"match your checkpoint before a real run.", file=sys.stderr)

    print(f"AX MoE fused down-proj A/B  (flag: {FLAG})")
    print(f"  model:    {args.model_dir or '<dry-run>'}")
    print(f"  manifest: {args.manifest}")
    print()

    parity_ok = True
    if not args.skip_parity:
        parity_ok = parity_phase(args)
        print()
    if not args.skip_throughput:
        throughput_phase(args)

    if not parity_ok:
        print("\nPARITY FAILED — do not enable the flag.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
