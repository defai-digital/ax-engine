#!/usr/bin/env python3
"""Run a direct-mode telemetry pass for graph-level optimization triage.

Collects direct-mode decode telemetry to identify whether host dispatch or
graph construction is material on the target row. This is the precondition
for Track C (graph-level / whole-layer fusion) work.

Output schema: ax.direct_telemetry_pass.v1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ax.direct_telemetry_pass.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Model artifact directory.",
    )
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=128,
        help="Prompt token count for the telemetry pass.",
    )
    p.add_argument(
        "--generation-tokens",
        type=int,
        default=128,
        help="Generation token count for the telemetry pass.",
    )
    p.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of benchmark repetitions.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON artifact to this path instead of stdout.",
    )
    p.add_argument(
        "--bench-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "bench_mlx_inference_stack.py",
        help="Path to bench_mlx_inference_stack.py.",
    )
    return p.parse_args()


def run_telemetry_pass(args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(args.bench_script),
        "--model-dir",
        str(args.model_dir),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--generation-tokens",
        str(args.generation_tokens),
        "--repetitions",
        str(args.repetitions),
        "--ax-direct",
        "--skip-mlx-lm",
        "--skip-llama-cpp",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "timeout",
            "error": "benchmark timed out after 600s",
        }

    if result.returncode != 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "error": f"benchmark failed with exit code {result.returncode}",
            "stderr": result.stderr[:2000],
        }

    try:
        bench_output = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "error": "failed to parse benchmark output",
            "stdout_preview": result.stdout[:2000],
        }

    ax_results = [
        r
        for r in bench_output.get("results", [])
        if str(r.get("engine", "")).startswith("ax_engine")
    ]

    if not ax_results:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "error": "no AX engine results in benchmark output",
        }

    telemetry_summary = extract_telemetry_summary(ax_results)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "model_dir": str(args.model_dir),
        "prompt_tokens": args.prompt_tokens,
        "generation_tokens": args.generation_tokens,
        "repetitions": args.repetitions,
        "telemetry_summary": telemetry_summary,
        "ax_row_count": len(ax_results),
        "bandwidth_accounting": bench_output.get("bandwidth_accounting"),
    }


def extract_telemetry_summary(ax_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_decode_s = 0.0
    total_prefill_s = 0.0
    count = 0

    for r in ax_results:
        decode_s = r.get("decode_s", 0)
        prefill_s = r.get("prefill_s", 0)
        if decode_s > 0:
            total_decode_s += decode_s
            count += 1
        if prefill_s > 0:
            total_prefill_s += prefill_s

    avg_decode_s = total_decode_s / count if count > 0 else 0
    avg_prefill_s = total_prefill_s / count if count > 0 else 0

    telemetry_keys = set()
    for r in ax_results:
        telemetry = r.get("ax_mlx_telemetry", {})
        telemetry_keys.update(telemetry.keys())

    return {
        "avg_decode_s": round(avg_decode_s, 4),
        "avg_prefill_s": round(avg_prefill_s, 4),
        "ax_row_count": len(ax_results),
        "telemetry_keys": sorted(telemetry_keys),
    }


def main() -> int:
    args = parse_args()
    result = run_telemetry_pass(args)

    output_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)

    if result["status"] != "success":
        print(
            f"\nTELEMETRY PASS FAILED: {result.get('error', 'unknown')}",
            file=sys.stderr,
        )
        return 1

    print("\nTELEMETRY PASS SUCCESS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
