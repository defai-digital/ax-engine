#!/usr/bin/env python3
"""Legacy MTP-mode n-gram benchmark: AX Engine vs injected MTPLX reference.

For new Qwen3.6 publication rows, prefer scripts/bench_qwen36_mtp_fair.py.
That harness excludes Youssofal optimized bundles and records sidecar
provenance for standard Qwen/Qwen3.6 shards plus mlx-community 4-bit bases.
This script is retained for older AX artifact assembly and MTPLX JSON injection.

Examples:

  # Run both suites — AX rows only
  python3 scripts/bench_mtp_compare.py \\
    --model-dir ~/.cache/huggingface/hub/models--ax-local--Qwen3.6-27B-MTP/snapshots/v1 \\
    --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all

  # Inject MTPLX reference numbers and build a combined summary
  python3 scripts/bench_mtp_compare.py \\
    --model-dir /path/to/model \\
    --mtplx-results benchmarks/results/mtp-compare/2026-05-23-mtplx-vs-ax/mtplx.json \\
    --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all

  # Run one suite only
  python3 scripts/bench_mtp_compare.py \\
    --model-dir /path/to/model \\
    --suites flappy \\
    --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-flappy

MTPLX reference JSON format (--mtplx-results):
  {
    "mtplx_version": "0.3.7",
    "hardware": "Apple M5 Max 128GB",
    "results": [
      {"model_bundle": "Speed", "suite": "flappy", "decode_tok_s": 59.2, "accept_rate": 0.995, "depth": 3},
      {"model_bundle": "Speed", "suite": "long_code", "decode_tok_s": 59.8, "accept_rate": 0.996, "depth": 3}
    ]
  }

The AX artifact from each suite run is saved at
  <output-dir>/<suite>/<suite>.json
and a combined summary markdown is written to
  <output-dir>/summary.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mlx_inference_stack.py"
DEFAULT_SUITES_DIR = REPO_ROOT / "benchmarks" / "prompts" / "mtp-suites"
DEFAULT_RESULTS_BASE = REPO_ROOT / "benchmarks" / "results" / "mtp-compare"

MTP_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
MTP_GENERATION_TOKENS = 1000
MTP_REPETITIONS = 5
MTP_WARMUP_REPS = 1  # bench_mlx_inference_stack always does 1 warmup
MTP_COOLDOWN = 15.0

KNOWN_SUITES = ["flappy", "long_code", "python_modules_long"]


def run_ax_suite(
    suite_name: str,
    suite_file: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    repetitions: int,
    cooldown: float,
    mtp_only: bool = False,
    mtp_max_depth: int | None = None,
    mtp_fast_tail_topk_sampling: bool = False,
    enable_thinking: bool = False,
) -> dict:
    """Run bench_mlx_inference_stack.py for one real-prompt suite.

    Returns the parsed artifact dict.
    """
    suite_out_dir = output_dir / suite_name
    suite_out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = suite_out_dir / f"{suite_name}.json"

    policy_flag = "--ax-ngram-accel" if mtp_only else "--ax-compare-policies"
    cmd = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--model-dir", str(model_dir),
        "--prompt-source", "real",
        "--real-prompt-suite", str(suite_file),
        "--generation-tokens", str(MTP_GENERATION_TOKENS),
        "--repetitions", str(repetitions),
        "--cooldown", str(cooldown),
        policy_flag,
        "--ax-sampling", json.dumps(MTP_SAMPLING),
        "--skip-mlx-lm",
        "--output", str(artifact_path),
    ]
    if mtp_max_depth is not None:
        cmd.extend(["--ax-mtp-max-depth", str(mtp_max_depth)])
    if mtp_fast_tail_topk_sampling:
        cmd.append("--ax-mtp-fast-tail-topk-sampling")
    if not enable_thinking:
        cmd.append("--no-thinking")
    print(f"\n=== MTP suite: {suite_name} ===", flush=True)
    print(f"  suite_file: {suite_file}", flush=True)
    print(f"  artifact:   {artifact_path}", flush=True)
    print(f"  sampling:   {MTP_SAMPLING}", flush=True)
    print(f"  command:    {' '.join(cmd)}\n", flush=True)

    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"bench_mlx_inference_stack.py exited {result.returncode} for suite {suite_name}")

    return json.loads(artifact_path.read_text())


def extract_suite_summary(artifact: dict, suite_name: str) -> dict:
    """Pull decode tok/s and n-gram accept rate from a suite artifact."""
    rows = artifact.get("results", [])
    direct_rows = [r for r in rows if r.get("engine") == "ax_engine_mlx" and r.get("prompt_case_id") is not None]
    ngram_rows = [r for r in rows if r.get("engine") == "ax_engine_mlx_ngram_accel" and r.get("prompt_case_id") is not None]

    def median_decode(rs: list) -> float | None:
        vals = [r["decode_tok_s"]["median"] for r in rs if r.get("decode_tok_s", {}).get("median") is not None]
        if not vals:
            return None
        vals.sort()
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2

    def mean_accept_rate(rs: list) -> float | None:
        rates = []
        for r in rs:
            telem = r.get("ngram_acceleration_telemetry", {})
            # Prefer MTP telemetry when present (MTP replaces n-gram on MTPLX models).
            mtp_drafted = telem.get("ax_mtp_draft_tokens", 0)
            mtp_accepted = telem.get("ax_mtp_accepted_tokens", 0)
            if mtp_drafted and mtp_drafted > 0:
                rates.append(mtp_accepted / mtp_drafted)
                continue
            # Fall back to n-gram telemetry for non-MTPLX models.
            ngram_accepted = telem.get("ax_ngram_accepted_tokens", 0)
            ngram_drafted = telem.get("ax_ngram_draft_tokens", 0)
            if ngram_drafted and ngram_drafted > 0:
                rates.append(ngram_accepted / ngram_drafted)
        return sum(rates) / len(rates) if rates else None

    def mean_mtp_accept_rate(rs: list) -> float | None:
        total_drafted = sum(
            r.get("ngram_acceleration_telemetry", {}).get("ax_mtp_draft_tokens", 0) for r in rs
        )
        total_accepted = sum(
            r.get("ngram_acceleration_telemetry", {}).get("ax_mtp_accepted_tokens", 0) for r in rs
        )
        return total_accepted / total_drafted if total_drafted > 0 else None

    return {
        "suite": suite_name,
        "case_count": len(set(r.get("prompt_case_id") for r in rows if r.get("prompt_case_id"))),
        "ax_direct_decode_tok_s": median_decode(direct_rows),
        "ax_ngram_decode_tok_s": median_decode(ngram_rows),
        "ax_ngram_accept_rate": mean_accept_rate(ngram_rows),
        "ax_mtp_accept_rate": mean_mtp_accept_rate(ngram_rows),
        # Provenance fields surfaced in summary warnings.
        "ax_mtp_max_depth": artifact.get("ax_mtp_max_depth"),
        "ax_mtp_fast_tail_topk_sampling": bool(artifact.get("ax_mtp_fast_tail_topk_sampling", False)),
        "build_dirty": bool(artifact.get("build", {}).get("git_tracked_dirty", False)),
        "build_commit": artifact.get("build", {}).get("commit", "unknown")[:12],
    }


def load_mtplx_results(path: Path) -> dict:
    """Load MTPLX reference result JSON."""
    data = json.loads(path.read_text())
    by_suite: dict[str, dict] = {}
    by_bundle_suite: dict[tuple[str, str], dict] = {}
    for entry in data.get("results", []):
        suite = entry.get("suite")
        if suite:
            if "model_bundle" not in entry:
                by_suite[suite] = entry
            else:
                bundle = str(entry["model_bundle"]).strip().lower()
                by_bundle_suite[(bundle, suite)] = entry
    return {"meta": data, "by_suite": by_suite, "by_bundle_suite": by_bundle_suite}


def infer_model_bundle(model_dir: Path) -> str | None:
    text = str(model_dir).lower()
    if "quality" in text:
        return "quality"
    if "speed" in text:
        return "speed"
    return None


def write_summary(
    suite_summaries: list[dict],
    output_dir: Path,
    *,
    mtplx_ref: dict | None = None,
    model_dir: Path,
    run_date: str,
    repetitions: int,
) -> Path:
    """Write a markdown summary table to <output_dir>/summary.md."""
    lines: list[str] = []
    lines.append("# MTP Benchmark Summary")
    lines.append("")
    lines.append(f"Date: {run_date}  ")
    lines.append(f"Model: `{model_dir.name}`  ")
    lines.append(f"Sampling: temperature={MTP_SAMPLING['temperature']}, top_p={MTP_SAMPLING['top_p']}, top_k={MTP_SAMPLING['top_k']}  ")
    lines.append(f"Generation tokens: {MTP_GENERATION_TOKENS}  ")
    lines.append(f"Repetitions: {repetitions} + 1 warmup  ")
    lines.append("")

    # Dirty build warning — surfaces when any suite artifact was built from uncommitted changes.
    any_dirty = any(s.get("build_dirty", False) for s in suite_summaries)
    if any_dirty:
        commit = suite_summaries[0].get("build_commit", "unknown") if suite_summaries else "unknown"
        lines.append(f"> **WARNING: dirty build** — this run used uncommitted source changes (base commit `{commit}`).  ")
        lines.append("> Numbers are not reproducible from any tagged commit. Do not promote to PERFORMANCE.md  ")
        lines.append("> or README until a clean build is confirmed.  ")
        lines.append("")

    if mtplx_ref:
        meta = mtplx_ref["meta"]
        lines.append(f"MTPLX reference: version={meta.get('mtplx_version', 'unknown')}, hardware={meta.get('hardware', 'unknown')}")
        lines.append("")
    model_bundle = infer_model_bundle(model_dir)

    # Experimental-flag caveat.
    fast_tail_used = any(s.get("ax_mtp_fast_tail_topk_sampling", False) for s in suite_summaries)
    if fast_tail_used:
        lines.append("> **Note:** `AX_MLX_MTP_FAST_TAIL_TOPK_SAMPLING=1` was active for one or more suites.  ")
        lines.append("> top-p is not applied on the fast tail-sampling path; these rows are diagnostic only  ")
        lines.append("> and **not** comparable to MTPLX standard rejection sampling.  ")
        lines.append("")

    lines.append("## Decode throughput (tok/s, median across cases)")
    lines.append("")

    if mtplx_ref:
        lines.append("| Suite | AX depth | AX direct | AX MTP | AX MTP accept rate | MTPLX reference | MTPLX accept rate | MTPLX depth |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| Suite | AX depth | AX direct | AX MTP | AX MTP accept rate |")
        lines.append("|---|---:|---:|---:|---:|")

    for s in suite_summaries:
        suite = s["suite"]
        ax_depth = str(s["ax_mtp_max_depth"]) if s.get("ax_mtp_max_depth") is not None else "default"
        direct = f"{s['ax_direct_decode_tok_s']:.1f}" if s["ax_direct_decode_tok_s"] is not None else "—"
        ngram = f"{s['ax_ngram_decode_tok_s']:.1f}" if s["ax_ngram_decode_tok_s"] is not None else "—"
        mtp_accept = s.get("ax_mtp_accept_rate")
        accept = f"{mtp_accept:.1%}" if mtp_accept is not None else (
            f"{s['ax_ngram_accept_rate']:.1%}" if s["ax_ngram_accept_rate"] is not None else "—"
        )

        if mtplx_ref:
            ref = {}
            if model_bundle is not None:
                ref = mtplx_ref.get("by_bundle_suite", {}).get((model_bundle, suite), {})
            if not ref:
                ref = mtplx_ref["by_suite"].get(suite, {})
            m_decode = f"{ref['decode_tok_s']:.1f}" if "decode_tok_s" in ref else "—"
            m_accept = f"{ref['accept_rate']:.1%}" if "accept_rate" in ref else "—"
            m_depth = str(ref.get("depth", "—"))
            lines.append(f"| {suite} | {ax_depth} | {direct} | {ngram} | {accept} | {m_decode} | {m_accept} | {m_depth} |")
        else:
            lines.append(f"| {suite} | {ax_depth} | {direct} | {ngram} | {accept} |")

    lines.append("")
    lines.append("## Artifact provenance")
    lines.append("")
    for s in suite_summaries:
        lines.append(f"- `{output_dir}/{s['suite']}/{s['suite']}.json` — {s['case_count']} prompt cases, AX direct + n-gram rows")
    if mtplx_ref:
        lines.append(f"- MTPLX reference: `{mtplx_ref['meta'].get('source_path', 'injected')}`")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/bench_mtp_compare.py \\")
    lines.append(f"  --model-dir {model_dir} \\")
    if mtplx_ref:
        lines.append(f"  --mtplx-results <path-to-mtplx-results.json> \\")
    lines.append(f"  --output-dir benchmarks/results/mtp-compare/$(date +%F)-ax-mtp-all")
    lines.append("```")
    lines.append("")

    out_path = output_dir / "summary.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to the AX-ready MTP model directory (must have model-manifest.json).",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=KNOWN_SUITES + ["all"],
        default=["all"],
        help="Which prompt suites to run (default: all).",
    )
    parser.add_argument(
        "--suites-dir",
        type=Path,
        default=DEFAULT_SUITES_DIR,
        help=f"Directory containing suite JSONL files (default: {DEFAULT_SUITES_DIR}).",
    )
    parser.add_argument(
        "--mtplx-results",
        type=Path,
        default=None,
        help=(
            "Path to a MTPLX reference result JSON. When provided, the summary "
            "table includes a MTPLX column. See script header for the expected format."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            f"Root output directory for this run (default: "
            f"{DEFAULT_RESULTS_BASE}/<today>-ax-mtp-all)."
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=MTP_REPETITIONS,
        help=f"Measured repetitions per case (default: {MTP_REPETITIONS}).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=MTP_COOLDOWN,
        help=f"Cooldown seconds between repetitions (default: {MTP_COOLDOWN}).",
    )
    parser.add_argument(
        "--mtp-only",
        action="store_true",
        default=False,
        help=(
            "Skip the direct (no-MTP) baseline and run only the MTP/ngram-accel policy. "
            "Cuts benchmark time roughly in half. ax_direct column will show '—' in summary."
        ),
    )
    parser.add_argument(
        "--ax-mtp-max-depth",
        type=int,
        default=None,
        help="Set AX_MLX_MTP_MAX_DEPTH for AX server rows while benchmarking.",
    )
    parser.add_argument(
        "--ax-mtp-fast-tail-topk-sampling",
        action="store_true",
        help=(
            "Set AX_MLX_MTP_FAST_TAIL_TOPK_SAMPLING=1 for AX server rows. "
            "Diagnostic only; top-p is not applied on the fast tail-sampling path."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=False,
        help=(
            "Enable thinking mode (default: disabled). The MTPLX reference uses "
            "enable_thinking=false; this flag overrides that to match a thinking-enabled "
            "server configuration."
        ),
    )
    args = parser.parse_args()
    if args.ax_mtp_max_depth is not None and args.ax_mtp_max_depth < 0:
        parser.error("--ax-mtp-max-depth must be >= 0")

    run_date = date.today().isoformat()
    suites = KNOWN_SUITES if "all" in args.suites else args.suites

    output_dir = args.output_dir
    if output_dir is None:
        suite_tag = "all" if len(suites) == len(KNOWN_SUITES) else "_".join(suites)
        output_dir = DEFAULT_RESULTS_BASE / f"{run_date}-ax-mtp-{suite_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    mtplx_ref: dict | None = None
    if args.mtplx_results:
        mtplx_ref = load_mtplx_results(args.mtplx_results)
        mtplx_ref["meta"]["source_path"] = str(args.mtplx_results)
        print(f"  [mtplx] loaded reference from {args.mtplx_results}", flush=True)

    suite_summaries: list[dict] = []
    for suite_name in suites:
        suite_file = args.suites_dir / f"{suite_name}.jsonl"
        if not suite_file.is_file():
            print(f"ERROR: suite file not found: {suite_file}", file=sys.stderr)
            sys.exit(1)

        artifact = run_ax_suite(
            suite_name,
            suite_file,
            args.model_dir,
            output_dir,
            repetitions=args.repetitions,
            cooldown=args.cooldown,
            mtp_only=args.mtp_only,
            mtp_max_depth=args.ax_mtp_max_depth,
            mtp_fast_tail_topk_sampling=args.ax_mtp_fast_tail_topk_sampling,
            enable_thinking=args.enable_thinking,
        )
        summary = extract_suite_summary(artifact, suite_name)
        suite_summaries.append(summary)

        print(f"\n  [result] {suite_name}:", flush=True)
        print(f"    ax_direct:       {summary['ax_direct_decode_tok_s']:.1f} tok/s" if summary["ax_direct_decode_tok_s"] else "    ax_direct:       —", flush=True)
        print(f"    ax_mtp:          {summary['ax_ngram_decode_tok_s']:.1f} tok/s" if summary["ax_ngram_decode_tok_s"] else "    ax_mtp:          —", flush=True)
        if summary.get("ax_mtp_accept_rate") is not None:
            print(f"    mtp_accept:      {summary['ax_mtp_accept_rate']:.1%}", flush=True)
        elif summary["ax_ngram_accept_rate"] is not None:
            print(f"    ngram_accept:    {summary['ax_ngram_accept_rate']:.1%}", flush=True)
        else:
            print("    accept_rate:     —", flush=True)

    summary_path = write_summary(
        suite_summaries,
        output_dir,
        mtplx_ref=mtplx_ref,
        model_dir=args.model_dir,
        run_date=run_date,
        repetitions=args.repetitions,
    )
    print(f"\n=== Summary written to {summary_path} ===", flush=True)


if __name__ == "__main__":
    main()
