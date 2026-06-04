#!/usr/bin/env python3
"""AX Engine QA Runner — test inference quality and generate HTML reports."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prompts import PROMPTS, get_prompt_by_id
from client import send_request
from checkers import run_all_checks
from reporter import generate_html_report


def get_git_info():
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"
    try:
        tag = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        tag = "unknown"
    return commit, tag


def run_qa_suite(
    base_url,
    model_id,
    modes,
    streams,
    max_tokens,
    temperature,
    repetition_penalty,
    prompt_ids,
    timeout,
):
    results = []
    prompts_to_run = (
        PROMPTS if not prompt_ids else [get_prompt_by_id(pid) for pid in prompt_ids]
    )
    prompts_to_run = [p for p in prompts_to_run if p is not None]

    total = len(prompts_to_run) * len(modes) * len(streams)
    current = 0

    for prompt in prompts_to_run:
        for mode in modes:
            for stream in streams:
                current += 1
                label = f"[{current}/{total}] {prompt.id} | {mode} | stream={stream}"
                print(f"  {label}...", end=" ", flush=True)

                rp = repetition_penalty
                if mode == "ngram":
                    pass  # ngram mode is set server-side via --mtp-optimistic or similar

                resp = send_request(
                    base_url=base_url,
                    model=model_id,
                    system=prompt.system,
                    user=prompt.user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                    repetition_penalty=rp,
                    timeout=timeout,
                )

                if resp.error:
                    print(f"ERROR: {resp.error}")
                    from checkers import QualityReport

                    report = QualityReport(
                        prompt_id=prompt.id, output_preview=f"ERROR: {resp.error}"
                    )
                else:
                    report = run_all_checks(resp.text, prompt)
                    status = "PASS" if report.auto_pass else "FAIL"
                    print(f"{status} ({report.summary}, {resp.elapsed_ms:.0f}ms)")

                results.append(
                    {
                        "prompt_id": prompt.id,
                        "mode": mode,
                        "stream": stream,
                        "response": resp,
                        "report": report,
                    }
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="AX Engine QA Test Runner")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="AX Engine server base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--model", default=None, help="Model ID (default: auto-detect from server)"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["direct"],
        choices=["direct", "ngram"],
        help="Generation modes to test (default: direct)",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=["both"],
        choices=["true", "false", "both"],
        help="Streaming modes to test (default: both)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max output tokens (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty (default: server default)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Specific prompt IDs to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML report path (default: qa-report-{timestamp}.html)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--list-prompts", action="store_true", help="List available prompts and exit"
    )
    args = parser.parse_args()

    if args.list_prompts:
        print("Available prompts:")
        for p in PROMPTS:
            print(f"  {p.id:25s} [{p.category:12s}] {p.description}")
        return

    streams = []
    if "both" in args.streams:
        streams = [True, False]
    else:
        streams = [s == "true" for s in args.streams]

    model_id = args.model or "auto"
    commit, tag = get_git_info()

    print(f"AX Engine QA Suite")
    print(f"  Server:   {args.base_url}")
    print(f"  Model:    {model_id}")
    print(f"  Modes:    {', '.join(args.modes)}")
    print(f"  Streams:  {streams}")
    print(f"  Version:  {tag} ({commit})")
    print(f"  Prompts:  {len(args.prompts) if args.prompts else len(PROMPTS)}")
    print()

    results = run_qa_suite(
        base_url=args.base_url,
        model_id=model_id,
        modes=args.modes,
        streams=streams,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        prompt_ids=args.prompts,
        timeout=args.timeout,
    )

    passed = sum(1 for r in results if r["report"].auto_pass)
    total = len(results)
    print(
        f"\nResults: {passed}/{total} passed ({passed / total * 100:.1f}%)"
        if total > 0
        else "\nNo results"
    )

    metadata = {
        "title": f"AX Engine QA Report — {tag}",
        "version": tag,
        "commit": commit,
    }
    html_content = generate_html_report(results, metadata)

    output_path = args.output
    if not output_path:
        ts = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"qa-report-{ts}.html"

    Path(output_path).write_text(html_content)
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
