#!/usr/bin/env python3
"""AX Engine QA Runner — test inference quality and generate HTML reports."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prompts import (
    DEFAULT_SAMPLE_SIZE,
    PROMPTS,
    all_categories,
    bank_size,
    describe_bank,
    get_prompt_by_id,
    sample_prompts,
)
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
    tokenizer=None,
    tokenizer_path=None,
    sample_size=None,
    seed=None,
    categories=None,
    run_all=False,
):
    from client import send_generate_request

    results = []
    if prompt_ids:
        prompts_to_run, seed_used = sample_prompts(
            prompt_ids=prompt_ids,
            seed=seed,
        )
    elif run_all:
        prompts_to_run, seed_used = sample_prompts(
            n=bank_size(),
            seed=seed,
            categories=categories,
            stratified=False,
        )
    else:
        n = DEFAULT_SAMPLE_SIZE if sample_size is None else sample_size
        prompts_to_run, seed_used = sample_prompts(
            n=n,
            seed=seed,
            categories=categories,
            stratified=True,
        )

    total = len(prompts_to_run) * len(modes) * len(streams)
    current = 0
    print(
        f"  Sample:   {len(prompts_to_run)}/{bank_size()} prompts "
        f"(seed={seed_used}{' all' if run_all else ''})"
    )
    print(f"  IDs:      {', '.join(p.id for p in prompts_to_run)}")
    print()

    for prompt in prompts_to_run:
        for mode in modes:
            for stream in streams:
                current += 1
                label = f"[{current}/{total}] {prompt.id} | {mode} | stream={stream}"
                print(f"  {label}...", end=" ", flush=True)

                rp = repetition_penalty

                if tokenizer is not None:
                    resp = send_generate_request(
                        base_url=base_url,
                        model=model_id,
                        system=prompt.system,
                        user=prompt.user,
                        tokenizer=tokenizer,
                        tokenizer_path=tokenizer_path,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=stream,
                        repetition_penalty=rp,
                        timeout=timeout,
                    )
                else:
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
                        "category": prompt.category,
                        "mode": mode,
                        "stream": stream,
                        "response": resp,
                        "report": report,
                    }
                )

    return results, seed_used, [p.id for p in prompts_to_run]


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
        choices=["direct", "ngram", "mtp", "mtp-ngram"],
        help="Generation modes to label in the report (default: direct)",
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
        help="Specific prompt IDs to run (overrides sampling)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        metavar="N",
        help=(
            f"Draw N prompts from the bank each run (default: {DEFAULT_SAMPLE_SIZE}). "
            "Uses stratified sampling across categories."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for sampling (default: random; print/replay this for reproducibility)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run the full question bank (no subset sampling)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Limit bank/sample to these categories",
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
        "--tokenizer",
        default=None,
        help="Path to tokenizer.json for client-side tokenization (uses /v1/generate endpoint)",
    )
    parser.add_argument(
        "--list-prompts", action="store_true", help="List available prompts and exit"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Show bank size and per-category counts, then exit",
    )
    args = parser.parse_args()

    if args.list_categories:
        print(describe_bank())
        return

    if args.list_prompts:
        print(describe_bank())
        print()
        print("Available prompts:")
        for p in PROMPTS:
            print(f"  {p.id:32s} [{p.category:14s}] {p.description}")
        return

    streams = []
    if "both" in args.streams:
        streams = [True, False]
    else:
        streams = [s == "true" for s in args.streams]

    model_id = args.model or "auto"
    commit, tag = get_git_info()

    tokenizer = None
    if args.tokenizer:
        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(args.tokenizer)
            print(f"  Tokenizer: {args.tokenizer}")
        except ImportError:
            print(
                "  WARNING: tokenizers library not installed, falling back to /v1/chat/completions"
            )
        except Exception as e:
            print(f"  WARNING: Failed to load tokenizer: {e}")

    print(f"AX Engine QA Suite")
    print(f"  Server:   {args.base_url}")
    print(f"  Model:    {model_id}")
    print(f"  Modes:    {', '.join(args.modes)}")
    print(f"  Streams:  {streams}")
    print(f"  Version:  {tag} ({commit})")
    print(f"  Bank:     {bank_size()} prompts / {len(all_categories())} categories")
    if args.prompts:
        print(f"  Select:   explicit ids ({len(args.prompts)})")
    elif args.run_all:
        print("  Select:   full bank")
    else:
        print(f"  Select:   sample {args.sample} (stratified)")
    print()

    results, seed_used, sampled_ids = run_qa_suite(
        base_url=args.base_url,
        model_id=model_id,
        modes=args.modes,
        streams=streams,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        prompt_ids=args.prompts,
        timeout=args.timeout,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer if tokenizer is not None else None,
        sample_size=args.sample,
        seed=args.seed,
        categories=args.categories,
        run_all=args.run_all,
    )

    passed = sum(1 for r in results if r["report"].auto_pass)
    total = len(results)
    print(
        f"\nResults: {passed}/{total} passed ({passed / total * 100:.1f}%)"
        if total > 0
        else "\nNo results"
    )
    print(f"Replay:  --seed {seed_used}" + (f" --sample {args.sample}" if not args.run_all and not args.prompts else ""))

    metadata = {
        "title": f"AX Engine QA Report — {tag}",
        "version": tag,
        "commit": commit,
        "seed": seed_used,
        "bank_size": bank_size(),
        "sampled_ids": sampled_ids,
        "sample_size": len(sampled_ids),
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
