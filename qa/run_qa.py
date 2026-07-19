#!/usr/bin/env python3
"""AX Engine QA Runner — inference quality checks with HTML + JSON reports."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prompts import (  # noqa: E402
    DEFAULT_SAMPLE_SIZE,
    PROMPTS,
    all_categories,
    bank_size,
    describe_bank,
    sample_prompts,
    validate_bank,
)
from client import send_request  # noqa: E402
from checkers import QualityReport, run_all_checks  # noqa: E402
from reporter import generate_html_report, generate_json_report  # noqa: E402

MODE_NOTE = (
    "mode_label is a report tag only; decode path (direct / n-gram / MTP) is "
    "controlled by how ax-engine-server was started, not by --mode."
)


def get_git_info() -> tuple[str, str]:
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
    base_url: str,
    model_id: str,
    mode_label: str,
    streams: list[bool],
    max_tokens: int,
    temperature: float,
    repetition_penalty: float | None,
    prompt_ids: list[str] | None,
    timeout: int,
    tokenizer=None,
    tokenizer_path: str | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
    categories: list[str] | None = None,
    run_all: bool = False,
):
    from client import send_generate_request

    results = []
    if prompt_ids:
        prompts_to_run, seed_used = sample_prompts(prompt_ids=prompt_ids, seed=seed)
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

    total = len(prompts_to_run) * len(streams)
    current = 0
    print(
        f"  Sample:   {len(prompts_to_run)}/{bank_size()} prompts "
        f"(seed={seed_used}{' all' if run_all else ''})"
    )
    print(f"  IDs:      {', '.join(p.id for p in prompts_to_run)}")
    print(f"  Mode:     {mode_label} (label only)")
    print()

    for prompt in prompts_to_run:
        for stream in streams:
            current += 1
            label = f"[{current}/{total}] {prompt.id} | {mode_label} | stream={stream}"
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
                report = QualityReport(
                    prompt_id=prompt.id,
                    auto_pass=False,
                    manual_review=True,
                    output_preview=f"ERROR: {resp.error}",
                )
            else:
                report = run_all_checks(resp.text, prompt)
                status = "PASS" if report.auto_pass else "FAIL"
                print(f"{status} ({report.summary}, {resp.elapsed_ms:.0f}ms)")

            results.append(
                {
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "mode": mode_label,
                    "stream": stream,
                    "response": resp,
                    "report": report,
                }
            )

    return results, seed_used, [p.id for p in prompts_to_run]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AX Engine QA Test Runner",
        epilog=MODE_NOTE,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:31418",
        help="AX Engine server base URL (default: http://localhost:31418)",
    )
    parser.add_argument(
        "--model", default=None, help="Model ID (default: auto-detect from server)"
    )
    parser.add_argument(
        "--mode",
        default="direct",
        choices=["direct", "ngram", "mtp", "mtp-ngram"],
        help=(
            "Report label for the decode path already configured on the server "
            "(default: direct). Does not change server behavior."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        choices=["direct", "ngram", "mtp", "mtp-ngram"],
        help=argparse.SUPPRESS,  # deprecated alias; kept for matrix/scripts
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=["false"],
        choices=["true", "false", "both"],
        help="Streaming modes to test (default: false). Use 'both' for stream+non-stream.",
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
        "--json-output",
        default=None,
        help="JSON report path (default: same stem as --output with .json)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip writing the machine-readable JSON report",
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
        help="Path to tokenizer.json for client-side tokenization (/v1/generate)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 even when hard checks fail (default: exit 1 on hard failure)",
    )
    parser.add_argument(
        "--list-prompts", action="store_true", help="List available prompts and exit"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Show bank size and per-category counts, then exit",
    )
    parser.add_argument(
        "--validate-bank",
        action="store_true",
        help="Validate question bank integrity and exit non-zero on errors",
    )
    args = parser.parse_args()

    if args.validate_bank:
        errors = validate_bank()
        if errors:
            print("Bank validation failed:")
            for err in errors:
                print(f"  - {err}")
            return 2
        print(f"Bank OK: {bank_size()} prompts, {len(all_categories())} categories")
        return 0

    if args.list_categories:
        print(describe_bank())
        return 0

    if args.list_prompts:
        print(describe_bank())
        print()
        print("Available prompts:")
        for p in PROMPTS:
            print(f"  {p.id:32s} [{p.category:14s}] {p.description}")
        return 0

    # Single mode label per run (honest: server controls decode path).
    if args.modes is not None:
        if len(args.modes) != 1:
            print(
                "ERROR: --modes with multiple values is no longer supported. "
                "Run once per server configuration and pass a single --mode label.",
                file=sys.stderr,
            )
            return 2
        mode_label = args.modes[0]
    else:
        mode_label = args.mode

    streams: list[bool]
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

    print("AX Engine QA Suite")
    print(f"  Server:   {args.base_url}")
    print(f"  Model:    {model_id}")
    print(f"  Mode:     {mode_label} (report label only)")
    print(f"  Note:     {MODE_NOTE}")
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
        mode_label=mode_label,
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

    hard_passed = sum(1 for r in results if r["report"].auto_pass)
    total = len(results)
    if total > 0:
        print(
            f"\nResults: {hard_passed}/{total} passed "
            f"({hard_passed / total * 100:.1f}%) [hard checks]"
        )
    else:
        print("\nNo results")
    replay = f"Replay:  --seed {seed_used}"
    if not args.run_all and not args.prompts:
        replay += f" --sample {args.sample}"
    print(replay)

    metadata = {
        "title": f"AX Engine QA Report — {tag}",
        "version": tag,
        "commit": commit,
        "seed": seed_used,
        "bank_size": bank_size(),
        "sampled_ids": sampled_ids,
        "sample_size": len(sampled_ids),
        "model": model_id,
        "base_url": args.base_url,
        "mode_label": mode_label,
        "mode_note": MODE_NOTE,
    }
    html_content = generate_html_report(results, metadata)

    output_path = args.output
    if not output_path:
        ts = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"qa-report-{ts}.html"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_content)
    print(f"Report: {out}")

    if not args.no_json:
        json_path = Path(args.json_output) if args.json_output else out.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(generate_json_report(results, metadata))
        print(f"JSON:   {json_path}")

    if total == 0:
        return 1
    if hard_passed < total and not args.allow_partial:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
