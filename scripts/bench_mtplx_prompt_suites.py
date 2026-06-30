#!/usr/bin/env python3
"""Run MTPLX MTP on repo prompt suites with AX-compatible repetitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mtplx.artifacts import inspect_model
from mtplx.benchmarks.schema import encode_prompt_case, load_prompt_suite
from mtplx.benchmarks.validators.basic import validate_benchmark_output
from mtplx.draft_lm_head import _install_draft_lm_head, draft_lm_head_spec_from_runtime_contract
from mtplx.draft_sampling import draft_sampler_spec_from_runtime_contract
from mtplx.generation import generate_mtpk
from mtplx.mtp_patch import MTPContract
from mtplx.profiles import apply_profile_env, get_profile
from mtplx.runtime import load
from mtplx.sampling import SamplerConfig
from mtplx.version import __version__ as MTPLX_VERSION


DEFAULT_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    complete_runs = [
        run for run in runs if bool(run.get("fixed_token_complete", True))
    ]
    decode = [float(run["decode_tok_s"]) for run in complete_runs]
    end_to_end = [float(run["end_to_end_tok_s"]) for run in complete_runs]
    generated = sum(int(run["generated_tokens"]) for run in runs)
    drafted = sum(int(run["drafted_tokens"]) for run in runs)
    accepted = sum(int(run["accepted_drafts"]) for run in runs)
    return {
        "runs": len(runs),
        "complete_runs": len(complete_runs),
        "incomplete_runs": len(runs) - len(complete_runs),
        "generated_tokens": generated,
        "decode_tok_s": {
            "median": median(decode),
            "p10": percentile(decode, 0.10),
            "p90": percentile(decode, 0.90),
            "values": decode,
        },
        "end_to_end_tok_s": {
            "median": median(end_to_end),
            "p10": percentile(end_to_end, 0.10),
            "p90": percentile(end_to_end, 0.90),
            "values": end_to_end,
        },
        "accepted_drafts": accepted,
        "drafted_tokens": drafted,
        "accept_rate": accepted / drafted if drafted else None,
    }


def runtime_contract_specs(model: str, profile_name: str) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    profile = get_profile(profile_name)
    fallback_draft_lm_head = (
        None
        if profile.draft_lm_head is None
        else {
            "bits": profile.draft_lm_head.bits,
            "group_size": profile.draft_lm_head.group_size,
            "mode": profile.draft_lm_head.mode,
        }
    )
    inspection = inspect_model(model).to_dict()
    compatibility = inspection.get("compatibility") or {}
    runtime_contract = compatibility.get("runtime_contract")
    draft_lm_head = draft_lm_head_spec_from_runtime_contract(
        runtime_contract,
        fallback=fallback_draft_lm_head,
    )
    draft_sampler = draft_sampler_spec_from_runtime_contract(runtime_contract)
    return inspection, draft_lm_head, draft_sampler


def require_model_compatibility(inspection: dict[str, Any], *, allow_unverified_model: bool) -> None:
    compatibility = inspection.get("compatibility") or {}
    if compatibility.get("can_run") is not False or allow_unverified_model:
        return

    details = [
        "MTPLX inspection rejected this model before benchmarking",
        "compatibility.can_run=false",
    ]
    runtime_compatibility = compatibility.get("runtime_compatibility")
    if runtime_compatibility:
        details.append(f"runtime_compatibility={runtime_compatibility}")
    message = compatibility.get("message")
    if message:
        details.append(str(message))
    runtime_contract_error = compatibility.get("runtime_contract_error")
    if runtime_contract_error:
        details.append(f"runtime_contract_error={runtime_contract_error}")
    if compatibility.get("unsafe_force_required"):
        details.append("unsafe_force_required=true")
    details.append("Regenerate the AX sidecar, or pass --allow-unverified-model for diagnostics only.")
    raise RuntimeError("; ".join(details))


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    prompt_suite = args.prompts.resolve()
    profile = get_profile(args.profile)
    apply_profile_env(args.profile)
    runtime_env = profile.env_dict()
    inspection, draft_lm_head, draft_sampler = runtime_contract_specs(args.model, args.profile)
    require_model_compatibility(
        inspection,
        allow_unverified_model=bool(args.allow_unverified_model),
    )

    rt = load(
        args.model,
        mtp=True,
        contract=MTPContract(
            mtp_quant_bits=args.mtp_quant_bits,
            mtp_quant_group_size=args.mtp_quant_group_size,
            mtp_quant_mode=args.mtp_quant_mode,
            mtp_quant_policy=args.mtp_quant_policy,
        ),
    )
    draft_lm_head_report = None
    if draft_lm_head is not None:
        draft_lm_head_report = _install_draft_lm_head(
            rt,
            bits=int(draft_lm_head["bits"]),
            group_size=int(draft_lm_head["group_size"]),
            mode=str(draft_lm_head["mode"]),
        )

    sampler = SamplerConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    draft_sampler_config = SamplerConfig(
        temperature=args.temperature if draft_sampler is None else float(draft_sampler["temperature"]),
        top_p=args.top_p if draft_sampler is None else float(draft_sampler["top_p"]),
        top_k=args.top_k if draft_sampler is None else int(draft_sampler["top_k"]),
    )

    cases = load_prompt_suite(prompt_suite)
    if args.limit is not None:
        cases = cases[: args.limit]

    encoded = [
        (
            case,
            encode_prompt_case(
                rt.tokenizer,
                case,
                chat_template=True,
                enable_thinking=False if args.disable_thinking else None,
            ),
        )
        for case in cases
    ]

    case_results = []
    for case_index, (case, token_ids) in enumerate(encoded):
        runs: list[dict[str, Any]] = []
        all_runs = args.warmup_repetitions + args.repetitions
        for rep in range(all_runs):
            measured = rep >= args.warmup_repetitions
            run_started = time.time()
            output = generate_mtpk(
                rt,
                token_ids,
                max_tokens=min(args.max_tokens, case.max_tokens),
                sampler=sampler,
                speculative_depth=args.depth,
                seed=args.seed + case_index * 1000 + rep,
                mtp_hidden_variant="post_norm",
                mtp_cache_policy="persistent",
                mtp_history_policy="committed",
                draft_sampler=draft_sampler_config,
                min_speculative_depth=1,
                verify_strategy="capture_commit",
                verify_core="linear-gdn-from-conv-tape",
                stop_token_ids=set() if args.ignore_eos else None,
            )
            run_ended = time.time()
            stats = output.stats
            requested_tokens = min(args.max_tokens, case.max_tokens)
            fixed_token_complete = int(stats.generated_tokens) == requested_tokens
            print(
                f"{case.id} rep {rep + 1}/{all_runs}: "
                f"decode={stats.decode_tok_s:.1f} tok/s "
                f"out={stats.generated_tokens}/{requested_tokens} "
                f"accept={stats.accepted_drafts}/{stats.drafted_tokens}",
                file=sys.stderr,
                flush=True,
            )
            run = {
                "measured": measured,
                "repetition": rep,
                "seed": args.seed + case_index * 1000 + rep,
                "started_at": run_started,
                "ended_at": run_ended,
                "requested_tokens": requested_tokens,
                "fixed_token_complete": fixed_token_complete,
                "fixed_token_rejected_reason": None
                if fixed_token_complete
                else "generated_tokens_lt_requested_tokens",
                "generated_tokens": stats.generated_tokens,
                "elapsed_s": stats.elapsed_s,
                "decode_elapsed_s": stats.decode_elapsed_s,
                "tok_s": stats.tok_s,
                "decode_tok_s": stats.decode_tok_s,
                "end_to_end_tok_s": stats.end_to_end_tok_s,
                "prompt_eval_time_s": stats.prompt_eval_time_s,
                "accepted_drafts": stats.accepted_drafts,
                "rejected_drafts": stats.rejected_drafts,
                "drafted_tokens": stats.drafted_tokens,
                "accepted_by_depth": stats.accepted_by_depth,
                "drafted_by_depth": stats.drafted_by_depth,
                "acceptance_by_depth": [
                    accepted / drafted if drafted else None
                    for accepted, drafted in zip(stats.accepted_by_depth, stats.drafted_by_depth)
                ],
                "verify_time_s": stats.verify_time_s,
                "draft_time_s": stats.draft_time_s,
                "target_forward_time_s": stats.target_forward_time_s,
                "peak_memory_bytes": stats.peak_memory_bytes,
                "text": output.text,
                "tokens": output.tokens,
            }
            if measured:
                runs.append(run)
            if args.cooldown > 0 and rep < all_runs - 1:
                time.sleep(args.cooldown)

        validations = [
            validation.__dict__
            for validation in validate_benchmark_output(
                runs[-1]["text"] if runs else "",
                category=case.category,
                prompt_id=case.id,
            )
        ]
        case_results.append(
            {
                "prompt_id": case.id,
                "category": case.category,
                "prompt_sha256": case.prompt_sha256,
                "prompt_tokens": len(token_ids),
                "max_tokens": min(args.max_tokens, case.max_tokens),
                "runs": runs,
                "summary": summarize_runs(runs),
                "validations": validations,
            }
        )
        if args.inter_case_cooldown > 0 and case_index < len(encoded) - 1:
            print(f"  [mtplx] inter-case cooldown {args.inter_case_cooldown:.0f}s", flush=True)
            time.sleep(args.inter_case_cooldown)

    measured_runs = [run for case in case_results for run in case["runs"]]
    validations = [v for case in case_results for v in case["validations"]]
    return {
        "schema": "ax.mtplx.prompt_suite_mtp.v1",
        "mtplx_version": MTPLX_VERSION,
        "engine": "mtplx",
        "model": args.model,
        "model_inspection": inspection,
        "profile": profile.to_dict(),
        "runtime_env": runtime_env,
        "prompt_suite": str(prompt_suite),
        "prompt_suite_sha256": file_sha256(prompt_suite),
        "suite": args.suite,
        "depth": args.depth,
        "sampling": asdict(sampler),
        "draft_sampling": asdict(draft_sampler_config),
        "max_tokens": args.max_tokens,
        "repetitions": args.repetitions,
        "warmup_repetitions": args.warmup_repetitions,
        "cooldown_s": args.cooldown,
        "ignore_eos": bool(args.ignore_eos),
        "disable_thinking": bool(args.disable_thinking),
        "allow_unverified_model": bool(args.allow_unverified_model),
        "draft_lm_head": draft_lm_head_report,
        "build": {
            "host": platform.node(),
            "platform": platform.platform(),
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_tracked_dirty": bool(git_value(["status", "--porcelain"])),
        },
        "results": case_results,
        "summary": {
            **summarize_runs(measured_runs),
            "case_count": len(case_results),
            "validations_passed": sum(1 for v in validations if v.get("passed")),
            "validations_total": len(validations),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="sustained")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=DEFAULT_SAMPLING["temperature"])
    parser.add_argument("--top-p", type=float, default=DEFAULT_SAMPLING["top_p"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_SAMPLING["top_k"])
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument(
        "--inter-case-cooldown",
        type=float,
        default=0.0,
        help="Extra sleep between prompt cases (seconds). Prevents GPU thermal throttling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Use an empty stop-token set so fixed-token throughput runs stop only at max_tokens.",
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--allow-unverified-model",
        action="store_true",
        help="Run even when MTPLX inspect_model reports compatibility.can_run=false. Diagnostics only.",
    )
    parser.add_argument("--mtp-quant-bits", type=int)
    parser.add_argument("--mtp-quant-group-size", type=int, default=64)
    parser.add_argument("--mtp-quant-mode", default="affine")
    parser.add_argument("--mtp-quant-policy")
    args = parser.parse_args()

    result = run_suite(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Saved to {args.output}", file=sys.stderr)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
