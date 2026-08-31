#!/usr/bin/env python3
"""Run OMLX MTP on the repository's real prompt suites.

OMLX is an optional external runtime. The import is intentionally deferred so
the repository benchmark checks do not require OMLX to be installed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        for key in ("id", "category", "prompt"):
            if key not in payload:
                raise ValueError(f"{path}:{line_number} is missing {key!r}")
        case_id = str(payload["id"])
        if case_id in seen:
            raise ValueError(f"{path}:{line_number} repeats prompt id {case_id!r}")
        seen.add(case_id)
        cases.append(
            {
                "id": case_id,
                "category": str(payload["category"]),
                "prompt": str(payload["prompt"]),
                "max_tokens": int(payload.get("max_tokens", 128)),
            }
        )
    if not cases:
        raise ValueError(f"{path} contains no prompt cases")
    return cases


def prompt_tokens(
    case: dict[str, Any],
    tokenizer: Any,
    token_dir: Path | None,
    generation_tokens: int,
    enable_thinking: bool,
) -> tuple[list[int], str, str]:
    if token_dir is not None:
        candidates = sorted(token_dir.glob(f"real-*-{case['id']}-gen-*.json"))
        if len(candidates) != 1:
            raise ValueError(
                f"expected one token artifact for {case['id']!r} in {token_dir}, "
                f"found {len(candidates)}"
            )
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        return (
            [int(token) for token in payload["token_ids"]],
            str(payload["prompt_text_sha256"]),
            str(payload["sha256"]),
        )

    kwargs = {} if enable_thinking else {"enable_thinking": False}
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": case["prompt"]}],
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    tokens = [int(token) for token in encoded]
    return (
        tokens,
        hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest(),
        hashlib.sha256(bytes().join(token.to_bytes(8, "little") for token in tokens)).hexdigest(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-repo-id", required=True)
    parser.add_argument("--suite", default="flappy")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-token-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-label", default="unknown")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from omlx.engine.batched import BatchedEngine
        from omlx.model_settings import ModelSettings
    except ImportError as error:
        raise SystemExit("OMLX is required for this runner; install omlx 0.6.4 or newer") from error

    cases = load_cases(args.prompts)
    settings = ModelSettings(
        mtp_enabled=True,
        mtp_num_draft_tokens=1,
        enable_thinking=args.enable_thinking,
    )
    engine = BatchedEngine(
        args.model,
        model_settings=settings,
        enable_thinking=args.enable_thinking,
    )
    await engine.start()
    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            tokens, prompt_hash, token_hash = prompt_tokens(
                case,
                engine.tokenizer,
                args.prompt_token_dir,
                args.max_tokens,
                args.enable_thinking,
            )
            trials: list[dict[str, Any]] = []
            output_hashes: set[str] = set()
            for measured in (False, True):
                count = args.repetitions if measured else args.warmup_repetitions
                for repetition in range(count):
                    started = time.perf_counter()
                    output = await engine.generate(
                        tokens,
                        max_tokens=min(args.max_tokens, int(case["max_tokens"])),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        seed=args.seed,
                    )
                    elapsed = time.perf_counter() - started
                    if measured:
                        trials.append(
                            {
                                "elapsed_s": elapsed,
                                "completion_tokens": output.completion_tokens,
                                "decode_tok_s": output.completion_tokens / elapsed,
                            }
                        )
                        output_hashes.add(hashlib.sha256(output.text.encode()).hexdigest())
                    phase = "measure" if measured else "warmup"
                    print(
                        f"{case['id']} {phase} {repetition + 1}/{count}: "
                        f"tokens={output.completion_tokens} "
                        f"tok_s={output.completion_tokens / elapsed:.2f}",
                        flush=True,
                    )
                    if measured and args.cooldown > 0:
                        await asyncio.sleep(args.cooldown)
            speeds = [trial["decode_tok_s"] for trial in trials]
            results.append(
                {
                    "prompt_case_id": case["id"],
                    "prompt_category": case["category"],
                    "prompt_tokens": len(tokens),
                    "prompt_text_sha256": prompt_hash,
                    "prompt_token_ids_sha256": token_hash,
                    "generation_tokens": args.max_tokens,
                    "trials": trials,
                    "decode_tok_s": {
                        "mean": statistics.mean(speeds),
                        "median": statistics.median(speeds),
                        "min": min(speeds),
                        "max": max(speeds),
                    },
                    "completion_tokens": sorted({trial["completion_tokens"] for trial in trials}),
                    "output_text_sha256": sorted(output_hashes),
                }
            )
    finally:
        await engine.stop()
    return {
        "schema_version": "ax.mtp_peer_omlx.v1",
        "host_label": args.host_label,
        "runtime": {
            "name": "OMLX",
            "version": "0.6.4-or-newer",
            "model_settings": {
                "mtp_enabled": True,
                "mtp_num_draft_tokens": 1,
                "enable_thinking": args.enable_thinking,
            },
        },
        "model": args.model,
        "model_repo_id": args.model_repo_id,
        "prompt_suite": args.suite,
        "prompt_source": str(args.prompts),
        "prompt_token_source": (
            "explicit token artifacts" if args.prompt_token_dir else "OMLX tokenizer"
        ),
        "warmup_repetitions": args.warmup_repetitions,
        "repetitions": args.repetitions,
        "generation_tokens": args.max_tokens,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "results": results,
    }


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
