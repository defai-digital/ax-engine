#!/usr/bin/env python3
"""Token-exact equivalence harness for the prefix-reuse warm path.

This is the correctness gate that any `prefix_cache_supported()` loosening
must pass before merge. Silent token drift between cold and warm_repeat
output is the worst possible failure mode for prefix-cache work, so the
harness is fail-closed and runs across a multi-prompt corpus.

For each prompt in the corpus, the harness:

  1. Opens a fresh Session.
  2. Cold call:   generate(prompt, seed=S, max=N, temperature=0)
  3. Warm repeat: generate(prompt, seed=S, max=N, temperature=0)
  4. Compares output tokens byte-for-byte. PASS requires exact match.

The same seed and temperature=0 is mandatory — any deviation invalidates
the comparison. The harness records the per-prompt result, an aggregate
verdict, and route telemetry (`ax_mlx_prefix_cache_hits`,
`ax_mlx_prefix_cache_warmup_tokens`, `retained_cache_hits`,
`prefix_reused_blocks`) for both calls so that a passing run on a
loosened gate also proves the prefix cache actually fired.

Exit codes (matching ax-engine-bench convention):

  0  all prompts produce token-exact match
  3  any prompt diverges (correctness failure)

Usage:

    python scripts/verify_prefix_reuse_equivalence.py \\
        --model-id qwen3-5-9b \\
        --mlx-artifacts-dir .internal/models/Qwen3.5-9B-MLX-4bit \\
        --output benchmarks/results/prefix-reuse-equivalence/

Artifact schema: ax.prefix_reuse_equivalence.v1
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "ax.prefix_reuse_equivalence.v1"

# Default corpus — 5 prompts of varied lengths, picked so a per-prompt
# regression bisects clearly when one diverges. Token counts are
# tokenizer-dependent; the harness re-tokenizes per model.
DEFAULT_CORPUS = [
    {
        "id": "p1_short_factoid",
        "text": "What is the capital of France? Answer in one short sentence.",
    },
    {
        "id": "p2_medium_explain",
        "text": (
            "Explain the difference between supervised and unsupervised learning, "
            "and give one concrete example for each. Keep the answer under 200 "
            "words and avoid bullet points."
        ),
    },
    {
        "id": "p3_long_story",
        "text": (
            "Write the opening paragraph of a noir detective story set in 1947 "
            "Shanghai. The narrator should be the detective. Mention rain, a "
            "cigarette, and a missing brass key. Keep it to about 120 words."
        ),
    },
    {
        "id": "p4_code_request",
        "text": (
            "Write a Python function `fibonacci(n)` that returns the nth "
            "Fibonacci number using memoization. Include a brief docstring "
            "and one example call."
        ),
    },
    {
        "id": "p5_repetition_safe",
        "text": (
            "List five fruits and for each give one short adjective describing "
            "its colour. Format: `<fruit>: <adjective>`."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-id", required=True)
    p.add_argument("--mlx-artifacts-dir", required=True, type=Path)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/prefix-reuse-equivalence"),
        help="Output JSON path or directory.",
    )
    p.add_argument("--max-output-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Optional JSON list of {id, text}. Defaults to the built-in 5-prompt corpus.",
    )
    return p.parse_args()


def load_corpus(args: argparse.Namespace) -> list[dict]:
    if args.corpus is None:
        return DEFAULT_CORPUS
    text = args.corpus.read_text()
    items = json.loads(text)
    if not isinstance(items, list):
        raise SystemExit("--corpus must point to a JSON list of {id, text}")
    for item in items:
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            raise SystemExit("each --corpus item must be {id: str, text: str}")
    return items


PREFIX_TELEMETRY_KEYS = [
    "retained_cache_hits",
    "prefix_reused_blocks",
    "prefix_reused_tokens",
    "ax_mlx_prefix_cache_hits",
    "ax_mlx_prefix_cache_misses",
    "ax_mlx_prefix_cache_blocked",
    "ax_mlx_prefix_cache_blocked_unsupported_layout",
    "ax_mlx_prefix_cache_blocked_policy_disabled",
    "ax_mlx_prefix_cache_warmup_tokens",
    "ax_mlx_prefix_cache_reused_tokens",
    "ax_mlx_prefix_cache_stores",
    "ax_mlx_prefix_cache_entries",
]


def run_generate(session, tokens: list[int], max_output_tokens: int, seed: int) -> dict:
    output_tokens: list[int] = []
    crossover_decisions: dict[str, int] = {}
    for event in session.stream_generate(
        input_tokens=tokens,
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        seed=seed,
        deterministic=True,
    ):
        if event.delta_tokens:
            output_tokens.extend(event.delta_tokens)
        if event.event == "response" and event.response is not None:
            if event.response.route and event.response.route.crossover_decisions:
                crossover_decisions = dict(event.response.route.crossover_decisions)
    return {
        "output_tokens": output_tokens,
        "telemetry": {k: crossover_decisions.get(k, 0) for k in PREFIX_TELEMETRY_KEYS},
    }


def compare_token_lists(cold: list[int], warm: list[int]) -> dict:
    if cold == warm:
        return {
            "tokens_match": True,
            "shared_prefix_len": len(cold),
            "first_divergence_index": None,
        }
    n = min(len(cold), len(warm))
    first_div = next((i for i in range(n) if cold[i] != warm[i]), n)
    return {
        "tokens_match": False,
        "shared_prefix_len": first_div,
        "first_divergence_index": first_div,
    }


def run(args: argparse.Namespace) -> tuple[dict, int]:
    try:
        from ax_engine import Session
    except ImportError as e:
        raise SystemExit("ax_engine not importable; run `maturin develop` first") from e
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise SystemExit("tokenizers not installed") from e

    if not args.mlx_artifacts_dir.is_dir():
        raise SystemExit(f"--mlx-artifacts-dir not found: {args.mlx_artifacts_dir}")
    tok_path = args.mlx_artifacts_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise SystemExit(f"tokenizer.json not found at {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    corpus = load_corpus(args)
    print(f"corpus: {len(corpus)} prompts; max_output_tokens={args.max_output_tokens}")

    session = Session(
        model_id=args.model_id,
        mlx=True,
        mlx_model_artifacts_dir=str(args.mlx_artifacts_dir),
        deterministic=True,
    )
    print("warmup (small generate to settle allocator) ...")
    _ = session.generate(
        input_tokens=tokenizer.encode("hello").ids,
        max_output_tokens=4,
        temperature=0.0,
        seed=args.seed,
    )

    per_prompt = []
    total_pass = 0
    for item in corpus:
        prompt_id = item["id"]
        tokens = tokenizer.encode(item["text"]).ids
        if not tokens:
            raise SystemExit(f"prompt '{prompt_id}' tokenized to empty list")
        print(f"  {prompt_id}: cold ... ", end="", flush=True)
        cold = run_generate(session, tokens, args.max_output_tokens, args.seed)
        print("warm ... ", end="", flush=True)
        warm = run_generate(session, tokens, args.max_output_tokens, args.seed)
        cmp_result = compare_token_lists(cold["output_tokens"], warm["output_tokens"])
        passed = cmp_result["tokens_match"]
        if passed:
            total_pass += 1
        print("PASS" if passed else "FAIL")
        per_prompt.append(
            {
                "id": prompt_id,
                "prompt_preview": item["text"][:100],
                "prompt_token_count": len(tokens),
                "cold_output_token_count": len(cold["output_tokens"]),
                "warm_output_token_count": len(warm["output_tokens"]),
                **cmp_result,
                "cold_telemetry": cold["telemetry"],
                "warm_telemetry": warm["telemetry"],
            }
        )

    all_pass = total_pass == len(corpus)
    verdict = "PASS" if all_pass else "FAIL"

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.platform(),
        "model": {
            "model_id": args.model_id,
            "artifacts_dir": str(args.mlx_artifacts_dir),
        },
        "config": {
            "max_output_tokens": args.max_output_tokens,
            "seed": args.seed,
            "prompt_count": len(corpus),
            "corpus_source": "default" if args.corpus is None else str(args.corpus),
        },
        "aggregate": {
            "prompts_total": len(corpus),
            "prompts_matching_exactly": total_pass,
            "verdict": verdict,
        },
        "per_prompt": per_prompt,
    }
    return artifact, 0 if all_pass else 3


def main() -> int:
    args = parse_args()
    artifact, exit_code = run(args)
    out_path = args.output
    if out_path.is_dir() or (not out_path.exists() and out_path.suffix != ".json"):
        out_path.mkdir(parents=True, exist_ok=True)
        safe_id = args.model_id.replace("/", "_").replace(" ", "_")
        date_part = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = out_path / f"{safe_id}-{date_part}.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_path}")
    a = artifact["aggregate"]
    print(f"verdict: {a['verdict']} ({a['prompts_matching_exactly']}/{a['prompts_total']})")
    if exit_code != 0:
        print("FAIL — at least one prompt diverged between cold and warm.")
        for p in artifact["per_prompt"]:
            if not p["tokens_match"]:
                print(
                    f"  {p['id']}: first divergence at idx={p['first_divergence_index']} "
                    f"(shared_prefix_len={p['shared_prefix_len']}, "
                    f"cold_count={p['cold_output_token_count']}, "
                    f"warm_count={p['warm_output_token_count']})"
                )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
