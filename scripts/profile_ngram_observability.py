#!/usr/bin/env python3
"""Profile n-gram speculative-decode behaviour split by prompt class.

This script produces the REQ-6 artifact required by the DS4 Reference
Learnings PRD (`.internal/planning/DS4-REFERENCE-LEARNINGS-PRD.md`). It
exercises one supported model with a mixed corpus of repeating and
non-repeating prompts, captures the per-request `ax_*` route decisions
already emitted by `ax-engine-mlx`, and aggregates accept-rate / fallback
ratios per `ax_prompt_class_code`.

The artifact is intentionally evidence-only. It does not modify the n-gram
policy. Its purpose is to make per-family tuning of
`AX_NGRAM_CONFIDENCE_THRESHOLD` (see `ngram_accel.rs::parse_confidence_threshold`)
debuggable from a benchmark row instead of a profiler attach.

Artifact schema: ax.ngram_observability.v1

    {
      "schema_version": "ax.ngram_observability.v1",
      "generated_at_utc": "...",
      "model": { "model_id": str, "artifacts_dir": str },
      "host": { "platform": str },
      "config": {
        "confidence_threshold_env": str | null,    # raw env value, if set
        "confidence_threshold_resolved": float | null,
        "max_output_tokens": int,
        "seed": int,
        "requests_per_class": int
      },
      "by_prompt_class": {
        "non_repeating": <stats>,
        "repeating": <stats>,
        "unset": <stats>
      }
    }

    <stats> = {
        "request_count": int,
        "total_draft_attempts": int,
        "total_draft_tokens": int,
        "total_accepted_tokens": int,
        "accept_rate": float | null,                      # accepted / draft tokens
        "no_candidate_fallback_steps": int,
        "confidence_filtered_fallback_steps": int,
        "short_output_fallback_steps": int,
        "cooldown_steps": int
    }

Usage:
    python scripts/profile_ngram_observability.py \\
        --model-id qwen3_dense \\
        --mlx-artifacts-dir /path/to/mlx-community/Qwen3-9B-4bit \\
        --output-root benchmarks/results/ngram-observability

    # Tune and re-run:
    AX_NGRAM_CONFIDENCE_THRESHOLD=0.55 python scripts/profile_ngram_observability.py ...

Prompt-class classification matches `ngram_accel::classify_prompt_class`:
4-gram self-similarity ratio ≤ 0.5 ⇒ repeating, else non-repeating. The
script builds prompts that are guaranteed to land in each class so the
artifact has at least one row per class.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "ax.ngram_observability.v1"
CONFIDENCE_THRESHOLD_ENV = "AX_NGRAM_CONFIDENCE_THRESHOLD"
PROMPT_CLASS_CODE_KEY = "ax_prompt_class_code"
PROMPT_CLASS_UNSET = 0
PROMPT_CLASS_NON_REPEATING = 1
PROMPT_CLASS_REPEATING = 2
CLASS_NAMES = {
    PROMPT_CLASS_UNSET: "unset",
    PROMPT_CLASS_NON_REPEATING: "non_repeating",
    PROMPT_CLASS_REPEATING: "repeating",
}
DEFAULT_REQUESTS_PER_CLASS = 4
DEFAULT_MAX_OUTPUT_TOKENS = 64
DEFAULT_SEED = 1234

# Source paragraphs for the non-repeating class. Different content per request
# avoids any 4-gram repeat across prompts that would push them into the
# repeating bucket. The token classifier is per-prompt, so cross-prompt
# diversity does not matter for classification, only intra-prompt diversity.
NON_REPEATING_SOURCES = [
    (
        "Margaret turned the brass key with steady fingers and pushed open the "
        "warehouse door, which groaned on hinges that had not seen oil for at "
        "least a decade, releasing a long breath of stale air into the cold "
        "morning. The sky over the harbour was the colour of pewter and the "
        "gulls wheeled silently above the wharves."
    ),
    (
        "Quantum error correction codes encode logical qubits across many "
        "physical qubits so that local noise can be detected and reversed "
        "without disturbing the encoded information. Surface codes have been "
        "favoured for hardware implementations because their stabiliser checks "
        "only involve nearest-neighbour interactions on a planar lattice."
    ),
    (
        "Bread depends on a small ecology of microbes interacting with starch "
        "and gluten under controlled hydration. A long, cool fermentation lets "
        "the dough develop complex aromas while the yeast slowly inflates "
        "the structure built by the kneaded protein matrix."
    ),
    (
        "The shipping forecast for sea area Viking, North Utsire, and South "
        "Utsire reports winds south-west five to seven, occasionally gale "
        "eight later; sea state moderate or rough; weather rain at times, "
        "fog patches; visibility good, occasionally very poor."
    ),
    (
        "A garden snail moves on a single muscular foot, propelled by waves of "
        "contraction that travel along its underside, and lubricates its path "
        "with a mucus that doubles as a defensive barrier against predators "
        "and as a glue when the animal anchors itself to a vertical surface."
    ),
    (
        "Index funds owe their popularity to a simple observation: most "
        "actively managed equity funds underperform their benchmark after "
        "fees over multi-decade horizons. Holding the benchmark directly "
        "captures the underlying market return at near-zero overhead."
    ),
    (
        "Late in the evening the printer started up unprompted and produced "
        "three pages of nothing but the letter Q, followed by a single blank "
        "sheet, after which it returned to its normal idle state and refused "
        "to acknowledge that anything unusual had happened."
    ),
    (
        "Glacial moraines record the maximum extent of an ice sheet because "
        "the rock fragments that the glacier transported are dropped along its "
        "leading edge when the ice retreats, forming a ridge that outlasts "
        "the climate change responsible for the retreat by many thousands of "
        "years."
    ),
]

# Source phrases for the repeating class. The classifier looks at 4-gram
# uniqueness; short cycles guarantee the ratio is well below 0.5.
REPEATING_TEMPLATES = [
    "Translate to French: hello hello hello hello hello hello hello hello "
    "hello hello hello hello hello hello hello hello hello hello hello hello.",
    "List the numbers: one two three one two three one two three one two "
    "three one two three one two three one two three one two three.",
    "Repeat after me: ping pong ping pong ping pong ping pong ping pong ping "
    "pong ping pong ping pong ping pong ping pong ping pong ping pong.",
    "Echo: alpha beta gamma alpha beta gamma alpha beta gamma alpha beta "
    "gamma alpha beta gamma alpha beta gamma alpha beta gamma alpha beta gamma.",
    "Sing the chorus: la la la la la la la la la la la la la la la la la la "
    "la la la la la la la la la la la la la la la la la la la la la la la.",
    "Loop: red green blue red green blue red green blue red green blue red "
    "green blue red green blue red green blue red green blue red green blue.",
    "Pattern: dot dash dot dash dot dash dot dash dot dash dot dash dot dash "
    "dot dash dot dash dot dash dot dash dot dash dot dash dot dash dot dash.",
    "Drill: left right left right left right left right left right left right "
    "left right left right left right left right left right left right.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-id", required=True)
    p.add_argument("--mlx-artifacts-dir", required=True, type=Path)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/results/ngram-observability"),
    )
    p.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    p.add_argument("--requests-per-class", type=int, default=DEFAULT_REQUESTS_PER_CLASS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def empty_stats() -> dict:
    return {
        "request_count": 0,
        "total_output_tokens": 0,
        "total_draft_attempts": 0,
        "total_draft_tokens": 0,
        "total_accepted_tokens": 0,
        "accept_rate": None,
        "no_candidate_fallback_steps": 0,
        "confidence_filtered_fallback_steps": 0,
        "short_output_fallback_steps": 0,
        "cooldown_steps": 0,
    }


def update_stats(stats: dict, decisions: dict[str, int], output_tokens: int) -> None:
    stats["request_count"] += 1
    stats["total_output_tokens"] += output_tokens
    stats["total_draft_attempts"] += decisions.get("ax_ngram_draft_attempts", 0)
    stats["total_draft_tokens"] += decisions.get("ax_ngram_draft_tokens", 0)
    stats["total_accepted_tokens"] += decisions.get("ax_ngram_accepted_tokens", 0)
    stats["no_candidate_fallback_steps"] += decisions.get(
        "ax_ngram_fallback_no_candidate_steps", 0
    )
    stats["confidence_filtered_fallback_steps"] += decisions.get(
        "ax_ngram_fallback_confidence_filtered_steps", 0
    )
    stats["short_output_fallback_steps"] += decisions.get(
        "ax_ngram_fallback_short_output_steps", 0
    )
    stats["cooldown_steps"] += decisions.get("ax_ngram_cooldown_steps", 0)


def finalize_stats(stats: dict) -> None:
    drafts = stats["total_draft_tokens"]
    stats["accept_rate"] = (
        round(stats["total_accepted_tokens"] / drafts, 4) if drafts > 0 else None
    )


def resolved_confidence_threshold(env_value: str | None) -> float | None:
    if env_value is None:
        return None
    try:
        parsed = float(env_value)
    except ValueError:
        raise SystemExit(
            f"{CONFIDENCE_THRESHOLD_ENV} must be a float; got {env_value!r}"
        )
    if not (0.0 <= parsed <= 1.0):
        raise SystemExit(
            f"{CONFIDENCE_THRESHOLD_ENV} must be in [0.0, 1.0]; got {parsed}"
        )
    return parsed


def run(args: argparse.Namespace) -> dict:
    try:
        from ax_engine import Session
    except ImportError as e:
        raise SystemExit(
            "ax_engine module not importable. Run `maturin develop` from the "
            "repo root to build and install the Python extension."
        ) from e

    if not args.mlx_artifacts_dir.is_dir():
        raise SystemExit(
            f"--mlx-artifacts-dir does not exist: {args.mlx_artifacts_dir}"
        )

    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise SystemExit(
            "tokenizers package not installed. Install with `pip install tokenizers`."
        ) from e
    tok_path = args.mlx_artifacts_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise SystemExit(f"tokenizer.json not found under {args.mlx_artifacts_dir}")
    tokenizer = Tokenizer.from_file(str(tok_path))

    env_value = os.environ.get(CONFIDENCE_THRESHOLD_ENV)
    resolved = resolved_confidence_threshold(env_value)

    session = Session(
        model_id=args.model_id,
        mlx=True,
        mlx_model_artifacts_dir=str(args.mlx_artifacts_dir),
        deterministic=True,
    )

    by_class: dict[str, dict] = {name: empty_stats() for name in CLASS_NAMES.values()}

    n = args.requests_per_class
    prompts = []
    for i in range(n):
        prompts.append(("non_repeating_expected", NON_REPEATING_SOURCES[i % len(NON_REPEATING_SOURCES)]))
        prompts.append(("repeating_expected", REPEATING_TEMPLATES[i % len(REPEATING_TEMPLATES)]))

    for _expected, text in prompts:
        input_tokens = tokenizer.encode(text).ids
        result = session.generate(
            input_tokens=input_tokens,
            max_output_tokens=args.max_output_tokens,
            temperature=0.0,
            seed=args.seed,
            deterministic=True,
        )
        decisions = (
            dict(result.route.crossover_decisions)
            if result.route and result.route.crossover_decisions
            else {}
        )
        class_code = decisions.get(PROMPT_CLASS_CODE_KEY, PROMPT_CLASS_UNSET)
        class_name = CLASS_NAMES.get(class_code, "unset")
        output_token_count = len(result.output_tokens) if result.output_tokens else 0
        update_stats(by_class[class_name], decisions, output_token_count)

    for stats in by_class.values():
        finalize_stats(stats)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "model_id": args.model_id,
            "artifacts_dir": str(args.mlx_artifacts_dir),
        },
        "host": {"platform": platform.platform()},
        "config": {
            "confidence_threshold_env": env_value,
            "confidence_threshold_resolved": resolved,
            "max_output_tokens": args.max_output_tokens,
            "seed": args.seed,
            "requests_per_class": n,
        },
        "by_prompt_class": by_class,
    }


def write_artifact(args: argparse.Namespace, artifact: dict) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    safe_id = args.model_id.replace("/", "_").replace(" ", "_")
    date_part = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = args.output_root / f"{safe_id}-{date_part}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    args = parse_args()
    artifact = run(args)
    out = write_artifact(args, artifact)
    print(f"wrote {out}")
    for name, stats in artifact["by_prompt_class"].items():
        if stats["request_count"] == 0:
            continue
        rate = stats["accept_rate"]
        rate_str = f"{rate:.3f}" if rate is not None else "n/a"
        print(
            f"  {name}: n={stats['request_count']} accept_rate={rate_str} "
            f"draft_tokens={stats['total_draft_tokens']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
