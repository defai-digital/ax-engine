#!/usr/bin/env python3
"""Multi-turn chat baseline for the Phase C prefix-cache decision.

Drives a single `ax_engine.Session` through N iterative chat turns and
captures per-turn TTFT, decode tok/s, and prefix-cache telemetry. Each
turn appends the previous turn's output plus a fresh user-turn delta,
matching the live-checkpoint growth pattern that DS4's
`ds4_session_sync` handles.

Companion plan: `.internal/planning/MLX-PHASE-C-MULTITURN-BASELINE-PLAN-<date>.md`.
That plan defines the go/no-go decision rules consumed by the next
artifact.

Output schema: `ax.kv_multiturn_chat_evidence.v1` with per-turn rows
plus a derived `ttft_growth_ratio = turn_N.ttft_s / turn_2.ttft_s` and
a coarse verdict suggestion. Artifacts also record selected environment
flags that can alter prefix-cache, n-gram, or TurboQuant paths.

Usage:

    python scripts/profile_kv_multiturn_chat_evidence.py \\
        --model-id gemma4 \\
        --mlx-artifacts-dir .internal/models/gemma-4-e2b-it-4bit \\
        --turns 10 \\
        --output benchmarks/results/kv-long-context/gemma4-multiturn-2026-05-14.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "ax.kv_multiturn_chat_evidence.v1"
DEFAULT_PROMPT_TOKENS = 2048
DEFAULT_TURNS = 10
DEFAULT_DECODE_TOKENS = 64
DEFAULT_USER_DELTA_TOKENS = 128

PROVENANCE_ENV_FLAGS = [
    "AX_ALLOW_MLA_PREFIX_RESTORE",
    "AX_DISABLE_TURBOQUANT_FUSED_DECODE",
    "AX_MLX_DIRECT_CLEAR_CACHE_CADENCE",
    "AX_MLX_NGRAM_POLICY",
    "AX_MLX_PREFIX_CACHE_MAX_BYTES",
    "AX_MLX_PREFIX_CACHE_MAX_ENTRIES",
    "AX_NGRAM_CONFIDENCE_THRESHOLD",
    "AX_NO_SPEC",
]

BOOLEAN_PROVENANCE_ENV_FLAGS = {
    "AX_ALLOW_MLA_PREFIX_RESTORE",
    "AX_DISABLE_TURBOQUANT_FUSED_DECODE",
    "AX_NO_SPEC",
}

# Telemetry keys this script consumes from `route.crossover_decisions`.
# Kept in sync with `profile_kv_long_context_evidence.py` so JSON
# artifacts from both harnesses are easy to diff.
TELEMETRY_KEYS = [
    # KV capacity / usage
    "ax_mlx_kv_capacity_tokens",
    "ax_mlx_kv_capacity_kib",
    "ax_mlx_kv_logical_tokens",
    "ax_mlx_kv_logical_kib",
    "ax_mlx_kv_request_snapshots",
    "ax_mlx_kv_full_attention_layers",
    "ax_mlx_kv_sliding_window_layers",
    "ax_mlx_kv_growth_count",
    "ax_mlx_kv_sliding_reclaimable_capacity_tokens",
    "ax_mlx_kv_sliding_retained_tokens",
    # Engine-level prefix reuse (logical)
    "prefix_reused_blocks",
    "prefix_reused_tokens",
    "prefix_reused_requests",
    "max_prefix_blocks_reused_per_request",
    "retained_cache_hits",
    "live_share_hits",
    "blocked_prefix_reuse_blocks",
    "blocked_prefix_reuse_tokens",
    "blocked_prefix_reuse_requests",
    # MLX-runner physical prefix cache
    "ax_mlx_prefix_cache_hits",
    "ax_mlx_prefix_cache_misses",
    "ax_mlx_prefix_cache_blocked",
    "ax_mlx_prefix_cache_blocked_policy_disabled",
    "ax_mlx_prefix_cache_blocked_unsupported_layout",
    "ax_mlx_prefix_cache_blocked_trim_failure",
    "ax_mlx_prefix_cache_stores",
    "ax_mlx_prefix_cache_evictions",
    "ax_mlx_prefix_cache_reused_tokens",
    "ax_mlx_prefix_cache_warmup_tokens",
    "ax_mlx_prefix_cache_entries",
    "ax_mlx_prefix_cache_bytes_kib",
    # Decode / prefill timing
    "ax_mlx_decode_steps",
    "ax_mlx_decode_wall_us",
    "ax_mlx_single_decode_steps",
    "ax_mlx_single_decode_wall_us",
    "ax_mlx_ngram_decode_steps",
    "ax_mlx_ngram_decode_wall_us",
    "ax_mlx_bonus_tokens",
    "ax_mlx_prefill_steps",
    "ax_mlx_prefill_wall_us",
    "ax_mlx_prefill_drain_async_evals",
    "ax_mlx_prefill_eval_barriers",
]


def parse_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def collect_environment_flags() -> dict[str, dict[str, object]]:
    return {
        name: {
            "set": name in os.environ,
            "value": os.environ.get(name),
            "truthy": (
                parse_truthy_env(os.environ.get(name))
                if name in BOOLEAN_PROVENANCE_ENV_FLAGS
                else None
            ),
        }
        for name in PROVENANCE_ENV_FLAGS
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-id", required=True)
    p.add_argument("--mlx-artifacts-dir", required=True, type=Path)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/kv-long-context"),
        help="Output JSON path or directory. If a directory, the filename "
        "is auto-generated as <model_id_safe>-multiturn-<date>.json.",
    )
    p.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    p.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    p.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS)
    p.add_argument(
        "--user-delta-tokens",
        type=int,
        default=DEFAULT_USER_DELTA_TOKENS,
        help="Approximate token count appended as the new user turn each round.",
    )
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


# Each user-turn delta uses a distinct text seed so the tokenizer cannot
# trivially collapse the new tokens into the previous turn's tail. The
# bench is about prefix reuse across genuine deltas, not against
# identical-suffix degeneracies.
USER_DELTA_TEXTS = [
    "Could you elaborate on the second point with a concrete example? ",
    "Now suppose we ran this on a smaller dataset with noisy labels. ",
    "What changes if we replace the activation with a gated variant? ",
    "Walk me through the failure mode when the cache is cold. ",
    "How would the latency budget shift on a thermally throttled laptop? ",
    "Tell me about the assumptions that break under heavy concurrency. ",
    "If we cut the context window in half, which step suffers most? ",
    "Suppose the upstream tokenizer changes; what regresses first? ",
    "Sketch the migration path when the model family changes. ",
    "Finally, summarize the three top risks for the production rollout. ",
]


def synthesize_prompt(target_tokens: int) -> str:
    base = (
        "The quick brown fox jumps over the lazy dog while the engine "
        "warms its caches and the operator records throughput metrics. "
    )
    repeats = max(1, (target_tokens // 16) + 4)
    return base * repeats


def synthesize_user_delta(turn_index: int, target_tokens: int) -> str:
    # Cycle through the canned deltas so each turn is genuinely different
    # in text but predictable to the bench operator.
    seed = USER_DELTA_TEXTS[turn_index % len(USER_DELTA_TEXTS)]
    repeats = max(1, (target_tokens // 16) + 2)
    return seed * repeats


def tokenize_to_target(tokenizer, text: str, target_tokens: int) -> list[int]:
    ids = tokenizer.encode(text).ids
    return ids[:target_tokens]


def run_turn(
    session, input_tokens: list[int], decode_tokens: int, seed: int
) -> dict:
    """Execute one turn and capture TTFT + telemetry."""
    t0 = time.monotonic()
    first_token_t: float | None = None
    output_tokens: list[int] = []
    crossover_decisions: dict[str, int] = {}
    runner_us_acc = 0
    for event in session.stream_generate(
        input_tokens=input_tokens,
        max_output_tokens=decode_tokens,
        temperature=0.0,
        seed=seed,
        deterministic=True,
    ):
        if event.delta_tokens:
            if first_token_t is None:
                first_token_t = time.monotonic()
            output_tokens.extend(event.delta_tokens)
        if event.event == "response" and event.response is not None:
            if event.response.route and event.response.route.crossover_decisions:
                crossover_decisions = dict(event.response.route.crossover_decisions)
        if event.event == "step" and event.step is not None:
            if event.step.runner_time_us:
                runner_us_acc += event.step.runner_time_us
    t1 = time.monotonic()
    total_wall_s = t1 - t0
    ttft_s = (first_token_t - t0) if first_token_t else total_wall_s
    decode_wall_s = (t1 - first_token_t) if first_token_t else 0.0
    return {
        "input_token_count": len(input_tokens),
        "output_token_count": len(output_tokens),
        "output_tokens": output_tokens,
        "ttft_s": round(ttft_s, 6),
        "total_wall_s": round(total_wall_s, 6),
        "decode_wall_s": round(decode_wall_s, 6),
        "decode_tok_s": (
            round(len(output_tokens) / decode_wall_s, 3)
            if decode_wall_s > 0
            else None
        ),
        "prefill_tok_s": (
            round(len(input_tokens) / ttft_s, 3) if ttft_s > 0 else None
        ),
        "runner_time_us_total": runner_us_acc,
        "telemetry": {k: crossover_decisions.get(k, 0) for k in TELEMETRY_KEYS},
    }


def derive_summary(turns: list[dict]) -> dict:
    """Compute coarse summary signals consumed by the decision rules in
    the companion planning artifact."""
    if not turns:
        return {"turns": 0, "verdict_hint": "no_turns_recorded"}
    ttft_first = turns[0]["ttft_s"]
    ttft_last = turns[-1]["ttft_s"]
    ttft_turn2 = turns[1]["ttft_s"] if len(turns) >= 2 else None
    growth_ratio = None
    if ttft_turn2 is not None and ttft_turn2 > 0:
        growth_ratio = round(ttft_last / ttft_turn2, 3)
    cache_hits_total = sum(
        t["telemetry"].get("ax_mlx_prefix_cache_hits", 0) for t in turns
    )
    cache_misses_total = sum(
        t["telemetry"].get("ax_mlx_prefix_cache_misses", 0) for t in turns
    )
    physical_reused_tokens_total = sum(
        t["telemetry"].get("ax_mlx_prefix_cache_reused_tokens", 0) for t in turns
    )

    # Map to the plan's decision table. These are hints, not verdicts —
    # the planning artifact's decision rules are authoritative.
    verdict_hint = "indeterminate"
    if growth_ratio is not None:
        if growth_ratio < 1.2:
            verdict_hint = "phase_c_skip__existing_infra_captures_win"
        elif growth_ratio < 2.0:
            verdict_hint = "phase_c_narrow_headroom__reassess_after_phase_b"
        elif growth_ratio < 4.0:
            verdict_hint = "phase_c_substantial_headroom__do"
        else:
            verdict_hint = "phase_c_immediate__growth_pathological"
    if cache_hits_total == 0 and len(turns) >= 2:
        verdict_hint = (
            "phase_c_bugfix__snapshot_never_fires_across_turns"
        )

    return {
        "turns": len(turns),
        "ttft_turn1_s": round(ttft_first, 6),
        "ttft_turn2_s": round(ttft_turn2, 6) if ttft_turn2 else None,
        "ttft_turn_last_s": round(ttft_last, 6),
        "ttft_growth_ratio": growth_ratio,
        "ax_mlx_prefix_cache_hits_total": cache_hits_total,
        "ax_mlx_prefix_cache_misses_total": cache_misses_total,
        "ax_mlx_prefix_cache_reused_tokens_total": physical_reused_tokens_total,
        "verdict_hint": verdict_hint,
    }


def run(args: argparse.Namespace) -> dict:
    try:
        from ax_engine import Session
    except ImportError as e:
        raise SystemExit(
            "ax_engine not importable; run `maturin develop` first"
        ) from e
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

    initial_prompt_text = synthesize_prompt(args.prompt_tokens)
    initial_tokens = tokenize_to_target(
        tokenizer, initial_prompt_text, args.prompt_tokens
    )

    print(
        f"initial prompt tokens: {len(initial_tokens)}; "
        f"turns: {args.turns}; user-delta tokens/turn ≈ {args.user_delta_tokens}"
    )
    print(f"loading {args.mlx_artifacts_dir} via ax_engine ...")
    session = Session(
        model_id=args.model_id,
        mlx=True,
        mlx_model_artifacts_dir=str(args.mlx_artifacts_dir),
        deterministic=True,
    )

    print("warmup ...")
    _ = session.generate(
        input_tokens=initial_tokens[:64],
        max_output_tokens=4,
        temperature=0.0,
        seed=args.seed,
    )

    turns: list[dict] = []
    accumulated_tokens: list[int] = list(initial_tokens)

    for turn_index in range(args.turns):
        print(
            f"turn {turn_index + 1}/{args.turns}: "
            f"input_tokens={len(accumulated_tokens)}"
        )
        turn_result = run_turn(
            session,
            accumulated_tokens,
            args.decode_tokens,
            args.seed + turn_index,
        )
        turn_result["turn_index"] = turn_index + 1
        turns.append(turn_result)

        # Build the next turn's input: previous accumulated + this turn's
        # output + a fresh user delta.
        delta_text = synthesize_user_delta(turn_index, args.user_delta_tokens)
        delta_tokens = tokenize_to_target(
            tokenizer, delta_text, args.user_delta_tokens
        )
        accumulated_tokens = (
            accumulated_tokens + turn_result["output_tokens"] + delta_tokens
        )

    summary = derive_summary(turns)

    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "environment_flags": collect_environment_flags(),
        "args": {
            "model_id": args.model_id,
            "mlx_artifacts_dir": str(args.mlx_artifacts_dir),
            "prompt_tokens": args.prompt_tokens,
            "turns": args.turns,
            "decode_tokens": args.decode_tokens,
            "user_delta_tokens": args.user_delta_tokens,
            "seed": args.seed,
        },
        "turns": turns,
        "summary": summary,
    }


def resolve_output_path(out: Path, model_id: str) -> Path:
    if out.is_dir() or (not out.exists() and out.suffix == ""):
        out.mkdir(parents=True, exist_ok=True)
        safe_id = model_id.replace("/", "-")
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return out / f"{safe_id}-multiturn-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main() -> int:
    args = parse_args()
    result = run(args)
    out_path = resolve_output_path(args.output, args.model_id)
    out_path.write_text(json.dumps(result, indent=2))
    summary = result["summary"]
    print(f"\nwrote {out_path}")
    print(
        f"ttft turn1={summary['ttft_turn1_s']}s "
        f"turn2={summary.get('ttft_turn2_s')}s "
        f"last={summary['ttft_turn_last_s']}s "
        f"growth_ratio={summary['ttft_growth_ratio']}"
    )
    print(f"verdict_hint: {summary['verdict_hint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
