#!/usr/bin/env python3
"""W1 evidence collection for KV-SCHEDULER-REVAMP-PRD §W1.

Runs three scenarios on a single model that exercise the prefix-cache /
KV-snapshot machinery, captures TTFT, decode tok/s, prefill tok/s, and the
route-decision telemetry that ADR 0018 W1 exit criteria require:

  cold              first call with a fresh prompt; no prefix reuse possible
  warm_repeat       same prompt again immediately; retained logical prefix
                    + physical MLX snapshot should both fire
  warm_extend       prompt = cold_prompt + new suffix; logical prefix should
                    cover the cold-prompt prefix; new suffix needs warmup

Output: ax.kv_long_context_evidence.v1 artifact with per-scenario telemetry
and a derived bottleneck verdict. Designed to satisfy the W1 exit criteria:

  > A checked-in artifact identifies the most important KV/prefix bottleneck.
  > Single-request throughput artifacts with zero prefix reuse do not count.

Usage:

    python scripts/profile_kv_long_context_evidence.py \\
        --model-id gemma4 \\
        --mlx-artifacts-dir .internal/models/gemma-4-e2b-it-4bit \\
        --output benchmarks/results/kv-long-context/gemma4-2026-05-11.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "ax.kv_long_context_evidence.v1"
DEFAULT_PROMPT_TOKENS = 2048
DEFAULT_DECODE_TOKENS = 64
DEFAULT_EXTENSION_TOKENS = 256


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
        help="Output JSON path or directory. If a directory, file name is "
        "auto-generated as <model_id_safe>-<date>.json.",
    )
    p.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    p.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS)
    p.add_argument(
        "--extension-tokens",
        type=int,
        default=DEFAULT_EXTENSION_TOKENS,
        help="Number of extra tokens appended for the warm_extend scenario.",
    )
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def synthesize_prompt(target_tokens: int) -> str:
    base = (
        "The quick brown fox jumps over the lazy dog while the engine "
        "warms its caches and the operator records throughput metrics. "
    )
    repeats = max(1, (target_tokens // 16) + 4)
    return base * repeats


def synthesize_extension(target_tokens: int) -> str:
    base = (
        "Margaret turned the brass key with steady fingers and pushed "
        "open the warehouse door. "
    )
    repeats = max(1, (target_tokens // 16) + 4)
    return base * repeats


def tokenize_to_target(tokenizer, text: str, target_tokens: int) -> list[int]:
    ids = tokenizer.encode(text).ids
    return ids[:target_tokens]


# Telemetry keys to extract from route decisions for the artifact.
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
    # MLX-runner physical prefix cache (KV snapshot store + restore)
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


def run_scenario(session, prompt_tokens: list[int], decode_tokens: int, seed: int) -> dict:
    t0 = time.monotonic()
    first_token_t: float | None = None
    output_tokens: list[int] = []
    crossover_decisions: dict[str, int] = {}
    runtime_us_acc = 0
    for event in session.stream_generate(
        input_tokens=prompt_tokens,
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
                runtime_us_acc += event.step.runner_time_us
    t1 = time.monotonic()
    total_wall_s = t1 - t0
    ttft_s = (first_token_t - t0) if first_token_t else total_wall_s
    decode_wall_s = (t1 - first_token_t) if first_token_t else 0.0
    return {
        "prompt_token_count": len(prompt_tokens),
        "output_token_count": len(output_tokens),
        "ttft_s": round(ttft_s, 6),
        "total_wall_s": round(total_wall_s, 6),
        "decode_wall_s": round(decode_wall_s, 6),
        "decode_tok_s": (
            round(len(output_tokens) / decode_wall_s, 3)
            if decode_wall_s > 0
            else None
        ),
        "prefill_tok_s": (
            round(len(prompt_tokens) / ttft_s, 3) if ttft_s > 0 else None
        ),
        "runner_time_us_total": runtime_us_acc,
        "telemetry": {
            k: crossover_decisions.get(k, 0) for k in TELEMETRY_KEYS
        },
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

    cold_prompt_text = synthesize_prompt(args.prompt_tokens)
    cold_tokens = tokenize_to_target(tokenizer, cold_prompt_text, args.prompt_tokens)
    extension_text = synthesize_extension(args.extension_tokens)
    extension_tokens = tokenize_to_target(
        tokenizer, extension_text, args.extension_tokens
    )
    extended_tokens = cold_tokens + extension_tokens

    print(f"prompt tokens: {len(cold_tokens)}; extension: {len(extension_tokens)}")
    print(f"loading {args.mlx_artifacts_dir} via ax_engine ...")
    session = Session(
        model_id=args.model_id,
        mlx=True,
        mlx_model_artifacts_dir=str(args.mlx_artifacts_dir),
        deterministic=True,
    )

    print("warmup ...")
    _ = session.generate(
        input_tokens=cold_tokens[:64],
        max_output_tokens=4,
        temperature=0.0,
        seed=args.seed,
    )

    print("scenario 1/3: cold (fresh prompt, no prefix reuse expected)")
    cold = run_scenario(session, cold_tokens, args.decode_tokens, args.seed)

    print("scenario 2/3: warm_repeat (same prompt, retained prefix + snapshot expected)")
    warm_repeat = run_scenario(session, cold_tokens, args.decode_tokens, args.seed + 1)

    print(f"scenario 3/3: warm_extend (cold prompt + {len(extension_tokens)} new tokens)")
    warm_extend = run_scenario(
        session, extended_tokens, args.decode_tokens, args.seed + 2
    )

    scenarios = {
        "cold": cold,
        "warm_repeat": warm_repeat,
        "warm_extend": warm_extend,
    }

    # Compute deltas / bottleneck verdict.
    cold_tel = cold["telemetry"]
    warm_tel = warm_repeat["telemetry"]
    prefix_blocks_warm = warm_tel.get("prefix_reused_blocks", 0)
    retained_hits_warm = warm_tel.get("retained_cache_hits", 0)
    ttft_speedup = (
        cold["ttft_s"] / warm_repeat["ttft_s"]
        if warm_repeat["ttft_s"] > 0
        else None
    )
    extend_prefix_blocks = warm_extend["telemetry"].get("prefix_reused_blocks", 0)

    bottleneck = "unknown"
    bottleneck_reason = ""
    if prefix_blocks_warm == 0 and retained_hits_warm == 0:
        bottleneck = "prefix_reuse_not_firing"
        bottleneck_reason = (
            "warm_repeat scenario shows zero retained-prefix hits and zero "
            "prefix_reused_blocks. Retained cache + MLX snapshot lookups are "
            "either disabled, not finding the entry, or being blocked. This "
            "is the most important fix before any structural cache change."
        )
    elif ttft_speedup is not None and ttft_speedup < 1.2:
        bottleneck = "prefix_reuse_low_value"
        bottleneck_reason = (
            f"warm_repeat TTFT speedup = {ttft_speedup:.2f}× — prefix reuse "
            "fires but yields little TTFT improvement. Either the warmup "
            "phase is heavy or the physical snapshot restore costs eat into "
            "the prefill saving."
        )
    elif extend_prefix_blocks == 0:
        bottleneck = "prefix_reuse_no_extension"
        bottleneck_reason = (
            "warm_extend scenario shows zero prefix-block reuse despite the "
            "cold-prompt prefix being identical. Prefix matching may be "
            "exact-match-only and not handling extension cases."
        )
    else:
        bottleneck = "prefix_reuse_working"
        bottleneck_reason = (
            f"warm_repeat TTFT speedup = {ttft_speedup:.2f}×; warm_extend "
            f"reused {extend_prefix_blocks} blocks. Prefix cache is "
            "functioning; bottleneck is elsewhere (decode kernel, KV growth, "
            "or scheduler eviction)."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.platform(),
        "model": {
            "model_id": args.model_id,
            "artifacts_dir": str(args.mlx_artifacts_dir),
        },
        "config": {
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "extension_tokens": args.extension_tokens,
            "seed": args.seed,
        },
        "scenarios": scenarios,
        "deltas": {
            "warm_repeat_vs_cold": {
                "ttft_speedup": round(ttft_speedup, 3) if ttft_speedup else None,
                "decode_tok_s_ratio": (
                    round(
                        warm_repeat["decode_tok_s"] / cold["decode_tok_s"], 3
                    )
                    if cold["decode_tok_s"] and warm_repeat["decode_tok_s"]
                    else None
                ),
            },
            "warm_extend_prefix_reused_blocks": extend_prefix_blocks,
            "warm_repeat_retained_cache_hits": retained_hits_warm,
            "warm_repeat_prefix_reused_blocks": prefix_blocks_warm,
        },
        "bottleneck_verdict": {
            "label": bottleneck,
            "reason": bottleneck_reason,
        },
    }


def main() -> int:
    args = parse_args()
    artifact = run(args)
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
    v = artifact["bottleneck_verdict"]
    print(f"bottleneck: {v['label']}")
    print(f"  {v['reason']}")
    for name, sc in artifact["scenarios"].items():
        print(
            f"  {name}: TTFT={sc['ttft_s']:.3f}s "
            f"decode_tok_s={sc['decode_tok_s']} "
            f"prefix_reused_blocks={sc['telemetry'].get('prefix_reused_blocks', 0)} "
            f"retained_cache_hits={sc['telemetry'].get('retained_cache_hits', 0)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
