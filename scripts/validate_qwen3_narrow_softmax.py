#!/usr/bin/env python3
"""Validate Qwen3 MoE narrow softmax token-for-token equivalence.

Runs the same prompts through the ax-engine with
``AX_MLX_QWEN3_MOE_NARROW_SOFTMAX=0`` (reference: full softmax_precise) and
``AX_MLX_QWEN3_MOE_NARROW_SOFTMAX=1`` (candidate: narrow argpartition-first)
and reports the token-level match rate.

Success criterion: >= 99.9% token-for-token match across all prompts.

Usage::

    python scripts/validate_qwen3_narrow_softmax.py \\
        --model-path /path/to/qwen3-coder-next \\
        --num-prompts 100

Requires ``ax_engine`` to be importable (``maturin develop``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CORPUS_SEED = 42
PROMPT_LENGTHS = (128, 512, 2048)
MAX_NEW_TOKENS = 64
MATCH_THRESHOLD = 0.999


def _build_prompts(num_prompts: int, rng_seed: int) -> list[str]:
    """Build a deterministic set of synthetic prompts."""
    import hashlib

    rng_state = rng_seed
    prompts: list[str] = []
    base_phrases = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing but code.",
        "Machine learning models require careful validation.",
        "The server processes thousands of requests per second.",
        "Rust provides memory safety without garbage collection.",
        "Attention mechanisms revolutionized natural language processing.",
        "Local inference on Apple Silicon is bandwidth-limited.",
        "Mixture of experts models route tokens to specialist networks.",
    ]
    while len(prompts) < num_prompts:
        for phrase in base_phrases:
            if len(prompts) >= num_prompts:
                break
            h = hashlib.sha256(f"{rng_state}:{phrase}".encode()).hexdigest()[:16]
            prompts.append(f"{phrase} [{h}]")
            rng_state += 1
    return prompts


def _generate_tokens(model, prompt: str, max_new_tokens: int) -> list[int]:
    """Generate tokens greededy (temperature=0) and return the token IDs."""
    tokens: list[int] = []
    try:
        result = model.generate(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        # Extract token IDs from the result.
        if hasattr(result, "token_ids"):
            tokens = list(result.token_ids)
        elif hasattr(result, "tokens"):
            tokens = [t.id if hasattr(t, "id") else int(t) for t in result.tokens]
        else:
            # Fallback: encode the output text.
            text = result.text if hasattr(result, "text") else str(result)
            tokens = list(model.encode(text))
    except Exception as exc:
        print(f"  Generation error: {exc}", file=sys.stderr)
    return tokens


def _match_rate(ref_tokens: list[int], cand_tokens: list[int]) -> float:
    """Compute the fraction of matching tokens between reference and candidate."""
    if not ref_tokens and not cand_tokens:
        return 1.0
    min_len = min(len(ref_tokens), len(cand_tokens))
    if min_len == 0:
        return 0.0
    matches = sum(1 for a, b in zip(ref_tokens[:min_len], cand_tokens[:min_len]) if a == b)
    return matches / max(len(ref_tokens), len(cand_tokens))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to the Qwen3 MoE model directory.",
    )
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Override AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR.",
    )
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir or os.environ.get("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR")
    if not artifacts_dir and not args.model_path:
        print(
            "Error: provide --model-path or set AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR.",
            file=sys.stderr,
        )
        return 1

    try:
        import ax_engine
    except ImportError:
        print("Error: ax_engine not importable. Run 'maturin develop' first.", file=sys.stderr)
        return 1

    model_path = str(args.model_path or artifacts_dir)
    prompts = _build_prompts(args.num_prompts, CORPUS_SEED)
    print(f"Validating narrow softmax with {len(prompts)} prompts, "
          f"max_new_tokens={args.max_new_tokens}")

    # --- Reference run (narrow OFF) ---
    os.environ["AX_MLX_QWEN3_MOE_NARROW_SOFTMAX"] = "0"
    print("\n[1/2] Reference run (AX_MLX_QWEN3_MOE_NARROW_SOFTMAX=0)...")
    ref_model = ax_engine.load_model(model_path)
    ref_results: list[list[int]] = []
    t0 = time.monotonic()
    for i, prompt in enumerate(prompts):
        tokens = _generate_tokens(ref_model, prompt, args.max_new_tokens)
        ref_results.append(tokens)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(prompts)} prompts done")
    ref_elapsed = time.monotonic() - t0
    print(f"  Reference: {len(prompts)} prompts in {ref_elapsed:.1f}s")

    # --- Candidate run (narrow ON) ---
    os.environ["AX_MLX_QWEN3_MOE_NARROW_SOFTMAX"] = "1"
    print("\n[2/2] Candidate run (AX_MLX_QWEN3_MOE_NARROW_SOFTMAX=1)...")
    cand_model = ax_engine.load_model(model_path)
    cand_results: list[list[int]] = []
    t0 = time.monotonic()
    for i, prompt in enumerate(prompts):
        tokens = _generate_tokens(cand_model, prompt, args.max_new_tokens)
        cand_results.append(tokens)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(prompts)} prompts done")
    cand_elapsed = time.monotonic() - t0
    print(f"  Candidate: {len(prompts)} prompts in {cand_elapsed:.1f}s")

    # --- Comparison ---
    total_tokens = 0
    matching_tokens = 0
    divergent_prompts: list[tuple[int, float, int]] = []
    for i, (ref, cand) in enumerate(zip(ref_results, cand_results)):
        rate = _match_rate(ref, cand)
        n = max(len(ref), len(cand))
        matches = int(rate * n)
        total_tokens += n
        matching_tokens += matches
        if rate < 1.0:
            divergent_prompts.append((i, rate, n))

    overall_rate = matching_tokens / total_tokens if total_tokens > 0 else 1.0

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Prompts:            {len(prompts)}")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Matching tokens:    {matching_tokens}")
    print(f"  Match rate:         {overall_rate:.6f} ({overall_rate*100:.3f}%)")
    print(f"  Divergent prompts: {len(divergent_prompts)}")
    for idx, rate, n in divergent_prompts[:10]:
        print(f"    prompt[{idx}]: {rate:.4f} ({n} tokens)")
    print(f"  Threshold:          {MATCH_THRESHOLD*100:.1f}%")
    print(f"  PASS:               {overall_rate >= MATCH_THRESHOLD}")
    print(f"{'='*60}")

    return 0 if overall_rate >= MATCH_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
