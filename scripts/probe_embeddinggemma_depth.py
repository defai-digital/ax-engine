#!/usr/bin/env python3
"""EmbeddingGemma per-layer depth probe comparison.

Runs the Gemma3 encoder through mlx-embeddings layer by layer, computes
masked mean-pooled hidden states after each transformer layer, and compares
against the AX Engine depth probe output to pinpoint the first layer where
numerical drift appears.

Usage:
    # Full comparison (requires both mlx-embeddings and AX probe output):
    python scripts/probe_embeddinggemma_depth.py \
        --model-dir /path/to/embeddinggemma-300m-8bit \
        --ax-depths ax_depths.txt

    # Reference-only (dump mlx-embeddings per-layer hidden states):
    python scripts/probe_embeddinggemma_depth.py \
        --model-dir /path/to/embeddinggemma-300m-8bit \
        --output ref_depths.txt

    # Compare two depth-dump files:
    python scripts/probe_embeddinggemma_depth.py \
        --compare ax_depths.txt ref_depths.txt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


PROMPT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def parse_depth_file(path: Path) -> dict[str, list[float]]:
    """Parse a depth probe output file into {label: [values]}."""
    checkpoints: dict[str, list[float]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # Format: "layer 0 0.123 0.456 ..." or "final 0.123 ..."
            if parts[0] == "layer":
                label = f"layer {parts[1]}"
                values = [float(x) for x in parts[2:]]
            else:
                label = parts[0]
                values = [float(x) for x in parts[1:]]
            checkpoints[label] = values
    return checkpoints


def run_reference_depth(
    model_dir: Path,
    sentences: list[str],
) -> dict[str, list[float]]:
    """Run mlx-embeddings Gemma3 encoder layer by layer."""
    import mlx.core as mx
    import mlx_embeddings
    import mlx.nn as nn
    from mlx_lm.models.gemma3_text import RMSNorm
    from transformers import AutoTokenizer

    model, _ = mlx_embeddings.load(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Tokenize.
    sentence = sentences[0]
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        token_ids = token_ids + [int(eos_id)]
    seq_len = len(token_ids)
    print(f"  prompt: {sentence!r}", file=sys.stderr)
    print(f"  token_ids ({seq_len}): {token_ids[:20]}{'...' if seq_len > 20 else ''}",
          file=sys.stderr)

    # Access the inner Gemma3Model.
    encoder = model.model
    embed_tokens = encoder.embed_tokens
    layers = encoder.layers
    final_norm = encoder.norm
    hidden_size = encoder.config.hidden_size

    # Embed tokens + scale.
    inputs = mx.array([token_ids])
    h = embed_tokens(inputs)
    scale = mx.array(hidden_size ** 0.5, h.dtype)
    h = h * scale

    # Build bidirectional mask (same as Model.__call__ in mlx-embeddings).
    attention_mask = mx.ones([1, seq_len])
    extended_mask = attention_mask[None, None, :, :]  # [1, 1, 1, seq]
    extended_mask = mx.broadcast_to(extended_mask, [1, 1, seq_len, seq_len])
    extended_mask = mx.where(
        extended_mask.astype(mx.bool_),
        mx.array(0.0),
        mx.array(-float("inf")),
    )
    extended_mask = extended_mask.astype(embed_tokens.weight.dtype)

    # Run layer by layer, collecting checkpoints.
    checkpoints: dict[str, list[float]] = {}

    # Embedding checkpoint (pre-layers).
    emb_cp = _masked_mean_pool(h, [seq_len])
    checkpoints["embed"] = emb_cp

    for i, layer in enumerate(layers):
        h = layer(h, extended_mask, None)

        cp = _masked_mean_pool(h, [seq_len])
        checkpoints[f"layer {i}"] = cp

    # Final norm checkpoint.
    h_normed = final_norm(h)
    final_cp = _masked_mean_pool(h_normed, [seq_len])
    checkpoints["final"] = final_cp

    return checkpoints


def _masked_mean_pool(hidden, actual_lens: list[int]) -> list[float]:
    """Masked mean-pool over the sequence dimension for batch row 0.

    Args:
        hidden: [1, seq, H] MLX array.
        actual_lens: list with one element (the real sequence length).

    Returns:
        list[float]: mean-pooled vector for row 0.
    """
    import mlx.core as mx

    length = actual_lens[0]
    # Slice to real tokens: [1, length, H].
    row = hidden[:, :length, :]
    # Mean over sequence dim: [1, H].
    mean_vec = mx.mean(row, axis=1)
    # Convert to f32 and materialize.
    mean_f32 = mean_vec.astype(mx.float32)
    return mean_f32.tolist()[0]


def write_depth_file(checkpoints: dict[str, list[float]], path: Path) -> None:
    """Write checkpoints to a depth-dump file."""
    with open(path, "w") as f:
        for label, values in checkpoints.items():
            vals_str = " ".join(f"{v:.6f}" for v in values)
            f.write(f"{label} {vals_str}\n")
    print(f"Wrote {path}", file=sys.stderr)


def compare_depths(
    ax: dict[str, list[float]],
    ref: dict[str, list[float]],
) -> None:
    """Print per-layer cosine similarity between AX and reference."""
    labels = sorted(
        set(ax.keys()) & set(ref.keys()),
        key=lambda l: (0 if l == "embed" else (9999 if l == "final" else int(l.split()[-1]))),
    )
    if not labels:
        print("No common checkpoints found.", file=sys.stderr)
        return

    print(f"\n{'Checkpoint':<12} {'Cosine':>10} {'Hidden':>8} {'Drift':>10}")
    print("-" * 42)

    prev_cos = None
    for label in labels:
        ax_v = ax[label]
        ref_v = ref[label]
        min_len = min(len(ax_v), len(ref_v))
        if min_len == 0:
            continue
        cos = cosine_sim(ax_v[:min_len], ref_v[:min_len])
        drift = ""
        if prev_cos is not None:
            delta = cos - prev_cos
            if abs(delta) > 0.0001:
                drift = f"{delta:+.4f}"
        prev_cos = cos
        print(f"{label:<12} {cos:>10.6f} {min_len:>8} {drift:>10}")

    print()
    # Summary.
    layer_cosines = [
        (label, cosine_sim(ax[label][:min(len(ax[label]), len(ref[label]))],
                           ref[label][:min(len(ax[label]), len(ref[label]))]))
        for label in labels
        if label.startswith("layer") and label in ax and label in ref
    ]
    if layer_cosines:
        first_drift = next(
            ((l, c) for l, c in layer_cosines if c < 0.999),
            None,
        )
        if first_drift:
            print(f"  >> First drift at {first_drift[0]}: cosine = {first_drift[1]:.6f}")
        else:
            print("  >> All layers above 0.999 cosine threshold")
        min_cos = min(layer_cosines, key=lambda x: x[1])
        print(f"  >> Worst layer: {min_cos[0]} cosine = {min_cos[1]:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Path to the EmbeddingGemma model directory.",
    )
    parser.add_argument(
        "--ax-depths",
        type=Path,
        help="Path to AX depth probe output file (from embed_gemma_depth_probe).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write reference depth checkpoints to this file.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        help="Compare two depth-dump files directly.",
    )
    args = parser.parse_args()

    if args.compare:
        a = parse_depth_file(Path(args.compare[0]))
        b = parse_depth_file(Path(args.compare[1]))
        compare_depths(a, b)
        return 0

    if not args.model_dir:
        parser.error("--model-dir is required (or use --compare)")
        return 2

    print(f"[reference] loading {args.model_dir}", file=sys.stderr)
    ref = run_reference_depth(args.model_dir, PROMPT_SENTENCES)

    if args.output:
        write_depth_file(ref, args.output)

    if args.ax_depths:
        print(f"\n[ax] loading {args.ax_depths}", file=sys.stderr)
        ax = parse_depth_file(args.ax_depths)
        compare_depths(ax, ref)
    elif not args.output:
        # Print reference checkpoints to stdout.
        for label, values in ref.items():
            vals_str = " ".join(f"{v:.6f}" for v in values)
            print(f"{label} {vals_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
