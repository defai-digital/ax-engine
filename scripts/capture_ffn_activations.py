#!/usr/bin/env python3
"""Capture per-layer FFN-input activation statistics for P2b §3b AWQ calibration.

Loads the target model via mlx_lm, wraps each layer's `.mlp.__call__` to
record per-input-channel max(|x|), runs the calibration corpus, and dumps a
single safetensors file with `activation_max.layer.{i}` vectors. Downstream:
`quantize_rotated_weights.py --activation-stats <file>` reads these and
computes the AWQ smoothing scale `s` from real activation magnitudes
(instead of the weight-magnitude proxy that did not reliably help).

Usage:

    python scripts/capture_ffn_activations.py \\
        --mlx-artifacts-dir .internal/models/Qwen3.5-9B-MLX-4bit \\
        --output benchmarks/results/activation-stats/qwen3_5_9b-2026-05-11.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

DEFAULT_CALIBRATION_CORPUS = [
    "The quick brown fox jumps over the lazy dog while the engine warms its caches and the operator records throughput metrics.",
    "Write one sentence about the moon.",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return",
    "List five fruits separated by commas: apple, banana, cherry, ",
    "Margaret turned the brass key with steady fingers and pushed open the warehouse door, which groaned on hinges that had not seen oil for at least a decade, releasing a long breath of stale air into the cold morning.",
    "Quantum error correction codes encode logical qubits across many physical qubits so that local noise can be detected and reversed without disturbing the encoded information.",
    "A garden snail moves on a single muscular foot, propelled by waves of contraction that travel along its underside, and lubricates its path with a mucus that doubles as a defensive barrier against predators.",
    "Index funds owe their popularity to a simple observation: most actively managed equity funds underperform their benchmark after fees over multi-decade horizons.",
    "Late in the evening the printer started up unprompted and produced three pages of nothing but the letter Q, followed by a single blank sheet.",
    "Glacial moraines record the maximum extent of an ice sheet because the rock fragments that the glacier transported are dropped along its leading edge when the ice retreats.",
    "Translate the following English sentence to French: 'The library will be closed on Sunday.'",
    "Summarise the second law of thermodynamics in two sentences for a high-school audience.",
    "import numpy as np\ndef softmax(x):\n    e = np.exp(x - np.max(x))\n    return",
    "Once upon a time in a kingdom by the sea, a young engineer discovered that her favorite teapot held exactly the right capacitance to balance a particular resonant circuit.",
    "The minutes of the meeting recorded that the committee had voted unanimously in favour of postponing the decision until the next quarter.",
    "SELECT name, salary FROM employees WHERE department = 'Engineering' AND salary >",
]


ROTATION_SEED_CONSTANT_INT = 0xA5A5A5A5A5A5A5A5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mlx-artifacts-dir", required=True, type=Path)
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for activation_max safetensors.",
    )
    p.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Optional newline-delimited prompts (one per line). Default uses "
        "an in-built 16-prompt calibration corpus.",
    )
    p.add_argument("--max-tokens-per-prompt", type=int, default=64)
    p.add_argument(
        "--rotation-dim",
        type=int,
        default=None,
        help="If set, apply the canonical AX rotation R of this dim to each "
        "captured activation BEFORE taking per-channel max. The captured "
        "stat then matches what the AX runtime's Apply-mode forward sees. "
        "If unset, captures un-rotated activations.",
    )
    return p.parse_args()


def build_orthonormal_hadamard_for_capture(dim: int):
    """Inline mirror of `quantize_rotated_weights.build_orthonormal_hadamard`
    so the capture script doesn't need to import the other module."""
    import numpy as np

    assert dim >= 2 and (dim & (dim - 1)) == 0, "dim must be power-of-2 >= 2"
    state = ROTATION_SEED_CONSTANT_INT
    MASK = 0xFFFFFFFFFFFFFFFF
    sign = np.empty(dim, dtype=np.float32)
    for i in range(dim):
        state = (state ^ ((state << 13) & MASK)) & MASK
        state = (state ^ (state >> 7)) & MASK
        state = (state ^ ((state << 17) & MASK)) & MASK
        sign[i] = 1.0 if (state & 1) == 0 else -1.0
    scale = 1.0 / np.sqrt(np.float32(dim))
    rows = np.arange(dim).reshape(-1, 1)
    cols = np.arange(dim).reshape(1, -1)
    popcount_parity = (np.bitwise_count(rows & cols) % 2).astype(np.float32)
    h = np.where(popcount_parity == 0, 1.0, -1.0).astype(np.float32)
    r = (sign.reshape(-1, 1) * h * sign.reshape(1, -1)) * scale
    return r


def load_corpus(args: argparse.Namespace) -> list[str]:
    if args.prompts_file is None:
        return DEFAULT_CALIBRATION_CORPUS
    text = args.prompts_file.read_text()
    return [ln for ln in text.splitlines() if ln.strip()]


def main() -> int:
    args = parse_args()
    try:
        from mlx_lm import load
    except ImportError as e:
        raise SystemExit("mlx_lm not installed in the venv") from e

    print(f"loading {args.mlx_artifacts_dir} via mlx_lm ...")
    model, tokenizer = load(str(args.mlx_artifacts_dir))
    layers = model.layers
    print(f"  {len(layers)} layers")

    captured: dict[int, mx.array] = {}
    rotation_R = None
    if args.rotation_dim is not None:
        print(f"  applying R of dim {args.rotation_dim} inside capture hook")
        rotation_R = mx.array(build_orthonormal_hadamard_for_capture(args.rotation_dim))

    # __call__ is looked up on the TYPE, not the instance, so per-instance
    # monkey-patching does nothing. Instead: tag each mlp instance with its
    # layer index, then replace the CLASS's __call__ once with a dispatcher
    # that captures before delegating.
    hooked_count = 0
    classes_to_patch: dict[type, "callable"] = {}
    for idx, layer in enumerate(layers):
        children = layer.children()
        mlp = children.get("mlp") if isinstance(children, dict) else None
        if mlp is None:
            continue
        setattr(mlp, "_ax_capture_layer_idx", idx)
        if type(mlp) not in classes_to_patch:
            classes_to_patch[type(mlp)] = type(mlp).__call__
        hooked_count += 1

    for cls, orig in classes_to_patch.items():
        def make_dispatch(orig_call):
            def dispatch(self, x):
                idx = getattr(self, "_ax_capture_layer_idx", None)
                if idx is not None:
                    x_for_stats = x if rotation_R is None else mx.matmul(x, rotation_R)
                    abs_x = mx.abs(x_for_stats)
                    reduced = abs_x
                    while reduced.ndim > 1:
                        reduced = mx.max(reduced, axis=0)
                    mx.eval(reduced)
                    prev = captured.get(idx)
                    captured[idx] = (
                        mx.maximum(prev, reduced) if prev is not None else reduced
                    )
                    mx.eval(captured[idx])
                return orig_call(self, x)

            return dispatch

        cls.__call__ = make_dispatch(orig)
    print(f"  hooked {hooked_count} mlp instances across {len(classes_to_patch)} class(es)")

    corpus = load_corpus(args)
    print(f"running {len(corpus)} calibration prompts (max {args.max_tokens_per_prompt} tokens each) ...")
    for i, text in enumerate(corpus):
        ids = tokenizer.encode(text)
        if len(ids) > args.max_tokens_per_prompt:
            ids = ids[: args.max_tokens_per_prompt]
        tokens = mx.array([ids])
        _ = model(tokens)
        mx.eval(_)
        print(f"  [{i+1}/{len(corpus)}] prompt_tokens={len(ids)}")

    if not captured:
        raise SystemExit("no activations captured — model.layers.*.mlp not hit")

    print(f"captured {len(captured)} layers; output channels: "
          f"{captured[next(iter(captured))].shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"activation_max.layer.{k}": v for k, v in captured.items()}
    mx.save_safetensors(
        str(args.output),
        arrays,
        metadata={
            "schema_version": "ax.activation_max.v1",
            "prompts_count": str(len(corpus)),
            "max_tokens_per_prompt": str(args.max_tokens_per_prompt),
            "reduction": "per_input_channel_max_abs",
        },
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
