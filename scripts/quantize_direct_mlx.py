#!/usr/bin/env python3
"""Direct safetensors quantization using mlx, bypassing mlx-lm model class.

Used when mlx-lm convert cannot load the model due to architecture mismatch.
Applies a mixed quantization recipe to all linear weight tensors directly.

Usage:
    python3 scripts/quantize_direct_mlx.py \
        --source <bf16-model-dir> \
        --output <output-dir> \
        --recipe mixed_3_4
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import mlx.core as mx


RECIPE_PARAMS: dict[str, tuple[int, int]] = {
    "mixed_3_4": (3, 4),
    "mixed_3_6": (3, 6),
    "mixed_4_6": (4, 6),
    "mixed_2_6": (2, 6),
}

HIGH_BITS_LAYER_SUFFIXES = (
    "v_proj",
    "v_a_proj",
    "v_b_proj",
    "down_proj",
    "lm_head",
)

GROUP_SIZE = 64


def should_quantize(key: str, shape: tuple[int, ...]) -> bool:
    """Return True for weight tensors that should be quantized (2D Linear weights)."""
    if not key.endswith(".weight"):
        return False
    # Must be exactly 2D with last dim divisible by group size
    if len(shape) != 2 or shape[-1] % GROUP_SIZE != 0:
        return False
    # Skip norm/embed layers
    excluded = (
        "norm",
        "embed_tokens",
        "rotary_emb",
        "k_norm",
        "q_norm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "pre_attention_layernorm",
        "post_attention_layernorm",
        "final_norm",
    )
    lower = key.lower()
    return not any(ex in lower for ex in excluded)


def bits_for_key(key: str, low_bits: int, high_bits: int) -> int:
    """Return the quantization bit width for a given weight key."""
    for suffix in HIGH_BITS_LAYER_SUFFIXES:
        if key.endswith(f"{suffix}.weight"):
            return high_bits
    return low_bits


def read_header(sf_path: Path) -> dict[str, Any]:
    with open(sf_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len).decode("utf-8"))


def quantize_safetensors(
    source_dir: Path,
    output_dir: Path,
    recipe: str,
) -> dict[str, dict]:
    low_bits, high_bits = RECIPE_PARAMS[recipe]
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = source_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map: dict[str, str] = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
    else:
        # Single file
        shard_files = ["model.safetensors"]
        header = read_header(source_dir / "model.safetensors")
        weight_map = {k: "model.safetensors" for k in header if k != "__metadata__"}

    # Group keys by output shard.
    # We'll produce one output shard per input shard.
    shard_keys: dict[str, list[str]] = {sf: [] for sf in shard_files}
    for key, sf in weight_map.items():
        shard_keys[sf].append(key)

    quant_config: dict[str, Any] = {"group_size": GROUP_SIZE, "bits": high_bits, "mode": "affine"}
    new_weight_map: dict[str, str] = {}
    total_quant = 0
    total_kept = 0
    bits_count: dict[int, int] = {}

    for shard_idx, sf_name in enumerate(shard_files):
        src_path = source_dir / sf_name
        print(f"  Loading {sf_name} ...")
        # mx.load handles bfloat16 natively when path has .safetensors extension.
        # Use the source_dir path (may be a symlink) to preserve the extension.
        weights_raw: dict[str, mx.array] = mx.load(str(source_dir / sf_name))
        out_tensors: dict[str, mx.array] = {}

        for key in shard_keys[sf_name]:
            arr_np = weights_raw[key]
            arr = mx.array(arr_np)

            if should_quantize(key, tuple(arr.shape)):
                bits = bits_for_key(key, low_bits, high_bits)
                # Ensure float for quantization
                if arr.dtype != mx.bfloat16:
                    arr = arr.astype(mx.bfloat16)
                q_w, scales, biases = mx.quantize(arr, group_size=GROUP_SIZE, bits=bits)
                mx.eval(q_w, scales, biases)
                out_tensors[key] = q_w
                out_tensors[key.replace(".weight", ".scales")] = scales
                out_tensors[key.replace(".weight", ".biases")] = biases
                # Record per-tensor quant config
                quant_config[key.replace(".weight", "")] = {
                    "group_size": GROUP_SIZE,
                    "bits": bits,
                    "mode": "affine",
                }
                bits_count[bits] = bits_count.get(bits, 0) + 1
                total_quant += 1
            else:
                out_tensors[key] = arr
                total_kept += 1

        # Write output shard via mx.save_safetensors (handles bfloat16 natively).
        out_name = sf_name
        out_path = output_dir / out_name
        print(f"  Writing {out_name} ...")
        for k in out_tensors:
            mx.eval(out_tensors[k])
            new_weight_map[k] = out_name
        mx.save_safetensors(str(out_path), out_tensors)
        print(f"    {len(out_tensors)} tensors written")

    print(f"\nQuantized {total_quant} weight tensors, kept {total_kept} unquantized.")
    for bits, count in sorted(bits_count.items()):
        print(f"  {bits}-bit: {count} tensors")

    total_weights = sum(bits_count.values())
    if total_weights > 0:
        avg_bits = sum(b * c for b, c in bits_count.items()) / total_weights
        print(f"  avg bits/weight: {avg_bits:.3f}")

    return quant_config, new_weight_map


def copy_config_files(source_dir: Path, output_dir: Path, quant_config: dict) -> None:
    for fname in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer.model",
    ]:
        src = source_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    # Update config.json with quantization info
    cfg_path = output_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["quantization"] = quant_config
    cfg_path.write_text(json.dumps(cfg, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct safetensors quantization via mlx")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--recipe",
        choices=list(RECIPE_PARAMS),
        default="mixed_3_4",
    )
    args = parser.parse_args()

    print(f"Recipe: {args.recipe}")
    low_bits, high_bits = RECIPE_PARAMS[args.recipe]
    print(f"  low_bits={low_bits}, high_bits={high_bits}")
    print(f"  sensitive layers ({', '.join(HIGH_BITS_LAYER_SUFFIXES)}) → {high_bits}-bit")
    print(f"  other linear weights → {low_bits}-bit")
    print()

    quant_config, weight_map = quantize_safetensors(args.source, args.output, args.recipe)
    copy_config_files(args.source, args.output, quant_config)

    # Write index if multi-shard
    unique_shards = sorted(set(weight_map.values()))
    if len(unique_shards) > 1:
        index = {"metadata": {"format": "pt"}, "weight_map": weight_map}
        (args.output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
