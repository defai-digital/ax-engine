#!/usr/bin/env python3
"""W1 spike — MXFP4 dequant correctness check for Phase B (GPT-OSS-PHASE-B-NATIVE-PRD.md).

Goal: confirm `openai/gpt-oss-20b` MoE expert weights can be loaded from
their on-disk MXFP4 layout (`<name>_blocks` uint8 + `<name>_scales`
uint8) and dequantized to BF16 with output matching the reference
produced by mlx-lm's load path. Sample 4 random layers × 4 random
experts × {gate_proj, up_proj, down_proj} = 16 tensors.

Pass criterion: max_abs_diff < 1e-3 across all 16 samples in BF16
(matches Phase B PRD §W1 exit bar).

KEY FINDING (recorded here so Phase B picker sees it immediately):
MLX-core natively supports `mx.dequantize(..., mode='mxfp4', bits=4,
group_size=32)` since mlx 0.30+. Ax-engine does **not** need to
implement FP4 E2M1 unpacking or E8M0 scale math from scratch — the
loader only needs to:

1. Read `<name>_blocks` as uint8 from safetensors.
2. Reinterpret as uint32 + flatten last two dims (mlx-lm sanitize).
3. De-interleave gate/up rows (even = gate, odd = up).
4. Call mx.dequantize() with mode='mxfp4'.

The Phase B W3 (MXFP4 loader) implementation surface is correspondingly
smaller: it's a safetensors→mlx-array adapter, not a new quantization
kernel. `mlx-sys` must expose `mlx_dequantize` with mode parameter.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


def load_mxfp4_pair(
    shards: list[Path], block_key: str, scale_key: str
) -> tuple[mx.array, mx.array]:
    """Read `<name>_blocks` (uint8) + `<name>_scales` (uint8) from safetensors."""
    blocks = None
    scales = None
    for shard in shards:
        with safe_open(shard, framework="numpy") as fh:
            keys = set(fh.keys())
            if block_key in keys:
                blocks = mx.array(fh.get_tensor(block_key))
            if scale_key in keys:
                scales = mx.array(fh.get_tensor(scale_key))
        if blocks is not None and scales is not None:
            break
    if blocks is None:
        raise KeyError(f"{block_key} not found in any shard")
    if scales is None:
        raise KeyError(f"{scale_key} not found in any shard")
    return blocks, scales


def sanitize_mxfp4(
    blocks_u8: mx.array, scales_u8: mx.array
) -> tuple[mx.array, mx.array]:
    """Apply mlx-lm gpt_oss sanitize transform: view u8 blocks as u32, flatten last two dims.

    Input  blocks_u8 shape: (..., out_rows, n_groups, block_size_bytes=16)
                            where block_size_bytes = group_size/2 = 16
    Output weight       shape: (..., out_rows, n_groups * 4 in u32)
                            (4 u8 bytes per u32; each u32 packs 8 FP4 values)
    Output scales       shape: (..., out_rows, n_groups) u8 (unchanged)
    """
    w = blocks_u8.view(mx.uint32)
    w = w.flatten(-2)
    return w, scales_u8


def own_dequant(
    shards: list[Path],
    layer: int,
    expert: int,
    which: str,
) -> mx.array:
    """Independent path that does NOT call mlx_lm.load — only safetensors I/O
    and mx.dequantize. This mirrors what ax-engine's loader will do."""
    prefix = f"model.layers.{layer}.mlp.experts"
    if which == "gate_proj" or which == "up_proj":
        b_key = f"{prefix}.gate_up_proj_blocks"
        s_key = f"{prefix}.gate_up_proj_scales"
        blocks, scales = load_mxfp4_pair(shards, b_key, s_key)
        w_u32, s_u8 = sanitize_mxfp4(blocks, scales)
        if which == "gate_proj":
            w_part = mx.contiguous(w_u32[..., ::2, :])
            s_part = mx.contiguous(s_u8[..., ::2, :])
        else:
            w_part = mx.contiguous(w_u32[..., 1::2, :])
            s_part = mx.contiguous(s_u8[..., 1::2, :])
        return mx.dequantize(
            w_part[expert],
            scales=s_part[expert],
            group_size=32,
            bits=4,
            mode="mxfp4",
        )
    elif which == "down_proj":
        b_key = f"{prefix}.down_proj_blocks"
        s_key = f"{prefix}.down_proj_scales"
        blocks, scales = load_mxfp4_pair(shards, b_key, s_key)
        w_u32, s_u8 = sanitize_mxfp4(blocks, scales)
        return mx.dequantize(
            w_u32[expert],
            scales=s_u8[expert],
            group_size=32,
            bits=4,
            mode="mxfp4",
        )
    else:
        raise ValueError(which)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path.home()
        / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee",
    )
    p.add_argument("--model-id", default="openai/gpt-oss-20b")
    p.add_argument("--layers", type=int, default=4, help="how many random layers to sample")
    p.add_argument("--experts", type=int, default=4, help="how many random experts per layer")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-layer", type=int, default=24, help="20B has 24 layers")
    p.add_argument("--max-expert", type=int, default=32, help="20B has 32 experts")
    p.add_argument("--pass-bar", type=float, default=1e-3)
    args = p.parse_args()

    shards = sorted(args.snapshot_dir.glob("model-*.safetensors"))
    # Filter out the original/ PyTorch shard which has the same prefix but different layout
    shards = [s for s in shards if "original" not in s.parts]
    if not shards:
        print(f"no shards under {args.snapshot_dir}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    layers = sorted(rng.sample(range(args.max_layer), args.layers))
    experts = sorted(rng.sample(range(args.max_expert), args.experts))

    print(f"shards: {[s.name for s in shards]}", flush=True)
    print(f"sampling layers={layers} experts={experts}", flush=True)
    print(f"loading mlx-lm reference (one-time)…", flush=True)
    t0 = time.monotonic()
    from mlx_lm import load
    ref_model, _ = load(args.model_id)
    load_s = time.monotonic() - t0
    print(f"loaded in {load_s:.2f}s", flush=True)

    results = []
    for layer in layers:
        layer_obj = ref_model.layers[layer]
        for which in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(layer_obj.mlp.experts, which)
            for expert in experts:
                ref = mx.dequantize(
                    lin.weight[expert],
                    scales=lin.scales[expert],
                    group_size=lin.group_size,
                    bits=lin.bits,
                    mode=lin.mode,
                )
                own = own_dequant(shards, layer, expert, which)
                mx.eval(ref, own)
                diff = mx.abs(ref - own)
                max_abs = float(mx.max(diff))
                mean_abs = float(mx.mean(diff))
                shape = tuple(ref.shape)
                rec = {
                    "layer": layer,
                    "expert": expert,
                    "tensor": which,
                    "shape": list(shape),
                    "max_abs_diff": max_abs,
                    "mean_abs_diff": mean_abs,
                    "ref_dtype": str(ref.dtype),
                    "own_dtype": str(own.dtype),
                }
                results.append(rec)
                print(
                    f"  layer={layer:3d} expert={expert:3d} {which:10s} "
                    f"shape={shape} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}",
                    flush=True,
                )

    aggregate_max = max(r["max_abs_diff"] for r in results)
    aggregate_mean = sum(r["mean_abs_diff"] for r in results) / len(results)
    passed = aggregate_max < args.pass_bar

    summary = {
        "model_id": args.model_id,
        "snapshot_dir": str(args.snapshot_dir),
        "shards": [s.name for s in shards],
        "sampled_layers": layers,
        "sampled_experts": experts,
        "samples_total": len(results),
        "aggregate_max_abs_diff": aggregate_max,
        "aggregate_mean_abs_diff": aggregate_mean,
        "pass_bar": args.pass_bar,
        "passed": passed,
        "mlx_lm_load_seconds": load_s,
        "spike_finding": "mlx-core supports mx.dequantize(..., mode='mxfp4') natively; ax-engine MXFP4 loader is a safetensors→mlx-array adapter, not a new kernel.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "per_sample": results}, indent=2)
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())
