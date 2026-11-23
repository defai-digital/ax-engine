"""Flash Next selected-attention allocation and output control on MLX 0.32.4."""

from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

import mlx.core as mx
import numpy as np


HEADS, KV, DIM, HISTORY = 24, 2, 256, 8192
SELECTED, WIDTH = 2051, (1 << 30) // (2051 * 2 * 256 * 2 * 2)


def data(queries):
    arrays = []
    for seed, shape in enumerate(((queries, HEADS, 1, DIM),
                                  (HISTORY * KV, DIM), (HISTORY * KV, DIM))):
        key = mx.random.key(seed)
        arrays.append(mx.random.uniform(-0.25, 0.25, shape, dtype=mx.bfloat16, key=key))
    generator = np.random.default_rng(42)
    tokens = np.stack([generator.choice(HISTORY, SELECTED, replace=False) for _ in range(queries)])
    tokens[1::4, -3:] = -1
    tokens[2::4, -2:] = -1
    tokens[3::4, -1:] = -1
    return *arrays, mx.array(tokens.astype(np.int32))


def build(inputs, width, gather, launches_early=False, transpose_gather=False):
    q, k, v, tokens = inputs
    parts = []
    for start in range(0, q.shape[0], width):
        chunk = tokens[start:start + width]
        safe = mx.maximum(chunk, 0)
        if transpose_gather:
            def gather_rows(rows):
                flat = rows.reshape(HISTORY, KV, DIM)
                return mx.contiguous(mx.take(flat, safe, 0).transpose(0, 2, 1, 3))
            keys, values = gather_rows(k), gather_rows(v)
        else:
            tile = safe[:, None, :] * KV + mx.arange(KV)[None, :, None]
            keys, values = mx.take(k, tile, 0), mx.take(v, tile, 0)
        if gather:
            parts.extend((keys, values))
            continue
        mask = (chunk >= 0)[:, None, None, :]
        out = mx.fast.scaled_dot_product_attention(
            q[start:start + width], keys, values, scale=DIM ** -0.5, mask=mask)
        if launches_early:
            mx.eval(out)
        parts.append(out)
    if gather:
        return tuple(parts)
    return (mx.concatenate(parts, axis=0),)


def measure(inputs, width, *, gather=False, launches_early=False, transpose_gather=False):
    initial = mx.get_active_memory()
    mx.reset_peak_memory()
    start = time.perf_counter_ns()
    out = build(inputs, width, gather, launches_early, transpose_gather)
    mx.eval(*out)
    mx.synchronize()
    elapsed = (time.perf_counter_ns() - start) / 1e6
    peak = mx.get_peak_memory() - initial
    # The bit check and host transfer deliberately follow the peak sample.
    digest = None if gather else hashlib.sha256(
        np.asarray(out[0].view(mx.uint16)).tobytes()).hexdigest()
    del out
    return peak, elapsed, digest


def run(queries, transpose_gather):
    assert mx.__version__ == "0.32.4" and mx.metal.is_available()
    mx.set_default_device(mx.gpu)
    inputs = data(queries)
    mx.eval(*inputs)
    # Compile the vector masked SDPA and gather outside measured repetitions.
    build_and_eval = build(tuple(a[:16] if i in (0, 3) else a for i, a in enumerate(inputs)),
                           16, False, transpose_gather=transpose_gather)
    mx.eval(*build_and_eval)
    del build_and_eval
    mx.synchronize()
    ports = []
    for width, gather, early in ((WIDTH, False, False), (64, False, False),
                                 (128, False, False), (192, False, False),
                                 (WIDTH, False, True), (WIDTH, True, False)):
        repetitions = [measure(inputs, width, gather=gather, launches_early=early,
                                transpose_gather=transpose_gather) for _ in range(3)]
        peaks, times, hashes = zip(*repetitions, strict=True)
        assert len(set(hashes)) == 1
        ports.append({
            "sub_batch_queries": width,
            "mode": "gather-only" if gather else "serial-eval" if early else "lazy",
            "peaks_bytes": peaks,
            "median_time_ms": median(times),
            "output_sha256": hashes[0],
        })
    old_charge = min(1 << 30, queries * SELECTED * KV * DIM * 2)
    return {
        "queries": queries, "selected": SELECTED, "head_dim": DIM,
        "transpose_gather": transpose_gather,
        "mlx_sdpa_blocks": int(os.environ.get("MLX_SDPA_BLOCKS", "0")),
        "query_heads": HEADS, "kv_heads": KV, "history": HISTORY,
        "one_tile_byte_cap": 1 << 30, "old_gather_charge_bytes": old_charge,
        "kv_pair_payload_bytes": queries * SELECTED * KV * DIM * 2 * 2,
        "modes": ports,
        "all_attention_outputs_exact": len({row["output_sha256"] for row in ports
                                            if row["output_sha256"] is not None}) == 1,
    }


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("queries", type=int, choices=(128, 256, 511, 512, 1024, 2048))
    cli.add_argument("--transpose-gather", action="store_true")
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    result = run(args.queries, args.transpose_gather)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
