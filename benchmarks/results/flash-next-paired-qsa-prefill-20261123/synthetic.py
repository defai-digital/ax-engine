"""Test exact paired QSA gather over the Flash Next prefill tile geometry."""
from argparse import ArgumentParser
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import mlx.core as mx
import numpy as np

HEADS, QUERY_HEADS, DIM = 2, 24, 256
# One query, KV head, selected position, and a vector of eight BF16 lanes.
VECTORS = DIM // 8
ORDER = ("stock", "paired", "paired", "stock") * 3
SHAPES = ((16, 2051, 2303, 0), (64, 2049, 4353, 4096),
          (255, 2051, 8191, 4096), (256, 2048, 32767, 28672))

SOURCE = Path("metal/kernels/flash_next_qsa_paired_gather.metal").read_text()


def inputs(queries, padded, tokens, sealed):
    rng = np.random.default_rng(queries + padded + tokens + sealed)
    def random(shape):
        return mx.array(rng.standard_normal(shape, dtype=np.float32)).astype(mx.bfloat16)
    def segments():
        prefix = random((sealed, HEADS, DIM)) if sealed else mx.zeros((1, HEADS, DIM), mx.bfloat16)
        return prefix, random((tokens - sealed, HEADS, DIM))
    keys, values = segments(), segments()
    idxs = rng.integers(0, tokens, (queries, padded), dtype=np.int32)
    if sealed:
        idxs[0, :4] = [0, sealed - 1, sealed, tokens - 1]
    idxs[1::2, -1] = -1
    idxs[::4, -2:] = -1
    q = random((queries, QUERY_HEADS, 1, DIM))
    idxs = mx.array(idxs)
    mx.eval(*keys, *values, idxs, q)
    return keys, values, idxs, q


def stock(keys, values, selection, sealed):
    rows, width = selection.shape
    safe = mx.maximum(selection, 0)
    def take(segments):
        x = mx.concatenate(segments, axis=0) if sealed else segments[1]
        indices = safe[:, None, :] * HEADS + mx.arange(HEADS, dtype=mx.int32)[None, :, None]
        return mx.take(x.reshape(-1, DIM), indices, axis=0)
    return take(keys), take(values)


def paired(kernel, keys, values, selection, sealed):
    rows, width = selection.shape
    out = (rows, HEADS, width, DIM)
    return kernel(
        inputs=[keys[0], values[0], keys[1], values[1], selection,
                mx.array([width, sealed, sealed + keys[1].shape[0]], dtype=mx.int32)],
        grid=(width * VECTORS, HEADS, rows), threadgroup=(256, 1, 1),
        output_shapes=[out, out], output_dtypes=[mx.bfloat16, mx.bfloat16],
    )


def differing(a, b):
    assert a.dtype == b.dtype == mx.bfloat16 and a.shape == b.shape
    return int(mx.sum((a.view(mx.uint16) != b.view(mx.uint16)).astype(mx.int32)).item())


def test(kernel, shape):
    rows, width, tokens, sealed = shape
    keys, values, selection, q = inputs(*shape)
    actions = {
        "stock": lambda: stock(keys, values, selection, sealed),
        "paired": lambda: paired(kernel, keys, values, selection, sealed),
    }
    def run(mode, attention):
        k, v = actions[mode]()
        if attention:
            return [mx.fast.scaled_dot_product_attention(
                q, k, v, scale=DIM ** -0.5, mask=(selection >= 0)[:, None, None, :],
            )]
        return [k, v]

    # Compare one pair at a time so the exactness check holds two tiles,
    # while the timed scope holds one. Keep the same selected row order.
    report = []
    for attention in (False, True):
        baseline, candidate = (run(mode, attention) for mode in actions)
        mx.eval(*baseline, *candidate)
        deltas = [differing(a, b) for a, b in zip(baseline, candidate, strict=True)]
        assert not any(deltas), (shape, attention, deltas)
        del baseline, candidate
        values_out = {mode: [] for mode in actions}
        for mode in ORDER:
            start = perf_counter_ns()
            mx.eval(*run(mode, attention))
            values_out[mode].append((perf_counter_ns() - start) / 1000)
        report.append({"scope": "attention" if attention else "gather", "differences": deltas,
                       "wall_us": values_out, "paired_over_stock": median(values_out["paired"]) / median(values_out["stock"])})
        mx.clear_cache()
    return {"geometry": shape, "comparisons": report}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert version("mlx") == version("mlx-metal") == "0.32.4"
    mx.set_default_device(mx.gpu)
    kernel = mx.fast.metal_kernel(
        name="ax_probe_flash_next_paired_gather",
        input_names=["key_prefix", "value_prefix", "key_tail", "value_tail", "selection", "geometry"],
        output_names=["key_out", "value_out"], source=SOURCE, ensure_row_contiguous=True,
    )
    cases = []
    for shape in SHAPES:
        cases.append(test(kernel, shape))
        print(shape, [comp["paired_over_stock"] for comp in cases[-1]["comparisons"]], flush=True)
    args.output.write_text(json.dumps({"mlx": version("mlx"), "order": ORDER, "cases": cases},
                                      indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
