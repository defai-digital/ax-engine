"""Compare separate and shared index graphs for identical MLX K/V gathers."""

import gc
import importlib.metadata
import json
from statistics import median
import time

import mlx.core as mx
import numpy as np


KV_HEADS = 2
HEAD_DIM = 256
CASES = ((48, 4096, 2051), (128, 8192, 2051), (255, 16384, 2051))
REPEATS = 8


def index_tile(safe, width, padded):
    positions = mx.reshape(safe, (width, 1, padded))
    strided = positions * mx.array(KV_HEADS, mx.int32)
    heads = mx.reshape(mx.arange(KV_HEADS, dtype=mx.int32), (1, KV_HEADS, 1))
    return strided + heads


def gather(rows, indices):
    return mx.take(mx.reshape(rows, (-1, HEAD_DIM)), indices, axis=0)


def original(keys, values, safe, width, padded):
    return (gather(keys, index_tile(safe, width, padded)),
            gather(values, index_tile(safe, width, padded)))


def shared(keys, values, safe, width, padded):
    indices = index_tile(safe, width, padded)
    return gather(keys, indices), gather(values, indices)


def run_case(width, context, padded):
    rng = np.random.default_rng(893)
    shape = (context, KV_HEADS, HEAD_DIM)
    # Integral tokens and nontrivial signed BF16 payloads, with padding and repeats.
    keys = mx.array(rng.integers(-8192, 8193, size=shape, dtype=np.int16)).astype(mx.bfloat16)
    values = mx.array(rng.integers(-8192, 8193, size=shape, dtype=np.int16)).astype(mx.bfloat16)
    selected = rng.integers(0, context, size=(width, padded), dtype=np.int32)
    selected[::2, -3:] = -1
    safe = mx.maximum(mx.array(selected, mx.int32), mx.array(0, mx.int32))
    mx.eval(keys, values, safe)

    def run(function):
        mx.synchronize()
        mx.clear_cache()
        active = mx.get_active_memory()
        mx.reset_peak_memory()
        started = time.perf_counter()
        k, v = function(keys, values, safe, width, padded)
        mx.eval(k, v)
        elapsed = time.perf_counter() - started
        return (k, v), {"seconds": elapsed, "peak_bytes": mx.get_peak_memory(), "active_before_bytes": active}

    before, _ = run(original)
    after, _ = run(shared)
    equal = [bool(mx.array_equal(x.view(mx.uint16), y.view(mx.uint16)).item())
             for x, y in zip(before, after, strict=True)]
    assert all(equal)
    del before, after
    gc.collect()
    stats = {"separate": [], "shared": []}
    for i in range(REPEATS):
        order = (("separate", original), ("shared", shared))
        for name, function in order if i % 2 == 0 else reversed(order):
            values_out, sample = run(function)
            stats[name].append(sample)
            del values_out
    return {"width": width, "context": context, "padded": padded, "equal_bits": equal,
            "samples": stats,
            "median_seconds": {name: median(x["seconds"] for x in arm) for name, arm in stats.items()},
            "median_extra_bytes": {name: median(x["peak_bytes"] - x["active_before_bytes"] for x in arm) for name, arm in stats.items()}}


if __name__ == "__main__":
    versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal")}
    assert set(versions.values()) == {"0.32.4"}
    assert mx.metal.is_available()
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(8 * 2**30)
    print(json.dumps({"versions": versions, "cases": [run_case(*case) for case in CASES]}, indent=2))
