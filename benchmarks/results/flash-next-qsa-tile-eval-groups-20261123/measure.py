"""Measure exact QSA attention with a bounded number of gathered tiles per evaluation."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from statistics import median
import time

import mlx.core as mx
import numpy as np

BASELINE = Path(__file__).resolve().parent.parent / "flash-next-qsa-prefill-scratch-20261123"
sys.path.insert(0, str(BASELINE))
from measure_tiles import DIM, KV, SELECTED, WIDTH, data


def attend(inputs, group):
    queries, keys, values, indices = inputs
    pending = []
    parts = []
    for start in range(0, queries.shape[0], WIDTH):
        tokens = indices[start:start + WIDTH]
        rows = mx.maximum(tokens, 0)[:, None, :] * KV + mx.arange(KV)[None, :, None]
        output = mx.fast.scaled_dot_product_attention(
            queries[start:start + WIDTH], mx.take(keys, rows, 0), mx.take(values, rows, 0),
            scale=DIM ** -0.5, mask=(tokens >= 0)[:, None, None, :],
        )
        parts.append(output)
        if group:
            pending.append(output)
            if len(pending) == group:
                mx.eval(*pending)
                pending.clear()
    if pending:
        mx.eval(*pending)
    return mx.concatenate(parts, axis=0)


def run(queries, history):
    # Same exact tile generator as the prior memory audit, with the range of
    # selected indices widened for the representative late-context case.
    import measure_tiles as frozen
    frozen.HISTORY = history
    arrays = data(queries)
    mx.eval(*arrays)
    warmup = attend((arrays[0][:16], arrays[1], arrays[2], arrays[3][:16]), 0)
    mx.eval(warmup)
    del warmup
    mx.synchronize()
    groups = (0, 1, 2, 4)
    measured = {str(group): [] for group in groups}
    for order in (groups, groups[::-1], (2, 0, 4, 1)):
        for group in order:
            initial = mx.get_active_memory()
            mx.reset_peak_memory()
            start = time.perf_counter_ns()
            outputs = attend(arrays, group)
            mx.eval(outputs)
            mx.synchronize()
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            peak = mx.get_peak_memory() - initial
            output_sha = hashlib.sha256(np.asarray(outputs.view(mx.uint16)).tobytes()).hexdigest()
            del outputs
            measured[str(group)].append({"time_ms": elapsed_ms, "peak_bytes": peak, "output_sha256": output_sha})
    assert len({row["output_sha256"] for rows in measured.values() for row in rows}) == 1
    return {
        "queries": queries, "history": history, "selected": SELECTED, "tile_width": WIDTH,
        "exact_bf16": True, "measurements": measured,
        "peak_bytes_by_group": {key: max(row["peak_bytes"] for row in rows) for key, rows in measured.items()},
        "median_ms_by_group": {key: median(row["time_ms"] for row in rows) for key, rows in measured.items()},
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    assert version("mlx") == version("mlx-metal") == "0.32.4" and mx.metal.is_available()
    mx.set_default_device(mx.gpu)
    result = [run(queries, history) for history in (8192, 32768) for queries in (512, 2048)]
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
