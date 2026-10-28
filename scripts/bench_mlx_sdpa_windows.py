#!/usr/bin/env python3
"""Measure SDPA window shapes and singleton parity, without executing a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Exact MLX release or source commit")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    outputs = {}
    for family, heads, kv_heads in [("qwen38_27b", 24, 4), ("flash_next", 24, 2)]:
        for context in [2048, 8192, 32768]:
            for queries in [1, 2, 3, 4, 6, 8, 9]:
                for mask_kind in ["none", "causal", "array"]:
                    rng = np.random.default_rng(540 + context + queries + kv_heads)
                    arrays = [mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
                              for shape in [(1, heads, queries, 256),
                                            (1, kv_heads, context, 256), (1, kv_heads, context, 256)]]
                    q, k, v = arrays
                    mask = None
                    if mask_kind == "causal":
                        mask = "causal"
                    elif mask_kind == "array":
                        keep = np.arange(context)[None, :] <= np.arange(context-queries, context)[:, None]
                        keep &= np.arange(context)[None, :] % 3 != 1
                        mask = mx.array(keep)
                        mx.eval(mask)
                    mx.eval(q, k, v)

                    def attend():
                        return mx.fast.scaled_dot_product_attention(q, k, v, scale=1/16, mask=mask)

                    for _ in range(3):
                        mx.eval(attend())
                    elapsed = []
                    for _ in range(args.iterations):
                        start = time.perf_counter_ns()
                        out = attend()
                        mx.eval(out)
                        elapsed.append((time.perf_counter_ns()-start)/1e6)
                    singleton = []
                    for index in range(queries):
                        length = context-queries+index+1 if mask_kind == "causal" else context
                        row_mask = mask[index:index+1] if mask_kind == "array" else None
                        singleton.append(mx.fast.scaled_dot_product_attention(
                            q[:, :, index:index+1], k[:, :, :length], v[:, :, :length],
                            scale=1/16, mask=row_mask))
                    reference = mx.concatenate(singleton, axis=2)
                    mx.eval(reference)
                    actual = np.array(out.astype(mx.float32))
                    expected = np.array(reference.astype(mx.float32))
                    key = f"{family}_{context}_{queries}_{mask_kind}"
                    outputs[key] = actual
                    rows.append({"key":key, "family":family, "context":context, "queries":queries,
                                 "gqa":heads//kv_heads, "head_dim":256, "dtype":"bfloat16", "mask":mask_kind,
                                 "split_shape_candidate":queries<=8 and 32<queries*(heads//kv_heads)<=128 and mask_kind!="array",
                                 "median_ms":statistics.median(elapsed), "samples_ms":elapsed,
                                 "singleton_bit_exact":bool(np.array_equal(actual, expected)),
                                 "singleton_max_abs":float(np.max(np.abs(actual-expected))),
                                 "sha256":hashlib.sha256(actual.tobytes()).hexdigest()})
                    del arrays, q, k, v, out, reference, singleton
    np.savez(args.output/"outputs.npz", **outputs)
    report = {"mlx_version":mx.__version__, "source":args.source, "iterations":args.iterations,
              "scope":"Synthetic shape timing; eligibility is not proof of the dispatched kernel.", "rows":rows}
    (args.output/"results.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
