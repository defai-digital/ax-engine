#!/usr/bin/env python3
"""Record deterministic synthetic MLX upgrade checks and a hardware-specific qmm probe.

Run independently under the old and candidate published wheels on the same host.
This checks upstream operations, not AX model qualification or MTP certification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    import mlx
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(31418)
    outputs = {}
    checks = []
    input_digests = {}

    def record_input(name, value):
        input_digests[name] = hashlib.sha256(value.tobytes()).hexdigest()
        return mx.array(value)

    def check(name, actual, expected, tolerance):
        mx.eval(actual, expected)
        actual_np = np.asarray(actual.astype(mx.float32)).copy()
        expected_np = np.asarray(expected.astype(mx.float32))
        error = float(np.max(np.abs(actual_np - expected_np)))
        if not np.isfinite(actual_np).all() or error > tolerance:
            raise AssertionError(f"{name}: max_abs={error}, tolerance={tolerance}")
        outputs[name] = actual_np
        checks.append({"name": name, "max_abs_error": error, "tolerance": tolerance})

    x = record_input("small_x", rng.normal(0, 0.1, (17, 128)).astype(np.float16))
    w = record_input("small_w", rng.normal(0, 0.1, (64, 128)).astype(np.float16))
    for bits in (4, 6, 8):
        packed, scales, biases = mx.quantize(w, group_size=64, bits=bits)
        dense = mx.dequantize(packed, scales, biases, group_size=64, bits=bits)
        check(f"affine_qmm_{bits}bit", mx.quantized_matmul(
            x, packed, scales, biases, group_size=64, bits=bits), x @ dense.T, 0.002)

    experts = record_input("experts", rng.normal(0, 0.1, (4, 64, 128)).astype(np.float32))
    experts = experts.astype(mx.bfloat16)
    routed_x = x[:1].astype(mx.bfloat16)
    indices = mx.array([0, 1, 2, 3], dtype=mx.uint32)
    packed, scales = mx.quantize(experts, group_size=32, bits=4, mode="mxfp4")
    dense = mx.dequantize(packed, scales, group_size=32, bits=4,
                          mode="mxfp4", dtype=mx.bfloat16)
    expected = mx.gather_mm(routed_x, mx.swapaxes(dense, -1, -2), rhs_indices=indices)
    for sorted_indices in (False, True):
        actual = mx.gather_qmm(routed_x, packed, scales, rhs_indices=indices,
                              group_size=32, bits=4, mode="mxfp4",
                              sorted_indices=sorted_indices)
        check(f"mxfp4_gather_sorted_{sorted_indices}", actual, expected, 0.008)

    z = mx.array(np.linspace(-12, 12, 257, dtype=np.float32))
    check("compiled_sigmoid", mx.compile(mx.sigmoid)(z), mx.sigmoid(z), 2e-7)
    q = record_input("q", rng.normal(0, 0.1, (1, 4, 8, 64)).astype(np.float32))
    k = record_input("k", rng.normal(0, 0.1, (1, 2, 128, 64)).astype(np.float32))
    v = record_input("v", rng.normal(0, 0.1, (1, 2, 128, 64)).astype(np.float32))
    kk, vv = mx.repeat(k, 2, axis=1), mx.repeat(v, 2, axis=1)
    expected = mx.softmax((q @ mx.swapaxes(kk, -1, -2)) * 0.125, axis=-1) @ vv
    actual = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
    check("gqa_sdpa", actual, expected, 2e-6)

    m, n, inner = 2048, 16384, 4096
    w = record_input("bench_w", rng.normal(0, 0.1, (n, inner)).astype(np.float16))
    x = record_input("bench_x", rng.normal(0, 0.1, (m, inner)).astype(np.float16))
    packed, scales, biases = mx.quantize(w, group_size=64, bits=4)
    mx.eval(x, packed, scales, biases)

    def run_qmm():
        return mx.quantized_matmul(x, packed, scales, biases, group_size=64, bits=4)

    for _ in range(5):
        mx.eval(run_qmm())
    mx.synchronize()
    trials = []
    for _ in range(3):
        times = []
        for _ in range(args.repetitions):
            start = time.perf_counter()
            mx.eval(run_qmm())
            mx.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        median_ms = statistics.median(times)
        trials.append({"latencies_ms": times, "median_ms": median_ms,
                       "tflops": 2 * m * inner * n / median_ms / 1e9})
    lib = Path(list(mlx.__path__)[0]) / "lib" / "libmlx.dylib"
    report = {
        "schema": "ax.mlx.runtime_upgrade_probe.v1",
        "scope": "Synthetic upstream operations and qmm only; no model or MTP certification",
        "host": {"chip": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
            "macos": platform.mac_ver()[0]},
        "runtime": {"mlx": importlib.metadata.version("mlx"),
                    "mlx_metal": importlib.metadata.version("mlx-metal"),
                    "python": platform.python_version(),
                    "libmlx_sha256": hashlib.sha256(lib.read_bytes()).hexdigest()},
        "input_sha256": input_digests, "checks": checks,
        "qmm": {"m": m, "k": inner, "n": n, "bits": 4, "group_size": 64,
                "warmups": 5, "trials": trials,
                "median_ms": statistics.median(t["median_ms"] for t in trials)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), **outputs)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
