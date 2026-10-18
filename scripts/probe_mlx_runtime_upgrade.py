#!/usr/bin/env python3
"""Record deterministic MLX admission checks and an optional hardware-specific qmm probe.

Run old and candidate published wheels separately on the same host. Failed checks
are retained in the report and cause a nonzero exit. Cross-version numerical drift
is reported separately from accuracy against references; it is not a ship gate.
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


def compare_reports(baseline_path, candidate, candidate_outputs):
    """Reject mismatched experiments before describing cross-version output drift."""
    import numpy as np

    baseline = json.loads(baseline_path.read_text())
    for field in ("schema", "probe_sha256", "host", "input_sha256"):
        if field not in baseline or field not in candidate or baseline[field] != candidate[field]:
            raise ValueError(f"comparison requires matching {field}")
    for field in ("python", "numpy"):
        if (not baseline["runtime"].get(field)
                or baseline["runtime"][field] != candidate["runtime"].get(field)):
            raise ValueError(f"comparison requires matching runtime {field}")
    baseline_checks = [(c["name"], c["tolerance"]) for c in baseline["checks"]]
    candidate_checks = [(c["name"], c["tolerance"]) for c in candidate["checks"]]
    if baseline_checks != candidate_checks:
        raise ValueError("comparison requires matching checks and tolerances")
    baseline_arrays = baseline_path.with_suffix(".npz")
    if hashlib.sha256(baseline_arrays.read_bytes()).hexdigest() != baseline.get("outputs_sha256"):
        raise ValueError("baseline output digest mismatch")
    differences = []
    with np.load(baseline_arrays, allow_pickle=False) as previous:
        if set(previous.files) != set(candidate_outputs):
            raise ValueError("comparison requires matching output names")
        for name, actual in candidate_outputs.items():
            old = previous[name]
            if old.shape != actual.shape or old.dtype != actual.dtype:
                raise ValueError(f"comparison requires matching shape and dtype: {name}")
            if not np.isfinite(old).all() or not np.isfinite(actual).all():
                raise ValueError(f"comparison requires finite outputs: {name}")
            differences.append({
                "name": name,
                "bit_identical": old.tobytes() == actual.tobytes(),
                "different_elements": int(np.count_nonzero(old != actual)),
                "elements": int(actual.size),
                "max_abs_difference": float(np.max(np.abs(old - actual))),
            })
    return {
        "baseline_runtime": baseline["runtime"],
        "baseline_report_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "baseline_outputs_sha256": hashlib.sha256(
            baseline_path.with_suffix(".npz").read_bytes()).hexdigest(),
        "scope": "Same-host numerical drift only; reference accuracy is checked separately",
        "outputs": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--skip-timing", action="store_true",
                        help="Run software correctness checks without throughput measurements")
    parser.add_argument("--compare-to", type=Path,
                        help="Compare with a same-source, same-host report and adjacent NPZ")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.compare_to and args.compare_to.resolve() == args.output.resolve():
        parser.error("baseline and output must be different files")

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
        if actual_np.shape != expected_np.shape:
            raise AssertionError(f"{name}: reference shape mismatch")
        finite = bool(np.isfinite(actual_np).all() and np.isfinite(expected_np).all())
        error = float(np.max(np.abs(actual_np - expected_np))) if finite else None
        outputs[name] = actual_np
        checks.append({"name": name, "max_abs_error": error, "tolerance": tolerance,
                       "passed": finite and error <= tolerance})

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
    # GPU FP32 matmul can select reduced-precision NAX arithmetic on M5.
    # Keep the accuracy oracle independent of that dispatch and its rounding.
    kk = np.repeat(np.asarray(k).astype(np.float64), 2, axis=1)
    vv = np.repeat(np.asarray(v).astype(np.float64), 2, axis=1)
    scores = (np.asarray(q).astype(np.float64) @ kk.swapaxes(-1, -2)) * 0.125
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = mx.array((probabilities @ vv).astype(np.float32))
    actual = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
    check("gqa_sdpa", actual, expected, 2e-6)

    # Cover vector and split-K dispatch with an independent float64 dot product.
    # BF16 rounding is assessed against dequantized weights, not the old runtime.
    w = record_input("splitk_w", rng.normal(0, 0.02, (1024, 2560)).astype(np.float32))
    x = record_input("splitk_x", rng.normal(0, 0.1, (128, 2560)).astype(np.float32))
    w, x = w.astype(mx.bfloat16), x.astype(mx.bfloat16)
    packed, scales = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
    dense = mx.dequantize(packed, scales, group_size=32, bits=4,
                          mode="mxfp4", dtype=mx.float32)
    mx.eval(x, packed, scales, dense)
    reference = np.asarray(x.astype(mx.float32)).astype(np.float64) @ np.asarray(dense).T
    memory = []
    for rows in (1, 2, 4, 16, 32, 128):
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        active_before = mx.get_active_memory()
        actual = mx.quantized_matmul(x[:rows], packed, scales, group_size=32,
                                     bits=4, mode="mxfp4")
        mx.eval(actual)
        peak = mx.get_peak_memory()
        memory.append({"rows": rows, "active_before_bytes": active_before,
                       "active_after_bytes": mx.get_active_memory(), "peak_bytes": peak,
                       "peak_increase_bytes": max(0, peak - active_before)})
        check(f"mxfp4_bf16_rows_{rows}", actual,
              mx.array(reference[:rows].astype(np.float32)), 0.004)
        del actual

    # M2 multi-query dispatch and pre-M5 D512 paths; float64 softmax oracle.
    for dim in (256, 512):
        for seq in (1, 4, 5, 32):
            prefix = f"sdpa_d{dim}_s{seq}"
            q_np = rng.normal(0, 0.1, (1, 8, seq, dim)).astype(np.float16)
            k_np = rng.normal(0, 0.1, (1, 2, 1024, dim)).astype(np.float16)
            v_np = rng.normal(0, 0.1, (1, 2, 1024, dim)).astype(np.float16)
            q, k, v = (record_input(prefix + suffix, value)
                       for suffix, value in (("_q", q_np), ("_k", k_np), ("_v", v_np)))
            kk = np.repeat(k_np.astype(np.float64), 4, axis=1)
            vv = np.repeat(v_np.astype(np.float64), 4, axis=1)
            scores = (q_np.astype(np.float64) @ kk.swapaxes(-1, -2)) * dim**-0.5
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            expected = mx.array((probabilities @ vv).astype(np.float32))
            actual = mx.fast.scaled_dot_product_attention(q, k, v, scale=dim**-0.5)
            check(prefix, actual, expected, 5e-5)

    # The temporary MLX array must not donate a foreign NumPy buffer to exp.
    foreign = np.linspace(-1, 1, 64, dtype=np.float32)
    original = foreign.copy()
    input_digests["foreign_buffer"] = hashlib.sha256(original.tobytes()).hexdigest()
    result = mx.exp(mx.array(foreign), stream=mx.cpu)
    mx.eval(result)
    check("foreign_buffer_preserved", mx.array(foreign.copy()), mx.array(original), 0.0)
    check("foreign_buffer_exp", result, mx.array(np.exp(original)), 2e-6)

    qmm = None
    if not args.skip_timing:
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
        qmm = {"m": m, "k": inner, "n": n, "bits": 4, "group_size": 64,
               "warmups": 5, "trials": trials,
               "median_ms": statistics.median(t["median_ms"] for t in trials)}

    lib_dir = Path(list(mlx.__path__)[0]) / "lib"
    report = {
        "schema": "ax.mlx.runtime_upgrade_probe.v2",
        "scope": "Synthetic upstream operations only; no model or MTP certification",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host": {"chip": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
            "macos": platform.mac_ver()[0]},
        "runtime": {"mlx": importlib.metadata.version("mlx"),
                    "mlx_metal": importlib.metadata.version("mlx-metal"),
                    "python": platform.python_version(), "numpy": np.__version__,
                    "asset_sha256": {name: hashlib.sha256((lib_dir / name).read_bytes()).hexdigest()
                                     for name in ("libmlx.dylib", "libjaccl.dylib", "mlx.metallib")}},
        "input_sha256": input_digests, "checks": checks,
        "passed": all(c["passed"] for c in checks),
        "mxfp4_memory": {"k": 2560, "n": 1024, "dtype": "bfloat16", "samples": memory},
        "qmm": qmm,
    }
    if args.compare_to:
        report["comparison"] = compare_reports(args.compare_to, report, outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), **outputs)
    report["outputs_sha256"] = hashlib.sha256(args.output.with_suffix(".npz").read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
