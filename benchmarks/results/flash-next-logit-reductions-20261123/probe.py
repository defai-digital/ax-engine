"""Compare Flash Next GPU logit reads on materialized rows."""
from hashlib import file_digest
from importlib.metadata import distribution, version
import json
from math import isfinite
from pathlib import Path
import platform
from statistics import median
import struct
import subprocess
from time import perf_counter_ns

import mlx.core as mx

WIDTH = 248_320


def result(token, *scores, fallback=None):
    mx.eval(token, *scores)
    values = [float(score.item()) for score in scores]
    if fallback is not None and not all(isfinite(value) and abs(value) >= 2**-126 for value in values):
        return fallback()
    if not all(isfinite(value) for value in values):
        return None
    value = values[0] if len(values) == 1 else max(values) - min(values)
    return int(token.item()), struct.pack("<f", value).hex()


def sorted_bonus(row):
    return result(mx.argmax(row.reshape(-1)), mx.topk(row, 1, axis=-1))


def gathered_bonus(row):
    token = mx.argmax(row.reshape(-1))
    return result(token, mx.take_along_axis(row, token.reshape(1, 1), axis=-1))


def maximum_bonus(row):
    return result(mx.argmax(row.reshape(-1)), mx.max(row))


def sorted_correction(row):
    token = mx.argmax(row.reshape(-1))
    top = mx.topk(row, 2, axis=-1)
    mx.eval(token, top)
    scores = [float(x) for x in top.reshape(-1).tolist()]
    if not all(isfinite(value) for value in scores):
        return None
    return int(token.item()), struct.pack("<f", max(scores) - min(scores)).hex()


def scatter_correction(row):
    token = mx.argmax(row.reshape(-1))
    index = token.reshape(1, 1)
    first = mx.take_along_axis(row, index, axis=-1)
    others = mx.put_along_axis(row, index, mx.full((1, 1), -mx.inf, dtype=row.dtype), axis=-1)
    next_index = mx.argmax(others.reshape(-1)).reshape(1, 1)
    second = mx.take_along_axis(others, next_index, axis=-1)
    return result(token, first, second, fallback=lambda: sorted_correction(row))


def mask_correction(row, indices):
    token = mx.argmax(row.reshape(-1))
    index = token.reshape(1, 1)
    first = mx.take_along_axis(row, index, axis=-1)
    return result(token, first, mx.max(mx.where(indices == index, -mx.inf, row)))


def percentile(samples, fraction):
    values = sorted(samples)
    return values[min(len(values) - 1, round(fraction * len(values)))]


def main():
    if version("mlx") != "0.32.4" or version("mlx-metal") != "0.32.4":
        raise SystemExit("Pinned MLX is required")
    mx.set_default_device(mx.gpu)
    mx.random.seed(24613)
    cases = {
        "ties": [3, 5, 1, 5], "signed_zero": [-0.0, 0.0, -0.0, 0.0],
        "mixed_nan": [0.5, float("nan"), 3, -10], "all_nan": [float("nan")] * 4,
        "positive_infinity": [2, float("inf"), 3, 1],
        "finite_first_infinite_runner_up": [5, -float("inf"), -float("inf"), -float("inf")],
        "finite_top_two_and_negative_infinity": [5, -float("inf"), 2, -float("inf")],
        "all_negative_infinity": [-float("inf")] * 4,
        "subnormal": [0.0, 2**-149, 2**-148, -2**-149],
    }
    wide = [mx.random.normal((1, WIDTH), dtype=mx.float32) for _ in range(68)]
    mx.eval(*wide)
    indices = mx.arange(WIDTH, dtype=mx.uint32).reshape(1, WIDTH)
    mx.eval(indices)
    modes = {
        "sorted_bonus": sorted_bonus, "gathered_bonus": gathered_bonus, "maximum_bonus": maximum_bonus,
        "sorted_correction": sorted_correction, "scatter_correction": scatter_correction,
        "mask_correction": lambda row: mask_correction(row, indices),
    }
    adversarial = {}
    for name, data in cases.items():
        row = mx.array([data], dtype=mx.float32)
        small_indices = mx.arange(len(data), dtype=mx.uint32).reshape(1, len(data))
        mx.eval(row, small_indices)
        adversarial[name] = {mode: fn(row) for mode, fn in modes.items() if mode != "mask_correction"}
        adversarial[name]["mask_correction"] = mask_correction(row, small_indices)
    detail = {name: [] for name in modes}
    outputs = {name: [] for name in modes}
    for index, row in enumerate(wide):
        order = list(modes)
        offset = index % len(order)
        order = order[offset:] + order[:offset]
        if (index // len(order)) % 2:
            order.reverse()
        for mode in order:
            start = perf_counter_ns()
            observed = modes[mode](row)
            microseconds = (perf_counter_ns() - start) / 1_000
            if index >= 8:
                outputs[mode].append(observed)
                detail[mode].append(microseconds)
    def observed_output(mode, value):
        if value is None or not mode.endswith("bonus"):
            return value
        return value[0]

    wide_adversarial = {}
    locations = (0, 511, 513, WIDTH - 4)
    for name, data in cases.items():
        for offset in locations:
            fill = -float("inf") if name in ("signed_zero", "subnormal", "all_negative_infinity") else -11.0
            row = mx.concatenate((mx.full((1, offset), fill, dtype=mx.float32),
                                  mx.array([data], dtype=mx.float32),
                                  mx.full((1, WIDTH - offset - len(data)), fill, dtype=mx.float32)), axis=1)
            mx.eval(row)
            wide_adversarial[f"{name}/{offset}"] = {mode: fn(row) for mode, fn in modes.items()}
    comparisons = {}
    for name in modes:
        baseline = "sorted_bonus" if name.endswith("bonus") else "sorted_correction"
        comparisons[name] = {
            "median_us": median(detail[name]), "p95_us": percentile(detail[name], 0.95),
            "finite_bit_differences": [i for i, (a, b) in enumerate(zip(outputs[name], outputs[baseline], strict=True)) if observed_output(name, a) != observed_output(baseline, b)],
            "adversarial_differences": [key for key, row in (adversarial | wide_adversarial).items()
                                        if observed_output(name, row[name]) != observed_output(baseline, row[baseline])],
        }
    lib = Path(distribution("mlx").locate_file("mlx/lib/libmlx.dylib"))
    with lib.open("rb") as f:
        digest = file_digest(f, "sha256").hexdigest()
    hardware = {"soc": subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                "memory_bytes": int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"])), "os": platform.mac_ver()[0]}
    print(json.dumps({"qualification": False, "hardware": hardware, "mlx": "0.32.4", "libmlx_sha256": digest,
                      "width": WIDTH, "dtype": "float32", "warmups": 8, "measurements": 60,
                      "summary": comparisons, "adversarial": adversarial, "wide_adversarial": wide_adversarial, "raw_us": detail, "outputs": outputs},
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
