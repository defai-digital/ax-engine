"""Probe a fused routed reduction with the pinned MLX BF16 boundaries."""
from argparse import ArgumentParser
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import mlx.core as mx
import numpy as np

COLUMNS = 2560
EXPERTS_PER_ROW = 10
SHAPES = ((511, 512, 37), (512, 512, 43), (512, 64, 53))
ORDER = ("eager", "fused", "fused", "eager") * 5

# The original AX sorted-reduction addressing comes from 236611eec.
# For the audited top-k=10 layout, the stock MLX reduction forms the
# two BF16 pairs (0,8) and (1,9), then folds the remaining partials.
EXACT_SOURCE = """
    const uint cell = thread_position_in_grid.x;
    const uint token = cell / Hidden;
    const uint column = cell - token * Hidden;
    const uint selection = token * 10;

    bfloat first = bfloat(0.0f);
    bfloat second = bfloat(0.0f);
    for (uint r : {0u, 8u}) {
        const uint p = selection + r;
        const auto value = down[inverse[p] * Hidden + column];
        first = bfloat(float(bfloat(float(value) * float(weights[p]))) + float(first));
    }
    for (uint r : {1u, 9u}) {
        const uint p = selection + r;
        const auto value = down[inverse[p] * Hidden + column];
        second = bfloat(float(bfloat(float(value) * float(weights[p]))) + float(second));
    }
    bfloat total = bfloat(float(second) + float(first));
    for (uint r = 2; r < 8; ++r) {
        const uint p = selection + r;
        const auto value = down[inverse[p] * Hidden + column];
        const bfloat product = bfloat(float(value) * float(weights[p]));
        const bfloat partial = bfloat(float(product) + 0.0f);
        total = bfloat(float(partial) + float(total));
    }
    const bfloat shared_value = bfloat(float(shared[cell]) * float(gate[token]));
    out[cell] = bfloat(float(total) + float(shared_value));
"""

FLOAT_SOURCE = """
    const uint cell = thread_position_in_grid.x;
    const uint token = cell / Hidden;
    const uint column = cell - token * Hidden;
    const uint selection = token * 10;
    float result = 0.0f;
    for (uint slot = 0; slot != 10; ++slot) {
        auto p = selection + slot;
        result += float(down[inverse[p] * Hidden + column]) * float(weights[p]);
    }
    out[cell] = bfloat(result + float(shared[cell]) * float(gate[token]));
"""


def kernel(source, name):
    return mx.fast.metal_kernel(
        name=name, input_names=["down", "inverse", "weights", "shared", "gate"],
        output_names=["out"], source=source, ensure_row_contiguous=True,
    )


def inputs(rows, pool, seed):
    rng = np.random.default_rng(seed)
    expert_ids = rng.integers(0, pool, size=(rows * EXPERTS_PER_ROW,), dtype=np.int32)
    order = np.argsort(expert_ids, kind="stable")
    inverse = mx.array(np.argsort(order).astype(np.uint32))
    down = mx.array(rng.standard_normal((rows * EXPERTS_PER_ROW, COLUMNS), dtype=np.float32)).astype(mx.bfloat16)
    shared = mx.array(rng.standard_normal((rows, COLUMNS), dtype=np.float32)).astype(mx.bfloat16)
    gates = mx.array(rng.uniform(0.0, 1.0, rows).astype(np.float32)).astype(mx.bfloat16)
    weights = mx.softmax(mx.array(rng.standard_normal((rows, EXPERTS_PER_ROW), dtype=np.float32)), axis=-1).astype(mx.bfloat16)
    mx.eval(down, inverse, weights, shared, gates)
    return down, inverse, weights, shared, gates


def eager(down, inverse, weights, shared, gate):
    rows = weights.shape[0]
    restored = mx.take(down, inverse, axis=0).reshape(rows, EXPERTS_PER_ROW, COLUMNS)
    return mx.sum(restored * weights[:, :, None], axis=1) + shared * gate[:, None]


def launch(fn, values):
    rows = values[2].shape[0]
    return fn(
        inputs=values, template=[("Hidden", COLUMNS)],
        grid=(rows * COLUMNS, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, COLUMNS)], output_dtypes=[mx.bfloat16],
    )[0]


def unequal(expected, observed):
    bits = mx.sum((expected.view(mx.uint16) != observed.view(mx.uint16)).astype(mx.int32))
    return int(bits.item())


def measure(values, fused, rounded_at_end):
    functions = {"eager": lambda: eager(*values), "fused": lambda: launch(fused, values)}
    old, new = (make() for make in functions.values())
    float_acc = launch(rounded_at_end, values)
    mx.eval(old, new, float_acc)
    differs = unequal(old, new)
    changed = unequal(old, float_acc)
    scales = {mode: [] for mode in functions}
    if differs == 0:
        for mode in ORDER:
            start = perf_counter_ns()
            mx.eval(functions[mode]())
            scales[mode].append((perf_counter_ns() - start) / 1000)
    return {"elements": old.size, "fused_bit_differences": differs,
            "float_acc_bit_differences": changed, "wall_us": scales,
            "fused_over_eager": median(scales["fused"]) / median(scales["eager"]) if scales["fused"] else None}


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    assert version("mlx") == version("mlx-metal") == "0.32.4"
    mx.set_default_device(mx.gpu)
    fused = kernel(EXACT_SOURCE, "ax_diag_exact_bf16_sorted_combine")
    rounded_at_end = kernel(FLOAT_SOURCE, "ax_diag_float_sorted_combine")
    rows = []
    for width, pool, seed in SHAPES:
        result = measure(inputs(width, pool, seed), fused, rounded_at_end)
        rows.append({"width": width, "pool": pool, "seed": seed, **result})
        print(width, pool, result["fused_bit_differences"], result["fused_over_eager"], flush=True)
    args.output.write_text(json.dumps({"mlx": version("mlx"), "order": ORDER, "cases": rows},
                                      indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
