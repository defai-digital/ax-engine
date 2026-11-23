"""Measure an exact Flash Next routed tail after real MXFP4 expert projections."""
from argparse import ArgumentParser
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter_ns

import mlx.core as mx
import numpy as np
import packed_probe as source
import probe as control
import rounding_probe as arithmetic


def compare(weights, rows, pool, seed, fused, rounded_at_end):
    rng = np.random.default_rng(seed)
    experts = rng.integers(0, pool, size=(rows * source.TOP_K,), dtype=np.uint32)
    order = np.argsort(experts, kind="stable")
    indices = mx.array(experts[order])
    inverse = mx.array(np.argsort(order).astype(np.uint32))
    raw = mx.array(rng.standard_normal((rows, source.HIDDEN), dtype=np.float32)).astype(mx.bfloat16)
    x = raw[mx.array(order // source.TOP_K)].reshape(rows * source.TOP_K, 1, source.HIDDEN)
    shared = mx.array(rng.standard_normal((rows, source.HIDDEN), dtype=np.float32)).astype(mx.bfloat16)
    gates = mx.array(rng.uniform(0.0, 1.0, rows).astype(np.float32)).astype(mx.bfloat16)
    routing = mx.softmax(mx.array(rng.standard_normal((rows, source.TOP_K), dtype=np.float32)), axis=-1).astype(mx.bfloat16)
    mx.eval(indices, inverse, x, shared, gates, routing)
    gate, up, down = weights

    def routed():
        def gather(array, pair):
            return source.gather(array, pair, indices, True)
        product = arithmetic.rounded(gather(x, gate), gather(x, up))
        return gather(product, down).reshape(rows * source.TOP_K, source.HIDDEN)

    precomputed = routed()
    mx.eval(precomputed)
    tail = control.measure((precomputed, inverse, routing, shared, gates), fused, rounded_at_end)
    assert tail["fused_bit_differences"] == 0 and tail["float_acc_bit_differences"] > 0

    def evaluate(mode):
        values = routed(), inverse, routing, shared, gates
        return control.eager(*values) if mode == "eager" else control.launch(fused, values)

    baseline, candidate = (evaluate(mode) for mode in ("eager", "fused"))
    mx.eval(baseline, candidate)
    assert control.unequal(baseline, candidate) == 0
    samples = {"eager": [], "fused": []}
    for mode in control.ORDER:
        start = perf_counter_ns()
        mx.eval(evaluate(mode))
        samples[mode].append((perf_counter_ns() - start) / 1000)
    ratio = median(samples["fused"]) / median(samples["eager"])
    return {
        "width": rows, "route_pool": pool, "seed": seed,
        "distinct_experts": len(np.unique(experts)), "tail": tail,
        "routed_projections_and_tail": {
            "fused_bit_differences": 0, "elements": baseline.size,
            "wall_us": samples, "fused_over_eager": ratio,
        },
    }


def main():
    cli = ArgumentParser(description=__doc__)
    for name in ("pack", "inventory", "out"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    assert version("mlx") == version("mlx-metal") == "0.32.4"
    mx.set_default_device(mx.gpu)
    fused = control.kernel(control.EXACT_SOURCE, "ax_diag_exact_bf16_sorted_combine")
    rounded_at_end = control.kernel(control.FLOAT_SOURCE, "ax_diag_float_sorted_combine")
    inventory = json.loads(args.inventory.read_bytes())
    index, shards = arithmetic.verify_shards(args.pack, inventory)
    reports = []
    pressure = [arithmetic.pressure()]
    for layer in source.LAYERS:
        headers = {}
        arrays = [source.read_tensor(args.pack, index, name, headers) for name in source.tensor_names(layer)]
        weights = [(arrays[i], arrays[i+1]) for i in range(0, len(arrays), 2)]
        cases = [compare(weights, rows, pool, seed, fused, rounded_at_end)
                 for rows, pool, seed in control.SHAPES]
        reports.append({"layer": layer, "cases": cases})
        pressure.append(arithmetic.pressure())
        print(layer, [(row["width"], row["routed_projections_and_tail"]["fused_over_eager"])
                      for row in cases], flush=True)
    assert arithmetic.verify_shards(args.pack, inventory) == (index, shards)
    info = {
        "ps": pressure, "layers": reports, "round_order": control.ORDER, "shards": shards,
        "pack_revision": source.REVISION, "mlx": version("mlx"), "qualification": False,
        "macos": subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip(),
        "soc": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "harness_sha256": source.sha(Path(__file__)), "kernel_sha256": source.sha(Path(control.__file__)),
    }
    args.out.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
