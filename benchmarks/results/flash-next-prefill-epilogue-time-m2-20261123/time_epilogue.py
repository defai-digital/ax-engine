"""Compare fusion of the exact routed SwiGLU epilogue at Flash Next prefill widths."""
from argparse import ArgumentParser
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter_ns

import mlx.core as mx
import packed_probe as reference
import rounding_probe as arithmetic

ORDER = ("eager", "compiled", "compiled", "eager") * 5


def timed(make):
    start = perf_counter_ns()
    output = make()
    mx.eval(output)
    return (perf_counter_ns() - start) / 1000


def compare(weights, rows, pool):
    gate, up, down = weights
    x, indices, experts = reference.inputs(rows, pool)
    sorted_indices = rows > 1
    def gather(array, weights):
        return reference.gather(array, weights, indices, sorted_indices)

    gate_values, up_values = gather(x, gate), gather(x, up)
    mx.eval(gate_values, up_values)
    operations = {"eager": arithmetic.rounded, "compiled": arithmetic.rounded_compiled}
    def run(mode, scope):
        fn = operations[mode]
        if scope == "epilogue_only":
            return fn(gate_values, up_values)
        return gather(fn(gather(x, gate), gather(x, up)), down)

    results = {}
    for scope in ("epilogue_only", "gate_up_epilogue_down"):
        old, new = (run(mode, scope) for mode in operations)
        mx.eval(old, new)
        assert reference.same_bits(old, new)
        for mode in operations:
            mx.eval(run(mode, scope))
        samples = {mode: [] for mode in operations}
        for mode in ORDER:
            samples[mode].append(timed(lambda: run(mode, scope)))
        results[scope] = {"bitwise_exact": True, "wall_us": samples,
                          "compiled_over_eager": median(samples["compiled"]) / median(samples["eager"])}
    return {"rows": rows, "route_pool": pool, "distinct_experts": experts, "scopes": results}


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("pack", "inventory", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    assert version("mlx") == version("mlx-metal") == "0.32.4"
    mx.set_default_device(mx.gpu)
    inventory = json.loads(args.inventory.read_bytes())
    index, shards = arithmetic.verify_shards(args.pack, inventory)
    memory_pressure = [arithmetic.pressure()]
    layers = []
    for layer in reference.LAYERS:
        headers = {}
        arrays = [reference.read_tensor(args.pack, index, name, headers)
                  for name in reference.tensor_names(layer)]
        weights = [(arrays[i], arrays[i + 1]) for i in range(0, len(arrays), 2)]
        layers.append({"layer": layer, "cases": [compare(weights, rows, pool)
                                                  for rows, pool in arithmetic.CASES]})
        memory_pressure.append(arithmetic.pressure())
        print(layers[-1]["layer"], [row["scopes"]["gate_up_epilogue_down"]["compiled_over_eager"]
                                    for row in layers[-1]["cases"]], flush=True)
    assert arithmetic.verify_shards(args.pack, inventory) == (index, shards)
    result = {"layers": layers, "pack_revision": reference.REVISION, "shards": shards,
              "round_order": ORDER, "qualification": False, "mlx": version("mlx"),
              "soc": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
              "macos": subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip(),
              "memory_pressure": memory_pressure, "probe_sha256": reference.sha(Path(__file__))}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
