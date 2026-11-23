"""Check the Flash Next routed expert epilogue's BF16 rounding boundary."""

from argparse import ArgumentParser
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import mlx.core as mx

import packed_probe as reference


CASES = ((1, reference.EXPERTS), (511, reference.EXPERTS),
         (512, reference.EXPERTS), (512, 64))


def rounded(gate, up):
    gate32 = gate.astype(mx.float32)
    gate_activation = (gate32 * mx.sigmoid(gate32)).astype(gate.dtype)
    return gate_activation * up


@mx.compile
def rounded_compiled(gate, up):
    return rounded(gate, up)


@mx.compile
def late_round(gate, up):
    gate32 = gate.astype(mx.float32)
    activated = gate32 * mx.sigmoid(gate32)
    return (activated * up.astype(mx.float32)).astype(gate.dtype)


def differences(old, new):
    assert old.dtype == new.dtype == mx.bfloat16 and old.shape == new.shape
    left, right = old.astype(mx.float32), new.astype(mx.float32)
    finite = mx.all(mx.isfinite(left)) & mx.all(mx.isfinite(right))
    diff = mx.sum((old.view(mx.uint16) != new.view(mx.uint16)).astype(mx.uint32))
    maximum = mx.max(mx.abs(left - right))
    mx.eval(finite, diff, maximum)
    assert finite.item()
    return {"elements": old.size, "different_bits": int(diff.item()),
            "max_abs_difference": float(maximum.item())}


def measure(gate, up, down, rows, pool):
    x, indices, chosen = reference.inputs(rows, pool)
    sorted_indices = rows != 1
    g = reference.gather(x, gate, indices, sorted_indices)
    u = reference.gather(x, up, indices, sorted_indices)
    mx.eval(g, u)
    actual = [fn(g, u) for fn in (rounded, rounded_compiled, late_round)]
    out = [reference.gather(a, down, indices, sorted_indices) for a in actual]
    mx.eval(*actual, *out)
    variants = {}
    for idx, name in enumerate(("compiled_with_bf16_activation", "late_bf16_round"), 1):
        variants[name] = {"activation": differences(actual[0], actual[idx]),
                          "down": differences(out[0], out[idx])}
    exact = variants["compiled_with_bf16_activation"]
    assert all(item["different_bits"] == 0 for item in exact.values())
    return {"rows": rows, "route_pool": pool, "distinct_experts": chosen, "variants": variants}


def verify_shards(pack, inventory):
    assert inventory["revision"] == reference.REVISION and len(inventory["members"]) == 49
    members = {row["name"]: row for row in inventory["members"]}
    path = pack / "model.safetensors.index.json"
    assert reference.sha(path) == members[path.name]["sha256"]
    index = json.loads(path.read_bytes())["weight_map"]
    shards = sorted({index[name] for layer in reference.LAYERS for name in reference.tensor_names(layer)})
    for name in shards:
        path = pack / name
        record = members[name]
        assert path.stat().st_size == record["size"]
        assert reference.sha(path) == record["publisher_sha256"] == record["sha256"]
    return index, shards


def pressure():
    level = int(subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip())
    assert level == 1
    return level


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("pack", "inventory", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    versions = {name: version(name) for name in ("mlx", "mlx-metal", "numpy")}
    assert versions["mlx"] == versions["mlx-metal"] == "0.32.4"
    mx.set_default_device(mx.gpu)
    inventory = json.loads(args.inventory.read_bytes())
    samples = [pressure()]
    index, shards = verify_shards(args.pack, inventory)
    reports = []
    for layer in reference.LAYERS:
        headers = {}
        gw, gs, uw, us, dw, ds = (reference.read_tensor(args.pack, index, name, headers)
                                  for name in reference.tensor_names(layer))
        samples.append(pressure())
        cases = [measure((gw, gs), (uw, us), (dw, ds), rows, pool) for rows, pool in CASES]
        samples.append(pressure())
        reports.append({"layer": layer, "cases": cases})
        print(f"Layer {layer}: {len(cases)} epilogue cases complete.", flush=True)
        del gw, gs, uw, us, dw, ds
        mx.clear_cache()
    assert verify_shards(args.pack, inventory) == (index, shards)
    samples.append(pressure())
    assert any(case["variants"]["late_bf16_round"]["down"]["different_bits"] > 0
               for report in reports for case in report["cases"])
    macos = subprocess.check_output(["/usr/bin/sw_vers", "-productVersion"], text=True).strip()
    result = {"scope": "Two publisher-backed routed expert layers with deterministic BF16 activations",
              "qualification": False, "versions": versions, "python": sys.version.split()[0], "macos": macos,
              "pack_revision": reference.REVISION, "inventory_sha256": reference.sha(args.inventory),
              "reused_probe_sha256": reference.sha(Path(reference.__file__)), "probe_sha256": reference.sha(Path(__file__)),
              "publisher_shards_verified_before_and_after": shards,
              "sampled_pressure_levels": samples, "layer_results": reports}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
