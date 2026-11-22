"""Verify resident shard inputs and recompute both mmap visitation comparisons."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median
import tarfile

ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT.parent / "qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json"
STARTUP_ARCHIVE = ROOT.parent / "flash-next-delivery-bound-startup-m5-20261122/evidence.tar.xz"
SHARDS = ("model-00021-of-00025.safetensors", "model-00024-of-00025.safetensors")
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"


def need(valid, message):
    if not valid:
        raise ValueError(message)


def hashed_json(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def replay(path, inventory_path, resident_path, startup_archive, fixed):
    result, _ = hashed_json(path)
    inventory, inventory_sha = hashed_json(inventory_path)
    resident, resident_sha = hashed_json(resident_path)
    with tarfile.open(startup_archive) as archive:
        raw = archive.extractfile("native-support/server-default.log").read()
    need(resident["startup_trace_sha256"] == sha256(raw).hexdigest(), "The baseline startup trace changed")
    by_shard = {name.decode(): {"tensors": int(count), "read_us": int(elapsed)} for name, count, elapsed in
                re.findall(rb'shard=Some\("(.*?)"\) tensor_count=(\d+) read_us=(\d+)', raw)}
    need(resident["shards"] == by_shard and len(by_shard) == 15, "Resident selection differs from startup")
    need(result["qualification"] is False and result["completed"] is True
         and result["revision"] == inventory["revision"] and result["inventory_sha256"] == inventory_sha
         and result["manifest_sha256"] == resident["manifest_sha256"] == MANIFEST
         and result["resident_shards_sha256"] == resident_sha
         and list(result["shards"]) == list(SHARDS), "The probe did not complete with the exact inputs")
    publisher = {entry["name"]: entry for entry in inventory["members"]}
    name_order = "mmap-fixed" if fixed else "mmap"
    physical_order = "mmap-offset-fixed" if fixed else "mmap-offset"
    normalized = {}
    for index, (name, shard) in enumerate(result["shards"].items()):
        entry = publisher[name]
        need(shard["publisher_sha256"] == entry["publisher_sha256"]
             and shard["source_bytes"] == entry["size"] == shard["selected_bytes"] + shard["header_bytes"]
             and shard["tensors"] == by_shard[name]["tensors"] and shard["name_order_backward_jumps"] > 0,
             "Selected resident shard differs from publisher or startup trace")
        sequence = (name_order, physical_order, physical_order, name_order)
        if index == 1:
            sequence = (physical_order, name_order, name_order, physical_order)
        arms = shard["arms"]
        need([arm["mode"] for arm in arms] == list(sequence), "Mmap arm order changed")
        need(all(arm["tensors"] == shard["tensors"] and arm["selected_bytes"] == shard["selected_bytes"]
                 and arm["weights_sha256"] == shard["weights_sha256"]
                 and arm["allocated_bytes"] >= arm["selected_bytes"]
                 and arm["pressure_before"] == arm["pressure_after"] == 1
                 and arm["file_pages_before"]["selected_pages"] > 0
                 and 0 <= arm["file_pages_before"]["resident_pages"] <= arm["file_pages_before"]["selected_pages"]
                 for arm in arms), "Tensor signature, allocation or sampled pressure changed")
        need(len({(arm["metal_sha256"], arm["allocated_bytes"]) for arm in arms}) == 1,
             "Metal output or allocated buffer sizes changed")
        if fixed:
            need(all(0 < arm["reserve_and_touch_seconds"]
                     and 0 < arm["copy_only_seconds"]
                     and arm["reserve_and_touch_seconds"] + arm["copy_only_seconds"] <= arm["elapsed_seconds"]
                     for arm in arms), "Allocator and mmap phase timing differs from total")
        metric = "copy_only_seconds" if fixed else "elapsed_seconds"
        arm_order_pairs = []
        for a, b in (arms[:2], arms[2:]):
            control, treatment = (a, b) if a["mode"] == name_order else (b, a)
            arm_order_pairs.append({
                "cold_source_pages": not (control["file_pages_before"]["resident_pages"]
                                          or treatment["file_pages_before"]["resident_pages"]),
                "control_seconds": control[metric],
                "physical_order_seconds": treatment[metric],
                "physical_over_name_time": treatment[metric] / control[metric],
            })
        normalized[name] = {
            "tensors": shard["tensors"],
            "selected_bytes": shard["selected_bytes"],
            "name_order_backward_jumps": shard["name_order_backward_jumps"],
            "measured_m5_default_copy_seconds": by_shard[name]["read_us"] / 1e6,
            "metric": metric,
            "same_layout": fixed,
            "name_order_total_median_seconds": median(arm["elapsed_seconds"] for arm in arms if arm["mode"] == name_order),
            "physical_order_total_median_seconds": median(arm["elapsed_seconds"] for arm in arms if arm["mode"] == physical_order),
            "pairs": arm_order_pairs,
        }
    return normalized


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--result", default=ROOT / "result.json", type=Path)
    p.add_argument("--inventory", default=INVENTORY, type=Path)
    p.add_argument("--resident-shards", default=ROOT / "resident-shards.json", type=Path)
    p.add_argument("--startup-archive", default=STARTUP_ARCHIVE, type=Path)
    p.add_argument("--fixed", action="store_true")
    args = p.parse_args()
    if args.result == ROOT / "result.json":
        build, _ = hashed_json(ROOT / "build.json")
        result, _ = hashed_json(args.result)
        need((ROOT / "status.txt").read_text() == "0\n"
             and build["mlx"] == build["mlx-metal"] == "0.32.4"
             and build["binary_sha256"] == result["binary_sha256"]
             and build["source_sha256"] == sha256((ROOT / "direct_weights.cpp").read_bytes()).hexdigest()
             and build["driver_sha256"] == result["driver_sha256"]
             == sha256((ROOT / "run_order.py").read_bytes()).hexdigest(), "Build binding changed")
    print(json.dumps(replay(args.result, args.inventory, args.resident_shards, args.startup_archive, args.fixed),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
