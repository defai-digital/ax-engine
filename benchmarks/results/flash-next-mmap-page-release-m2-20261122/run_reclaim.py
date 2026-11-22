"""Measure targeted file-cache release after the original Flash Next mapped copy."""
from argparse import ArgumentParser
from hashlib import file_digest, sha256
import json
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory

from base_probe import INVENTORY_SHA, PAGE, REVISION, clone, page_state, pressure, range_plan, source_digest, sysctl

MODES = ("mmap", "mmap-reclaim-shared")
IDENTITY = ("weights_sha256", "metal_sha256", "allocated_bytes", "selected_bytes", "tensors")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_receipt(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def full_page_count(regions):
    ranges = sorted((start, start + length) for start, length, _, _ in regions)
    groups = []
    for start, end in ranges:
        if groups and start <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], end)
        else:
            groups.append([start, end])
    return sum(max(0, end // PAGE - (start + PAGE - 1) // PAGE) for start, end in groups)


def invoke(binary, file, plan, mode):
    return json.loads(subprocess.run([binary, file, plan, mode], check=True, capture_output=True,
                                     text=True, timeout=240).stdout)


def run(args):
    root = Path(__file__).resolve().parent
    raw_inventory = (root / "inventory.json").read_bytes()
    inventory = json.loads(raw_inventory)
    if sha256(raw_inventory).hexdigest() != INVENTORY_SHA or inventory["revision"] != REVISION:
        raise ValueError("The publisher inventory changed")
    build = json.loads((root / "build.json").read_bytes())
    if build["binary_sha256"] != digest(args.binary) or PAGE != 16384 or pressure() != 1:
        raise ValueError("The diagnostic requires the pinned binary, 16 KiB pages and normal pressure")
    synthetic = json.loads((root / "synthetic.json").read_bytes())
    if not synthetic["targeted_release_observed"]["mmap-reclaim-shared"]:
        raise ValueError("The direct shared-mapping page control must pass")
    if sysctl("hw.memsize") != str(192 * 1024**3):
        raise ValueError("This functional run is bound to the 192 GB M2 device")

    selected_file = root / "selection.json"
    measurements = {
        "status": "running", "qualification": False, "scope": "M2 Ultra selected mapped cache release",
        "hardware": {"soc": sysctl("machdep.cpu.brand_string"), "macos": platform.mac_ver()[0],
                     "memory_bytes": int(sysctl("hw.memsize")), "storage": "external SSD"},
        "build": build, "inventory_sha256": INVENTORY_SHA, "selection_sha256": digest(selected_file),
        "synthetic_sha256": digest(root / "synthetic.json"), "page_bytes": PAGE, "arms": [],
    }
    write_receipt(args.output, measurements)
    members = {row["name"]: row for row in inventory["members"]}
    selections = json.loads(selected_file.read_bytes())["files"]
    if len(selections) != 2:
        raise ValueError("The campaign requires its two anchored shard selections")
    for shard_index, selection in enumerate(selections):
        name = selection["name"]
        file = args.pack / name
        if digest(file) != members[name]["publisher_sha256"]:
            raise ValueError(f"The selected publisher shard changed: {name}")
        regions = range_plan(file, selection["names"])
        selected_bytes = sum(length for _, length, _, _ in regions)
        if selected_bytes != selection["selected_bytes"]:
            raise ValueError("The selected byte count changed")
        expected_weight_sha = source_digest(file, regions)
        eligible_pages = full_page_count(regions)
        order = MODES if shard_index == 0 else MODES[::-1]
        identity = None
        with TemporaryDirectory(dir=args.scratch) as work:
            path = Path(work)
            plan = path / "geometry.txt"
            plan.write_text("".join(" ".join(map(str, (start, length, dtype, len(shape), *shape))) + "\n"
                                    for start, length, dtype, shape in regions))
            for condition in ("nonresident", "primed"):
                for trial, mode in enumerate((*order, *order[::-1]), 1):
                    if pressure() != 1:
                        raise ValueError("The host is under memory pressure before the run")
                    target = path / f"{condition}-{trial}.safetensors"
                    clone(file, target)
                    if condition == "primed" and source_digest(target, regions) != expected_weight_sha:
                        raise ValueError("The bytes used to prime the clone changed")
                    before = page_state(target, regions)
                    if before["resident_pages"] != (0 if condition == "nonresident" else before["selected_pages"]):
                        raise ValueError("The clone did not begin in its intended page-cache condition")
                    actual = invoke(args.binary, target, plan, mode)
                    after = page_state(target, regions)
                    restart = invoke(args.binary, target, plan, "mmap")
                    restart_after = page_state(target, regions)
                    target.unlink()
                    if (actual["reclaim_errno"] != 0 or restart["reclaim_errno"] != 0
                            or actual["requested_invalidate_pages"] != (eligible_pages if mode != "mmap" else 0)
                            or actual["weights_sha256"] != expected_weight_sha
                            or actual["selected_bytes"] != selected_bytes
                            or actual["mode"] != mode or restart["mode"] != "mmap"):
                        raise ValueError("The measured reader fell through or altered the selected payload")
                    stable = tuple(actual[field] for field in IDENTITY)
                    identity = identity or stable
                    if stable != identity or stable != tuple(restart[field] for field in IDENTITY) or pressure() != 1:
                        raise ValueError("Observed MLX identity, restart bytes, or host pressure changed")
                    measurements["arms"].append({
                        "shard": name, "condition": condition, "trial": trial, "eligible_pages": eligible_pages,
                        "before": before, "after": after, "restart_after": restart_after, "pressure": 1,
                        "initial_read": actual, "mapped_restart": restart,
                    })
                    write_receipt(args.output, measurements)
                    print(name, condition, mode, after["resident_pages"], restart["elapsed_seconds"], flush=True)
    measurements["status"] = "complete"
    write_receipt(args.output, measurements)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("binary", "pack", "scratch", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    run(parser.parse_args())
