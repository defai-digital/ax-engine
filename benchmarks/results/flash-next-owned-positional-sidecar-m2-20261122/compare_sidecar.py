"""Compare both owned-storage readers using the complete pinned MTP sidecar."""
from argparse import ArgumentParser
from hashlib import file_digest, sha256
import json
from math import isfinite
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory

from run_probe import (INVENTORY_SHA, PAGE, REVISION, clone, page_state,
                       pressure, range_plan, source_digest, sysctl)

EXPECTED_BINARY = "92070e6c2635331e66a8e941969ac9ab5d1f49dc53085aff0acc02d013bf8b9b"
ORDER = (("clone", "mmap"), ("clone", "pread"), ("clone", "pread"), ("clone", "mmap"),
         ("reused", "pread"), ("reused", "mmap"), ("reused", "mmap"), ("reused", "pread"))


def run(args):
    inventory_bytes = args.inventory.read_bytes()
    binary_hash = sha256(args.binary.read_bytes()).hexdigest()
    if sha256(inventory_bytes).hexdigest() != INVENTORY_SHA or binary_hash != EXPECTED_BINARY:
        raise ValueError("Bound inventory or resident allocation probe changed")
    inventory = json.loads(inventory_bytes)
    if inventory["revision"] != REVISION or pressure() != 1:
        raise ValueError("Wrong publisher revision or above-normal starting pressure")
    original = args.pack / "mtp.safetensors"
    entry, = (row for row in inventory["members"] if row["name"] == original.name)
    with original.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
        if digest != entry["publisher_sha256"] or original.stat().st_size != entry["size"]:
            raise ValueError("MTP sidecar publisher bytes changed")
        source.seek(0)
        header_bytes = int.from_bytes(source.read(8), "little")
        if not 0 < header_bytes <= 100_000_000:
            raise ValueError("Invalid sidecar header length")
        names = sorted(set(json.loads(source.read(header_bytes))) - {"__metadata__"})
    regions = range_plan(original, names)
    if len(regions) != 31 or {row[2] for row in regions} != {"BF16"}:
        raise ValueError("Unexpected MTP sidecar tensor geometry")
    independent_digest = source_digest(original, regions)
    payload_bytes = sum(length for _, length, _, _ in regions)
    result = {"scope": "Complete pinned MTP sidecar owned-storage read diagnostic",
              "revision": REVISION, "inventory_sha256": INVENTORY_SHA,
              "binary_sha256": binary_hash, "file_bytes": entry["size"], "publisher_sha256": digest,
              "payload_sha256": independent_digest, "page_size": PAGE,
              "hardware": {"soc": sysctl("machdep.cpu.brand_string"), "os": platform.mac_ver()[0],
                           "memory_bytes": int(sysctl("hw.memsize")), "storage": "external SSD"}, "arms": []}
    with TemporaryDirectory(dir=args.scratch) as directory:
        scratch = Path(directory)
        plan = scratch / "geometry.txt"
        plan.write_text("".join(" ".join(map(str, (start, extent, dtype, len(shape), *shape))) + "\n"
                                for start, extent, dtype, shape in regions))
        for cache, reader in ORDER:
            target = original if cache == "reused" else scratch / original.name
            before = pressure()
            if before != 1:
                raise ValueError("Non-normal preflight pressure")
            if cache == "clone":
                clone(original, target)
            try:
                pages = page_state(target, regions)
                proc = subprocess.run([args.binary, target, plan, reader], capture_output=True,
                                      text=True, check=True, timeout=240)
                raw = json.loads(proc.stdout)
            finally:
                if cache == "clone":
                    target.unlink()
            after = pressure()
            if (after != 1 or raw["weights_sha256"] != independent_digest or raw["mode"] != reader
                    or raw["selected_bytes"] != payload_bytes or raw["tensors"] != len(names)
                    or not 0 < raw["elapsed_seconds"] or not isfinite(raw["elapsed_seconds"])):
                raise ValueError("Sidecar output bytes, timing or pressure differed")
            row = {"cache": cache, "file_pages_before": pages, "pressure_before": before,
                   "pressure_after": after, **raw}
            result["arms"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(cache, reader, f'{row["elapsed_seconds"]:.3f} s', flush=True)
    if len({(row["allocated_bytes"], row["metal_sha256"]) for row in result["arms"]}) != 1:
        raise ValueError("Metal output or allocation size varied by sidecar route")
    result["completed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("binary", "inventory", "pack", "scratch", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
