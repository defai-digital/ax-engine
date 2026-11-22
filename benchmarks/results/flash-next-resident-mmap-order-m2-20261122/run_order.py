"""Compare the same native mapped reader in name order and byte order."""
from argparse import ArgumentParser
from hashlib import file_digest, sha256
import json
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory

from run_probe import INVENTORY_SHA, PAGE, REVISION, clone, page_state, pressure, range_plan, source_digest, sysctl

SHARDS = ("model-00021-of-00025.safetensors", "model-00024-of-00025.safetensors")
MANIFEST_SHA = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
ARMS = ("mmap", "mmap-offset", "mmap-offset", "mmap")


def require(valid, message):
    if not valid:
        raise ValueError(message)


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("pack", "resident-shards", "inventory", "binary", "scratch", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    raw_inventory = args.inventory.read_bytes()
    inventory = json.loads(raw_inventory)
    require(sha256(raw_inventory).hexdigest() == INVENTORY_SHA and inventory["revision"] == REVISION,
            "Publisher inventory changed")
    raw_resident = args.resident_shards.read_bytes()
    resident = json.loads(raw_resident)
    require(resident["manifest_sha256"] == MANIFEST_SHA
            and sha256((args.pack / "model-manifest.json").read_bytes()).hexdigest() == MANIFEST_SHA,
            "Pack manifest differs from the observed ordinary startup")
    require(pressure() == 1, "Normal memory pressure required")
    record = {
        "scope": "Owned MLX tensor copy order, single-shard diagnostic",
        "qualification": False,
        "hardware": {"soc": sysctl("machdep.cpu.brand_string"), "memory_bytes": int(sysctl("hw.memsize")),
                     "form_factor": "Mac Studio", "storage": "external APFS SSD", "os": platform.mac_ver()[0]},
        "revision": REVISION,
        "inventory_sha256": INVENTORY_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "resident_shards_sha256": sha256(raw_resident).hexdigest(),
        "binary_sha256": sha256(args.binary.read_bytes()).hexdigest(),
        "driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "page_size": PAGE,
        "shards": {},
    }
    entries = {entry["name"]: entry for entry in inventory["members"]}

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output)

    for index, name in enumerate(SHARDS):
        entry = entries[name]
        publisher = entry["publisher_sha256"]
        source = args.pack / name
        with source.open("rb") as reader:
            header_len = int.from_bytes(reader.read(8), "little")
            require(0 < header_len < 100_000_000, "Safetensors header is out of bounds")
            header = reader.read(header_len)
            require(len(header) == header_len, "Incomplete safetensors header")
            reader.seek(0)
            require(file_digest(reader, "sha256").hexdigest() == publisher, "Publisher bytes changed")
        names = sorted(set(json.loads(header)) - {"__metadata__"})
        require(len(names) == resident["shards"][name]["tensors"],
                "Shard does not match the resident startup selection")
        regions = range_plan(source, names)
        expected = source_digest(source, regions)
        selected_bytes = sum(item[1] for item in regions)
        require(selected_bytes == entry["size"] - 8 - header_len, "Selected ranges do not cover the full shard")
        offsets = [row[0] for row in regions]
        backwards = sum(right < left for left, right in zip(offsets, offsets[1:]))
        require(backwards > 0, "The name order does not exercise backward jumps")
        rows = []
        record["shards"][name] = {
            "publisher_sha256": publisher, "header_sha256": sha256(header).hexdigest(),
            "source_bytes": entry["size"], "selected_bytes": selected_bytes, "header_bytes": 8 + header_len,
            "name_order_backward_jumps": backwards, "tensors": len(regions), "weights_sha256": expected,
            "arms": rows,
        }
        save()
        signature = None
        with TemporaryDirectory(dir=args.scratch) as tempdir:
            root = Path(tempdir)
            plan = root / "geometry.txt"
            plan.write_text("".join(" ".join(map(str, (start, size, dtype, len(shape), *shape))) + "\n"
                                    for start, size, dtype, shape in regions))
            sequence = ARMS if index == 0 else tuple("mmap-offset" if arm == "mmap" else "mmap" for arm in ARMS)
            for mode in sequence:
                before = pressure()
                require(before == 1, "Diagnostic reached pressure before measurement")
                target = root / "fresh.safetensors"
                clone(source, target)
                try:
                    pages = page_state(target, regions)
                    call = subprocess.run([args.binary, target, plan, mode], capture_output=True, text=True,
                                          check=True, timeout=240)
                    data = json.loads(call.stdout)
                finally:
                    target.unlink()
                after = pressure()
                require(data["mode"] == mode and data["tensors"] == len(regions)
                        and data["selected_bytes"] == selected_bytes and data["weights_sha256"] == expected
                        and after == 1, "Tensor bytes, label or pressure changed")
                result = (data["metal_sha256"], data["allocated_bytes"])
                if signature is None:
                    signature = result
                require(result == signature, "Metal output or allocated size changed")
                rows.append(data | {"file_pages_before": pages, "pressure_before": before, "pressure_after": after})
                save()
                print(f"{name} {mode} {data['elapsed_seconds']:.3f} s ({pages['resident_pages']} resident pages)",
                      flush=True)
    record["completed"] = True
    save()


if __name__ == "__main__":
    main()
