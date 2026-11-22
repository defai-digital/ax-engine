"""Compare synchronous writes into MLX-owned storage with the mapped copy."""
from argparse import ArgumentParser
import ctypes
from hashlib import file_digest, sha256
import json
import mmap
import os
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory

REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
INVENTORY_SHA = "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"
PAGE = os.sysconf("SC_PAGESIZE")
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
LIBC.clonefile.restype = ctypes.c_int
LIBC.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
LIBC.mincore.restype = ctypes.c_int
CHUNK = 4 * 1024 * 1024
SIZES = {"U8": 1, "U32": 4, "BF16": 2, "F32": 4}


def sysctl(name):
    return subprocess.check_output(["/usr/sbin/sysctl", "-n", name], text=True).strip()


def pressure():
    return int(sysctl("kern.memorystatus_vm_pressure_level"))


def range_plan(path, names):
    with path.open("rb") as file:
        header_len = int.from_bytes(file.read(8), "little")
        if not 0 < header_len <= 100_000_000:
            raise ValueError("Invalid safetensors header")
        meta = json.loads(file.read(header_len))
    regions = []
    for name in sorted(names):
        tensor = meta[name]
        start, end = tensor["data_offsets"]
        dtype = tensor["dtype"]
        shape = tensor["shape"]
        itemsize = SIZES[dtype]
        for extent in shape:
            itemsize *= extent
        if not 0 <= start < end or end - start != itemsize:
            raise ValueError("Invalid selected tensor extent")
        regions.append((8 + header_len + start, end - start, dtype, shape))
    return regions


def source_digest(path, regions):
    digest = sha256()
    with path.open("rb", buffering=0) as file:
        for offset, length, _, _ in regions:
            file.seek(offset)
            left = length
            while left:
                chunk = file.read(min(left, CHUNK))
                if not chunk:
                    raise ValueError("Early EOF in independent checksum")
                digest.update(chunk)
                left -= len(chunk)
    return digest.hexdigest()


def page_state(path, regions):
    wanted = {p for start, length, _, _ in regions
              for p in range(start // PAGE, (start + length - 1) // PAGE + 1)}
    with path.open("rb") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY) as backing:
            vector = (ctypes.c_ubyte * ((len(backing) + PAGE - 1) // PAGE))()
            address = ctypes.c_char.from_buffer(backing)
            try:
                if LIBC.mincore(ctypes.addressof(address), len(backing), vector):
                    raise OSError(ctypes.get_errno(), "Could not sample selected pages")
            finally:
                del address
            return {"selected_pages": len(wanted),
                    "resident_pages": sum(vector[p] & 1 for p in wanted)}


def clone(source, target):
    if LIBC.clonefile(os.fsencode(source), os.fsencode(target), 0):
        raise OSError(ctypes.get_errno(), "Independent APFS clone failed")


def run(args):
    raw_inventory = args.inventory.read_bytes()
    raw_selection = args.selection.read_bytes()
    inventory = json.loads(raw_inventory)
    if sha256(raw_inventory).hexdigest() != INVENTORY_SHA or inventory["revision"] != REVISION:
        raise ValueError("Publisher inventory changed")
    if pressure() != 1:
        raise ValueError("Diagnostic requires normal prelaunch pressure")
    pack = {entry["name"]: entry for entry in inventory["members"]}
    source = json.loads(raw_selection)
    cases = source["files"][:2]
    if len(cases) != 2:
        raise ValueError("Expected two different shard controls")
    measurement = {"scope": "Synchronous resident shard diagnostic",
                   "hardware": {"soc": sysctl("machdep.cpu.brand_string"), "os": platform.mac_ver()[0],
                                "memory_bytes": int(sysctl("hw.memsize")), "storage": "external SSD"},
                   "revision": REVISION, "inventory_sha256": INVENTORY_SHA,
                   "selection_sha256": sha256(raw_selection).hexdigest(),
                   "binary_sha256": sha256(args.binary.read_bytes()).hexdigest(),
                   "page_size": PAGE, "arms": []}
    eof = subprocess.run([args.binary, "--eof-self-test"], capture_output=True, text=True, check=True)
    if eof.stdout.strip() != "Partial read rejected before array construction.":
        raise ValueError("EOF safety control failed")
    measurement["short_read_rejected"] = True
    args.output.write_text(json.dumps(measurement, indent=2) + "\n")
    for index, selection in enumerate(cases):
        name = selection["name"]
        file = args.pack / name
        entry = pack[name]
        with file.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != entry["publisher_sha256"]:
                raise ValueError(f"Unexpected publisher hash for {name}")
        regions = range_plan(file, selection["names"])
        selected = sum(row[1] for row in regions)
        unaligned = sum(start % SIZES[dtype] != 0 for start, _, dtype, _ in regions)
        if selected != selection["selected_bytes"]:
            raise ValueError("Selected byte count changed")
        payload = source_digest(file, regions)
        with TemporaryDirectory(dir=args.scratch) as directory:
            scratch = Path(directory)
            plan = scratch / "geometry.txt"
            plan.write_text("".join(" ".join(map(str, (start, length, dtype, len(shape), *shape))) + "\n"
                                    for start, length, dtype, shape in regions))
            if index == 0:
                sequence = (("clone", "mmap"), ("clone", "pread"),
                            ("reused", "pread"), ("reused", "mmap"))
            else:
                sequence = (("clone", "pread"), ("clone", "mmap"),
                            ("reused", "mmap"), ("reused", "pread"))
            signature = None
            for reuse, mode in sequence:
                before = pressure()
                if before != 1:
                    raise ValueError("Diagnostic reached memory pressure before measurement")
                target = file
                if reuse == "clone":
                    target = scratch / "fresh.safetensors"
                    clone(file, target)
                try:
                    cache = page_state(target, regions)
                    completed = subprocess.run([args.binary, target, plan, mode], capture_output=True,
                                               text=True, timeout=240)
                    if completed.returncode != 0:
                        raise ValueError(f"Resident reader failed: {completed.stderr.strip()}")
                    data = json.loads(completed.stdout)
                finally:
                    if reuse == "clone":
                        target.unlink()
                after = pressure()
                if data["weights_sha256"] != payload or data["selected_bytes"] != selected:
                    raise ValueError("Selected model bytes differ from the publisher source")
                row = (data["allocated_bytes"], data["metal_sha256"], data["tensors"])
                if signature is None:
                    signature = row
                if row != signature or after != 1 or data["mode"] != mode:
                    raise ValueError("Allocator signature, Metal output, label, or pressure changed")
                measurement["arms"].append({"shard": name, "publisher_sha256": entry["publisher_sha256"],
                                            "cache": reuse, "file_pages_before": cache,
                                            "unaligned_source_tensors": unaligned,
                                            "pressure_before": before, "pressure_after": after, **data})
                args.output.write_text(json.dumps(measurement, indent=2) + "\n")
                print(name, reuse, mode, f'{data["elapsed_seconds"]:.3f} s', flush=True)
    measurement["completed"] = True
    args.output.write_text(json.dumps(measurement, indent=2) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("binary", "selection", "inventory", "pack", "scratch", "output"):
        parser.add_argument(key, type=Path)
    run(parser.parse_args())
