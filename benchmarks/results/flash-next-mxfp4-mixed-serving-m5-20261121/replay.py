"""Verify and replay the archived Flash Next mixed-serving evidence."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parent
AUDITOR = ROOT.parents[2] / "scripts" / "audit_flash_next_mixed_serving.py"


def unpack(bundle, destination):
    metadata = json.loads((bundle / "sources.json").read_bytes())
    if metadata["schema"] != "ax-engine.flash-next.mixed-serving-archive.v1":
        raise ValueError("Unknown mixed-serving archive schema")
    inventory = metadata["retained_files"]
    seen = set()
    with tarfile.open(bundle / "evidence.tar.xz", "r:xz") as archive:
        for member in archive:
            name = member.name
            path = Path(name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts or name in seen:
                raise ValueError(f"Invalid archived pathname or type: {name!r}")
            if name not in inventory or member.size != inventory[name]["size"]:
                raise ValueError(f"Unexpected archive member or size: {name!r}")
            with archive.extractfile(member) as handle:
                data = handle.read()
            if sha256(data).hexdigest() != inventory[name]["sha256"]:
                raise ValueError(f"Archive member hash mismatch: {name}")
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)
            seen.add(name)
    if seen != inventory.keys():
        raise ValueError("Archive is missing an input or receipt")
    return metadata


def replay(bundle, auditor):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        metadata = unpack(bundle, root)
        if metadata["partial"] is not False:
            raise ValueError("A partial campaign cannot be replayed as a complete result")
        holdouts = json.loads((root / "holdout-source.json").read_bytes())
        if set(holdouts["holdouts"]) != {"2048", "2049", "2050", "8192"}:
            raise ValueError("The four physical holdout proofs are required")
        build = json.loads((root / "build.json").read_bytes())
        if (build["server_sha256"] != holdouts["binary_sha256"]
                or sha256((root / "mxfp4-inventory.json").read_bytes()).hexdigest()
                != holdouts["inventory_sha256"]):
            raise ValueError("The holdout executable or pack inventory changed")
        for length, proof in holdouts["holdouts"].items():
            folder = root / f"current-prefill-split-admission-{length}"
            raw = (folder / "result.json").read_bytes()
            result = json.loads(raw)
            if (sha256(raw).hexdigest() != proof["receipt_sha256"]
                    or sha256((folder / "audit.json").read_bytes()).hexdigest() != proof["audit_sha256"]
                    or result["binary_sha256"] != holdouts["binary_sha256"]
                    or result["inventory_sha256"] != holdouts["inventory_sha256"]):
                raise ValueError(f"Unbound physical holdout at {length} tokens")
        return subprocess.check_output([sys.executable, "-B", str(auditor), str(root)], text=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auditor", type=Path, default=AUDITOR)
    parser.add_argument("--bundle", type=Path, default=ROOT)
    args = parser.parse_args()
    print(replay(args.bundle, args.auditor), end="")
