"""Replay the complete M2 sidecar owned-storage parity comparison."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-owned-positional-shards-m2-20261122"
INVENTORY = ROOT.parent / "qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json"
INPUTS_SHA = "7e2f69482596abc06807499b8e1d6c02650965573d7017fa6448b8f3152fb8b5"
ORDER = (("clone", "mmap"), ("clone", "pread"), ("clone", "pread"), ("clone", "mmap"),
         ("reused", "pread"), ("reused", "mmap"), ("reused", "mmap"), ("reused", "pread"))


def replay(root=ROOT):
    assert (root / "phase-status.txt").read_bytes() == b"sidecar-owned-storage 0\n"
    raw_manifest = (root / "inputs.sha256").read_bytes()
    assert sha256(raw_manifest).hexdigest() == INPUTS_SHA
    manifest = {name: value for value, name in (line.split() for line in raw_manifest.decode().splitlines())}
    for path in (root / "compare_sidecar.py", PRIOR / "run_probe.py", INVENTORY):
        assert sha256(path.read_bytes()).hexdigest() == manifest[path.name]
    raw = (root / "result.json").read_bytes()
    result = json.loads(raw)
    assert result["hardware"] == {"soc":"Apple M2 Ultra", "memory_bytes":192 * 2**30,
                                  "os":"27.2.1", "storage":"external SSD"}
    inventory = json.loads(INVENTORY.read_bytes())
    entry, = [row for row in inventory["members"] if row["name"] == "mtp.safetensors"]
    assert result["revision"] == inventory["revision"]
    prior = json.loads((PRIOR / "result.json").read_bytes())
    assert result["inventory_sha256"] == manifest["pack-inventory.json"] == prior["inventory_sha256"]
    assert result["binary_sha256"] == manifest["direct_weights"] == prior["binary_sha256"]
    assert result["publisher_sha256"] == entry["publisher_sha256"] and result["file_bytes"] == entry["size"]
    assert result["completed"] is True and result["page_size"] == 16384
    arms = result["arms"]
    assert tuple((row["cache"], row["mode"]) for row in arms) == ORDER
    assert {row["weights_sha256"] for row in arms} == {result["payload_sha256"]}
    assert len(bytes.fromhex(result["payload_sha256"])) == 32
    for field in ("metal_sha256","allocated_bytes","selected_bytes","tensors"):
        assert len({row[field] for row in arms}) == 1
    assert len(bytes.fromhex(arms[0]["metal_sha256"])) == 32
    assert arms[0]["tensors"] == 31 and arms[0]["selected_bytes"] == 5214301696
    observations = {}
    for row in arms:
        assert row["pressure_before"] == row["pressure_after"] == 1
        elapsed = row["elapsed_seconds"]
        assert elapsed > 0 and isfinite(elapsed)
        pages = row["file_pages_before"]
        assert pages["selected_pages"] == 318256
        assert pages["resident_pages"] == (0 if row["cache"] == "clone" else 318256)
        observations.setdefault(f'{row["cache"]}_{row["mode"]}', []).append(elapsed)
    return {"scope": result["scope"], "tensors": 31, "selected_bytes": arms[0]["selected_bytes"],
            "metal_sha256": arms[0]["metal_sha256"], "receipt_sha256": sha256(raw).hexdigest(),
            "read_seconds": observations}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True))
