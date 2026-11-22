"""Replay the two-shard Flash Next sidecar read-ahead component measurements."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
IDENTITY = ("weights_sha256", "metal_sha256", "allocated_bytes", "selected_bytes", "tensors")
SOURCE_HASHES = {
    "pipeline_weights.cpp": "reader_source_sha256",
    "base_probe.py": "base_probe_sha256",
    "run_overlap.py": "runner_sha256",
}
ORDERS = (("serial", "overlap", "overlap", "serial"), ("overlap", "serial", "serial", "overlap"))
SIDECAR_BYTES = 5_214_301_696


def require(condition, error):
    if not condition:
        raise ValueError(error)


def digest(data):
    return sha256(data).hexdigest()


def read_archive(path):
    expected = {"build.json", "inventory.json", "selection.json", "synthetic.json",
                "result.json", "status.txt", *SOURCE_HASHES}
    payloads = {}
    with tarfile.open(path) as archive:
        for member in archive:
            require(member.isfile() and member.name in expected - payloads.keys(),
                    "Unexpected evidence member")
            with archive.extractfile(member) as stream:
                payloads[member.name] = stream.read()
    require(payloads.keys() == expected, "An evidence member is missing")
    return payloads


def identity(arm, part):
    return tuple(arm[part][key] for key in IDENTITY)


def verify_arm(arm, mode, sidecar_bytes, measured):
    enabled = mode == "overlap"
    require(arm["buffered_prefetch"] is enabled
            and arm["prefetched_bytes"] == (sidecar_bytes if enabled else 0),
            "The selected sidecar read does not match the tested mode")
    times = tuple(arm[key] for key in ("trunk_seconds", "prefetch_seconds", "overlap_seconds",
                                     "join_wait_seconds", "sidecar_copy_seconds", "combined_seconds"))
    require(all(type(value) in (int, float) and isfinite(value) and value >= 0 for value in times),
            "A phase recorded invalid wall time")
    trunk, read, overlap, wait, copy, total = times
    require((0 <= overlap <= min(trunk, read) if enabled else overlap == read == 0)
            and (not enabled or not measured or overlap > 0)
            and abs(total - trunk - wait - copy) <= 0.02,
            "The reader overlap and load phase timing do not agree")


def replay(root=ROOT):
    payload = read_archive(root / "evidence.tar.xz")
    raw, build, inventory, selection, synthetic = (
        json.loads(payload[f"{label}.json"])
        for label in ("result", "build", "inventory", "selection", "synthetic")
    )
    require(payload["status.txt"] == b"0\n" and raw["status"] == "complete"
            and raw["qualification"] is False
            and raw["hardware"]["soc"] == "Apple M2 Ultra"
            and raw["hardware"]["memory_bytes"] == 192 * 1024**3
            and raw["hardware"]["storage"] == "external SSD",
            "The measurement lacks the expected terminal host receipt")
    require(raw["pack_revision"] == inventory["revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
            and raw["inventory_sha256"] == digest(payload["inventory.json"])
            == "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"
            and raw["selection_sha256"] == digest(payload["selection.json"])
            and raw["synthetic_sha256"] == digest(payload["synthetic.json"])
            and raw["build"] == build and build["mlx"] == build["mlx_metal"] == "0.32.4"
            and all(digest(payload[name]) == build[key] for name, key in SOURCE_HASHES.items()),
            "The measured model, paths or software inputs changed")
    publisher = {row["name"]: row["publisher_sha256"] for row in inventory["members"]}
    sidecar = raw["sidecar_geometry"]
    require(len(publisher) == 49 and sidecar["sha256"] == publisher["mtp.safetensors"]
            and sidecar["bytes"] == SIDECAR_BYTES and len(set(sidecar["names"])) == 31
            and sidecar["names"] == sorted(sidecar["names"])
            and all(name.startswith("mtp.") for name in sidecar["names"]),
            "The publisher sidecar extent changed")
    require(synthetic["copied_bytes_match"] is True and synthetic["short_extent_rejected"] is True
            and synthetic["expected_sidecar_bytes"] == 444936, "A synthetic control failed")
    for mode, arm in synthetic["arms"].items():
        verify_arm(arm, mode, 444936, False)
    for part in ("trunk", "sidecar"):
        require(identity(synthetic["arms"]["serial"], part) == identity(synthetic["arms"]["overlap"], part),
                "The synthetic CPU and Metal payloads diverged")

    measured = root.parent
    original = json.loads((measured / "flash-next-page-aligned-reader-m2-20261122" / "result.json").read_bytes())
    original_trunk = {row["shard"]: row for row in original["arms"]}
    earlier_sidecar_root = measured / "flash-next-owned-positional-sidecar-m2-20261122"
    earlier_sidecar = json.loads((earlier_sidecar_root / "comparison.json").read_bytes())
    earlier_sidecar_payload = json.loads((earlier_sidecar_root / "result.json").read_bytes())
    require(earlier_sidecar_payload["publisher_sha256"] == sidecar["sha256"]
            and earlier_sidecar["receipt_sha256"] == digest((earlier_sidecar_root / "result.json").read_bytes()),
            "The prior complete sidecar control changed")
    require(original["selection_sha256"] == raw["selection_sha256"] and len(raw["arms"]) == 8
            and len(selection["files"]) == len(ORDERS), "The two shard selections or arm count changed")
    results = {}
    for index, selected in enumerate(selection["files"]):
        name = selected["name"]
        earlier = original_trunk[name]
        require(publisher[name] == earlier["publisher_sha256"]
                and selected["selected_bytes"] == earlier["selected_bytes"],
                "The measured trunk extent differs from the pinned prior selection")
        records = raw["arms"][index * 4:index * 4 + 4]
        phases = []
        for trial, (mode, arm) in enumerate(zip(ORDERS[index], records, strict=True), 1):
            require((arm["shard"], arm["trial"], arm["mode"], arm["pressure"]) == (name, trial, mode, 1),
                    "The counterbalanced process order or memory pressure changed")
            verify_arm(arm, mode, SIDECAR_BYTES, True)
            require(identity(arm, "trunk") == identity(records[0], "trunk")
                    and identity(arm, "sidecar") == identity(raw["arms"][0], "sidecar")
                    and all(arm["trunk"][key] == earlier[key] for key in ("weights_sha256", "metal_sha256"))
                    and arm["sidecar"]["metal_sha256"] == earlier_sidecar["metal_sha256"]
                    and arm["sidecar"]["weights_sha256"] == earlier_sidecar_payload["payload_sha256"]
                    and arm["sidecar"]["selected_bytes"] == SIDECAR_BYTES
                    and arm["sidecar"]["tensors"] == len(sidecar["names"]),
                    "Owned buffers differ from another arm or an earlier exact control")
            for before, after in zip(arm["page_state_before"], arm["page_state_after"], strict=True):
                require(0 == before["resident_pages"]
                        < before["selected_pages"] == after["selected_pages"] == after["resident_pages"],
                        "The clone's virtual page observations do not close")
            phases.append({key: arm[key] for key in (
                "mode", "combined_seconds", "trunk_seconds", "sidecar_copy_seconds",
                "prefetch_seconds", "overlap_seconds", "join_wait_seconds",
            )})
        results[name] = phases
    return {"scope": raw["scope"], "hardware": raw["hardware"], "qualification": False,
            "exact_arrays": sum(raw_arm["trunk"]["tensors"] + raw_arm["sidecar"]["tensors"]
                                for raw_arm in raw["arms"]),
            "sidecar_bytes": SIDECAR_BYTES, "measured_arms": results}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
