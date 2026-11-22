"""Verify the matched native payload-descriptor experiment from sealed records."""

import gzip
from hashlib import sha256
import json
from pathlib import Path
import tarfile


HERE = Path(__file__).resolve().parent
INVENTORY = HERE.parent / "qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json"
ORDER = [["no-cache-first", "no-cache"], ["buffered-first", "buffered"],
         ["buffered-second", "buffered"], ["no-cache-second", "no-cache"]]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def replay(archive):
    def read(name):
        return archive.extractfile(name).read()

    def load(name):
        return json.loads(read(name))

    def lines(name):
        return [json.loads(line) for line in read(name).splitlines()]

    sealed_names = {item.name for item in archive if item.isfile()}
    verified_sources = 0
    for line in read("inputs.sha256").decode().splitlines():
        digest, name = line.split()
        if name in sealed_names:
            require(sha256(read(name)).hexdigest() == digest, f"Input changed: {name}")
            verified_sources += 1
    require(verified_sources == 10 and read("status.txt") == b"0\n", "Incomplete paired experiment")
    run = load("result.json")
    graph = gzip.decompress(read("graph.json.gz"))
    metadata = json.loads(graph)
    inventory = INVENTORY.read_bytes()
    pack = json.loads(inventory)
    inventory_digest = sha256(inventory).hexdigest()
    require(pack["revision"] == metadata["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
            and len(pack["members"]) == 49 and inventory_digest == metadata["inventory_sha256"],
            "The graph and publisher inventory differ")
    require(run["order"] == ORDER and run["checksum_mode"] == "no-cache"
            and run["native_manifest_sha256"] == "06cc592252dfe740b078519091995e706f2a3ae4b2e8b863720bea179d7d0e76",
            "The isolated arm order or pack profile changed")
    payload = {row["name"]: (row["size"], row["sha256"]) for row in pack["members"]
               if row["name"].endswith(".safetensors")}
    results = []
    for arm, mode in ORDER:
        def record(name):
            return load(f"{arm}/{name}.json")

        prelaunch, status, descriptors, reference = map(record, (
            "prelaunch", "status", "payload-descriptors", "reference-result"))
        bound = run["arms"][arm]
        require(prelaunch["sample"]["pressure"] == 1 and prelaunch["quiet"]["cpu_idle_percent"] >= 90,
                f"{arm}: ineligible prelaunch")
        require(bound["mode"] == status["payload_cache_mode"] == descriptors["mode"] == mode
                and bound["receipt"] == status and status["checksum_mode"] == "no-cache"
                and status["ordinary_qualification"] is False and status["wired_override"] is False
                and status["graph_sha256"] == sha256(graph).hexdigest(), f"{arm}: wrong controlled variables")
        checksums = lines(f"{arm}/checksum-events.jsonl")
        require(len(checksums) == bound["completed_checksums"] == len(payload) == 26
                and {row["member"]: (row["file_bytes"], row["sha256"]) for row in checksums} == payload
                and sum(row["file_bytes"] for row in checksums) == bound["checksummed_bytes"]
                and all(row["mode"] == "no-cache" and row["cached_before"] == row["cached_after"] == 0
                        for row in checksums), f"{arm}: missing or nonmatching publisher payload")
        require(descriptors["descriptors_modified"] == 25
                and descriptors["members"] == {name: 1 for name in payload if name != "mtp.safetensors"},
                f"{arm}: native descriptors differ from the publisher payloads")
        require(reference["reference"]["commit"] == metadata["reference_commit"]
                and reference["versions"] == metadata["versions"] and reference["versions"]["mlx"] == "0.32.2"
                and reference["hardware"]["soc"] == "Apple M5 Max"
                and reference["hardware"]["memory_bytes"] == 128 << 30
                and reference["inventory_sha256"] == inventory_digest and reference["phase"] == "loading"
                and reference["completed"] is False and reference["release_ready"] is False,
                f"{arm}: ordinary reference scope changed or reached first prefill")
        samples = lines(f"{arm}/samples.jsonl")
        phases = lines(f"{arm}/phase-events.jsonl")
        warning = samples[-1]
        starts = [row["batch"] for row in phases if row["phase"] == "batch_start"]
        ends = [row["batch"] for row in phases if row["phase"] == "batch_end"]
        require(starts == list(range(len(ends) + 1)) and ends == list(range(len(ends)))
                and all(row["pressure"] == 1 for row in samples[:-1]) and warning["pressure"] == 2
                and status["exit_code"] == -15 and status["stop_reason"] == "memory pressure above normal",
                f"{arm}: pressure boundary or batch progression changed")
        results.append({"arm": arm, "mode": mode, "ready": False, "batches_completed": len(ends),
                        "footprint_at_warning_bytes": warning["physical_footprint_bytes"],
                        "compressor_at_warning_bytes": warning["compressor_bytes"]})
    return results


if __name__ == "__main__":
    with tarfile.open(HERE / "evidence.tar.xz", "r:xz") as evidence:
        result = replay(evidence)
    require(result == json.loads((HERE / "comparison.json").read_bytes()), "Replayed comparison changed")
    print(json.dumps(result, indent=2, allow_nan=False))
