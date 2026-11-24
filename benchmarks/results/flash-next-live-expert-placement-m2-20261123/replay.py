#!/usr/bin/env python3
"""Replay the Flash Next startup placement control from its signed receipt."""

import hashlib
import json
from pathlib import Path
import tarfile


GIB = 2**30


def replay(archive):
    def get(name):
        file = archive.extractfile(name)
        assert file is not None
        return file.read()

    def read(name):
        return json.loads(get(name))

    def digest(name):
        return hashlib.sha256(get(name)).hexdigest()

    receipt = read("postflight.json")
    build = read("build.json")
    assert build == receipt["build"]
    assert build["codesign_verified"] is True and build["profile"] == "release-server"
    assert build["mlx"] == build["mlx_metal"] == "0.32.4"
    assert "1.99.1" in build["rustc"] and "1.99.1" in build["cargo"]
    assert digest("source.patch") == build["patch_sha256"]
    assert digest("inputs.sha256") == receipt["input_receipt_sha256"]
    sealed = dict(line.split(maxsplit=1)[::-1]
                  for line in get("inputs.sha256").decode().splitlines())
    for name in ("build.json", "source.patch", "mxfp4-inventory.json"):
        assert digest(name) == sealed[name]
    assert build["binary_sha256"] == sealed["ax-engine-server"]
    assert len(read("mxfp4-inventory.json")["members"]) == 49
    assert receipt["pack_manifest"]["model-manifest.json"] == build["manifest_sha256"]
    assert receipt["soc"] == "Apple M2 Ultra" and receipt["memory_bytes"] == 192 * GIB
    assert receipt["storage"] == {"BusProtocol": "USB", "Internal": False, "SolidState": True}
    assert receipt["qualification"] is False
    assert get("original-status.txt").strip() == b"1"

    arms = {}
    counts = {}
    for policy, streaming in (("auto", False), ("on", True)):
        observed = receipt["arms"][policy]
        assert digest(f"{policy}/memory.json") == observed["memory_sha256"]
        assert digest(f"{policy}/server-samples.jsonl") == observed["samples_sha256"]
        memory = read(f"{policy}/memory.json")
        assert memory["models"] == [{"id": "qwen3.8-flash-next",
                                     "expert_streaming_active": streaming}]
        samples = [json.loads(line) for line in get(f"{policy}/server-samples.jsonl").splitlines()]
        assert len(samples) == observed["pressure_samples"] and len(samples) > 0
        assert all(sample["pressure"] == 1 for sample in samples)
        peak = max(sample["footprint_bytes"] for sample in samples)
        assert peak == observed["peak_physical_bytes"] and peak <= 96 * GIB
        startup = read(f"{policy}/startup.json")
        assert startup["original_sha256"] == observed["startup_log_sha256"]
        rows = {name: int(count) for name, count in startup["resident_shards"]}
        assert len(rows) == len(startup["resident_shards"]) == 15
        assert rows == observed["loaded_tensors_by_shard"]
        assert sum(rows.values()) == observed["resident_tensor_count"]
        counts[policy] = rows
        arms[policy] = {"expert_streaming_active": streaming,
                        "resident_tensor_count": sum(rows.values()),
                        "sampled_peak_footprint_bytes": peak,
                        "normal_pressure_samples": len(samples)}

    auto, streamed = counts["auto"], counts["on"]
    assert auto.keys() == streamed.keys()
    assert all(streamed[name] <= auto[name] for name in auto)
    removed = sum(auto.values()) - sum(streamed.values())
    assert removed == 288
    return {"qualification": False, "arms": arms, "resident_tensors_removed": removed,
            "original_wrapper_exit": 1, "pack_members_revalidated": 49}


if __name__ == "__main__":
    with tarfile.open(Path(__file__).with_name("evidence.tar.xz")) as archive:
        print(json.dumps(replay(archive), indent=2, sort_keys=True))
